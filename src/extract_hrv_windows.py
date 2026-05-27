"""
Windowed HRV feature extraction from ECG CSV files.

Loads each ECG CSV produced by extract_ecg.py, applies a bandpass filter to
the full signal, then slides a window across the recording and extracts
time-domain HRV features for each window using the same R-peak detection and
artifact-rejection logic as the existing DROZY pipeline.

Output
------
outputs/hrv_windows.csv — one row per window across all subjects and datasets

Subject ID format (prevents collisions when combining datasets):
  DROZY : drozy_s01 … drozy_s14
  DDD   : ddd_01M  … ddd_10M

Usage
-----
    python src/extract_hrv_windows.py
    python src/extract_hrv_windows.py --dataset drozy --window-sec 300 --step-sec 60
    python src/extract_hrv_windows.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    ECG_CSV_DIR,
    HRV_WINDOWS_CSV,
    OUTPUTS_DIR,
    FILTER_LOWCUT,
    FILTER_HIGHCUT,
    FILTER_ORDER,
    WINDOW_SEC,
    STEP_SEC,
)
from hrv_features import extract_hrv_features

log = logging.getLogger(__name__)


# ── bandpass filter ───────────────────────────────────────────────────────────

def bandpass_filter(signal: np.ndarray, fs: float) -> np.ndarray:
    """Zero-phase Butterworth bandpass using config cutoffs."""
    sos = butter(
        FILTER_ORDER,
        [FILTER_LOWCUT, FILTER_HIGHCUT],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    return sosfiltfilt(sos, signal)


# ── file discovery ────────────────────────────────────────────────────────────

def _parse_drozy_stem(stem: str) -> tuple[str, int]:
    """'1-1' → ('drozy_s01', 1)"""
    subj, trial = stem.split("-")
    return f"drozy_s{int(subj):02d}", int(trial)


def _parse_ddd_stem(stem: str) -> tuple[str, int]:
    """'01M_1' → ('ddd_01M', 1)"""
    volunteer, trial = stem.split("_")
    return f"ddd_{volunteer}", int(trial)


def discover_csv_files(dataset: str) -> list[tuple[Path, str, int, str]]:
    """
    Return list of (csv_path, subject_id, trial_id, dataset_tag) tuples.
    """
    entries = []
    if dataset in ("drozy", "all"):
        d = ECG_CSV_DIR / "drozy"
        if d.exists():
            for p in sorted(d.glob("*.csv")):
                try:
                    subj_id, trial_id = _parse_drozy_stem(p.stem)
                    entries.append((p, subj_id, trial_id, "drozy"))
                except Exception:
                    log.warning("Skipping unrecognised DROZY CSV: %s", p.name)
    if dataset in ("ddd", "all"):
        d = ECG_CSV_DIR / "ddd"
        if d.exists():
            for p in sorted(d.glob("*.csv")):
                try:
                    subj_id, trial_id = _parse_ddd_stem(p.stem)
                    entries.append((p, subj_id, trial_id, "ddd"))
                except Exception:
                    log.warning("Skipping unrecognised DDD CSV: %s", p.name)
    return entries


# ── per-file windowed extraction ──────────────────────────────────────────────

def process_file(
    csv_path: Path,
    subject_id: str,
    trial_id: int,
    dataset: str,
    window_sec: int,
    step_sec: int,
) -> list[dict]:
    """
    Load one ECG CSV, filter it, slide windows, extract HRV per window.
    Returns a list of row dicts (may be empty on failure).
    """
    log.info("[%s] Loading %s", subject_id, csv_path.name)

    try:
        df_ecg = pd.read_csv(csv_path, dtype={"timestamp_sec": float, "ecg": float})
    except Exception as exc:
        log.error("[%s] Failed to load CSV: %s", subject_id, exc)
        return []

    if "timestamp_sec" not in df_ecg.columns or "ecg" not in df_ecg.columns:
        log.error("[%s] CSV missing required columns", subject_id)
        return []

    timestamps = df_ecg["timestamp_sec"].values
    ecg_raw    = df_ecg["ecg"].values.astype(np.float64)

    # Infer sampling rate from timestamps
    if len(timestamps) < 2:
        log.warning("[%s] Too few samples (%d)", subject_id, len(timestamps))
        return []

    dt = timestamps[1] - timestamps[0]
    if dt <= 0:
        log.error("[%s] Non-positive timestamp delta: %.6f", subject_id, dt)
        return []

    fs = round(1.0 / dt)  # round to nearest integer Hz
    total_sec = len(ecg_raw) / fs
    log.info("[%s] fs=%d Hz  duration=%.1f s  samples=%d",
             subject_id, fs, total_sec, len(ecg_raw))

    # Filter the full signal once to avoid edge effects at window boundaries
    try:
        ecg_filt = bandpass_filter(ecg_raw, fs)
    except Exception as exc:
        log.error("[%s] Bandpass filter failed: %s", subject_id, exc)
        return []

    window_samples = int(window_sec * fs)
    step_samples   = int(step_sec * fs)
    n_samples      = len(ecg_filt)

    rows   = []
    w_idx  = 0
    n_skip = 0

    start = 0
    while start + window_samples <= n_samples:
        end   = start + window_samples
        seg   = ecg_filt[start:end]
        t_start = float(timestamps[start])
        t_end   = float(timestamps[end - 1])

        features = extract_hrv_features(seg, fs=fs)

        if features is None:
            log.debug("[%s] Window %d skipped (insufficient clean RR intervals)",
                      subject_id, w_idx)
            n_skip += 1
        else:
            row = {
                "dataset":          dataset,
                "subject_id":       subject_id,
                "trial_id":         trial_id,
                "window_index":     w_idx,
                "window_start_sec": round(t_start, 3),
                "window_end_sec":   round(t_end, 3),
                "sampling_rate":    fs,
                **features.to_feature_dict(),
            }
            rows.append(row)

        w_idx += 1
        start += step_samples

    log.info("[%s] Windows extracted=%d  skipped=%d", subject_id, len(rows), n_skip)
    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def run(
    dataset: str = "all",
    window_sec: int = WINDOW_SEC,
    step_sec: int = STEP_SEC,
) -> pd.DataFrame:
    log.info("=== Windowed HRV Extraction ===")
    log.info("Window=%ds  Step=%ds  Dataset=%s", window_sec, step_sec, dataset)

    files = discover_csv_files(dataset)
    if not files:
        log.error("No ECG CSV files found. Run extract_ecg.py first.")
        return pd.DataFrame()

    log.info("Found %d ECG CSV files", len(files))

    all_rows: list[dict] = []
    stats: dict[str, dict] = {}

    for csv_path, subject_id, trial_id, ds_tag in files:
        rows = process_file(
            csv_path, subject_id, trial_id, ds_tag,
            window_sec=window_sec, step_sec=step_sec,
        )
        all_rows.extend(rows)
        if ds_tag not in stats:
            stats[ds_tag] = {"windows": 0, "subjects": set()}
        stats[ds_tag]["windows"] += len(rows)
        stats[ds_tag]["subjects"].add(subject_id)

    if not all_rows:
        log.error("No windows extracted. Check ECG CSV files and window parameters.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(HRV_WINDOWS_CSV, index=False)
    log.info("HRV windows saved -> %s  (%d rows)", HRV_WINDOWS_CSV, len(df))

    print("\n=== Extraction Summary ===")
    for ds, s in stats.items():
        n_subj = len(s["subjects"])
        print(f"  {ds.upper():6s}: {s['windows']:5d} windows  {n_subj} subjects")
    print(f"  TOTAL : {len(df):5d} windows")
    print(f"  Output: {HRV_WINDOWS_CSV}")

    per_subj = df.groupby("subject_id")["window_index"].count().rename("n_windows")
    print("\nPer-subject window counts:")
    print(per_subj.to_string())
    print()

    return df


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract windowed HRV features from ECG CSVs"
    )
    parser.add_argument("--dataset", choices=["drozy", "ddd", "all"], default="all")
    parser.add_argument("--window-sec", type=int, default=WINDOW_SEC,
                        help=f"Window size in seconds (default: {WINDOW_SEC})")
    parser.add_argument("--step-sec", type=int, default=STEP_SEC,
                        help=f"Step size in seconds (default: {STEP_SEC})")
    args = parser.parse_args()

    setup_logging()
    run(
        dataset=args.dataset,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
    )
