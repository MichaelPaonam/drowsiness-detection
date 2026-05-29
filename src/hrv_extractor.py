"""
Windowed HRV feature extraction pipeline.

Processes ECG CSV files to extract time-domain HRV features over sliding windows.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from config import (
    ECG_CSV_DIR,
    FILTER_HIGHCUT,
    FILTER_LOWCUT,
    FILTER_ORDER,
    HRV_WINDOWS_CSV,
    OUTPUTS_DIR,
    STEP_SEC,
    WINDOW_SEC,
)
from hrv_metrics import compute_window_hrv

log = logging.getLogger(__name__)


def apply_bandpass_filter(signal: np.ndarray, fs: float) -> np.ndarray:
    """Applies a zero-phase Butterworth bandpass filter."""
    sos = butter(
        FILTER_ORDER,
        [FILTER_LOWCUT, FILTER_HIGHCUT],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    return sosfiltfilt(sos, signal)


def get_ecg_files(dataset: str) -> List[Tuple[Path, str, int, str]]:
    """Discovers and identifies ECG CSV files."""
    entries = []

    # DROZY parsing
    if dataset in ("drozy", "all"):
        path = ECG_CSV_DIR / "drozy"
        if path.exists():
            for p in sorted(path.glob("*.csv")):
                try:
                    subj, trial = p.stem.split("-")
                    entries.append((p, f"drozy_s{int(subj):02d}", int(trial), "drozy"))
                except ValueError:
                    log.warning("Skipping malformed DROZY CSV: %s", p.name)

    # DDD parsing
    if dataset in ("ddd", "all"):
        path = ECG_CSV_DIR / "ddd"
        if path.exists():
            for p in sorted(path.glob("*.csv")):
                try:
                    volunteer, trial = p.stem.split("_")
                    entries.append((p, f"ddd_{volunteer}", int(trial), "ddd"))
                except ValueError:
                    log.warning("Skipping malformed DDD CSV: %s", p.name)

    return entries


# pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
def process_ecg_file(
    csv_path: Path,
    subject_id: str,
    trial_id: int,
    dataset: str,
    window_sec: int,
    step_sec: int,
) -> List[Dict]:
    """Processes a single ECG file, extracting features per window."""
    try:
        df = pd.read_csv(csv_path, dtype={"timestamp_sec": float, "ecg": float})
    except (OSError, ValueError, KeyError) as exc:
        log.error("[%s] Failed to load %s: %s", subject_id, csv_path.name, exc)
        return []

    timestamps = df["timestamp_sec"].values
    ecg_raw = df["ecg"].values.astype(np.float64)

    if len(timestamps) < 2 or (dt := timestamps[1] - timestamps[0]) <= 0:
        log.warning("[%s] Invalid timestamps", subject_id)
        return []

    fs = round(1.0 / dt)
    ecg_filt = apply_bandpass_filter(ecg_raw, fs)

    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)
    n_samples = len(ecg_filt)

    rows = []
    w_idx = 0
    start = 0
    while start + window_samples <= n_samples:
        end = start + window_samples

        features = compute_window_hrv(
            ecg_filt[start:end],
            fs=fs,
            subject_id=subject_id,
            trial_id=trial_id
        )

        if features:
            row = {
                "dataset": dataset,
                "subject_id": subject_id,
                "trial_id": trial_id,
                "window_index": w_idx,
                "window_start_sec": round(float(timestamps[start]), 3),
                "window_end_sec": round(float(timestamps[end - 1]), 3),
                "sampling_rate": fs,
                **features.to_dict(),
            }
            rows.append(row)

        w_idx += 1
        start += step_samples

    return rows


def run_hrv_extraction(
    dataset: str = "all",
    window_sec: int = WINDOW_SEC,
    step_sec: int = STEP_SEC,
) -> pd.DataFrame:
    """Runs the full HRV extraction pipeline."""
    files = get_ecg_files(dataset)
    if not files:
        log.error("No files found.")
        return pd.DataFrame()

    all_rows = []
    for csv_path, subj, trial, ds_tag in files:
        all_rows.extend(
            process_ecg_file(csv_path, subj, trial, ds_tag, window_sec, step_sec)
        )

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(HRV_WINDOWS_CSV, index=False)
    log.info("Saved %d windows to %s", len(df), HRV_WINDOWS_CSV)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract windowed HRV features")
    parser.add_argument("--dataset", choices=["drozy", "ddd", "all"], default="all")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_hrv_extraction(dataset=args.dataset)
