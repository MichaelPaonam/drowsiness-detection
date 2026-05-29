"""
Extract raw ECG signals from DROZY and DDD EDF files to CSV.

Output layout:
  outputs/ecg_csv/drozy/SUBJECT-TEST.csv   columns: timestamp_sec, ecg
  outputs/ecg_csv/ddd/VOLUNTEER_TRIAL.csv  columns: timestamp_sec, ecg

ECG channel discovery is flexible: exact match on "ECG" / "EKG", then
case-insensitive substring match, so it handles both datasets transparently.

Usage
-----
    python src/extract_ecg.py
    python src/extract_ecg.py --dataset drozy
    python src/extract_ecg.py --dataset ddd --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyedflib

sys.path.insert(0, str(Path(__file__).parent))

# pylint: disable=wrong-import-position
from config import DDD_DIR, DROZY_ROOT, ECG_CSV_DIR

log = logging.getLogger(__name__)

PSG_DIR = DROZY_ROOT / "psg"


# ── data container ────────────────────────────────────────────────────────────


class EdfInfo(NamedTuple):
    """Container for EDF file metadata and parsing info."""

    path: Path
    dataset: str  # "drozy" or "ddd"
    stem: str  # output filename stem, e.g. "1-1" or "01M_1"
    subject_id: str  # unified: "drozy_s01" or "ddd_01M"
    trial_id: int


# ── EDF discovery ─────────────────────────────────────────────────────────────


def discover_drozy_edfs() -> list[EdfInfo]:
    """Discover and parse DROZY EDF files from PSG directory."""
    if not PSG_DIR.exists():
        log.warning("DROZY PSG directory not found: %s", PSG_DIR)
        return []
    infos = []
    for edf_path in sorted(PSG_DIR.glob("*.edf")):
        stem = edf_path.stem
        parts = stem.split("-")
        if len(parts) != 2:
            log.warning("Skipping unrecognised DROZY filename: %s", edf_path.name)
            continue
        try:
            subj_int = int(parts[0])
            trial_int = int(parts[1])
        except ValueError:
            log.warning("Cannot parse subject/test from: %s", edf_path.name)
            continue
        infos.append(
            EdfInfo(
                path=edf_path,
                dataset="drozy",
                stem=stem,
                subject_id=f"drozy_s{subj_int:02d}",
                trial_id=trial_int,
            )
        )
    return infos


def discover_ddd_edfs() -> list[EdfInfo]:
    """Discover and parse DDD EDF files from DDD directory."""
    if not DDD_DIR.exists():
        log.warning("DDD directory not found: %s", DDD_DIR)
        return []
    infos = []
    for edf_path in sorted(DDD_DIR.glob("*.edf")):
        if "_annotations" in edf_path.name:
            continue
        stem = edf_path.stem  # e.g. "01M_1"
        parts = stem.split("_")
        if len(parts) != 2:
            log.warning("Skipping unrecognised DDD filename: %s", edf_path.name)
            continue
        volunteer_id = parts[0]  # "01M"
        try:
            trial_int = int(parts[1])
        except ValueError:
            log.warning("Cannot parse trial from: %s", edf_path.name)
            continue
        infos.append(
            EdfInfo(
                path=edf_path,
                dataset="ddd",
                stem=stem,
                subject_id=f"ddd_{volunteer_id}",
                trial_id=trial_int,
            )
        )
    return infos


# ── ECG channel search ────────────────────────────────────────────────────────


def find_ecg_channel_index(labels: list[str]) -> int | None:
    """
    Return the index of the ECG channel in `labels`.

    Priority:
      1. Exact match: "ECG" or "EKG" (stripped, any case)
      2. Substring match: label contains "ecg" or "ekg" (case-insensitive)
    Returns None if not found.
    """
    clean = [label.strip() for label in labels]
    # Exact match first
    for exact in ("ECG", "EKG", "ecg", "ekg"):
        if exact in clean:
            return clean.index(exact)
    # Substring match
    for i, label in enumerate(clean):
        low = label.lower()
        if "ecg" in low or "ekg" in low:
            return i
    return None


# ── extraction ────────────────────────────────────────────────────────────────


def extract_and_save(info: EdfInfo, out_dir: Path) -> bool:
    """
    Open the EDF, find the ECG channel, extract raw signal, save to CSV.
    Returns True on success.
    """
    out_path = out_dir / f"{info.stem}.csv"
    if out_path.exists():
        log.info("  [skip] already exists: %s", out_path.name)
        return True

    try:
        with pyedflib.EdfReader(str(info.path)) as edf:
            labels = [label.strip() for label in edf.getSignalLabels()]
            ecg_idx = find_ecg_channel_index(labels)

            if ecg_idx is None:
                log.warning("  [%s] No ECG/EKG channel found. Available: %s", info.stem, labels)
                return False

            fs = float(edf.getSampleFrequency(ecg_idx))
            ecg_raw = edf.readSignal(ecg_idx).astype(np.float32)
            duration_sec = len(ecg_raw) / fs

            log.info(
                "  [%s] channel='%s'  fs=%.0f Hz  duration=%.1f s  samples=%d",
                info.stem,
                labels[ecg_idx],
                fs,
                duration_sec,
                len(ecg_raw),
            )

        # Build CSV — timestamp_sec is exact (float64) but ecg is float32
        n = len(ecg_raw)
        timestamps = (np.arange(n) / fs).astype(np.float64)
        df = pd.DataFrame(
            {
                "timestamp_sec": timestamps,
                "ecg": ecg_raw,
            }
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, float_format="%.7g")
        log.info("  -> saved %s  (%.1f MB)", out_path.name, out_path.stat().st_size / 1e6)
        return True

    except (OSError, ValueError) as exc:
        log.error("  [%s] failed: %s", info.stem, exc)
        return False


# ── main ──────────────────────────────────────────────────────────────────────


def run(dataset: str = "all") -> None:
    """Run ECG extraction pipeline for specified dataset(s)."""
    infos: list[EdfInfo] = []
    if dataset in ("drozy", "all"):
        infos += discover_drozy_edfs()
    if dataset in ("ddd", "all"):
        infos += discover_ddd_edfs()

    log.info("=== ECG Extraction ===")
    log.info("Files found: %d  (dataset=%s)", len(infos), dataset)

    n_ok = n_fail = 0
    for info in infos:
        out_dir = ECG_CSV_DIR / info.dataset
        log.info("[%s] %s", info.dataset.upper(), info.path.name)
        ok = extract_and_save(info, out_dir)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    drozy_dir = ECG_CSV_DIR / "drozy"
    ddd_dir = ECG_CSV_DIR / "ddd"
    nd = len(list(drozy_dir.glob("*.csv"))) if drozy_dir.exists() else 0
    nw = len(list(ddd_dir.glob("*.csv"))) if ddd_dir.exists() else 0

    print("\n=== Summary ===")
    print(f"  Files found   : {len(infos)}")
    print(f"  Processed OK  : {n_ok}")
    print(f"  Failed        : {n_fail}")
    print(f"  DROZY CSVs    : {nd}")
    print(f"  DDD CSVs      : {nw}")
    print()


def setup_logging() -> None:
    """Configure logging for console output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ECG signals from EDF files to CSV")
    parser.add_argument(
        "--dataset",
        choices=["drozy", "ddd", "all"],
        default="all",
        help="Which dataset(s) to process (default: all)",
    )
    args = parser.parse_args()
    setup_logging()
    run(dataset=args.dataset)
