"""
Per-subject z-score normalization of HRV features.

For each subject, each HRV feature is standardized relative to that subject's
own distribution:  z = (x - subject_mean) / subject_std

This removes inter-subject baseline differences (e.g. naturally higher HR in
some subjects) so that the clustering in the next step reflects within-subject
state changes rather than between-subject physiology.

Raw columns are kept unchanged; normalized columns are added with _znorm suffix.
Subjects with std == 0 on any feature (constant signal) get znorm == 0 for that
feature — a warning is logged.

Output
------
outputs/hrv_windows_normalized.csv

Usage
-----
    python src/normalize_features.py
    python src/normalize_features.py --exclude-subjects drozy_s04
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import HRV_FEATURE_COLS, HRV_WINDOWS_CSV, HRV_WINDOWS_NORMALIZED_CSV, OUTPUTS_DIR

log = logging.getLogger(__name__)

ZNORM_COLS = [f"{c}_znorm" for c in HRV_FEATURE_COLS]


# ── normalization ─────────────────────────────────────────────────────────────

def normalize_per_subject(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add *_znorm columns in-place (returns a copy).
    Subjects with fewer than 2 windows cannot have a meaningful std;
    they receive znorm = 0.0 with a warning.
    """
    df = df.copy()
    for col, zcol in zip(HRV_FEATURE_COLS, ZNORM_COLS):
        df[zcol] = np.nan

    for subj, grp in df.groupby("subject_id"):
        idx = grp.index
        n   = len(grp)
        for col, zcol in zip(HRV_FEATURE_COLS, ZNORM_COLS):
            vals = grp[col].values.astype(float)
            mu   = vals.mean()
            sigma = vals.std(ddof=1) if n > 1 else 0.0
            if sigma == 0.0:
                if n > 1:
                    log.warning("[%s] Feature '%s' has zero std — znorm set to 0",
                                subj, col)
                df.loc[idx, zcol] = 0.0
            else:
                df.loc[idx, zcol] = (vals - mu) / sigma

    return df


# ── per-subject summary ───────────────────────────────────────────────────────

def print_subject_summary(df: pd.DataFrame) -> None:
    print("\n=== Per-Subject Feature Summary (raw values) ===")
    fmt_header = f"  {'Subject':15s} {'N':>4s}"
    for col in HRV_FEATURE_COLS:
        fmt_header += f"  {col[:10]:>12s}(mu/sd)"
    print(fmt_header)
    print("-" * (20 + 26 * len(HRV_FEATURE_COLS)))

    for subj, grp in df.sort_values("subject_id").groupby("subject_id"):
        row = f"  {str(subj):15s} {len(grp):>4d}"
        for col in HRV_FEATURE_COLS:
            mu = grp[col].mean()
            sd = grp[col].std(ddof=1) if len(grp) > 1 else 0.0
            row += f"  {mu:7.2f}/{sd:5.2f}"
        print(row)
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    log.info("=== Feature Normalization ===")

    if not HRV_WINDOWS_CSV.exists():
        raise FileNotFoundError(
            f"hrv_windows.csv not found at {HRV_WINDOWS_CSV}. "
            "Run extract_hrv_windows.py first."
        )

    df = pd.read_csv(HRV_WINDOWS_CSV)
    log.info("Loaded hrv_windows.csv: %d rows, %d subjects",
             len(df), df["subject_id"].nunique())

    # Drop rows missing HRV features
    before = len(df)
    df = df.dropna(subset=HRV_FEATURE_COLS).copy()
    if len(df) < before:
        log.warning("Dropped %d rows with NaN HRV features", before - len(df))

    print_subject_summary(df)

    df = normalize_per_subject(df)

    # Verify no NaN znorm columns remain
    bad = df[ZNORM_COLS].isnull().sum()
    if bad.any():
        log.error("NaN in znorm columns: %s", bad[bad > 0].to_dict())

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(HRV_WINDOWS_NORMALIZED_CSV, index=False)
    log.info("Normalized features saved -> %s  (%d rows)", HRV_WINDOWS_NORMALIZED_CSV, len(df))

    print(f"Output: {HRV_WINDOWS_NORMALIZED_CSV}")
    print(f"Columns: {list(df.columns)}")
    return df


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-subject z-score normalization of HRV features"
    )
    args = parser.parse_args()

    setup_logging()
    run()
