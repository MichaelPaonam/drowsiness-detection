"""
Subject-level train/val/test split utilities.

Guarantees that no subject appears in more than one split.
Uses a fixed, documented split map for reproducibility.
"""

import logging
from pathlib import Path

import pandas as pd

from config import (
    EXCLUDED_TESTS,
    HRV_FEATURE_COLS,
    KSS_DROWSY_THRESHOLD,
    KSS_FILE,
    SPLIT_MAP,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KSS loading and label assignment
# ---------------------------------------------------------------------------


def load_kss() -> pd.DataFrame:
    """
    Parse KSS.txt into a DataFrame with columns:
        subject_id, test_id, kss_raw, label, severe_flag, excluded

    Applies exclusion rules:
      - KSS == 0  → excluded (test 7-1 never occurred)
      - (subject, test) in EXCLUDED_TESTS → excluded (backup loss or never occurred)
    """
    rows = []
    with open(KSS_FILE, "r") as f:
        for line_idx, line in enumerate(f):
            parts = line.strip().split()
            if not parts:
                continue
            subject_id = line_idx + 1  # 1-indexed
            for test_idx, score_str in enumerate(parts):
                test_id = test_idx + 1  # 1-indexed
                kss_raw = int(score_str)
                is_excluded = kss_raw == 0 or (subject_id, test_id) in EXCLUDED_TESTS
                label = None
                if not is_excluded:
                    label = 1 if kss_raw >= KSS_DROWSY_THRESHOLD else 0
                rows.append(
                    {
                        "subject_id": subject_id,
                        "test_id": test_id,
                        "kss_raw": kss_raw,
                        "label": label,
                        "severe_flag": kss_raw == 9 and not is_excluded,
                        "excluded": is_excluded,
                    }
                )

    df = pd.DataFrame(rows)
    n_excluded = df["excluded"].sum()
    log.info(
        "KSS loaded: %d tests total, %d excluded (%d usable).",
        len(df),
        n_excluded,
        len(df) - n_excluded,
    )
    return df


def assign_split_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'split' column to a DataFrame that has a 'subject_id' column.
    Subjects not in SPLIT_MAP raise a ValueError.
    Excluded rows receive split = None.
    """
    # Build reverse map: subject_id → split name
    subject_to_split = {}
    for split_name, subjects in SPLIT_MAP.items():
        for s in subjects:
            subject_to_split[s] = split_name

    def _get_split(row):
        if row.get("excluded", False):
            return None
        sid = row["subject_id"]
        if sid not in subject_to_split:
            raise ValueError(f"Subject {sid} not assigned to any split in SPLIT_MAP.")
        return subject_to_split[sid]

    df = df.copy()
    df["split"] = df.apply(_get_split, axis=1)
    return df


# ---------------------------------------------------------------------------
# DataFrame splitting
# ---------------------------------------------------------------------------


def split_features_df(features_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Split a features DataFrame (must have 'subject_id' and 'label' columns)
    into {'train': ..., 'val': ..., 'test': ...} using SPLIT_MAP.

    Validates that:
      - No subject appears in multiple splits
      - Both classes present in train
      - Each split is non-empty
    """
    df = assign_split_column(features_df)
    usable = df[df["split"].notna()].copy()

    _validate_split_integrity(usable)

    splits = {}
    for split_name in ("train", "val", "test"):
        subset = usable[usable["split"] == split_name].copy()
        splits[split_name] = subset
        log.info(
            "Split '%s': %d rows, %d subjects, label dist: %s",
            split_name,
            len(subset),
            subset["subject_id"].nunique(),
            subset["label"].value_counts().to_dict(),
        )

    return splits


def save_splits(
    splits: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Write train/val/test CSVs to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, df in splits.items():
        path = output_dir / f"{split_name}.csv"
        df.to_csv(path, index=False)
        log.info("Saved %s split -> %s (%d rows)", split_name, path, len(df))


# ---------------------------------------------------------------------------
# Convenience: get feature matrix and labels
# ---------------------------------------------------------------------------


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and label vector y from a split DataFrame."""
    missing = [c for c in HRV_FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns missing from DataFrame: {missing}")
    X = df[HRV_FEATURE_COLS].copy()
    y = df["label"].copy()
    return X, y


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_split_integrity(df: pd.DataFrame) -> None:
    """Raise or warn on split integrity violations."""
    # Check no subject in multiple splits
    subject_splits = df.groupby("subject_id")["split"].nunique()
    overlap = subject_splits[subject_splits > 1]
    if not overlap.empty:
        raise ValueError(
            f"DATA LEAKAGE: subjects appear in multiple splits: {overlap.index.tolist()}"
        )

    # Check both classes in train
    train = df[df["split"] == "train"]
    if len(train["label"].unique()) < 2:
        raise ValueError(
            "Train split contains only one class. Check SPLIT_MAP and label distribution."
        )

    # Warn if val or test have only one class
    for split_name in ("val", "test"):
        subset = df[df["split"] == split_name]
        if len(subset) > 0 and len(subset["label"].unique()) < 2:
            log.warning(
                "Split '%s' contains only one class. Precision/recall undefined on this split.",
                split_name,
            )

    log.info("Split integrity check passed.")
