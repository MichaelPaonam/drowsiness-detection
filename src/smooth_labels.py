"""
Temporal smoothing of pseudo-KSS labels.

Within each subject's recording (same subject_id + trial_id), applies a median
filter to the pseudo_kss sequence to remove isolated label flips.

Optional --enforce-monotonic flag applies a cumulative maximum within each
session, preventing drowsiness from "recovering" mid-session. This reflects the
typical experimental design where participants become progressively more drowsy.

Output
------
outputs/hrv_windows_final.csv
outputs/pseudo_kss/fig_temporal_smoothing.png

Usage
-----
    python src/smooth_labels.py
    python src/smooth_labels.py --kernel-size 5 --enforce-monotonic
    python src/smooth_labels.py --method gmm
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import medfilt

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    HRV_WINDOWS_FINAL_CSV,
    HRV_WINDOWS_PSEUDO_CSV,
    KSS_DROWSY_THRESHOLD,
    OUTPUTS_DIR,
    PSEUDO_KSS_DIR,
)

log = logging.getLogger(__name__)


# ── smoothing logic ───────────────────────────────────────────────────────────


def _median_smooth(series: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Median filter with the given kernel size.
    medfilt uses 'reflect' padding at edges, so boundary windows are handled.
    Kernel size is forced to be odd (rounds up).
    """
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    smoothed = medfilt(series.astype(float), kernel_size=k)
    # medfilt can output floats; round back to nearest valid pseudo-KSS level
    valid_levels = np.array([2, 5, 8])
    rounded = np.array([valid_levels[np.argmin(np.abs(valid_levels - v))] for v in smoothed])
    return rounded


def _monotonic_constraint(series: np.ndarray) -> np.ndarray:
    """Apply cumulative maximum: drowsiness can only stay same or increase."""
    return np.maximum.accumulate(series)


def smooth_session(
    kss_seq: np.ndarray,
    kernel_size: int,
    enforce_monotonic: bool,
) -> np.ndarray:
    """Apply median filtering and optional monotonic constraint to a KSS sequence."""
    smoothed = _median_smooth(kss_seq, kernel_size)
    if enforce_monotonic:
        smoothed = _monotonic_constraint(smoothed)
    return smoothed


# ── temporal smoothing figure ─────────────────────────────────────────────────


def fig_temporal_smoothing(df: pd.DataFrame, n_examples: int = 4) -> None:
    """Generate and save a figure showing raw vs. smoothed pseudo-KSS for example sessions."""
    PSEUDO_KSS_DIR.mkdir(parents=True, exist_ok=True)

    # Pick subjects with the most windows for clarity
    session_counts = cast(pd.Series, df.groupby(["subject_id", "trial_id"]).size())
    ordered_sessions = session_counts.sort_values(ascending=False).head(n_examples)
    top_sessions = cast(list[tuple[object, object]], list(ordered_sessions.index))

    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 3 * n_examples), sharex=False)
    if n_examples == 1:
        axes = [axes]

    for ax, (subj, trial) in zip(axes, top_sessions):
        sub = cast(
            pd.DataFrame,
            df.loc[(df["subject_id"] == subj) & (df["trial_id"] == trial)],
        ).copy()
        sub = sub.sort_values(by="window_index")
        x = np.asarray(sub["window_start_sec"].values, dtype=np.float64) / 60.0

        ax.plot(
            x,
            sub["pseudo_kss"].values,
            "o--",
            color="#4C9BE8",
            linewidth=1.2,
            markersize=4,
            label="Raw pseudo-KSS",
            alpha=0.8,
        )
        ax.plot(
            x,
            sub["pseudo_kss_smoothed"].values,
            "s-",
            color="#E8614C",
            linewidth=1.8,
            markersize=5,
            label="Smoothed",
            alpha=0.9,
        )
        ax.set_yticks([2, 5, 8])
        ax.set_yticklabels(["2 (Alert)", "5 (Trans.)", "8 (Drowsy)"], fontsize=8)
        ax.set_xlabel("Time (minutes)")
        ax.set_title(f"{subj} — Trial {trial}", fontweight="bold", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_ylim(0, 10)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Temporal Smoothing of Pseudo-KSS Labels", fontweight="bold")
    fig.tight_layout()
    out = PSEUDO_KSS_DIR / "fig_temporal_smoothing.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved -> %s", out)


# ── main ──────────────────────────────────────────────────────────────────────


def run(
    method: str = "kmeans",
    kernel_size: int = 3,
    enforce_monotonic: bool = False,
) -> pd.DataFrame:
    """Run temporal smoothing of pseudo-KSS labels and save results to CSV."""
    log.info("=== Temporal Smoothing ===")
    log.info(
        "method=%s  kernel_size=%d  enforce_monotonic=%s", method, kernel_size, enforce_monotonic
    )

    if not HRV_WINDOWS_PSEUDO_CSV.exists():
        raise FileNotFoundError(
            f"Pseudo-labeled CSV not found: {HRV_WINDOWS_PSEUDO_CSV}. "
            "Run generate_pseudo_kss.py first."
        )

    df = pd.read_csv(HRV_WINDOWS_PSEUDO_CSV)
    log.info("Loaded: %d rows, %d subjects", len(df), df["subject_id"].nunique())

    # Select the source column for smoothing
    src_col = f"pseudo_kss_{method}"
    if src_col not in df.columns:
        src_col = "pseudo_kss"
        log.warning("Column '%s' not found; falling back to 'pseudo_kss'", f"pseudo_kss_{method}")
    log.info("Smoothing column: %s", src_col)

    df = df.sort_values(["subject_id", "trial_id", "window_index"]).copy()
    df["pseudo_kss_smoothed"] = df[src_col].copy()

    changed_total = 0
    for session_key, grp in df.groupby(["subject_id", "trial_id"]):
        subj, trial = cast(tuple[object, object], session_key)
        idx = grp.index
        raw = np.asarray(grp[src_col].values, dtype=np.float64)
        smoothed = smooth_session(raw, kernel_size, enforce_monotonic)
        df.loc[idx, "pseudo_kss_smoothed"] = smoothed
        changed = int((smoothed != raw).sum())
        if changed:
            log.debug("[%s t%s] %d/%d windows changed label", subj, trial, changed, len(raw))
        changed_total += changed

    df["pseudo_label_smoothed"] = (df["pseudo_kss_smoothed"] >= KSS_DROWSY_THRESHOLD).astype(int)

    # Summary
    total = len(df)
    pct = changed_total / total * 100 if total > 0 else 0.0
    print("\n=== Smoothing Summary ===")
    print(f"  Windows total     : {total}")
    print(f"  Windows changed   : {changed_total}  ({pct:.1f}%)")

    for ds, grp in df.groupby("dataset"):
        smoothed_values = grp["pseudo_kss_smoothed"].values.astype(float)
        source_values = grp[src_col].values.astype(float)
        ds_changed = int((smoothed_values != source_values).astype(int).sum())
        print(f"  {str(ds).upper():6s} changed: {ds_changed}/{len(grp)}")

    label_dist = df["pseudo_label_smoothed"].value_counts().sort_index()
    print("\n  Smoothed label distribution:")
    print(f"    Alert  (0): {label_dist.get(0, 0)}")
    print(f"    Drowsy (1): {label_dist.get(1, 0)}")
    print()

    # Figure
    fig_temporal_smoothing(df, n_examples=min(4, df.groupby(["subject_id", "trial_id"]).ngroups))

    # Save
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(HRV_WINDOWS_FINAL_CSV, index=False)
    log.info("Final CSV saved -> %s  (%d rows)", HRV_WINDOWS_FINAL_CSV, len(df))
    return df


def setup_logging() -> None:
    """Configure logging for console output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal smoothing of pseudo-KSS labels")
    parser.add_argument(
        "--method",
        choices=["kmeans", "gmm"],
        default="kmeans",
        help="Which clustering method's pseudo_kss to smooth (default: kmeans)",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=3,
        help="Median filter kernel size in windows (default: 3)",
    )
    parser.add_argument(
        "--enforce-monotonic",
        action="store_true",
        help="Apply cumulative-max constraint (drowsiness only increases)",
    )
    args = parser.parse_args()

    setup_logging()
    run(
        method=args.method,
        kernel_size=args.kernel_size,
        enforce_monotonic=args.enforce_monotonic,
    )
