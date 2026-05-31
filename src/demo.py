"""Drowsiness Detection Demo Script.

Loads a sample ECG recording, runs the inference pipeline, and saves
visualisations to outputs/demo/.

Usage
-----
    python src/demo.py
    python src/demo.py --ecg-file data/preprocessed/ecg_csv/drozy/1-1.csv
    python src/demo.py --ecg-file data/preprocessed/ecg_csv/ddd/01M_1.csv --sampling-rate 128
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from inference import DrowsinessPredictor

from config import ECG_CSV_DIR, OUTPUTS_DIR, SELECTED_FEATURES

log = logging.getLogger(__name__)

_MODEL_PATH = OUTPUTS_DIR / "models" / "best_model.json"
_SCALER_PATH = OUTPUTS_DIR / "models" / "scaler.pkl"
_DEFAULT_ECG = ECG_CSV_DIR / "drozy" / "1-1.csv"
_DEMO_DIR = OUTPUTS_DIR / "demo"
_FIG_DPI = 150
_THRESHOLD = 0.5
_COLOR_ALERT = "#2ca02c"
_COLOR_DROWSY = "#d62728"

_FEATURE_LABELS = ["SDNN (z)", "CV_RR (z)", "RMSSD (z)", "pNN50 (z)"]
_FEATURE_LINE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with ecg_file (Path | None) and sampling_rate (float | None).
    """
    parser = argparse.ArgumentParser(description="ECG-based drowsiness detection demo")
    parser.add_argument(
        "--ecg-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="ECG CSV file to analyse (default: ECG_CSV_DIR/drozy/1-1.csv)",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=None,
        metavar="HZ",
        help="ECG sampling rate in Hz (auto-detected from timestamps if omitted)",
    )
    return parser.parse_args()


def _infer_sampling_rate(timestamps: np.ndarray) -> float:
    """Estimate sampling rate in Hz from the first up to 100 inter-sample deltas.

    Args:
        timestamps: Recording timestamps in seconds.

    Returns:
        Rounded sampling rate estimate in Hz.
    """
    n = min(100, len(timestamps))
    dts = np.diff(timestamps[:n])
    median_dt = float(np.median(dts[dts > 0]))
    return round(1.0 / median_dt)


def _load_ecg(
    ecg_file: Path,
    sampling_rate: float | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load an ECG CSV and return signal arrays plus the resolved sampling rate.

    Args:
        ecg_file: Path to CSV with columns timestamp_sec and ecg.
        sampling_rate: Sampling rate in Hz, or None to auto-detect from timestamps.

    Returns:
        Tuple of (timestamps_sec, ecg_signal, sampling_rate_hz).

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    if not ecg_file.exists():
        raise FileNotFoundError(f"ECG file not found: {ecg_file}")

    df = pd.read_csv(ecg_file, dtype={"timestamp_sec": float, "ecg": float})
    timestamps = df["timestamp_sec"].values
    ecg_signal = df["ecg"].values.astype(np.float64)

    fs = sampling_rate if sampling_rate is not None else _infer_sampling_rate(timestamps)
    log.info(
        "Loaded %s — %d samples, %.1f min, %.0f Hz",
        ecg_file.name,
        len(ecg_signal),
        float(timestamps[-1]) / 60.0,
        fs,
    )
    return timestamps, ecg_signal, float(fs)


def plot_ecg_signal(
    timestamps: np.ndarray,
    ecg_signal: np.ndarray,
    out_path: Path,
) -> None:
    """Save a plot of the first 30 seconds of the raw ECG signal.

    Args:
        timestamps: Full recording timestamps in seconds.
        ecg_signal: Full raw ECG signal array.
        out_path: Destination PNG file path.
    """
    mask = timestamps <= 30.0
    t_plot = timestamps[mask]
    e_plot = ecg_signal[mask]

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t_plot, e_plot, color="#1f77b4", linewidth=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ECG (a.u.)")
    ax.set_title("Raw ECG Signal — first 30 seconds", fontweight="bold")
    ax.set_xlim(float(t_plot[0]) if len(t_plot) else 0.0, 30.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved -> %s", out_path)


def plot_drowsiness_timeline(
    windows: list[dict],
    out_path: Path,
) -> None:
    """Save a confidence timeline with green/alert and red/drowsy background shading.

    Args:
        windows: List of window-result dicts from DrowsinessPredictor.predict_from_ecg.
        out_path: Destination PNG file path.
    """
    centers_min = [(w["window_start_sec"] + w["window_end_sec"]) / 2.0 / 60.0 for w in windows]
    confidences = [w["confidence"] for w in windows]
    predictions = [w["prediction"] for w in windows]

    step_min = (centers_min[1] - centers_min[0]) if len(centers_min) > 1 else 1.0

    fig, ax = plt.subplots(figsize=(12, 4))

    for center, pred in zip(centers_min, predictions):
        color = _COLOR_DROWSY if pred == 1 else _COLOR_ALERT
        ax.axvspan(center - step_min / 2, center + step_min / 2, alpha=0.25, color=color)

    ax.plot(centers_min, confidences, color="black", linewidth=1.5, label="Confidence")
    ax.axhline(
        _THRESHOLD,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label=f"Threshold ({_THRESHOLD})",
    )
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Confidence score")
    ax.set_title("Drowsiness Timeline", fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved -> %s", out_path)


def plot_feature_evolution(
    windows: list[dict],
    out_path: Path,
) -> None:
    """Save a 4-panel plot of z-normalised HRV feature values over time.

    Each subplot corresponds to one feature from SELECTED_FEATURES, with
    alert/drowsy background shading matching the drowsiness timeline.

    Args:
        windows: List of window-result dicts from DrowsinessPredictor.predict_from_ecg.
        out_path: Destination PNG file path.
    """
    centers_min = [(w["window_start_sec"] + w["window_end_sec"]) / 2.0 / 60.0 for w in windows]
    predictions = [w["prediction"] for w in windows]
    step_min = (centers_min[1] - centers_min[0]) if len(centers_min) > 1 else 1.0

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    for ax, feat_key, label, line_color in zip(
        axes, SELECTED_FEATURES, _FEATURE_LABELS, _FEATURE_LINE_COLORS
    ):
        values = [w[feat_key] for w in windows]

        for center, pred in zip(centers_min, predictions):
            bg = _COLOR_DROWSY if pred == 1 else _COLOR_ALERT
            ax.axvspan(center - step_min / 2, center + step_min / 2, alpha=0.15, color=bg)

        ax.plot(centers_min, values, color=line_color, linewidth=1.5)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.set_ylabel(label)
        ax.set_title(label, fontweight="bold", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (min)")
    fig.suptitle("HRV Feature Evolution (z-normalised)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved -> %s", out_path)


def print_summary(
    ecg_file: Path,
    timestamps: np.ndarray,
    sampling_rate: float,
    windows: list[dict],
) -> None:
    """Print a console summary of the detection results.

    Args:
        ecg_file: Path to the analysed ECG file.
        timestamps: Recording timestamps in seconds.
        sampling_rate: ECG sampling rate in Hz.
        windows: List of window-result dicts from DrowsinessPredictor.predict_from_ecg.
    """
    duration_min = float(timestamps[-1]) / 60.0
    n_windows = len(windows)
    n_drowsy = sum(1 for w in windows if w["prediction"] == 1)
    n_alert = n_windows - n_drowsy
    pct_drowsy = 100.0 * n_drowsy / n_windows if n_windows > 0 else 0.0
    avg_conf = float(np.mean([w["confidence"] for w in windows])) if windows else 0.0

    first_drowsy_min: float | None = None
    for w in windows:
        if w["prediction"] == 1:
            first_drowsy_min = (w["window_start_sec"] + w["window_end_sec"]) / 2.0 / 60.0
            break

    sep = "=" * 52
    print(f"\n{sep}")
    print("  Drowsiness Detection Demo — Summary")
    print(sep)
    print(f"  ECG file   : {ecg_file}")
    print(f"  Duration   : {duration_min:.1f} min  |  Sampling: {sampling_rate:.0f} Hz")
    print(sep)
    print(f"  Windows analysed  : {n_windows}")
    print(f"  Drowsy windows    : {n_drowsy:3d}  ({pct_drowsy:.1f} %)")
    print(f"  Alert windows     : {n_alert:3d}  ({100.0 - pct_drowsy:.1f} %)")
    print(f"  Avg confidence    : {avg_conf:.3f}")
    if first_drowsy_min is not None:
        print(f"  First drowsy at   : {first_drowsy_min:.1f} min")
    else:
        print("  First drowsy at   : (none detected)")
    print(sep)
    print(f"  Plots saved to    : {_DEMO_DIR}")
    print("    fig_ecg_signal.png")
    print("    fig_drowsiness_timeline.png")
    print("    fig_feature_evolution.png")
    print(sep)


def _setup_logging() -> None:
    """Configure logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    """Run the drowsiness detection demo pipeline.

    Loads an ECG file, runs inference, generates three visualisation plots,
    and prints a results summary to the console.
    """
    _setup_logging()
    args = _parse_args()

    ecg_file = args.ecg_file if args.ecg_file is not None else _DEFAULT_ECG

    if not _MODEL_PATH.exists():
        print(f"Model not found at {_MODEL_PATH}.")
        print("Run python src/inference.py --train first.")
        sys.exit(1)

    timestamps, ecg_signal, sampling_rate = _load_ecg(ecg_file, args.sampling_rate)

    predictor = DrowsinessPredictor(model_path=_MODEL_PATH, scaler_path=_SCALER_PATH)
    windows = predictor.predict_from_ecg(ecg_signal, sampling_rate)

    if not windows:
        print("No valid HRV windows extracted — recording may be too short.")
        sys.exit(1)

    _DEMO_DIR.mkdir(parents=True, exist_ok=True)

    plot_ecg_signal(timestamps, ecg_signal, _DEMO_DIR / "fig_ecg_signal.png")
    plot_drowsiness_timeline(windows, _DEMO_DIR / "fig_drowsiness_timeline.png")
    plot_feature_evolution(windows, _DEMO_DIR / "fig_feature_evolution.png")
    print_summary(ecg_file, timestamps, sampling_rate, windows)


if __name__ == "__main__":
    main()
