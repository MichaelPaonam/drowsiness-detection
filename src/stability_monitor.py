"""
Prediction stability monitor for the drowsiness detection model.

Runs DrowsinessPredictor on all available ECG recordings, computes per-recording
and cross-recording stability metrics, and saves a full report plus CSV and figures
to outputs/stability/.

Usage
-----
    python src/stability_monitor.py
    python src/stability_monitor.py --dataset drozy
    python src/stability_monitor.py --dataset ddd
    python src/stability_monitor.py --dataset all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

from config import ECG_CSV_DIR, OUTPUTS_DIR  # noqa: E402
from inference import DrowsinessPredictor  # noqa: E402

log = logging.getLogger(__name__)

DROZY_ECG_DIR: Path = ECG_CSV_DIR / "drozy"
DDD_ECG_DIR: Path = ECG_CSV_DIR / "ddd"
MODELS_DIR: Path = OUTPUTS_DIR / "models"
STABILITY_DIR: Path = OUTPUTS_DIR / "stability"

UNSTABLE_FLIP_THRESHOLD: float = 0.30
DPI: int = 150


def detect_sampling_rate(df: pd.DataFrame) -> int:
    """Auto-detect ECG sampling rate from consecutive timestamp differences.

    Args:
        df: DataFrame with a ``timestamp_sec`` column.

    Returns:
        Sampling rate in Hz, rounded to the nearest integer.

    Raises:
        ValueError: If fewer than two rows are present or the median delta is
            non-positive.
    """
    if len(df) < 2:
        raise ValueError("Cannot detect sampling rate: fewer than 2 rows.")
    deltas = np.diff(df["timestamp_sec"].to_numpy(dtype=np.float64))
    dt = float(np.median(deltas))
    if dt <= 0:
        raise ValueError(f"Non-positive median timestamp delta: {dt}")
    return int(round(1.0 / dt))


def _max_streak(predictions: list[str]) -> int:
    """Return the length of the longest run of identical consecutive predictions.

    Args:
        predictions: Ordered list of prediction labels.

    Returns:
        Length of the longest consecutive run; 0 if the list is empty.
    """
    if not predictions:
        return 0
    best = current = 1
    for i in range(1, len(predictions)):
        current = current + 1 if predictions[i] == predictions[i - 1] else 1
        best = max(best, current)
    return best


def _drowsy_pct(windows: list[dict[str, Any]]) -> float:
    """Return the percentage of windows predicted as drowsy.

    Args:
        windows: Per-window prediction dicts (must have a ``prediction`` key).

    Returns:
        Drowsy percentage in [0, 100]; 0.0 if the list is empty.
    """
    if not windows:
        return 0.0
    return sum(1 for w in windows if w["prediction"] == "drowsy") / len(windows) * 100


def compute_recording_metrics(
    results: list[dict[str, Any]],
    recording_name: str,
) -> dict[str, Any]:
    """Compute per-recording stability metrics from prediction results.

    Args:
        results: Per-window prediction dicts from DrowsinessPredictor.
        recording_name: Identifier for this recording.

    Returns:
        Dict with keys: recording, n_windows, drowsy_pct, mean_confidence,
        std_confidence, max_streak, n_flips, flip_rate.
    """
    n = len(results)
    preds = [r["prediction"] for r in results]
    confs = [r["confidence"] for r in results]

    drowsy_p = _drowsy_pct(results)
    mean_conf = float(np.mean(confs)) if confs else 0.0
    std_conf = float(np.std(confs, ddof=1)) if len(confs) > 1 else 0.0
    n_flips = sum(1 for i in range(1, n) if preds[i] != preds[i - 1])
    flip_rate = n_flips / n if n > 0 else 0.0

    return {
        "recording": recording_name,
        "n_windows": n,
        "drowsy_pct": round(drowsy_p, 2),
        "mean_confidence": round(mean_conf, 4),
        "std_confidence": round(std_conf, 4),
        "max_streak": _max_streak(preds),
        "n_flips": n_flips,
        "flip_rate": round(flip_rate, 4),
    }


def run_inference_on_directory(
    ecg_dir: Path,
    predictor: DrowsinessPredictor,
) -> dict[str, list[dict[str, Any]]]:
    """Run inference on every ECG CSV in a directory.

    Args:
        ecg_dir: Directory of ECG CSVs (columns: ``timestamp_sec``, ``ecg``).
        predictor: Loaded DrowsinessPredictor instance.

    Returns:
        Dict mapping recording stem to its per-window prediction dicts.
        Recordings that fail inference are skipped with a warning.
    """
    csv_files = sorted(ecg_dir.glob("*.csv"))
    if not csv_files:
        log.warning("No CSV files found in %s", ecg_dir)
        return {}

    results: dict[str, list[dict[str, Any]]] = {}
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            fs = detect_sampling_rate(df)
            ecg_signal = df["ecg"].to_numpy(dtype=np.float64)
            log.info(
                "Running inference on %s (fs=%d Hz, %d samples)",
                csv_path.name,
                fs,
                len(ecg_signal),
            )
            results[csv_path.stem] = predictor.predict_from_ecg(ecg_signal, fs)
        except Exception as exc:  # noqa: BLE001
            log.warning("Skipped %s: %s", csv_path.name, exc)

    return results


def temporal_split_analysis(
    ddd_results: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Compare drowsy rates in first vs second half of each DDD recording.

    Splits windows at the midpoint of the recording's time span to compare
    the alert first half against the fatigued second half.

    Args:
        ddd_results: Dict mapping recording stem to per-window prediction dicts.

    Returns:
        List of dicts with keys: recording, first_half_drowsy_pct,
        second_half_drowsy_pct, expected_pattern (True when second > first).
    """
    rows: list[dict[str, Any]] = []
    for name, windows in ddd_results.items():
        if len(windows) < 2:
            log.warning("Skipping temporal split for %s: only %d windows", name, len(windows))
            continue

        end_times = [w["window_end_sec"] for w in windows]
        mid_time = (min(end_times) + max(end_times)) / 2.0
        first_half = [w for w in windows if w["window_end_sec"] <= mid_time]
        second_half = [w for w in windows if w["window_end_sec"] > mid_time]

        fp = _drowsy_pct(first_half)
        sp = _drowsy_pct(second_half)
        rows.append(
            {
                "recording": name,
                "first_half_drowsy_pct": round(fp, 2),
                "second_half_drowsy_pct": round(sp, 2),
                "expected_pattern": sp > fp,
            }
        )
    return rows


def plot_confidence_distribution(all_confs: list[float], output_path: Path) -> None:
    """Save a histogram of confidence scores across all recordings.

    Args:
        all_confs: Per-window confidence values in [0, 1].
        output_path: File path for the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_confs, bins=40, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(
        0.5,
        color="tomato",
        linestyle="--",
        linewidth=1.5,
        label="Decision threshold (0.5)",
    )
    ax.set_xlabel("Confidence (P(drowsy))")
    ax.set_ylabel("Window count")
    ax.set_title("Confidence Score Distribution — All Recordings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
    log.info("Saved: %s", output_path)


def plot_flip_rate(per_rec: list[dict[str, Any]], output_path: Path) -> None:
    """Save a bar chart of prediction flip rate per recording.

    Args:
        per_rec: Per-recording metric dicts (must have ``recording``, ``flip_rate``).
        output_path: File path for the saved PNG.
    """
    names = [r["recording"] for r in per_rec]
    rates = [r["flip_rate"] for r in per_rec]
    colors = ["tomato" if r > UNSTABLE_FLIP_THRESHOLD else "steelblue" for r in rates]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.5), 5))
    ax.bar(range(len(names)), rates, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(
        UNSTABLE_FLIP_THRESHOLD,
        color="tomato",
        linestyle="--",
        linewidth=1.5,
        label=f"Unstable threshold ({UNSTABLE_FLIP_THRESHOLD:.0%})",
    )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Flip rate (transitions / windows)")
    ax.set_title("Prediction Flip Rate per Recording")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
    log.info("Saved: %s", output_path)


def plot_temporal_pattern(temporal_rows: list[dict[str, Any]], output_path: Path) -> None:
    """Save a grouped bar chart of first/second half drowsy rates for DDD recordings.

    Args:
        temporal_rows: Output from :func:`temporal_split_analysis`.
        output_path: File path for the saved PNG.
    """
    if not temporal_rows:
        log.warning("No DDD temporal data to plot.")
        return

    names = [r["recording"] for r in temporal_rows]
    first_vals = [r["first_half_drowsy_pct"] for r in temporal_rows]
    second_vals = [r["second_half_drowsy_pct"] for r in temporal_rows]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.6), 5))
    ax.bar(
        x - width / 2,
        first_vals,
        width,
        label="First half",
        color="steelblue",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        second_vals,
        width,
        label="Second half",
        color="salmon",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Drowsy %")
    ax.set_title("DDD Temporal Pattern: First vs Second Half Drowsy Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
    log.info("Saved: %s", output_path)


def build_report(
    per_rec_metrics: list[dict[str, Any]],
    all_confs: list[float],
    temporal_rows: list[dict[str, Any]],
    unstable_recordings: list[str],
    dataset: str,
) -> str:
    """Build the full text stability report.

    Args:
        per_rec_metrics: Per-recording metric dicts.
        all_confs: All confidence values across all recordings.
        temporal_rows: DDD temporal analysis rows.
        unstable_recordings: Recording names flagged as unstable.
        dataset: Dataset label (``"drozy"``, ``"ddd"``, or ``"all"``).

    Returns:
        Multi-line report string.
    """
    sep = "=" * 70
    lines: list[str] = [sep, "PREDICTION STABILITY REPORT", f"Dataset: {dataset.upper()}", sep, ""]

    total_windows = sum(r["n_windows"] for r in per_rec_metrics)
    total_drowsy = sum(r["n_windows"] * r["drowsy_pct"] / 100.0 for r in per_rec_metrics)
    overall_rate = total_drowsy / total_windows * 100 if total_windows > 0 else 0.0

    lines += [
        f"Recordings processed : {len(per_rec_metrics)}",
        f"Total windows        : {total_windows}",
        f"Overall drowsy rate  : {overall_rate:.2f}%",
        "",
    ]

    if all_confs:
        arr = np.asarray(all_confs)
        lines += [
            "Confidence distribution (across all windows):",
            f"  Mean   : {arr.mean():.4f}",
            f"  Std    : {float(arr.std(ddof=1)) if len(arr) > 1 else 0.0:.4f}",
            f"  Min    : {float(arr.min()):.4f}",
            f"  Max    : {float(arr.max()):.4f}",
            f"  Median : {float(np.median(arr)):.4f}",
            "",
        ]

    threshold_label = f"{UNSTABLE_FLIP_THRESHOLD:.0%}"
    lines.append(f"Unstable recordings (flip_rate > {threshold_label}):")
    if unstable_recordings:
        lines += [f"  [UNSTABLE] {name}" for name in unstable_recordings]
    else:
        lines.append("  None")
    lines.append("")

    if temporal_rows:
        n_expected = sum(1 for r in temporal_rows if r["expected_pattern"])
        lines += [
            "DDD temporal pattern (fatigue builds over time):",
            f"  Expected pattern (2nd half > 1st half): {n_expected} / {len(temporal_rows)}",
            "",
            f"  {'Recording':<20} {'1st half %':>12} {'2nd half %':>12} {'Expected?':>10}",
            f"  {'-' * 20} {'-' * 12} {'-' * 12} {'-' * 10}",
        ]
        for row in temporal_rows:
            marker = "YES" if row["expected_pattern"] else "NO"
            lines.append(
                f"  {row['recording']:<20}"
                f" {row['first_half_drowsy_pct']:>12.2f}"
                f" {row['second_half_drowsy_pct']:>12.2f}"
                f" {marker:>10}"
            )
        lines.append("")

    col_hdr = (
        f"  {'Recording':<20} {'Windows':>7} {'Drowsy%':>8}"
        f" {'MeanConf':>9} {'StdConf':>8} {'Streak':>7} {'Flips':>6} {'FlipRate':>9}"
    )
    col_sep = f"  {'-' * 20} {'-' * 7} {'-' * 8} {'-' * 9} {'-' * 8} {'-' * 7} {'-' * 6} {'-' * 9}"
    lines += ["Per-recording summary:", col_hdr, col_sep]
    for r in per_rec_metrics:
        flag = " [!]" if r["flip_rate"] > UNSTABLE_FLIP_THRESHOLD else ""
        lines.append(
            f"  {r['recording']:<20}"
            f" {r['n_windows']:>7}"
            f" {r['drowsy_pct']:>8.2f}"
            f" {r['mean_confidence']:>9.4f}"
            f" {r['std_confidence']:>8.4f}"
            f" {r['max_streak']:>7}"
            f" {r['n_flips']:>6}"
            f" {r['flip_rate']:>9.4f}{flag}"
        )
    lines += ["", sep]
    return "\n".join(lines)


def run_stability_monitor(
    dataset: str,
    predictor: DrowsinessPredictor,
    stability_dir: Path,
) -> None:
    """Run the full stability analysis pipeline and save all outputs.

    Args:
        dataset: One of ``"drozy"``, ``"ddd"``, or ``"all"``.
        predictor: Loaded DrowsinessPredictor instance.
        stability_dir: Directory to write all output files into.
    """
    stability_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[dict[str, Any]]] = {}
    ddd_results: dict[str, list[dict[str, Any]]] = {}

    if dataset in ("drozy", "all"):
        log.info("Processing DROZY recordings from %s", DROZY_ECG_DIR)
        all_results.update(run_inference_on_directory(DROZY_ECG_DIR, predictor))

    if dataset in ("ddd", "all"):
        log.info("Processing DDD recordings from %s", DDD_ECG_DIR)
        ddd_results = run_inference_on_directory(DDD_ECG_DIR, predictor)
        all_results.update(ddd_results)

    if not all_results:
        log.error("No recordings processed. Check ECG CSV directories.")
        sys.exit(1)

    per_rec_metrics = sorted(
        [compute_recording_metrics(v, k) for k, v in all_results.items()],
        key=lambda r: r["recording"],
    )
    all_confs = [w["confidence"] for ws in all_results.values() for w in ws]
    unstable = [r["recording"] for r in per_rec_metrics if r["flip_rate"] > UNSTABLE_FLIP_THRESHOLD]

    temporal_rows: list[dict[str, Any]] = []
    if ddd_results:
        temporal_rows = temporal_split_analysis(ddd_results)

    report_text = build_report(per_rec_metrics, all_confs, temporal_rows, unstable, dataset)
    print(report_text)

    report_path = stability_dir / "stability_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    log.info("Saved: %s", report_path)

    csv_path = stability_dir / "stability_per_recording.csv"
    pd.DataFrame(per_rec_metrics).to_csv(csv_path, index=False)
    log.info("Saved: %s", csv_path)

    plot_confidence_distribution(all_confs, stability_dir / "fig_confidence_distribution.png")
    plot_flip_rate(per_rec_metrics, stability_dir / "fig_flip_rate.png")

    if ddd_results:
        plot_temporal_pattern(temporal_rows, stability_dir / "fig_temporal_pattern.png")
    else:
        log.info("Skipping fig_temporal_pattern.png (no DDD data in this run).")


def setup_logging() -> None:
    """Configure root logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prediction stability monitor for the drowsiness detection model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["drozy", "ddd", "all"],
        default="all",
        help="Which ECG dataset(s) to analyse.",
    )
    args = parser.parse_args()
    setup_logging()

    _model_path = MODELS_DIR / "best_model.json"
    _scaler_path = MODELS_DIR / "scaler.pkl"

    try:
        _predictor = DrowsinessPredictor(model_path=_model_path, scaler_path=_scaler_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}. Run 'python src/inference.py --train' first.", file=sys.stderr)
        sys.exit(1)

    run_stability_monitor(args.dataset, _predictor, STABILITY_DIR)
