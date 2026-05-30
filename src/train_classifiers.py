"""
Classifier training script for drowsiness detection (XGBoost and CatBoost).

Trains XGBoost and CatBoost classifiers on z-normalised HRV features from the
DROZY/DDD combined dataset. Supports two evaluation modes:

  fixed split — 80-10-10 subject-level split (default)
  LOSO-CV     — Leave-One-Subject-Out cross-validation (--loso)

Outputs
-------
outputs/classifier_comparison.txt            — model comparison summary table
outputs/classifier_results.csv               — per-fold or per-split metrics
outputs/classifier_plots/fig_roc_curves.png  — ROC curves for both models
outputs/classifier_plots/fig_pr_curves.png   — precision-recall curves
outputs/classifier_plots/fig_feature_importance.png

Usage
-----
    python src/train_classifiers.py           # fixed split mode
    python src/train_classifiers.py --loso    # LOSO-CV mode
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))

from config import OUTPUTS_DIR
from dataloader import SubjectWiseDataLoader

log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

RANDOM_STATE: int = 42
RECALL_FLOOR: float = 0.3
FIG_DPI: int = 150

PLOTS_DIR = OUTPUTS_DIR / "classifier_plots"
COMPARISON_TXT = OUTPUTS_DIR / "classifier_comparison.txt"
RESULTS_CSV = OUTPUTS_DIR / "classifier_results.csv"

MODEL_NAMES: tuple[str, ...] = ("XGBoost", "CatBoost")
_MODEL_COLORS: dict[str, str] = {
    "XGBoost": "#E8614C",
    "CatBoost": "#4C9BE8",
}

# ── model builders ────────────────────────────────────────────────────────────


def build_xgb(scale_pos_weight: float) -> XGBClassifier:
    """
    Build an XGBoost classifier configured for imbalanced binary classification.

    Args:
        scale_pos_weight: Ratio of negatives to positives in the training set.

    Returns:
        Configured XGBClassifier (unfitted).
    """
    return XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )


def build_catboost() -> CatBoostClassifier:
    """
    Build a CatBoost classifier with auto class-weight balancing.

    Returns:
        Configured CatBoostClassifier (unfitted).
    """
    return CatBoostClassifier(
        auto_class_weights="Balanced",
        verbose=0,
        random_seed=RANDOM_STATE,
    )


def _scale_pos_weight(y: np.ndarray) -> float:
    """
    Compute XGBoost class-imbalance weight: n_negative / n_positive.

    Args:
        y: Binary label array.

    Returns:
        Float ratio of negative to positive samples.

    Raises:
        ValueError: If y contains no positive samples.
    """
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError(
            "Training labels contain no positive samples — cannot compute scale_pos_weight."
        )
    return (len(y) - n_pos) / n_pos


# ── metric helpers ────────────────────────────────────────────────────────────


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Compute standard binary classification metrics at the given threshold.

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_proba: Predicted probabilities for the positive class.
        threshold: Decision threshold used to binarise probabilities.

    Returns:
        Dictionary with keys: precision, recall, f1, roc_auc, pr_auc, confusion_matrix.
    """
    y_pred = (y_proba >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true, y_proba))
        pr_auc = float(average_precision_score(y_true, y_proba))

    cm = confusion_matrix(y_true, y_pred)

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    recall_floor: float = RECALL_FLOOR,
) -> float:
    """
    Find the decision threshold that maximises precision subject to recall >= recall_floor.

    Searches over thresholds from sklearn's precision_recall_curve. Falls back to 0.5
    if no threshold satisfies the recall constraint.

    Args:
        y_true: Ground-truth binary labels.
        y_proba: Predicted probabilities for the positive class.
        recall_floor: Minimum acceptable recall value.

    Returns:
        Optimal probability threshold as a float.
    """
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve appends (precision=1, recall=0) with no matching threshold
    feasible = recall_vals[:-1] >= recall_floor
    if not feasible.any():
        log.warning("No threshold achieves recall >= %.2f; falling back to 0.5", recall_floor)
        return 0.5
    best_idx = int(np.argmax(np.where(feasible, precision_vals[:-1], 0.0)))
    return float(thresholds[best_idx])


# ── data preparation ──────────────────────────────────────────────────────────


def fit_and_scale(X_train: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """
    Fit a StandardScaler on X_train and transform X_train plus any additional arrays.

    Args:
        X_train: Training feature matrix used to fit the scaler.
        *arrays: Additional arrays to transform with the fitted scaler.

    Returns:
        Tuple of scaled arrays: (X_train_scaled, *arrays_scaled).
    """
    scaler = StandardScaler()
    result: list[np.ndarray] = [scaler.fit_transform(X_train)]
    result.extend(scaler.transform(a) for a in arrays)
    return tuple(result)


# ── fixed-split training ──────────────────────────────────────────────────────


def _train_and_eval_fixed(loader: SubjectWiseDataLoader) -> dict[str, Any]:
    """
    Train both models on the fixed train split and evaluate on val and test.

    Args:
        loader: Dataloader with splits already prepared via prepare().

    Returns:
        Nested dict {model_name: {"val": split_result, "test": split_result, "model": model}}
        where each split_result contains metrics, y_true, y_proba, n_samples, n_pos.
    """
    X_train, y_train = loader.get_split("train", return_arrays=True)
    X_val, y_val = loader.get_split("val", return_arrays=True)
    X_test, y_test = loader.get_split("test", return_arrays=True)

    X_train_s, X_val_s, X_test_s = fit_and_scale(X_train, X_val, X_test)
    log.info("Fixed split sizes: train=%d  val=%d  test=%d", len(y_train), len(y_val), len(y_test))

    spw = _scale_pos_weight(y_train)
    model_instances: dict[str, Any] = {
        "XGBoost": build_xgb(spw),
        "CatBoost": build_catboost(),
    }
    results: dict[str, Any] = {}

    for name, model in model_instances.items():
        log.info("Training %s (fixed split)...", name)
        model.fit(X_train_s, y_train)
        model_entry: dict[str, Any] = {"model": model}

        for split_name, X_s, y_true in [
            ("val", X_val_s, y_val),
            ("test", X_test_s, y_test),
        ]:
            y_proba = model.predict_proba(X_s)[:, 1]
            metrics = compute_metrics(y_true, y_proba)
            model_entry[split_name] = {
                **metrics,
                "y_true": y_true,
                "y_proba": y_proba,
                "n_samples": int(len(y_true)),
                "n_pos": int(y_true.sum()),
            }
            log.info(
                "%s / %s | prec=%.4f  rec=%.4f  f1=%.4f  roc_auc=%.4f  pr_auc=%.4f",
                name,
                split_name,
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
                metrics["roc_auc"],
                metrics["pr_auc"],
            )

        results[name] = model_entry

    return results


# ── LOSO training ─────────────────────────────────────────────────────────────


def _train_loso_fold(
    fold: dict[str, pd.DataFrame],
    feature_cols: list[str],
    target_col: str,
) -> dict[str, Any] | None:
    """
    Train both models on one LOSO fold and return out-of-fold predictions.

    Skips folds where the test set contains only a single class.

    Args:
        fold: Dict with "train" and "test" DataFrames.
        feature_cols: Feature column names.
        target_col: Target label column name.

    Returns:
        Dict with "y_test" and per-model {"y_proba", "model"} entries,
        or None if the fold is single-class and should be skipped.
    """
    train_df = fold["train"]
    test_df = fold["test"]

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[target_col].values.astype(np.int64)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[target_col].values.astype(np.int64)

    if len(np.unique(y_test)) < 2:
        return None

    X_train_s, X_test_s = fit_and_scale(X_train, X_test)
    spw = _scale_pos_weight(y_train)
    model_instances: dict[str, Any] = {
        "XGBoost": build_xgb(spw),
        "CatBoost": build_catboost(),
    }

    result: dict[str, Any] = {"y_test": y_test}
    for name, model in model_instances.items():
        model.fit(X_train_s, y_train)
        result[name] = {
            "y_proba": model.predict_proba(X_test_s)[:, 1],
            "model": model,
        }

    return result


def _run_loso(loader: SubjectWiseDataLoader) -> dict[str, Any]:
    """
    Run full LOSO-CV and pool out-of-fold predictions for final evaluation.

    Args:
        loader: Dataloader with data already loaded via load_data().

    Returns:
        Dict keyed by model name. Each value contains: y_true, y_proba,
        overall (metrics at opt_threshold), opt_threshold, fold_records,
        mean_precision, mean_recall, mean_f1, model (last fold's fitted model).
    """
    folds = loader.get_loso_folds()
    feature_cols = loader.feature_cols
    target_col = loader.target_col

    pooled: dict[str, dict[str, list]] = {
        name: {"y_true": [], "y_proba": [], "fold_records": []} for name in MODEL_NAMES
    }
    models_last: dict[str, Any] = {}

    for fold_idx, fold in enumerate(folds):
        held_out_subject = fold["test"]["subject_id"].iloc[0]
        n_test = len(fold["test"])
        n_pos = int(fold["test"][target_col].sum())

        result = _train_loso_fold(fold, feature_cols, target_col)
        if result is None:
            log.warning(
                "Fold %d (subject=%s): single-class test (%d pos / %d total) — skipped.",
                fold_idx + 1,
                held_out_subject,
                n_pos,
                n_test,
            )
            continue

        y_test = result["y_test"]
        for name in MODEL_NAMES:
            y_proba = result[name]["y_proba"]
            models_last[name] = result[name]["model"]

            metrics = compute_metrics(y_test, y_proba)
            # Flatten confusion matrix for CSV compatibility
            cm = metrics.pop("confusion_matrix")
            metrics["tn"] = int(cm[0, 0])
            metrics["fp"] = int(cm[0, 1])
            metrics["fn"] = int(cm[1, 0])
            metrics["tp"] = int(cm[1, 1])

            pooled[name]["y_true"].extend(y_test.tolist())
            pooled[name]["y_proba"].extend(y_proba.tolist())
            pooled[name]["fold_records"].append(
                {
                    "model": name,
                    "fold": fold_idx + 1,
                    "held_out_subject": held_out_subject,
                    "n_train": len(fold["train"]),
                    "n_test": n_test,
                    "n_pos": n_pos,
                    **metrics,
                }
            )

        xgb_rec = pooled["XGBoost"]["fold_records"][-1]
        cb_rec = pooled["CatBoost"]["fold_records"][-1]
        log.info(
            "Fold %2d/%d | %s | XGB prec=%.3f rec=%.3f | CB prec=%.3f rec=%.3f",
            fold_idx + 1,
            len(folds),
            held_out_subject,
            xgb_rec["precision"],
            xgb_rec["recall"],
            cb_rec["precision"],
            cb_rec["recall"],
        )

    aggregated: dict[str, Any] = {}
    for name in MODEL_NAMES:
        if len(pooled[name]["y_true"]) == 0:
            log.warning("All folds skipped for %s — no metrics to compute.", name)
            continue

        y_true_all = np.array(pooled[name]["y_true"])
        y_proba_all = np.array(pooled[name]["y_proba"])
        records = pooled[name]["fold_records"]

        opt_threshold = find_optimal_threshold(y_true_all, y_proba_all)
        overall = compute_metrics(y_true_all, y_proba_all, threshold=opt_threshold)

        fold_df = pd.DataFrame(records)
        aggregated[name] = {
            "y_true": y_true_all,
            "y_proba": y_proba_all,
            "overall": overall,
            "opt_threshold": opt_threshold,
            "fold_records": records,
            "model": models_last.get(name),
            "mean_precision": float(fold_df["precision"].mean()),
            "mean_recall": float(fold_df["recall"].mean()),
            "mean_f1": float(fold_df["f1"].mean()),
        }
        log.info(
            "%s LOSO | roc_auc=%.4f  pr_auc=%.4f  opt_threshold=%.4f",
            name,
            overall["roc_auc"],
            overall["pr_auc"],
            opt_threshold,
        )

    return aggregated


# ── plotting ──────────────────────────────────────────────────────────────────


def _ensure_plots_dir() -> None:
    """Create the classifier_plots output directory if it does not exist."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_roc_curves(
    model_curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
    suffix: str = "",
) -> None:
    """
    Plot and save ROC curves for all models.

    Args:
        model_curves: {model_name: (fpr_array, tpr_array, roc_auc_scalar)}.
        suffix: Optional filename suffix, e.g. "_fixed" or "_loso".
    """
    _ensure_plots_dir()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")

    for name, (fpr, tpr, auc_val) in model_curves.items():
        ax.plot(
            fpr,
            tpr,
            color=_MODEL_COLORS[name],
            linewidth=1.8,
            label=f"{name} (AUC={auc_val:.3f})",
        )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = PLOTS_DIR / f"fig_roc_curves{suffix}.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved -> %s", out)


def plot_pr_curves(
    model_curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
    suffix: str = "",
) -> None:
    """
    Plot and save precision-recall curves for all models.

    Args:
        model_curves: {model_name: (precision_array, recall_array, pr_auc_scalar)}.
        suffix: Optional filename suffix, e.g. "_fixed" or "_loso".
    """
    _ensure_plots_dir()
    fig, ax = plt.subplots(figsize=(6, 5))

    for name, (prec, rec, auc_val) in model_curves.items():
        ax.plot(
            rec,
            prec,
            color=_MODEL_COLORS[name],
            linewidth=1.8,
            label=f"{name} (AUC={auc_val:.3f})",
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    fig.tight_layout()

    out = PLOTS_DIR / f"fig_pr_curves{suffix}.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved -> %s", out)


def plot_feature_importance(
    models: dict[str, Any],
    feature_cols: list[str],
) -> None:
    """
    Plot and save a bar chart of feature importances for each model.

    Args:
        models: {model_name: fitted model instance (may be None)}.
        feature_cols: Feature column names in the same order as model inputs.
    """
    _ensure_plots_dir()
    short_names = [c.replace("_znorm", "") for c in feature_cols]
    valid_models = {
        name: m
        for name, m in models.items()
        if m is not None and hasattr(m, "feature_importances_")
    }

    if not valid_models:
        log.warning("No models with feature_importances_ available — skipping importance plot.")
        return

    n_models = len(valid_models)
    n_features = len(feature_cols)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 4), sharey=False)
    if n_models == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, valid_models.items()):
        importances: np.ndarray = model.feature_importances_
        order = np.argsort(importances)[::-1]
        ax.bar(
            range(n_features),
            importances[order],
            color=_MODEL_COLORS.get(name, "#888888"),
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_xticks(range(n_features))
        ax.set_xticklabels([short_names[i] for i in order], rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Feature importance")
        ax.set_title(f"{name} — Feature Importance", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Feature Importance Comparison", fontweight="bold")
    fig.tight_layout()

    out = PLOTS_DIR / "fig_feature_importance.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved -> %s", out)


# ── output formatting ─────────────────────────────────────────────────────────

_SEP = "=" * 80
_DIV = "-" * 80


def _format_cm(cm: np.ndarray) -> str:
    return f"[[{cm[0,0]:4d}, {cm[0,1]:4d}], [{cm[1,0]:4d}, {cm[1,1]:4d}]]"


def _format_comparison_fixed(results: dict[str, Any]) -> str:
    """
    Format a fixed-split comparison summary as a plain-text table.

    Args:
        results: Output of _train_and_eval_fixed().

    Returns:
        Formatted multi-line string.
    """
    header = "  %-10s %-5s %5s %6s %6s %6s %8s %7s %20s" % (
        "Model",
        "Split",
        "N",
        "Prec",
        "Rec",
        "F1",
        "ROC-AUC",
        "PR-AUC",
        "CM",
    )
    lines = [_SEP, "CLASSIFIER COMPARISON (FIXED SPLIT)", _SEP, "", header, _DIV]

    for name in MODEL_NAMES:
        for split_name in ("val", "test"):
            m = results[name][split_name]
            lines.append(
                "  %-10s %-5s %5d %6.4f %6.4f %6.4f %8.4f %7.4f   %s"
                % (
                    name,
                    split_name,
                    m["n_samples"],
                    m["precision"],
                    m["recall"],
                    m["f1"],
                    m["roc_auc"],
                    m["pr_auc"],
                    _format_cm(m["confusion_matrix"]),
                )
            )

    lines.append(_SEP)
    return "\n".join(lines)


def _format_comparison_loso(results: dict[str, Any]) -> str:
    """
    Format a LOSO-CV comparison summary as a plain-text table.

    Args:
        results: Output of _run_loso().

    Returns:
        Formatted multi-line string.
    """
    header = "  %-10s %8s %7s %8s %9s %8s %7s %7s %20s" % (
        "Model",
        "Windows",
        "PR-AUC",
        "ROC-AUC",
        "Prec@opt",
        "Rec@opt",
        "F1@opt",
        "Thresh",
        "CM",
    )
    lines = [_SEP, "CLASSIFIER COMPARISON (LOSO-CV)", _SEP, "", header, _DIV]

    for name in MODEL_NAMES:
        r = results[name]
        m = r["overall"]
        n = len(r["y_true"])
        lines.append(
            "  %-10s %8d %7.4f %8.4f %9.4f %8.4f %7.4f %7.4f   %s"
            % (
                name,
                n,
                m["pr_auc"],
                m["roc_auc"],
                m["precision"],
                m["recall"],
                m["f1"],
                r["opt_threshold"],
                _format_cm(m["confusion_matrix"]),
            )
        )

    mean_header = "  %-12s %10s %9s %8s" % ("Model", "Mean Prec", "Mean Rec", "Mean F1")
    lines += ["", "  --- Mean per-fold metrics (threshold = 0.5) ---", "", mean_header, _DIV]

    for name in MODEL_NAMES:
        r = results[name]
        lines.append(
            "  %-12s %10.4f %9.4f %8.4f"
            % (name, r["mean_precision"], r["mean_recall"], r["mean_f1"])
        )

    lines.append(_SEP)
    return "\n".join(lines)


def save_comparison_txt(text: str) -> None:
    """
    Write the model comparison summary to outputs/classifier_comparison.txt.

    Args:
        text: Formatted comparison string.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_TXT.write_text(text, encoding="utf-8")
    log.info("Saved -> %s", COMPARISON_TXT)


def save_results_csv(records: list[dict[str, Any]]) -> None:
    """
    Write per-fold or per-split metric records to outputs/classifier_results.csv.

    Args:
        records: List of metric dicts, one per fold or split.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(RESULTS_CSV, index=False)
    log.info("Saved -> %s", RESULTS_CSV)


# ── run functions ─────────────────────────────────────────────────────────────


def run_fixed_split() -> None:
    """Run training and evaluation in fixed 80-10-10 subject-level split mode."""
    log.info("=== Fixed Split Mode ===")

    loader = SubjectWiseDataLoader()
    loader.prepare()
    feature_cols = loader.feature_cols

    results = _train_and_eval_fixed(loader)

    roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    pr_data: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    fitted_models: dict[str, Any] = {}
    csv_records: list[dict[str, Any]] = []

    for name in MODEL_NAMES:
        y_true = results[name]["test"]["y_true"]
        y_proba = results[name]["test"]["y_proba"]

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_data[name] = (fpr, tpr, results[name]["test"]["roc_auc"])

        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        pr_data[name] = (prec, rec, results[name]["test"]["pr_auc"])

        fitted_models[name] = results[name]["model"]

        for split_name in ("val", "test"):
            m = results[name][split_name]
            csv_records.append(
                {
                    "model": name,
                    "split": split_name,
                    "n_samples": m["n_samples"],
                    "n_pos": m["n_pos"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "roc_auc": m["roc_auc"],
                    "pr_auc": m["pr_auc"],
                    "tn": int(m["confusion_matrix"][0, 0]),
                    "fp": int(m["confusion_matrix"][0, 1]),
                    "fn": int(m["confusion_matrix"][1, 0]),
                    "tp": int(m["confusion_matrix"][1, 1]),
                }
            )

    comparison_text = _format_comparison_fixed(results)
    print(comparison_text)

    save_comparison_txt(comparison_text)
    save_results_csv(csv_records)
    plot_roc_curves(roc_data)
    plot_pr_curves(pr_data)
    plot_feature_importance(fitted_models, feature_cols)


def run_loso() -> None:
    """Run training and evaluation in Leave-One-Subject-Out cross-validation mode."""
    log.info("=== LOSO-CV Mode ===")

    loader = SubjectWiseDataLoader()
    loader.load_data()
    feature_cols = loader.feature_cols

    results = _run_loso(loader)

    roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    pr_data: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    fitted_models: dict[str, Any] = {}
    csv_records: list[dict[str, Any]] = []

    for name in MODEL_NAMES:
        r = results[name]
        y_true = r["y_true"]
        y_proba = r["y_proba"]

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_data[name] = (fpr, tpr, r["overall"]["roc_auc"])

        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        pr_data[name] = (prec, rec, r["overall"]["pr_auc"])

        fitted_models[name] = r["model"]
        csv_records.extend(r["fold_records"])

    comparison_text = _format_comparison_loso(results)
    print(comparison_text)

    save_comparison_txt(comparison_text)
    save_results_csv(csv_records)
    plot_roc_curves(roc_data)
    plot_pr_curves(pr_data)
    plot_feature_importance(fitted_models, feature_cols)


# ── entry point ───────────────────────────────────────────────────────────────


def setup_logging() -> None:
    """Configure logging for console output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train XGBoost and CatBoost classifiers for drowsiness detection"
    )
    parser.add_argument(
        "--loso",
        action="store_true",
        help="Use Leave-One-Subject-Out CV instead of the fixed 80-10-10 split",
    )
    args = parser.parse_args()

    setup_logging()
    if args.loso:
        run_loso()
    else:
        run_fixed_split()
