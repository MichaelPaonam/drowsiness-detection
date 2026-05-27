"""
Train XGBoost and CatBoost classifiers on pseudo-labeled HRV windows.

Uses Leave-One-Subject-Out CV with the unified subject_id column (covers both
DROZY and DDD subjects). Raw HRV features are used as input; a fresh
StandardScaler is fit inside each fold to prevent leakage.

Aggregate metrics are reported for:
  - All windows combined
  - DROZY-only windows
  - DDD-only windows   (cross-dataset generalization check)

Outputs
-------
outputs/pseudo_model_comparison.txt
outputs/pseudo_model_results.csv
outputs/pseudo_kss/fig_roc_curves.png
outputs/pseudo_kss/fig_pr_curves.png
outputs/pseudo_kss/fig_feature_importance.png

Usage
-----
    python src/train_pseudo_model.py
    python src/train_pseudo_model.py --exclude-subjects drozy_s04
    python src/train_pseudo_model.py --exclude-flagged --method kmeans
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    HRV_FEATURE_COLS,
    HRV_WINDOWS_FINAL_CSV,
    OUTPUTS_DIR,
    PSEUDO_KSS_DIR,
    PSEUDO_MODEL_COMPARISON_TXT,
    PSEUDO_MODEL_RESULTS_CSV,
)
log = logging.getLogger(__name__)


# ── threshold selection ───────────────────────────────────────────────────────

def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float = 0.30,
) -> tuple[float, float, float]:
    """
    Find the threshold on the PR curve that maximises precision subject to
    recall >= min_recall.  Returns (threshold, precision, recall).
    """
    from sklearn.metrics import precision_recall_curve, precision_score, recall_score
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    precisions = precisions[:-1]
    recalls    = recalls[:-1]

    mask = recalls >= min_recall
    if not mask.any():
        log.warning("No threshold achieves recall >= %.2f; falling back to 0.5.", min_recall)
        pred = (y_prob >= 0.5).astype(int)
        return 0.5, float(precision_score(y_true, pred, zero_division=0)), \
               float(recall_score(y_true, pred, zero_division=0))

    best_idx = int(np.argmax(precisions[mask]))
    return (
        float(thresholds[mask][best_idx]),
        float(precisions[mask][best_idx]),
        float(recalls[mask][best_idx]),
    )

FEATURE_LABELS = {
    "mean_rr": "Mean RR",
    "mean_hr": "Mean HR",
    "sdnn":    "SDNN",
    "rmssd":   "RMSSD",
    "nn50":    "NN50",
    "pnn50":   "pNN50",
    "cv_rr":   "CV-RR",
}


# ── model factories ───────────────────────────────────────────────────────────

def _make_xgb(scale_pos_weight: float) -> Any:
    from xgboost import XGBClassifier
    return XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False, eval_metric="logloss",
        learning_rate=0.1, max_depth=3, n_estimators=200,
        subsample=0.8, random_state=42, n_jobs=-1, verbosity=0,
    )


def _make_catboost(scale_pos_weight: float) -> Any:
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        auto_class_weights="Balanced",
        iterations=200, learning_rate=0.1, depth=4,
        random_seed=42, verbose=0, thread_count=-1,
    )


def _get_importance(clf: Any, n_features: int) -> np.ndarray | None:
    if hasattr(clf, "feature_importances_"):
        return clf.feature_importances_
    if hasattr(clf, "coef_"):
        return np.abs(clf.coef_[0])
    return None


# ── LOSO loop ─────────────────────────────────────────────────────────────────

def run_loso(
    df: pd.DataFrame,
    clf_factory: Callable[[], Any],
    model_name: str,
    min_recall: float = 0.30,
) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """
    Leave-One-Subject-Out CV.

    Returns
    -------
    fold_df   : per-fold records (subject, metrics at 0.5 threshold)
    summary   : aggregate metrics dict
    imp_mean  : mean feature importance across folds (length n_features)
    """
    X      = df[HRV_FEATURE_COLS].values.astype(float)
    y      = df["pseudo_label_smoothed"].values.astype(int)
    groups = df["subject_id"].values
    ds_arr = df["dataset"].values

    logo   = LeaveOneGroupOut()
    n_fold = logo.get_n_splits(X, y, groups)
    log.info("[%s] LOSO-CV: %d folds", model_name, n_fold)

    fold_records   = []
    all_y_true     = []
    all_y_prob     = []
    all_datasets   = []
    imp_accum      = np.zeros(len(HRV_FEATURE_COLS))
    imp_count      = 0

    for fold_idx, (tr, te) in enumerate(logo.split(X, y, groups)):
        held = str(groups[te[0]])
        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]

        if len(np.unique(y_tr)) < 2:
            log.warning("[%s] Fold %d (%s): single-class train — skipping",
                        model_name, fold_idx + 1, held)
            continue

        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_tr)
        X_te_sc  = scaler.transform(X_te)

        clf = clf_factory()
        try:
            clf.fit(X_tr_sc, y_tr)
        except Exception as exc:
            log.error("[%s] Fold %d fit failed: %s", model_name, fold_idx + 1, exc)
            continue

        y_prob = clf.predict_proba(X_te_sc)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        single_class = len(np.unique(y_te)) < 2
        if single_class:
            prec = rec = f1 = float("nan")
        else:
            prec = float(precision_score(y_te, y_pred, zero_division=0))
            rec  = float(recall_score(y_te, y_pred, zero_division=0))
            f1   = float(f1_score(y_te, y_pred, zero_division=0))

        fold_records.append({
            "model":            model_name,
            "fold":             fold_idx + 1,
            "held_out_subject": held,
            "dataset":          str(ds_arr[te[0]]),
            "n_train":          len(y_tr),
            "n_test":           len(y_te),
            "test_pos":         int(y_te.sum()),
            "precision_0.5":    prec,
            "recall_0.5":       rec,
            "f1_0.5":           f1,
        })

        all_y_true.extend(y_te.tolist())
        all_y_prob.extend(y_prob.tolist())
        all_datasets.extend([str(ds_arr[te[0]])] * len(y_te))

        imp = _get_importance(clf, len(HRV_FEATURE_COLS))
        if imp is not None:
            imp_accum += imp
            imp_count += 1

        log.info("[%s] Fold %2d | %s | prec=%.3f rec=%.3f f1=%.3f",
                 model_name, fold_idx + 1, held,
                 prec if not np.isnan(prec) else -1,
                 rec  if not np.isnan(rec)  else -1,
                 f1   if not np.isnan(f1)   else -1)

    fold_df    = pd.DataFrame(fold_records)
    y_true_arr = np.array(all_y_true)
    y_prob_arr = np.array(all_y_prob)
    ds_result  = np.array(all_datasets)

    if len(fold_df) == 0 or len(np.unique(y_true_arr)) < 2:
        raise RuntimeError(f"[{model_name}] No valid folds or single-class predictions.")

    def _metrics(yt, yp, tag="all"):
        roc = float(roc_auc_score(yt, yp))
        pr  = float(average_precision_score(yt, yp))
        opt_thr, opt_prec, opt_rec = find_optimal_threshold(yt, yp, min_recall)
        opt_pred = (yp >= opt_thr).astype(int)
        return {
            f"{tag}_roc_auc": round(roc, 4),
            f"{tag}_pr_auc":  round(pr,  4),
            f"{tag}_threshold": round(opt_thr, 4),
            f"{tag}_precision_opt": round(opt_prec, 4),
            f"{tag}_recall_opt":    round(opt_rec,  4),
            f"{tag}_f1_opt":        round(float(f1_score(yt, opt_pred, zero_division=0)), 4),
        }

    summary = {
        "model": model_name,
        "n_folds": len(fold_df),
        "n_windows": len(y_true_arr),
        "mean_precision_0.5": round(float(fold_df["precision_0.5"].mean(skipna=True)), 4),
        "mean_recall_0.5":    round(float(fold_df["recall_0.5"].mean(skipna=True)), 4),
        "mean_f1_0.5":        round(float(fold_df["f1_0.5"].mean(skipna=True)), 4),
    }
    summary.update(_metrics(y_true_arr, y_prob_arr, tag="all"))

    # Cross-dataset breakdown
    for ds_tag in ("drozy", "ddd"):
        mask = ds_result == ds_tag
        if mask.sum() < 10 or len(np.unique(y_true_arr[mask])) < 2:
            log.warning("[%s] Skipping %s-only metrics (insufficient data)", model_name, ds_tag)
            continue
        summary.update(_metrics(y_true_arr[mask], y_prob_arr[mask], tag=ds_tag))

    # Store pooled predictions for plotting (attach to fold_df for convenience)
    summary["_y_true"] = y_true_arr
    summary["_y_prob"] = y_prob_arr
    summary["_ds"]     = ds_result

    imp_mean = imp_accum / max(imp_count, 1)
    return fold_df, summary, imp_mean


# ── figures ───────────────────────────────────────────────────────────────────

def fig_roc_curves(summaries: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#4C9BE8", "#E8614C"]
    for summary, color in zip(summaries, colors):
        yt = summary["_y_true"]
        yp = summary["_y_prob"]
        fpr, tpr, _ = roc_curve(yt, yp)
        auc = summary["all_roc_auc"]
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{summary['model']}  AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Pseudo-Label Models (LOSO-CV)", fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = PSEUDO_KSS_DIR / "fig_roc_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved -> %s", out)


def fig_pr_curves(summaries: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#4C9BE8", "#E8614C"]
    for summary, color in zip(summaries, colors):
        yt = summary["_y_true"]
        yp = summary["_y_prob"]
        prec, rec, _ = precision_recall_curve(yt, yp)
        auprc = summary["all_pr_auc"]
        ax.plot(rec, prec, color=color, lw=2,
                label=f"{summary['model']}  AUPRC={auprc:.4f}")
    # Baseline
    pos_rate = summaries[0]["_y_true"].mean()
    ax.axhline(pos_rate, color="gray", linestyle="--", lw=1,
               label=f"Baseline ({pos_rate:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curves — Pseudo-Label Models (LOSO-CV)", fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = PSEUDO_KSS_DIR / "fig_pr_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved -> %s", out)


def fig_feature_importance(
    imp_dict: dict[str, np.ndarray],
    feature_cols: list[str],
) -> None:
    feat_labels = [FEATURE_LABELS.get(f, f) for f in feature_cols]
    n_models    = len(imp_dict)
    colors = ["#4C9BE8", "#E8614C"][:n_models]

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (name, imp), color in zip(axes, imp_dict.items(), colors):
        norm = imp / (imp.sum() + 1e-12)
        idx  = np.argsort(norm)
        ax.barh(
            [feat_labels[i] for i in idx], norm[idx],
            color=color, edgecolor="black", linewidth=0.5,
        )
        ax.set_title(name, fontweight="bold", fontsize=10)
        ax.set_xlabel("Normalised importance")

    fig.suptitle("Feature Importance — Pseudo-Label Models\n(mean over LOSO folds)",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    out = PSEUDO_KSS_DIR / "fig_feature_importance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved -> %s", out)


# ── comparison table ──────────────────────────────────────────────────────────

def print_and_save_comparison(summaries: list[dict], fold_df: pd.DataFrame) -> None:
    lines = [
        "=" * 80,
        "PSEUDO-LABEL MODEL COMPARISON (LOSO-CV)",
        "=" * 80,
        "",
        f"  {'Model':15s} {'Windows':>8s} {'PR-AUC':>8s} {'ROC-AUC':>8s} "
        f"{'Prec@opt':>9s} {'Rec@opt':>8s} {'F1@opt':>7s}",
        "-" * 80,
    ]
    for s in summaries:
        lines.append(
            f"  {s['model']:15s} {s['n_windows']:>8d} "
            f"{s.get('all_pr_auc', 0.0):>8.4f} {s.get('all_roc_auc', 0.0):>8.4f} "
            f"{s.get('all_precision_opt', 0.0):>9.4f} "
            f"{s.get('all_recall_opt', 0.0):>8.4f} "
            f"{s.get('all_f1_opt', 0.0):>7.4f}"
        )
    lines += ["", "  --- Cross-dataset breakdown ---", ""]
    for s in summaries:
        lines.append(f"  {s['model']}:")
        for ds in ("all", "drozy", "ddd"):
            pr  = s.get(f"{ds}_pr_auc")
            roc = s.get(f"{ds}_roc_auc")
            if pr is None:
                continue
            lines.append(f"    {ds.upper():6s}: PR-AUC={pr:.4f}  ROC-AUC={roc:.4f}")
    lines += ["", "=" * 80]

    text = "\n".join(lines)
    print(text)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PSEUDO_MODEL_COMPARISON_TXT.write_text(text, encoding="utf-8")
    log.info("Comparison saved -> %s", PSEUDO_MODEL_COMPARISON_TXT)

    fold_df.to_csv(PSEUDO_MODEL_RESULTS_CSV, index=False)
    log.info("Per-fold results -> %s", PSEUDO_MODEL_RESULTS_CSV)


# ── main ──────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if not HRV_WINDOWS_FINAL_CSV.exists():
        raise FileNotFoundError(
            f"{HRV_WINDOWS_FINAL_CSV} not found. Run smooth_labels.py first."
        )
    df = pd.read_csv(HRV_WINDOWS_FINAL_CSV)
    log.info("Loaded: %d rows, %d subjects", len(df), df["subject_id"].nunique())

    before = len(df)
    df = df.dropna(subset=HRV_FEATURE_COLS + ["pseudo_label_smoothed"]).copy()
    if len(df) < before:
        log.warning("Dropped %d rows with NaN features/labels", before - len(df))

    if len(df) == 0:
        raise RuntimeError("No samples in the final dataset.")
    return df


def run(min_recall: float = 0.30) -> None:
    log.info("=== Pseudo-Label Model Training ===")

    df = load_data()
    log.info("Training on %d windows, %d subjects, label dist: %s",
             len(df), df["subject_id"].nunique(),
             df["pseudo_label_smoothed"].value_counts().to_dict())

    n_pos = int(df["pseudo_label_smoothed"].sum())
    n_neg = int((df["pseudo_label_smoothed"] == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    log.info("scale_pos_weight=%.2f", scale_pos_weight)

    models: dict[str, Callable] = {}
    try:
        from xgboost import XGBClassifier  # noqa: F401
        models["XGBoost"] = lambda: _make_xgb(scale_pos_weight)
    except ImportError:
        log.warning("xgboost not installed — skipping.")

    try:
        from catboost import CatBoostClassifier  # noqa: F401
        models["CatBoost"] = lambda: _make_catboost(scale_pos_weight)
    except ImportError:
        log.warning("catboost not installed — skipping. Install with: pip install catboost")

    if not models:
        raise RuntimeError("No models available. Install xgboost and/or catboost.")

    all_fold_dfs = []
    all_summaries = []
    all_importances: dict[str, np.ndarray] = {}

    for name, factory in models.items():
        log.info("=== Training: %s ===", name)
        fold_df, summary, imp = run_loso(df, factory, name, min_recall=min_recall)
        all_fold_dfs.append(fold_df)
        all_summaries.append(summary)
        all_importances[name] = imp
        log.info("[%s] PR-AUC=%.4f  ROC-AUC=%.4f",
                 name, summary.get("all_pr_auc", 0), summary.get("all_roc_auc", 0))

    combined_fold_df = pd.concat(all_fold_dfs, ignore_index=True)
    print_and_save_comparison(all_summaries, combined_fold_df)

    # Figures
    PSEUDO_KSS_DIR.mkdir(parents=True, exist_ok=True)
    if len(all_summaries) >= 1:
        fig_roc_curves(all_summaries)
        fig_pr_curves(all_summaries)
    fig_feature_importance(all_importances, list(HRV_FEATURE_COLS))

    log.info("=== Pseudo-Label Model Training complete ===")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train XGBoost + CatBoost on pseudo-labeled HRV windows"
    )
    parser.add_argument("--min-recall", type=float, default=0.30)
    args = parser.parse_args()

    setup_logging()
    run(min_recall=args.min_recall)
