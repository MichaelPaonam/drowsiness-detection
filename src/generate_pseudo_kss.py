"""
Pseudo-KSS label generation via unsupervised clustering.

Uses z-score-normalized HRV features to cluster windows into three drowsiness
states (alert / transitional / drowsy) using KMeans and/or GMM.

Cluster → pseudo-KSS mapping (domain-driven ordering):
  Drowsiness is associated with decreased HRV (lower SDNN, RMSSD) and
  slower, more regular heart rate. Clusters are ranked by mean sdnn_znorm
  (descending), so the highest-SDNN cluster is labelled "alert".

  alert       → pseudo_kss = 2
  transitional→ pseudo_kss = 5
  drowsy      → pseudo_kss = 8   (pseudo_label = 1 because 8 >= 7)

For DROZY windows, alignment with the real session-level KSS is reported
(accuracy, Cohen's kappa, confusion matrix) as an unsupervised validity check.

Outputs
-------
outputs/hrv_windows_pseudo_labeled.csv
outputs/pseudo_kss/fig_cluster_scatter.png
outputs/pseudo_kss/fig_cluster_distribution.png
outputs/pseudo_kss/fig_cluster_centroids.png
outputs/pseudo_kss/fig_silhouette.png

Usage
-----
    python src/generate_pseudo_kss.py
    python src/generate_pseudo_kss.py --method kmeans --exclude-subjects drozy_s04
    python src/generate_pseudo_kss.py --n-clusters 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    silhouette_samples,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    HRV_FEATURE_COLS,
    HRV_WINDOWS_NORMALIZED_CSV,
    HRV_WINDOWS_PSEUDO_CSV,
    KSS_DROWSY_THRESHOLD,
    N_CLUSTERS,
    OUTPUTS_DIR,
    PSEUDO_KSS_DIR,
)

log = logging.getLogger(__name__)

ZNORM_COLS = [f"{c}_znorm" for c in HRV_FEATURE_COLS]
PSEUDO_SCORES = {0: 2, 1: 5, 2: 8}  # ordered label index → pseudo-KSS value
CLUSTER_NAMES = {0: "alert", 1: "transitional", 2: "drowsy"}
CLUSTER_COLORS = {"alert": "#4C9BE8", "transitional": "#F5A623", "drowsy": "#E8614C"}


# ── cluster helpers ───────────────────────────────────────────────────────────


def _order_clusters_by_sdnn(centroids: np.ndarray, sdnn_col_idx: int) -> np.ndarray:
    """
    Return a permutation array that maps original cluster indices to ordered
    indices 0=alert, 1=transitional, 2=drowsy, sorted by SDNN descending
    (higher SDNN = more alert).
    """
    sdnn_vals = centroids[:, sdnn_col_idx]
    # argsort ascending → last element has highest SDNN (most alert)
    rank = np.argsort(sdnn_vals)[::-1]  # [most-alert-idx, mid-idx, drowsy-idx]
    # perm[original_cluster] = ordered_label_index
    perm = np.empty(len(rank), dtype=int)
    for ordered_idx, orig_idx in enumerate(rank):
        perm[orig_idx] = ordered_idx
    return perm


def _labels_to_pseudo_kss(ordered_labels: np.ndarray) -> np.ndarray:
    """Convert ordered label indices (0/1/2) to pseudo-KSS values (2/5/8)."""
    mapping = np.array([PSEUDO_SCORES[i] for i in range(3)])
    return mapping[ordered_labels]


def run_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int,
    sdnn_idx: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit KMeans; return (ordered_labels, centroids_in_scaled_space, silhouette)."""
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    raw_labels = km.fit_predict(X_scaled)
    perm = _order_clusters_by_sdnn(km.cluster_centers_, sdnn_idx)
    ordered = perm[raw_labels]
    sil = float(silhouette_score(X_scaled, raw_labels))
    log.info(
        "KMeans silhouette=%.4f  cluster sizes=%s",
        sil,
        dict(zip(*np.unique(ordered, return_counts=True))),
    )
    return ordered, km.cluster_centers_, sil


def run_gmm(
    X_scaled: np.ndarray,
    n_clusters: int,
    sdnn_idx: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit GMM; return (ordered_labels, means_in_scaled_space, silhouette)."""
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        random_state=random_state,
        n_init=10,
        max_iter=300,
    )
    gmm.fit(X_scaled)
    raw_labels = gmm.predict(X_scaled)
    perm = _order_clusters_by_sdnn(gmm.means_, sdnn_idx)
    ordered = perm[raw_labels]
    sil = float(silhouette_score(X_scaled, raw_labels))
    log.info(
        "GMM silhouette=%.4f  cluster sizes=%s",
        sil,
        dict(zip(*np.unique(ordered, return_counts=True))),
    )
    return ordered, gmm.means_, sil


# ── DROZY ground-truth alignment ──────────────────────────────────────────────


def _parse_drozy_ids(subject_id: str) -> int | None:
    """'drozy_s01' → 1, 'ddd_01M' → None"""
    if subject_id.startswith("drozy_s"):
        try:
            return int(subject_id[7:])
        except ValueError:
            return None
    return None


def validate_against_drozy(df: pd.DataFrame, method: str) -> None:
    """
    For DROZY windows where we have real session-level KSS, compare pseudo_label
    against the ground-truth binary label.
    """
    try:
        from split_utils import load_kss
    except ImportError:
        log.warning("split_utils not importable — skipping DROZY ground-truth validation.")
        return

    kss_df = load_kss()
    usable = kss_df[~kss_df["excluded"]].copy()

    drozy_mask = df["subject_id"].str.startswith("drozy_")
    drozy_df = df[drozy_mask].copy()
    if drozy_df.empty:
        log.info("No DROZY windows found — skipping ground-truth validation.")
        return

    # Map each DROZY window to its session's real label
    def lookup_label(row):
        subj_int = _parse_drozy_ids(row["subject_id"])
        if subj_int is None:
            return np.nan
        trial = int(row["trial_id"])
        match = usable[(usable["subject_id"] == subj_int) & (usable["test_id"] == trial)]
        if match.empty:
            return np.nan
        return int(match["label"].iloc[0])

    drozy_df["gt_label"] = drozy_df.apply(lookup_label, axis=1)
    valid = drozy_df.dropna(subset=["gt_label"])
    if valid.empty:
        log.warning("Could not match any DROZY windows to ground-truth labels.")
        return

    pseudo_col = f"pseudo_label_{method}"
    if pseudo_col not in valid.columns:
        pseudo_col = "pseudo_label"

    y_true = valid["gt_label"].astype(int).values
    y_pred = valid[pseudo_col].astype(int).values

    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n=== DROZY Ground-Truth Alignment ({method.upper()}) ===")
    print(f"  Windows compared : {len(valid)}")
    print(f"  Accuracy         : {acc:.4f}")
    print(f"  Cohen's kappa    : {kappa:.4f}")
    print("  Confusion matrix (rows=true, cols=pred):")
    print(f"    TN={cm[0, 0]}  FP={cm[0, 1]}")
    print(f"    FN={cm[1, 0]}  TP={cm[1, 1]}")
    print()


# ── figures ───────────────────────────────────────────────────────────────────


def _cluster_colors(ordered_labels: np.ndarray) -> list[str]:
    """Map ordered cluster labels to their hex color codes for visualization."""
    names = [CLUSTER_NAMES[cluster_id] for cluster_id in ordered_labels]
    return [CLUSTER_COLORS[n] for n in names]


def fig_cluster_scatter(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    labels_km: np.ndarray | None,
    labels_gmm: np.ndarray | None,
) -> None:
    """Generate and save a 2-D PCA scatter plot of clustered HRV windows."""
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_

    n_panels = sum(x is not None for x in [labels_km, labels_gmm])
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, (method, labels) in zip(axes, [("KMeans", labels_km), ("GMM", labels_gmm)]):
        if labels is None:
            continue
        colors = _cluster_colors(labels)
        ax.scatter(X2[:, 0], X2[:, 1], c=colors, alpha=0.5, s=12, linewidths=0)
        ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var[1] * 100:.1f}%)")
        ax.set_title(f"{method} — PCA Projection", fontweight="bold")
        # Legend
        for name, color in CLUSTER_COLORS.items():
            ax.scatter([], [], c=color, label=name.capitalize(), s=40)
        ax.legend(title="Cluster", loc="best", fontsize=8)

    fig.suptitle("2-D PCA of HRV Windows by Cluster", fontweight="bold")
    fig.tight_layout()
    out = PSEUDO_KSS_DIR / "fig_cluster_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved -> %s", out)


def fig_cluster_distribution(df: pd.DataFrame) -> None:
    """Generate and save a bar chart of pseudo-KSS distribution by dataset and method."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, method in zip(axes, ["kmeans", "gmm"]):
        col = f"pseudo_kss_{method}"
        if col not in df.columns:
            ax.set_visible(False)
            continue
        for ds, grp in df.groupby("dataset"):
            counts = grp[col].value_counts().sort_index()
            ax.bar(
                [x + (0.2 if ds == "ddd" else -0.2) for x in counts.index],
                counts.values,
                width=0.35,
                label=ds.upper(),
                alpha=0.8,
            )
        ax.set_xticks([2, 5, 8])
        ax.set_xticklabels(["2 (Alert)", "5 (Trans.)", "8 (Drowsy)"])
        ax.set_xlabel("Pseudo-KSS")
        ax.set_ylabel("Window count")
        ax.set_title(f"{method.upper()} — Pseudo-KSS Distribution", fontweight="bold")
        ax.legend()

    fig.tight_layout()
    out = PSEUDO_KSS_DIR / "fig_cluster_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved -> %s", out)


def fig_cluster_centroids(
    centroids_km: np.ndarray | None,
    centroids_gmm: np.ndarray | None,
    feature_labels: list[str],
    n_clusters: int,
) -> None:
    """Bar chart of normalised centroid values per cluster, grouped by feature."""
    panels = [
        (c, m) for c, m in [(centroids_km, "KMeans"), (centroids_gmm, "GMM")] if c is not None
    ]
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 4), sharey=False)
    if len(panels) == 1:
        axes = [axes]

    bar_colors = [CLUSTER_COLORS[CLUSTER_NAMES[i]] for i in range(n_clusters)]
    x = np.arange(len(feature_labels))
    width = 0.8 / n_clusters

    for ax, (centroids, method) in zip(axes, panels):
        for ci in range(n_clusters):
            offsets = x + (ci - n_clusters / 2 + 0.5) * width
            ax.bar(
                offsets,
                centroids[ci],
                width * 0.9,
                color=bar_colors[ci],
                edgecolor="black",
                linewidth=0.4,
                label=CLUSTER_NAMES[ci].capitalize(),
            )
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(feature_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Centroid value (scaled space)")
        ax.set_title(f"{method} Cluster Centroids", fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle("Cluster Centroid Feature Profiles", fontweight="bold")
    fig.tight_layout()
    out = PSEUDO_KSS_DIR / "fig_cluster_centroids.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved -> %s", out)


def fig_silhouette(
    X_scaled: np.ndarray,
    labels_km: np.ndarray | None,
    labels_gmm: np.ndarray | None,
    n_clusters: int,
) -> None:
    """Generate and save a silhouette plot for clustering quality assessment."""
    panels = [
        (labels_item, method)
        for labels_item, method in [(labels_km, "KMeans"), (labels_gmm, "GMM")]
        if labels_item is not None
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]

    bar_colors = [CLUSTER_COLORS[CLUSTER_NAMES[i]] for i in range(n_clusters)]

    for ax, (labels, method) in zip(axes, panels):
        sil_vals = silhouette_samples(X_scaled, labels)
        y_lower = 10
        for ci in range(n_clusters):
            ci_sil = np.sort(sil_vals[labels == ci])
            size = len(ci_sil)
            y_upper = y_lower + size
            ax.fill_betweenx(
                np.arange(y_lower, y_upper), 0, ci_sil, color=bar_colors[ci], alpha=0.7
            )
            ax.text(-0.05, (y_lower + y_upper) / 2, CLUSTER_NAMES[ci][:3], fontsize=8, va="center")
            y_lower = y_upper + 10

        avg = float(silhouette_score(X_scaled, labels))
        ax.axvline(avg, color="red", linestyle="--", linewidth=1.2)
        ax.text(avg + 0.01, y_lower * 0.95, f"avg={avg:.3f}", color="red", fontsize=8)
        ax.set_xlabel("Silhouette coefficient")
        ax.set_yticks([])
        ax.set_title(f"{method} Silhouette Plot", fontweight="bold")

    fig.suptitle("Silhouette Analysis of HRV Clustering", fontweight="bold")
    fig.tight_layout()
    out = PSEUDO_KSS_DIR / "fig_silhouette.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved -> %s", out)


# ── main ──────────────────────────────────────────────────────────────────────


def run(
    method: str = "both",
    n_clusters: int = N_CLUSTERS,
) -> pd.DataFrame:
    """Run pseudo-KSS label generation via HRV clustering and save results to CSV."""
    log.info("=== Pseudo-KSS Generation ===")
    log.info("Method=%s  n_clusters=%d", method, n_clusters)

    if not HRV_WINDOWS_NORMALIZED_CSV.exists():
        raise FileNotFoundError(
            f"Normalized CSV not found: {HRV_WINDOWS_NORMALIZED_CSV}. "
            "Run normalize_features.py first."
        )

    df = pd.read_csv(HRV_WINDOWS_NORMALIZED_CSV)
    log.info("Loaded: %d windows, %d subjects", len(df), df["subject_id"].nunique())

    # Drop rows missing any znorm feature
    before = len(df)
    df = df.dropna(subset=ZNORM_COLS).copy()
    if len(df) < before:
        log.warning("Dropped %d rows with NaN znorm features", before - len(df))

    # Scale znorm features for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[ZNORM_COLS].values)

    sdnn_idx = ZNORM_COLS.index("sdnn_znorm")

    # Run requested methods
    labels_km = centroids_km = sil_km = None
    labels_gmm = centroids_gmm = sil_gmm = None

    if method in ("kmeans", "both"):
        labels_km, centroids_km, sil_km = run_kmeans(X_scaled, n_clusters, sdnn_idx)
        df["cluster_kmeans"] = labels_km
        df["pseudo_kss_kmeans"] = _labels_to_pseudo_kss(labels_km)
        df["pseudo_label_kmeans"] = (df["pseudo_kss_kmeans"] >= KSS_DROWSY_THRESHOLD).astype(int)

    if method in ("gmm", "both"):
        labels_gmm, centroids_gmm, sil_gmm = run_gmm(X_scaled, n_clusters, sdnn_idx)
        df["cluster_gmm"] = labels_gmm
        df["pseudo_kss_gmm"] = _labels_to_pseudo_kss(labels_gmm)
        df["pseudo_label_gmm"] = (df["pseudo_kss_gmm"] >= KSS_DROWSY_THRESHOLD).astype(int)

    # Add a canonical "pseudo_kss" column (kmeans preferred; gmm if kmeans not run)
    primary = "kmeans" if method in ("kmeans", "both") else "gmm"
    df["pseudo_kss"] = df[f"pseudo_kss_{primary}"]
    df["pseudo_label"] = df[f"pseudo_label_{primary}"]

    # Print summary
    print("\n=== Cluster Sizes ===")
    if labels_km is not None:
        print("KMeans:")
        for ci in range(n_clusters):
            n = int((labels_km == ci).sum())
            print(f"  {CLUSTER_NAMES[ci]:12s}: {n:5d}  pseudo_kss={PSEUDO_SCORES[ci]}")
        print(f"  Silhouette: {sil_km:.4f}")
    if labels_gmm is not None:
        print("GMM:")
        for ci in range(n_clusters):
            n = int((labels_gmm == ci).sum())
            print(f"  {CLUSTER_NAMES[ci]:12s}: {n:5d}  pseudo_kss={PSEUDO_SCORES[ci]}")
        print(f"  Silhouette: {sil_gmm:.4f}")

    print("\n=== Pseudo-KSS Distribution by Dataset ===")
    for ds, grp in df.groupby("dataset"):
        print(f"  {ds.upper()}:")
        vc = grp["pseudo_kss"].value_counts().sort_index()
        for kss_val, cnt in vc.items():
            pct = cnt / len(grp) * 100
            print(f"    pseudo_kss={kss_val}: {cnt:4d} windows ({pct:.1f}%)")

    # DROZY ground-truth alignment
    if method in ("kmeans", "both"):
        validate_against_drozy(df, "kmeans")
    if method in ("gmm", "both"):
        validate_against_drozy(df, "gmm")

    # Save output
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(HRV_WINDOWS_PSEUDO_CSV, index=False)
    log.info("Pseudo-labeled CSV saved -> %s  (%d rows)", HRV_WINDOWS_PSEUDO_CSV, len(df))

    # Set matplotlib backend before generating figures
    # (deferred to avoid side effects during import)
    import matplotlib

    matplotlib.use("Agg")

    # Figures
    PSEUDO_KSS_DIR.mkdir(parents=True, exist_ok=True)
    fig_cluster_scatter(df, X_scaled, labels_km, labels_gmm)
    fig_cluster_distribution(df)
    fig_cluster_centroids(
        centroids_km,
        centroids_gmm,
        feature_labels=[c.replace("_znorm", "") for c in ZNORM_COLS],
        n_clusters=n_clusters,
    )
    fig_silhouette(X_scaled, labels_km, labels_gmm, n_clusters)

    return df


def setup_logging() -> None:
    """Configure logging for console output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pseudo-KSS labels via HRV clustering")
    parser.add_argument("--method", choices=["kmeans", "gmm", "both"], default="both")
    parser.add_argument("--n-clusters", type=int, default=N_CLUSTERS)
    args = parser.parse_args()

    setup_logging()
    run(
        method=args.method,
        n_clusters=args.n_clusters,
    )
