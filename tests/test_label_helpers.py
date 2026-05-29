import numpy as np
import pytest

from smooth_labels import smooth_session


def test_run_kmeans_orders_clusters_by_sdnn_descending():
    """
    Test that run_kmeans orders clusters by SDNN descending (highest SDNN → label 0).
    This validates the critical invariant that drowsiness is associated with lower SDNN.
    """
    from generate_pseudo_kss import run_kmeans

    # Create synthetic data with three clear clusters:
    # Cluster A: high SDNN (alert)
    # Cluster B: mid SDNN (transitional)
    # Cluster C: low SDNN (drowsy)
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(10, 2) + np.array([5.0, 3.0]),  # high SDNN at index 1
        np.random.randn(10, 2) + np.array([0.0, 1.0]),  # mid SDNN
        np.random.randn(10, 2) + np.array([-5.0, -3.0]),  # low SDNN
    ])
    
    labels, centroids, _ = run_kmeans(X, n_clusters=3, sdnn_idx=1, random_state=42)
    
    # Extract the cluster assignments for each data point
    # Points from the first cluster (high SDNN) should map to label 0 (alert)
    first_cluster_labels = labels[:10]
    assert np.all(first_cluster_labels == 0), "High SDNN cluster should be label 0 (alert)"
    
    # Last cluster (low SDNN) should map to label 2 (drowsy)
    last_cluster_labels = labels[20:]
    assert np.all(last_cluster_labels == 2), "Low SDNN cluster should be label 2 (drowsy)"


def test_run_kmeans_produces_valid_label_range():
    """
    Test that run_kmeans produces labels in the range [0, 1, 2],
    which map to pseudo-KSS values [2, 5, 8].
    """
    from generate_pseudo_kss import run_kmeans

    np.random.seed(42)
    X = np.random.randn(30, 2)
    
    labels, _, _ = run_kmeans(X, n_clusters=3, sdnn_idx=0, random_state=42)
    
    assert np.all((labels >= 0) & (labels <= 2)), "Labels should be in range [0, 1, 2]"
    assert set(labels) == {0, 1, 2}, "All three labels should be present"


def test_parse_drozy_ids_extracts_subject_numbers():
    """
    Test the critical invariant that DROZY subject IDs follow a specific format.
    This is a domain-critical parsing utility used throughout the pipeline.
    """
    from generate_pseudo_kss import _parse_drozy_ids

    assert _parse_drozy_ids("drozy_s01") == 1
    assert _parse_drozy_ids("drozy_s14") == 14
    assert _parse_drozy_ids("ddd_01M") is None
    assert _parse_drozy_ids("drozy_sxx") is None


def test_smooth_session_can_enforce_monotonic_drowsiness():
    """Test that monotonic smoothing increases (or maintains) labels over time."""
    raw = np.array([2, 5, 2, 8, 5])

    smoothed = smooth_session(raw, kernel_size=1, enforce_monotonic=True)

    assert smoothed.tolist() == [2, 5, 5, 8, 8]
