import numpy as np
import pytest

from generate_pseudo_kss import _labels_to_pseudo_kss, _order_clusters_by_sdnn, _parse_drozy_ids
from smooth_labels import smooth_session


def test_order_clusters_by_sdnn_maps_highest_sdnn_to_alert():
    centroids = np.array(
        [
            [0.0, -1.0],
            [0.0, 3.0],
            [0.0, 1.0],
        ]
    )

    permutation = _order_clusters_by_sdnn(centroids, sdnn_col_idx=1)

    assert permutation.tolist() == [2, 0, 1]


def test_labels_to_pseudo_kss_preserves_domain_mapping():
    labels = np.array([0, 1, 2, 2, 0])

    assert _labels_to_pseudo_kss(labels).tolist() == [2, 5, 8, 8, 2]


@pytest.mark.parametrize(
    ("subject_id", "expected"),
    [("drozy_s01", 1), ("drozy_s14", 14), ("ddd_01M", None), ("drozy_sxx", None)],
)
def test_parse_drozy_ids(subject_id, expected):
    assert _parse_drozy_ids(subject_id) == expected


def test_smooth_session_can_enforce_monotonic_drowsiness():
    raw = np.array([2, 5, 2, 8, 5])

    smoothed = smooth_session(raw, kernel_size=1, enforce_monotonic=True)

    assert smoothed.tolist() == [2, 5, 5, 8, 8]
