import numpy as np
import pytest

from hrv_metrics import _calculate_time_domain_features


def test_calculate_time_domain_features_uses_expected_formulas():
    rr_ms = np.array([1000.0, 1020.0, 970.0, 1040.0])

    features = _calculate_time_domain_features(rr_ms)

    assert features.mean_rr == pytest.approx(1007.5)
    assert features.mean_hr == pytest.approx(60000.0 / 1007.5)
    assert features.sdnn == pytest.approx(np.std(rr_ms, ddof=1))
    assert features.rmssd == pytest.approx(np.sqrt(np.mean(np.array([20.0, 50.0, 70.0]) ** 2)))
    assert features.nn50 == 1
    assert features.pnn50 == pytest.approx(100.0 / 3.0)
    assert features.cv_rr == pytest.approx(np.std(rr_ms, ddof=1) / 1007.5)


def test_hrv_features_to_dict_keeps_feature_contract():
    features = _calculate_time_domain_features(np.array([1000.0, 1010.0, 990.0]))

    assert list(features.to_dict()) == [
        "mean_rr",
        "mean_hr",
        "sdnn",
        "rmssd",
        "nn50",
        "pnn50",
        "cv_rr",
    ]
