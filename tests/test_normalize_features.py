import pandas as pd
import pytest

from config import HRV_FEATURE_COLS
from normalize_features import ZNORM_COLS, normalize_per_subject


def _feature_row(subject_id, base):
    return {
        "subject_id": subject_id,
        **{col: base + idx for idx, col in enumerate(HRV_FEATURE_COLS)},
    }


def test_normalize_per_subject_adds_zero_mean_unit_std_columns():
    df = pd.DataFrame(
        [
            _feature_row("s1", 10.0),
            _feature_row("s1", 20.0),
            _feature_row("s2", 100.0),
            _feature_row("s2", 130.0),
        ]
    )

    result = normalize_per_subject(df)

    for zcol in ZNORM_COLS:
        assert zcol in result.columns
        for _, group in result.groupby("subject_id"):
            assert group[zcol].mean() == pytest.approx(0.0)
            assert group[zcol].std(ddof=1) == pytest.approx(1.0)


def test_normalize_per_subject_sets_constant_features_to_zero():
    df = pd.DataFrame(
        [
            {"subject_id": "s1", **{col: 5.0 for col in HRV_FEATURE_COLS}},
            {"subject_id": "s1", **{col: 5.0 for col in HRV_FEATURE_COLS}},
        ]
    )

    result = normalize_per_subject(df)

    assert (result[ZNORM_COLS] == 0.0).all().all()
