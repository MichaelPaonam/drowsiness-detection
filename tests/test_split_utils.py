import pandas as pd
import pytest

from config import HRV_FEATURE_COLS
from split_utils import assign_split_column, get_xy, split_features_df


def _row(subject_id, label, excluded=False):
    return {
        "subject_id": subject_id,
        "label": label,
        "excluded": excluded,
        **{col: float(subject_id) for col in HRV_FEATURE_COLS},
    }


def test_assign_split_column_respects_fixed_split_and_exclusions():
    df = pd.DataFrame([_row(1, 0), _row(3, 0), _row(12, 1), _row(99, 1, excluded=True)])

    result = assign_split_column(df)

    assert result.loc[0, "split"] == "train"
    assert result.loc[1, "split"] == "val"
    assert result.loc[2, "split"] == "test"
    assert pd.isna(result.loc[3, "split"])


def test_split_features_df_keeps_subjects_in_single_splits():
    df = pd.DataFrame([_row(1, 0), _row(2, 1), _row(3, 0), _row(12, 1)])

    splits = split_features_df(df)

    assert set(splits) == {"train", "val", "test"}
    assert set(splits["train"]["subject_id"]) == {1, 2}
    assert set(splits["val"]["subject_id"]) == {3}
    assert set(splits["test"]["subject_id"]) == {12}


def test_get_xy_requires_all_feature_columns():
    df = pd.DataFrame([_row(1, 0)]).drop(columns=[HRV_FEATURE_COLS[0]])

    with pytest.raises(ValueError, match="Feature columns missing"):
        get_xy(df)
