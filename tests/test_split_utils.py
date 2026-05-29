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


def test_get_xy_positive_path():
    # Construct a dataframe with all required feature columns and a label column
    df = pd.DataFrame([_row(1, 0), _row(2, 1), _row(3, 0)])

    X, y = get_xy(df)

    # X should contain exactly the HRV feature columns in the given order
    assert list(X.columns) == HRV_FEATURE_COLS

    # All rows should be included (get_xy does not filter)
    assert len(X) == 3
    assert len(y) == 3

    # Indices should be preserved
    assert list(X.index) == [0, 1, 2]
    assert list(y.index) == [0, 1, 2]

    # y should match the label column for all rows
    assert list(y) == [0, 1, 0]

    # Feature values should correspond to subject_id (as set by _row helper)
    expected_values = [1.0, 2.0, 3.0]
    for col in HRV_FEATURE_COLS:
        assert list(X[col]) == expected_values
