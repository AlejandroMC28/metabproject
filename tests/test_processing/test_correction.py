"""Tests for sample weight correction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metabproject.processing.correction import weight_correct


def _make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"s1": [100.0, 200.0], "s2": [300.0, 400.0]},
        index=pd.Index([1, 2], name="feature_id"),
    )


def _make_meta(weights: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"group": "ctrl", "weight": list(weights.values())},
        index=pd.Index(list(weights.keys()), name="sample_name"),
    )


def test_divides_by_known_weights():
    df = _make_df()
    meta = _make_meta({"s1": 2.0, "s2": 4.0})
    result = weight_correct(df, meta)

    assert result.loc[1, "s1"] == pytest.approx(50.0)   # 100 / 2
    assert result.loc[2, "s1"] == pytest.approx(100.0)  # 200 / 2
    assert result.loc[1, "s2"] == pytest.approx(75.0)   # 300 / 4
    assert result.loc[2, "s2"] == pytest.approx(100.0)  # 400 / 4


def test_weight_one_is_identity():
    df = _make_df()
    meta = _make_meta({"s1": 1.0, "s2": 1.0})
    result = weight_correct(df, meta)
    pd.testing.assert_frame_equal(result, df)


def test_missing_weight_column_errors():
    df = _make_df()
    meta = pd.DataFrame(
        {"group": ["ctrl", "ctrl"]},
        index=pd.Index(["s1", "s2"], name="sample_name"),
    )
    with pytest.raises(ValueError, match="missing 'weight' column"):
        weight_correct(df, meta)


def test_nonpositive_weight_errors():
    df = _make_df()
    meta = _make_meta({"s1": 0.0, "s2": 1.0})
    with pytest.raises(ValueError, match="weight <= 0"):
        weight_correct(df, meta)


def test_negative_weight_errors():
    df = _make_df()
    meta = _make_meta({"s1": -1.5, "s2": 1.0})
    with pytest.raises(ValueError, match="weight <= 0"):
        weight_correct(df, meta)


def test_returns_copy():
    df = _make_df()
    meta = _make_meta({"s1": 1.0, "s2": 1.0})
    result = weight_correct(df, meta)
    result.iloc[0, 0] = -999.0
    assert df.iloc[0, 0] == 100.0
