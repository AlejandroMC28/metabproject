"""Tests for data transformation / scaling methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metabproject.processing.transformation import (
    transform,
    transform_log2,
    transform_log10,
    transform_sqrt,
)


def _positive_df(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.lognormal(mean=10, sigma=1.0, size=(10, 4)) + 1.0
    return pd.DataFrame(
        data,
        index=pd.Index(range(10), name="feature_id"),
        columns=["s1", "s2", "s3", "s4"],
    )


def _df_with_zeros() -> pd.DataFrame:
    df = _positive_df()
    df.iloc[0, 0] = 0.0
    return df


# ---------------------------------------------------------------------------
# log2
# ---------------------------------------------------------------------------


def test_log2_matches_manual():
    df = _positive_df()
    result = transform_log2(df)
    expected = np.log2(df)
    pd.testing.assert_frame_equal(result, expected)


def test_log2_errors_on_zeros():
    df = _df_with_zeros()
    with pytest.raises(ValueError, match="log2"):
        transform_log2(df)


def test_log2_errors_on_nan():
    df = _positive_df()
    df.iloc[1, 1] = np.nan
    with pytest.raises(ValueError, match="log2"):
        transform_log2(df)


# ---------------------------------------------------------------------------
# log10
# ---------------------------------------------------------------------------


def test_log10_matches_manual():
    df = _positive_df()
    result = transform_log10(df)
    expected = np.log10(df)
    pd.testing.assert_frame_equal(result, expected)


def test_log10_errors_on_zeros():
    df = _df_with_zeros()
    with pytest.raises(ValueError, match="log10"):
        transform_log10(df)


# ---------------------------------------------------------------------------
# sqrt
# ---------------------------------------------------------------------------


def test_sqrt_matches_manual():
    df = _positive_df()
    result = transform_sqrt(df)
    expected = np.sqrt(df)
    pd.testing.assert_frame_equal(result, expected)


def test_sqrt_allows_zeros():
    df = _positive_df()
    df.iloc[0, 0] = 0.0
    result = transform_sqrt(df)
    assert result.iloc[0, 0] == pytest.approx(0.0)


def test_sqrt_errors_on_negative():
    df = _positive_df()
    df.iloc[0, 0] = -1.0
    with pytest.raises(ValueError, match="negative"):
        transform_sqrt(df)


# ---------------------------------------------------------------------------
# none / dispatch
# ---------------------------------------------------------------------------


def test_none_returns_copy():
    df = _positive_df()
    result = transform(df, method="none")
    pd.testing.assert_frame_equal(result, df)
    assert result is not df


def test_unknown_method_raises():
    df = _positive_df()
    with pytest.raises(ValueError, match="Unknown transformation"):
        transform(df, method="magic")
