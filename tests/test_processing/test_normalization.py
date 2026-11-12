"""Tests for sample normalization methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metabproject.processing.normalization import normalize, normalize_median, normalize_pqn, normalize_tic


def _make_df(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.lognormal(mean=10, sigma=1.0, size=(20, 4))
    return pd.DataFrame(
        data,
        index=pd.Index(range(20), name="feature_id"),
        columns=["s1", "s2", "s3", "s4"],
    )


def test_none_returns_copy():
    df = _make_df()
    result = normalize(df, method="none")
    pd.testing.assert_frame_equal(result, df)
    assert result is not df


def test_tic_equalizes_totals():
    """After TIC normalization all column sums are equal to median_tic."""
    df = _make_df()
    result = normalize_tic(df)
    col_sums = result.sum(axis=0)
    # All column sums must equal each other (within float tolerance).
    assert col_sums.std() / col_sums.mean() < 1e-10


def test_tic_preserves_shape(tmp=None):
    df = _make_df()
    result = normalize_tic(df)
    assert result.shape == df.shape


def test_pqn_quotient_based():
    """PQN: the median of (sample / reference) quotients should be ~1 after normalisation."""
    df = _make_df()
    result = normalize_pqn(df)
    reference = df.median(axis=1)
    for col in result.columns:
        quotients = result[col] / reference
        # Median quotient of normalized column vs original reference should be close to 1.
        assert quotients.median() == pytest.approx(1.0, abs=0.01)


def test_pqn_preserves_shape():
    df = _make_df()
    assert normalize_pqn(df).shape == df.shape


def test_median_normalization():
    """Median normalization: per-sample medians are equalized."""
    df = _make_df()
    result = normalize_median(df)
    sample_medians = result.median(axis=0)
    # All per-sample medians should be equal (global median of original).
    assert sample_medians.std() / sample_medians.mean() < 1e-10


def test_median_preserves_shape():
    df = _make_df()
    assert normalize_median(df).shape == df.shape


def test_unknown_method_raises():
    df = _make_df()
    with pytest.raises(ValueError, match="Unknown normalization method"):
        normalize(df, method="bogus")
