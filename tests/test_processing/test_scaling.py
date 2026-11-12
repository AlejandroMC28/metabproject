"""Tests for centering / scaling methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metabproject.processing.scaling import (
    scale,
    scale_auto,
    scale_pareto,
)


def _positive_df(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.lognormal(mean=10, sigma=1.0, size=(10, 4)) + 1.0
    return pd.DataFrame(
        data,
        index=pd.Index(range(10), name="feature_id"),
        columns=["s1", "s2", "s3", "s4"],
    )


# ---------------------------------------------------------------------------
# pareto
# ---------------------------------------------------------------------------


def test_pareto_centering():
    """After Pareto scaling, per-feature mean is ~0."""
    df = _positive_df()
    result = scale_pareto(df)
    assert result.mean(axis=1).abs().max() < 1e-10


def test_pareto_shape():
    df = _positive_df()
    assert scale_pareto(df).shape == df.shape


# ---------------------------------------------------------------------------
# auto (z-score)
# ---------------------------------------------------------------------------


def test_autoscaling_zero_mean():
    df = _positive_df()
    result = scale_auto(df)
    assert result.mean(axis=1).abs().max() < 1e-10


def test_autoscaling_unit_std():
    df = _positive_df()
    result = scale_auto(df)
    assert (result.std(axis=1) - 1.0).abs().max() < 1e-10


# ---------------------------------------------------------------------------
# none / dispatch
# ---------------------------------------------------------------------------


def test_none_returns_copy():
    df = _positive_df()
    result = scale(df, method="none")
    pd.testing.assert_frame_equal(result, df)
    assert result is not df


def test_unknown_method_raises():
    df = _positive_df()
    with pytest.raises(ValueError, match="Unknown scaling"):
        scale(df, method="magic")
