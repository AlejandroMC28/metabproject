"""Tests for imputation methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metabproject.processing.imputation import (
    impute,
    impute_half_min,
    impute_lod,
    impute_min,
    impute_second_min,
)

N_FEATURES = 10
N_SAMPLES = 4


def _make_table(seed: int = 0) -> pd.DataFrame:
    """Small feature table with known zeros (missing values)."""
    rng = np.random.default_rng(seed)
    data = rng.lognormal(mean=10, sigma=1.5, size=(N_FEATURES, N_SAMPLES))
    # Punch in zeros at specific positions.
    data[0, 0] = 0.0
    data[1, 1] = 0.0
    data[2, 2] = np.nan
    return pd.DataFrame(
        data,
        index=pd.Index(range(N_FEATURES), name="feature_id"),
        columns=[f"sample_{i}" for i in range(N_SAMPLES)],
    )


# ---------------------------------------------------------------------------
# half-min
# ---------------------------------------------------------------------------


def test_half_min_seeded_reproducible():
    """Same seed yields identical imputed values."""
    df = _make_table()
    result1, _ = impute_half_min(df, seed=42)
    result2, _ = impute_half_min(df, seed=42)
    pd.testing.assert_frame_equal(result1, result2)


def test_half_min_different_seeds_differ():
    df = _make_table()
    r1, _ = impute_half_min(df, seed=1)
    r2, _ = impute_half_min(df, seed=2)
    assert not r1.equals(r2)


def test_half_min_value_range():
    """Imputed values fall within [0.1*global_min, 0.5*global_min]."""
    df = _make_table()
    positive_vals = df.values[np.isfinite(df.values) & (df.values > 0)]
    global_min = float(positive_vals.min())

    result, n = impute_half_min(df, seed=42)
    assert n == 3  # three missing values planted

    # Find the imputed positions (were zero or NaN) and check only those.
    was_missing = ((df == 0) | df.isna()).values
    imputed_vals = result.values[was_missing]
    for val in imputed_vals:
        assert global_min * 0.1 <= val <= global_min * 0.5


def test_half_min_no_zeros_remain():
    df = _make_table()
    result, _ = impute_half_min(df, seed=42)
    assert (result == 0).sum().sum() == 0
    assert result.isna().sum().sum() == 0


def test_half_min_no_missing_returns_unchanged():
    df = pd.DataFrame(
        np.ones((3, 3)) * 100.0,
        index=pd.Index([1, 2, 3], name="feature_id"),
        columns=["a", "b", "c"],
    )
    result, n = impute_half_min(df, seed=42)
    assert n == 0
    pd.testing.assert_frame_equal(result, df)


# ---------------------------------------------------------------------------
# min
# ---------------------------------------------------------------------------


def test_min_fills_with_global_min():
    df = _make_table()
    positive_vals = df.values[np.isfinite(df.values) & (df.values > 0)]
    global_min = float(positive_vals.min())

    result, n = impute_min(df)
    assert n == 3
    was_missing = ((df == 0) | df.isna()).values
    for val in result.values[was_missing]:
        assert val == pytest.approx(global_min)


def test_min_deterministic():
    df = _make_table()
    r1, _ = impute_min(df)
    r2, _ = impute_min(df)
    pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# lod
# ---------------------------------------------------------------------------


def test_lod_fills_with_specified_value():
    df = _make_table()
    result, n = impute_lod(df, lod_value=5.0)
    assert n == 3
    was_missing = ((df == 0) | df.isna()).values
    for val in result.values[was_missing]:
        assert val == pytest.approx(5.0)


def test_lod_requires_value_via_dispatch():
    df = _make_table()
    with pytest.raises(ValueError, match="LOD imputation requires"):
        impute(df, method="lod", lod_value=None)


# ---------------------------------------------------------------------------
# none
# ---------------------------------------------------------------------------


def test_none_returns_original():
    df = _make_table()
    result, n = impute(df, method="none")
    assert n == 0
    pd.testing.assert_frame_equal(result, df)


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def test_unknown_method_raises():
    df = _make_table()
    with pytest.raises(ValueError, match="Unknown imputation method"):
        impute(df, method="magic")


# ---------------------------------------------------------------------------
# second-min
# ---------------------------------------------------------------------------


def test_second_min_fills_within_bounds_and_is_seeded():
    df = _make_table()
    positive_vals = df.values[np.isfinite(df.values) & (df.values > 0)]
    second_min = float(np.unique(positive_vals)[1])

    result_a, n_a = impute_second_min(df, seed=42)
    result_b, n_b = impute_second_min(df, seed=42)

    # Reproducibility
    pd.testing.assert_frame_equal(result_a, result_b)
    assert n_a == n_b == 3

    # No remaining zeros / NaN
    assert not ((result_a == 0) | result_a.isna()).any().any()

    # Imputed cells fall within (0, second_min]
    mask = (df == 0) | df.isna()
    imputed_cells = result_a.values[mask.values]
    assert (imputed_cells > 0).all()
    assert (imputed_cells <= second_min).all()


def test_second_min_dispatch():
    df = _make_table()
    result, n = impute(df, method="second-min", seed=42)
    assert n == 3
    assert not ((result == 0) | result.isna()).any().any()
