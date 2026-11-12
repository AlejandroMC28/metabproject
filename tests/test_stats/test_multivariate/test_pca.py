"""Tests for ``run_pca``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metabproject.stats.multivariate.pca import PCAResult, run_pca


def test_run_pca_separates_planted_groups(planted_feature_table: pd.DataFrame) -> None:
    res = run_pca(planted_feature_table, n_components=5)

    assert isinstance(res, PCAResult)
    assert res.scores.shape == (planted_feature_table.shape[1], 5)
    assert res.loadings.shape == (planted_feature_table.shape[0], 5)
    assert res.n_components_used == 5
    assert list(res.scores.columns) == [f"PC{i + 1}" for i in range(5)]

    # PC1 should separate the two planted groups.
    a_samples = [s for s in res.scores.index if s.startswith("a")]
    b_samples = [s for s in res.scores.index if s.startswith("b")]
    mean_a = res.scores.loc[a_samples, "PC1"].mean()
    mean_b = res.scores.loc[b_samples, "PC1"].mean()
    assert np.sign(mean_a) != np.sign(mean_b)
    assert abs(mean_a - mean_b) > 1.0


def test_run_pca_explained_variance_is_monotone(planted_feature_table: pd.DataFrame) -> None:
    res = run_pca(planted_feature_table, n_components=5)
    evr = res.explained_variance_ratio
    assert np.all(np.diff(evr) <= 1e-12)
    assert 0.0 <= evr.sum() <= 1.0 + 1e-12


def test_run_pca_clamps_components_to_matrix_shape() -> None:
    # 4 features × 3 samples → max components = 3.
    ft = pd.DataFrame(
        np.arange(12, dtype=float).reshape(4, 3) + np.random.default_rng(0).normal(scale=0.1, size=(4, 3)),
        index=pd.Index([f"f{i}" for i in range(4)], name="feature_id"),
        columns=["s1", "s2", "s3"],
    )
    res = run_pca(ft, n_components=10)
    assert res.n_components_used == 3
    assert res.scores.shape == (3, 3)
    assert res.loadings.shape == (4, 3)


def test_run_pca_rejects_nan_input() -> None:
    ft = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, np.nan, 6.0]],
        index=pd.Index(["f1", "f2"], name="feature_id"),
        columns=["s1", "s2", "s3"],
    )
    with pytest.raises(ValueError, match="NaN"):
        run_pca(ft)


def test_run_pca_rejects_degenerate_shapes() -> None:
    one_sample = pd.DataFrame(
        [[1.0], [2.0]], index=["f1", "f2"], columns=["s1"]
    )
    with pytest.raises(ValueError, match="at least 2 features and 2 samples"):
        run_pca(one_sample)

    ft = pd.DataFrame(np.ones((3, 3)), index=["f1", "f2", "f3"], columns=["s1", "s2", "s3"])
    with pytest.raises(ValueError, match="n_components"):
        run_pca(ft, n_components=0)


def test_run_pca_does_not_mutate_input(planted_feature_table: pd.DataFrame) -> None:
    snapshot = planted_feature_table.copy()
    run_pca(planted_feature_table, scale=True)
    pd.testing.assert_frame_equal(planted_feature_table, snapshot)


def test_run_pca_unscaled_still_runs(planted_feature_table: pd.DataFrame) -> None:
    res = run_pca(planted_feature_table, n_components=3, scale=False)
    assert res.n_components_used == 3
