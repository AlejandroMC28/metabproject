"""Tests for ``run_hca``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metabproject.stats.multivariate.hca import HCAResult, run_hca


def test_run_hca_recovers_planted_split(planted_feature_table: pd.DataFrame) -> None:
    res = run_hca(planted_feature_table, n_clusters=2)

    assert isinstance(res, HCAResult)
    assert res.linkage_matrix.shape == (planted_feature_table.shape[1] - 1, 4)
    assert res.cluster_labels is not None
    assert res.cluster_labels.index.tolist() == list(planted_feature_table.columns)

    # Purity check: the two planted groups should land in different clusters.
    a_labels = res.cluster_labels.loc[[s for s in res.cluster_labels.index if s.startswith("a")]]
    b_labels = res.cluster_labels.loc[[s for s in res.cluster_labels.index if s.startswith("b")]]
    a_mode = a_labels.mode().iloc[0]
    b_mode = b_labels.mode().iloc[0]
    assert a_mode != b_mode
    purity = ((a_labels == a_mode).sum() + (b_labels == b_mode).sum()) / len(res.cluster_labels)
    assert purity >= 0.9


def test_run_hca_without_n_clusters_returns_no_labels(
    planted_feature_table: pd.DataFrame,
) -> None:
    res = run_hca(planted_feature_table)
    assert res.cluster_labels is None
    assert res.metric == "euclidean"
    assert res.linkage_method == "complete"


def test_run_hca_rejects_bad_metric(planted_feature_table: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Unsupported metric"):
        run_hca(planted_feature_table, metric="bogus")


def test_run_hca_rejects_bad_linkage(planted_feature_table: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Unsupported linkage_method"):
        run_hca(planted_feature_table, linkage_method="bogus")


def test_run_hca_rejects_ward_with_non_euclidean(planted_feature_table: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="ward"):
        run_hca(planted_feature_table, metric="cityblock", linkage_method="ward")


def test_run_hca_rejects_bad_n_clusters(planted_feature_table: pd.DataFrame) -> None:
    n_samples = planted_feature_table.shape[1]
    with pytest.raises(ValueError, match=r"n_clusters must be in"):
        run_hca(planted_feature_table, n_clusters=0)
    with pytest.raises(ValueError, match=r"n_clusters must be in"):
        run_hca(planted_feature_table, n_clusters=n_samples + 1)


def test_run_hca_rejects_nan(planted_feature_table: pd.DataFrame) -> None:
    ft = planted_feature_table.copy()
    ft.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        run_hca(ft)


def test_run_hca_rejects_single_sample() -> None:
    ft = pd.DataFrame(
        [[1.0], [2.0]], index=["f1", "f2"], columns=["s1"]
    )
    with pytest.raises(ValueError, match="at least 2 samples"):
        run_hca(ft)
