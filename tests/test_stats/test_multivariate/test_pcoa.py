"""Tests for ``run_pcoa``.

Skipped when ``scikit-bio`` is not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("skbio")

from metabproject.stats.multivariate.pcoa import PCoAResult, run_pcoa  # noqa: E402


def test_run_pcoa_recovers_planted_group_structure(
    planted_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ft, md = planted_dataset
    res = run_pcoa(ft, md, group_col="group", permutations=199, seed=0)

    assert isinstance(res, PCoAResult)
    assert res.scores.shape[0] == ft.shape[1]
    assert list(res.scores.columns)[0] == "PC1"
    assert res.distance_matrix.shape == (ft.shape[1], ft.shape[1])

    assert res.permanova["sample_size"] == ft.shape[1]
    assert res.permanova["number_of_groups"] == 2
    assert res.permanova["p_value"] < 0.05
    assert 0.0 <= res.permanova["R2"] <= 1.0

    # Manual R² formula check.
    f = res.permanova["test_statistic"]
    n = res.permanova["sample_size"]
    k = res.permanova["number_of_groups"]
    expected = (f * (k - 1) / (n - k)) / (1 + f * (k - 1) / (n - k))
    assert np.isclose(res.permanova["R2"], expected)

    assert 0.0 <= res.permdisp["p_value"] <= 1.0


def test_run_pcoa_rejects_unknown_metric(planted_dataset: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    ft, md = planted_dataset
    with pytest.raises(ValueError, match="Unsupported metric"):
        run_pcoa(ft, md, group_col="group", metric="cosine")


def test_run_pcoa_rejects_missing_group_col(
    planted_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ft, md = planted_dataset
    with pytest.raises(ValueError, match="metadata has no column"):
        run_pcoa(ft, md, group_col="treatment")


def test_run_pcoa_rejects_undersized_group(
    planted_feature_table: pd.DataFrame,
) -> None:
    md = pd.DataFrame(
        {"group": ["A"] * 19 + ["B"]},
        index=planted_feature_table.columns,
    )
    md.index.name = "sample_name"
    with pytest.raises(ValueError, match="<2 samples"):
        run_pcoa(planted_feature_table, md, group_col="group")


def test_run_pcoa_rejects_single_level_group(
    planted_feature_table: pd.DataFrame,
) -> None:
    md = pd.DataFrame(
        {"group": ["A"] * planted_feature_table.shape[1]},
        index=planted_feature_table.columns,
    )
    md.index.name = "sample_name"
    with pytest.raises(ValueError, match=">=2 distinct levels"):
        run_pcoa(planted_feature_table, md, group_col="group")


def test_run_pcoa_rejects_nan(
    planted_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ft, md = planted_dataset
    ft = ft.copy()
    ft.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        run_pcoa(ft, md, group_col="group")


def test_run_pcoa_rejects_samples_absent_from_metadata(
    planted_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ft, md = planted_dataset
    md = md.iloc[:5].copy()
    with pytest.raises(ValueError, match="absent from metadata"):
        run_pcoa(ft, md, group_col="group")
