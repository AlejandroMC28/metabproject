"""Tests for the per-feature post-hoc tests (Tukey, Games-Howell, Dunn)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metabproject.stats.posthoc import dunn, games_howell, tukey_hsd


SEED = 0


def _build_three_group_dataset(
    n_features: int,
    n_per_group: int,
    means: tuple[float, float, float] = (0.0, 5.0, 10.0),
    scales: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    groups = ["A"] * n_per_group + ["B"] * n_per_group + ["C"] * n_per_group
    samples = [f"s{i + 1}" for i in range(len(groups))]
    a = rng.normal(loc=means[0], scale=scales[0], size=(n_features, n_per_group))
    b = rng.normal(loc=means[1], scale=scales[1], size=(n_features, n_per_group))
    c = rng.normal(loc=means[2], scale=scales[2], size=(n_features, n_per_group))
    ft = pd.DataFrame(
        np.hstack([a, b, c]),
        index=pd.Index([f"f{i + 1}" for i in range(n_features)], name="feature_id"),
        columns=samples,
    )
    meta = pd.DataFrame({"group": groups}, index=pd.Index(samples, name="sample"))
    return ft, meta


def test_tukey_returns_three_pairs_with_expected_signs() -> None:
    ft, meta = _build_three_group_dataset(n_features=3, n_per_group=6)
    res = tukey_hsd(ft, meta, factor="group", features_to_test=["f1", "f2", "f3"])

    assert list(res.columns) == [
        "feature", "factor", "group_A", "group_B",
        "mean_A", "mean_B", "diff", "p_value", "significant", "hedges_g",
    ]
    # 3 features × 3 pairwise comparisons = 9 rows.
    assert len(res) == 9
    assert set(res["feature"]) == {"f1", "f2", "f3"}
    assert (res["factor"] == "group").all()
    # diff = mean_B - mean_A by convention. Alphabetical A < B < C and planted
    # means A < B < C → mean_B > mean_A on every pair, so diff > 0.
    for _, r in res.iterrows():
        assert r["diff"] > 0
    # With strong separation every pairwise comparison should be significant.
    assert res["significant"].all()


def test_tukey_empty_features_to_test_returns_empty_frame() -> None:
    ft, meta = _build_three_group_dataset(n_features=2, n_per_group=6)
    res = tukey_hsd(ft, meta, factor="group", features_to_test=[])
    assert len(res) == 0
    assert list(res.columns) == [
        "feature", "factor", "group_A", "group_B",
        "mean_A", "mean_B", "diff", "p_value", "significant", "hedges_g",
    ]


def test_games_howell_under_heteroscedasticity_finds_pair() -> None:
    # Same means, very different variances → Welch-style adjustment matters.
    ft, meta = _build_three_group_dataset(
        n_features=4, n_per_group=8, means=(0.0, 5.0, 10.0), scales=(0.5, 2.0, 5.0),
    )
    res = games_howell(ft, meta, factor="group", features_to_test=["f1", "f2", "f3", "f4"])
    assert list(res.columns) == [
        "feature", "factor", "group_A", "group_B",
        "mean_A", "mean_B", "diff", "p_value", "significant", "hedges_g",
    ]
    # All 3 pairwise comparisons exist per feature.
    assert len(res) == 12
    # A vs C (largest separation) should be significant for all features.
    ac = res[(res["group_A"] == "A") & (res["group_B"] == "C")]
    assert ac["significant"].all()


def test_dunn_returns_rank_mean_diff_and_pairs() -> None:
    ft, meta = _build_three_group_dataset(n_features=2, n_per_group=6)
    res = dunn(ft, meta, factor="group", features_to_test=["f1", "f2"])
    assert list(res.columns) == [
        "feature", "factor", "group_A", "group_B",
        "rank_mean_diff", "p_value", "significant",
    ]
    # 2 features × 3 unique pairs = 6 rows.
    assert len(res) == 6
    # A < C in rank mean → rank_mean_diff (A − C) should be negative.
    ac = res[(res["group_A"] == "A") & (res["group_B"] == "C")]
    assert (ac["rank_mean_diff"] < 0).all()
    # Strong separation → at minimum the widest pair (A vs C) is significant
    # for every feature under default p_adjust='fdr_bh'.
    assert ac["significant"].all()


def test_dunn_empty_features_to_test_returns_empty_frame() -> None:
    ft, meta = _build_three_group_dataset(n_features=2, n_per_group=6)
    res = dunn(ft, meta, factor="group", features_to_test=[])
    assert len(res) == 0
    assert "rank_mean_diff" in res.columns


def test_tukey_subset_of_features_only() -> None:
    ft, meta = _build_three_group_dataset(n_features=5, n_per_group=6)
    res = tukey_hsd(ft, meta, factor="group", features_to_test=["f2", "f4"])
    assert set(res["feature"]) == {"f2", "f4"}
    assert len(res) == 6  # 2 features × 3 pairs


def _pg_long(ft: pd.DataFrame, meta: pd.DataFrame, feature: str) -> pd.DataFrame:
    row = ft.loc[feature]
    return pd.DataFrame({"value": row.values, "group": meta["group"].loc[row.index].values})


def test_tukey_matches_pingouin() -> None:
    pg = pytest.importorskip("pingouin")
    ft, meta = _build_three_group_dataset(n_features=3, n_per_group=6)
    res = tukey_hsd(ft, meta, factor="group", features_to_test=list(ft.index))
    for feature in ft.index:
        ref = pg.pairwise_tukey(data=_pg_long(ft, meta, feature), dv="value", between="group")
        sub = res[res["feature"] == feature]
        for _, ref_row in ref.iterrows():
            a, b = str(ref_row["A"]), str(ref_row["B"])
            match = sub[(sub["group_A"] == a) & (sub["group_B"] == b)]
            assert len(match) == 1
            row = match.iloc[0]
            # pingouin uses diff = mean_A - mean_B; we use diff = mean_B - mean_A.
            assert abs(row["diff"] + float(ref_row["diff"])) < 1e-6
            assert abs(row["p_value"] - float(ref_row["p_tukey"])) < 1e-4
            assert abs(row["hedges_g"] + float(ref_row["hedges"])) < 1e-3


def test_gameshowell_matches_pingouin() -> None:
    pg = pytest.importorskip("pingouin")
    ft, meta = _build_three_group_dataset(
        n_features=3, n_per_group=8, means=(0.0, 2.0, 4.0), scales=(0.5, 2.0, 5.0),
    )
    res = games_howell(ft, meta, factor="group", features_to_test=list(ft.index))
    for feature in ft.index:
        ref = pg.pairwise_gameshowell(
            data=_pg_long(ft, meta, feature), dv="value", between="group"
        )
        sub = res[res["feature"] == feature]
        for _, ref_row in ref.iterrows():
            a, b = str(ref_row["A"]), str(ref_row["B"])
            match = sub[(sub["group_A"] == a) & (sub["group_B"] == b)]
            assert len(match) == 1
            row = match.iloc[0]
            assert abs(row["diff"] - float(ref_row["diff"])) < 1e-6
            assert abs(row["p_value"] - float(ref_row["pval"])) < 1e-4
            assert abs(row["hedges_g"] - float(ref_row["hedges"])) < 1e-3
