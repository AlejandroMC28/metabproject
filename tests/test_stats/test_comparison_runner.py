"""Tests for the Comparison runner: dispatch + 2-way Tukey routing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metabproject.stats.comparison_runner import ComparisonResult, run_comparison
from metabproject.stats.comparisons import Comparison


SEED = 0


def _three_group_dataset(
    n_features: int = 6, n_per: int = 6, planted: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    groups = ["A"] * n_per + ["B"] * n_per + ["C"] * n_per
    samples = [f"s{i + 1}" for i in range(len(groups))]
    a = rng.normal(loc=0.0, scale=0.5, size=(planted, n_per))
    b = rng.normal(loc=5.0, scale=0.5, size=(planted, n_per))
    c = rng.normal(loc=10.0, scale=0.5, size=(planted, n_per))
    null = rng.normal(size=(n_features - planted, len(groups)))
    ft = pd.DataFrame(
        np.vstack([np.hstack([a, b, c]), null]),
        index=pd.Index([f"f{i + 1}" for i in range(n_features)], name="feature_id"),
        columns=samples,
    )
    meta = pd.DataFrame({"group": groups}, index=pd.Index(samples, name="sample"))
    return ft, meta


def _two_way_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    samples = []
    f1_col = []
    f2_col = []
    for l1 in ["X", "Y"]:
        for l2 in ["P", "Q"]:
            for r in range(6):
                samples.append(f"{l1}{l2}{r + 1}")
                f1_col.append(l1)
                f2_col.append(l2)
    meta = pd.DataFrame(
        {"f1": f1_col, "f2": f2_col},
        index=pd.Index(samples, name="sample"),
    )

    f1_arr = np.array(f1_col)
    f2_arr = np.array(f2_col)
    n = len(samples)
    rows = np.vstack([
        # f1: factor1_only
        np.where(f1_arr == "Y", 5.0, 0.0) + rng.normal(scale=0.3, size=n),
        # f2: factor2_only
        np.where(f2_arr == "Q", 5.0, 0.0) + rng.normal(scale=0.3, size=n),
        # f3: both_no_interaction
        (
            np.where(f1_arr == "Y", 5.0, 0.0)
            + np.where(f2_arr == "Q", 5.0, 0.0)
            + rng.normal(scale=0.3, size=n)
        ),
        # f4: interaction
        np.where(
            ((f1_arr == "X") & (f2_arr == "P")) | ((f1_arr == "Y") & (f2_arr == "Q")),
            5.0,
            0.0,
        ) + rng.normal(scale=0.3, size=n),
        # f5: ns
        rng.normal(scale=1.0, size=n),
    ])
    ft = pd.DataFrame(
        rows,
        index=pd.Index([f"f{i + 1}" for i in range(5)], name="feature_id"),
        columns=samples,
    )
    return ft, meta


# ---------------------------------------------------------------------------


def test_run_comparison_anova_1way_with_tukey() -> None:
    ft, meta = _three_group_dataset()
    cmp = Comparison(
        name="x", kind="anova_1way", factors=["group"], posthoc="tukey",
    )
    result = run_comparison(cmp, ft, meta)

    assert isinstance(result, ComparisonResult)
    assert result.comparison_name == "x"
    assert result.kind == "anova_1way"
    # Planted features must be significant.
    assert result.omnibus.loc[[f"f{i}" for i in range(1, 5)], "significant"].all()
    # Post-hoc only on significant features → 4 features × 3 pairs = 12 rows.
    assert result.posthoc is not None
    assert len(result.posthoc) == 4 * 3
    assert set(result.posthoc["feature"]) == {"f1", "f2", "f3", "f4"}


def test_run_comparison_no_posthoc_returns_none() -> None:
    ft, meta = _three_group_dataset()
    cmp = Comparison(name="x", kind="anova_1way", factors=["group"], posthoc=None)
    result = run_comparison(cmp, ft, meta)
    assert result.posthoc is None


def test_run_comparison_welch_anova_with_games_howell() -> None:
    ft, meta = _three_group_dataset()
    cmp = Comparison(
        name="x", kind="welch_anova", factors=["group"], posthoc="games_howell",
    )
    result = run_comparison(cmp, ft, meta)
    assert "F" in result.omnibus.columns
    assert result.posthoc is not None
    assert "hedges_g" in result.posthoc.columns


def test_run_comparison_kruskal_with_dunn() -> None:
    ft, meta = _three_group_dataset()
    cmp = Comparison(
        name="x", kind="kruskal", factors=["group"], posthoc="dunn",
    )
    result = run_comparison(cmp, ft, meta)
    assert "H" in result.omnibus.columns
    assert result.posthoc is not None
    assert "rank_mean_diff" in result.posthoc.columns


def test_run_comparison_anova_2way_routes_tukey_by_bucket() -> None:
    ft, meta = _two_way_dataset()
    cmp = Comparison(
        name="x", kind="anova_2way", factors=["f1", "f2"], posthoc="tukey",
    )
    result = run_comparison(cmp, ft, meta)
    assert result.posthoc is not None

    by_feature_factor = result.posthoc.groupby(["feature", "factor"]).size().reset_index()
    # f1 (factor1_only) → tukey on f1 only.
    f1_factors = set(result.posthoc.loc[result.posthoc["feature"] == "f1", "factor"])
    assert f1_factors == {"f1"}
    # f2 (factor2_only) → tukey on f2 only.
    f2_factors = set(result.posthoc.loc[result.posthoc["feature"] == "f2", "factor"])
    assert f2_factors == {"f2"}
    # f3 (both_no_interaction) → tukey on both f1 AND f2.
    f3_factors = set(result.posthoc.loc[result.posthoc["feature"] == "f3", "factor"])
    assert f3_factors == {"f1", "f2"}
    # f4 (interaction or interaction_only) → tukey on the combined factor.
    f4_factors = set(result.posthoc.loc[result.posthoc["feature"] == "f4", "factor"])
    assert f4_factors == {"_combined_f1_x_f2"}
    # f5 (ns) → no post-hoc rows.
    assert "f5" not in set(result.posthoc["feature"])
    # by_feature_factor is silent sanity-check the groupby ran cleanly.
    assert len(by_feature_factor) > 0


def test_run_comparison_anova_2way_no_posthoc() -> None:
    ft, meta = _two_way_dataset()
    cmp = Comparison(
        name="x", kind="anova_2way", factors=["f1", "f2"], posthoc=None,
    )
    result = run_comparison(cmp, ft, meta)
    assert result.posthoc is None


def test_run_comparison_validates_factor_missing() -> None:
    ft, meta = _three_group_dataset()
    cmp = Comparison(name="x", kind="anova_1way", factors=["does_not_exist"])
    with pytest.raises(ValueError, match="not found in metadata.columns"):
        run_comparison(cmp, ft, meta)
