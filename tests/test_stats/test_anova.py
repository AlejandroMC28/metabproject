"""Tests for the per-feature ANOVA routines (1-way, Welch, 2-way)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metabproject.stats.anova import anova_1way, anova_2way, welch_anova


SEED = 0


def _make_table(n_features: int, sample_names: list[str]) -> pd.DataFrame:
    idx = pd.Index([f"f{i + 1}" for i in range(n_features)], name="feature_id")
    return pd.DataFrame(np.zeros((n_features, len(sample_names))), index=idx, columns=sample_names)


def _metadata(samples: list[str], factors: dict[str, list[str]]) -> pd.DataFrame:
    return pd.DataFrame(factors, index=pd.Index(samples, name="sample"))


# ---------------------------------------------------------------------------
# 1-way ANOVA


def test_anova_1way_recovers_planted_group_effect() -> None:
    rng = np.random.default_rng(SEED)
    n_features = 30
    n_per = 6
    groups = ["A"] * n_per + ["B"] * n_per + ["C"] * n_per
    samples = [f"s{i + 1}" for i in range(len(groups))]
    ft = _make_table(n_features, samples)
    # 20 features with a strong group effect; 10 nulls.
    n_signal = 20
    a = rng.normal(loc=0.0, scale=0.5, size=(n_signal, n_per))
    b = rng.normal(loc=5.0, scale=0.5, size=(n_signal, n_per))
    c = rng.normal(loc=10.0, scale=0.5, size=(n_signal, n_per))
    ft.iloc[:n_signal] = np.hstack([a, b, c])
    ft.iloc[n_signal:] = rng.normal(size=(n_features - n_signal, len(groups)))
    meta = _metadata(samples, {"group": groups})

    res = anova_1way(ft, meta, factor="group")

    assert list(res.columns) == [
        "F", "p_value", "p_adj", "significant", "group_means",
    ]
    # All signal features should be detected.
    assert res["significant"].iloc[:n_signal].all()
    # Most nulls should not be significant.
    assert (~res["significant"].iloc[n_signal:]).mean() >= 0.8
    # group_means is a dict per row with the right keys.
    assert set(res.loc["f1", "group_means"].keys()) == {"A", "B", "C"}


def test_anova_1way_skips_undersized_group() -> None:
    samples = ["a1", "a2"] + [f"b{i + 1}" for i in range(6)] + [f"c{i + 1}" for i in range(6)]
    groups = ["A"] * 2 + ["B"] * 6 + ["C"] * 6
    rng = np.random.default_rng(SEED)
    ft = pd.DataFrame(
        rng.normal(size=(5, len(samples))),
        index=pd.Index([f"f{i + 1}" for i in range(5)], name="feature_id"),
        columns=samples,
    )
    meta = _metadata(samples, {"group": groups})

    res = anova_1way(ft, meta, factor="group", min_samples_per_group=3)

    assert res["p_value"].isna().all()
    assert res["F"].isna().all()
    assert (~res["significant"]).all()
    assert res.loc["f1", "group_means"] == {}


def test_anova_1way_rejects_single_level() -> None:
    samples = ["s1", "s2", "s3"]
    ft = _make_table(3, samples)
    meta = _metadata(samples, {"group": ["A", "A", "A"]})
    with pytest.raises(ValueError, match="at least 2 distinct"):
        anova_1way(ft, meta, factor="group")


# ---------------------------------------------------------------------------
# Welch ANOVA


def test_welch_anova_unequal_variances() -> None:
    rng = np.random.default_rng(SEED)
    n_features = 20
    n_per = 8
    groups = ["A"] * n_per + ["B"] * n_per + ["C"] * n_per
    samples = [f"s{i + 1}" for i in range(len(groups))]
    ft = _make_table(n_features, samples)
    a = rng.normal(loc=0.0, scale=0.5, size=(n_features, n_per))
    b = rng.normal(loc=5.0, scale=2.0, size=(n_features, n_per))
    c = rng.normal(loc=10.0, scale=5.0, size=(n_features, n_per))
    ft.iloc[:] = np.hstack([a, b, c])
    meta = _metadata(samples, {"group": groups})

    res = welch_anova(ft, meta, factor="group")

    assert list(res.columns) == [
        "F", "p_value", "p_adj", "significant", "group_means",
    ]
    # All features have a strong group effect.
    assert res["significant"].all()


# ---------------------------------------------------------------------------
# 2-way ANOVA with classification


def _build_2way_dataset() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """4 features planted to land in each of the four interesting buckets."""
    rng = np.random.default_rng(SEED)
    # 2 × 2 factorial × 6 reps per cell = 24 samples.
    levels1 = ["X", "Y"]
    levels2 = ["P", "Q"]
    n_rep = 6
    samples: list[str] = []
    f1_col: list[str] = []
    f2_col: list[str] = []
    for l1 in levels1:
        for l2 in levels2:
            for r in range(n_rep):
                samples.append(f"{l1}{l2}{r + 1}")
                f1_col.append(l1)
                f2_col.append(l2)
    meta = _metadata(samples, {"f1": f1_col, "f2": f2_col})

    n_features = 4
    ft = _make_table(n_features, samples)
    f1_arr = np.array(f1_col)
    f2_arr = np.array(f2_col)

    # f1: factor1_only — effect of f1 (X=0, Y=5), no f2 effect, no interaction.
    row_f1_only = np.where(f1_arr == "Y", 5.0, 0.0) + rng.normal(scale=0.3, size=len(samples))
    # f2: factor2_only.
    row_f2_only = np.where(f2_arr == "Q", 5.0, 0.0) + rng.normal(scale=0.3, size=len(samples))
    # f3: both_no_interaction.
    row_both = (
        np.where(f1_arr == "Y", 5.0, 0.0)
        + np.where(f2_arr == "Q", 5.0, 0.0)
        + rng.normal(scale=0.3, size=len(samples))
    )
    # f4: interaction — only the X,P and Y,Q cells are elevated.
    row_ix = np.where(
        ((f1_arr == "X") & (f2_arr == "P")) | ((f1_arr == "Y") & (f2_arr == "Q")),
        5.0,
        0.0,
    ) + rng.normal(scale=0.3, size=len(samples))

    ft.iloc[0] = row_f1_only
    ft.iloc[1] = row_f2_only
    ft.iloc[2] = row_both
    ft.iloc[3] = row_ix
    return ft, meta, ["f1", "f2"], samples


def test_anova_2way_columns_and_classification() -> None:
    ft, meta, factors, _ = _build_2way_dataset()
    res = anova_2way(ft, meta, factors=factors)

    expected = [
        "F_f1", "p_f1", "p_f1_adj", "sig_f1",
        "F_f2", "p_f2", "p_f2_adj", "sig_f2",
        "F_interaction", "p_interaction", "p_interaction_adj", "sig_interaction",
        "classification",
    ]
    assert list(res.columns) == expected
    assert res.loc["f1", "classification"] == "factor1_only"
    assert res.loc["f2", "classification"] == "factor2_only"
    assert res.loc["f3", "classification"] == "both_no_interaction"
    # The interaction row may classify as 'interaction' (both mains
    # significant due to marginal means differing) or 'interaction_only'
    # depending on power; both are acceptable factorial outcomes.
    assert res.loc["f4", "classification"] in {"interaction", "interaction_only"}
    assert bool(res.loc["f4", "sig_interaction"])


def test_anova_2way_ns_feature() -> None:
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
    meta = _metadata(samples, {"f1": f1_col, "f2": f2_col})
    ft = _make_table(20, samples)
    ft.iloc[:] = rng.normal(size=(20, len(samples)))
    res = anova_2way(ft, meta, factors=["f1", "f2"])
    # With no planted effect, almost all features should be 'ns'.
    assert (res["classification"] == "ns").mean() >= 0.8


def test_anova_2way_rejects_wrong_factor_count() -> None:
    samples = ["s1", "s2", "s3", "s4"]
    ft = _make_table(2, samples)
    meta = _metadata(samples, {"f1": ["a", "a", "b", "b"]})
    with pytest.raises(ValueError, match="exactly 2"):
        anova_2way(ft, meta, factors=["f1"])


def test_anova_2way_rejects_duplicate_factors() -> None:
    samples = ["s1", "s2", "s3", "s4"]
    ft = _make_table(2, samples)
    meta = _metadata(samples, {"f1": ["a", "a", "b", "b"]})
    with pytest.raises(ValueError, match="must be distinct"):
        anova_2way(ft, meta, factors=["f1", "f1"])


def test_anova_2way_skips_undersized_cell() -> None:
    # f1=X, f2=P has only 1 sample → all features skipped.
    samples = []
    f1_col = []
    f2_col = []
    for l1, l2, n in [("X", "P", 1), ("X", "Q", 6), ("Y", "P", 6), ("Y", "Q", 6)]:
        for r in range(n):
            samples.append(f"{l1}{l2}{r + 1}")
            f1_col.append(l1)
            f2_col.append(l2)
    meta = _metadata(samples, {"f1": f1_col, "f2": f2_col})
    rng = np.random.default_rng(SEED)
    ft = pd.DataFrame(
        rng.normal(size=(3, len(samples))),
        index=pd.Index(["f1", "f2", "f3"], name="feature_id"),
        columns=samples,
    )
    res = anova_2way(ft, meta, factors=["f1", "f2"], min_samples_per_cell=2)
    assert res["p_f1"].isna().all()
    assert res["p_interaction"].isna().all()
    assert (res["classification"] == "ns").all()


# ---------------------------------------------------------------------------
# Pingouin-comparison tests (dev-only). Skip cleanly when pingouin missing.


def _long_for_pg(ft: pd.DataFrame, meta: pd.DataFrame, feature: str, factor: str) -> pd.DataFrame:
    row = ft.loc[feature]
    return pd.DataFrame({"value": row.values, factor: meta[factor].loc[row.index].values})


def test_anova_1way_matches_pingouin() -> None:
    pg = pytest.importorskip("pingouin")
    rng = np.random.default_rng(SEED)
    n_per = 6
    groups = ["A"] * n_per + ["B"] * n_per + ["C"] * n_per
    samples = [f"s{i + 1}" for i in range(len(groups))]
    ft = _make_table(5, samples)
    ft.iloc[:] = rng.normal(size=(5, len(samples)))
    ft.iloc[0, n_per:2 * n_per] += 3.0
    meta = _metadata(samples, {"group": groups})

    res = anova_1way(ft, meta, factor="group")
    for feature in ft.index:
        long = _long_for_pg(ft, meta, feature, "group")
        ref = pg.anova(data=long, dv="value", between="group", detailed=False)
        assert abs(res.loc[feature, "F"] - float(ref["F"].iloc[0])) < 1e-4
        assert abs(res.loc[feature, "p_value"] - float(ref["p_unc"].iloc[0])) < 1e-4


def test_welch_anova_matches_pingouin() -> None:
    pg = pytest.importorskip("pingouin")
    rng = np.random.default_rng(SEED)
    n_per = 8
    groups = ["A"] * n_per + ["B"] * n_per + ["C"] * n_per
    samples = [f"s{i + 1}" for i in range(len(groups))]
    ft = _make_table(5, samples)
    a = rng.normal(loc=0.0, scale=0.5, size=(5, n_per))
    b = rng.normal(loc=2.0, scale=2.0, size=(5, n_per))
    c = rng.normal(loc=4.0, scale=5.0, size=(5, n_per))
    ft.iloc[:] = np.hstack([a, b, c])
    meta = _metadata(samples, {"group": groups})

    res = welch_anova(ft, meta, factor="group")
    for feature in ft.index:
        long = _long_for_pg(ft, meta, feature, "group")
        ref = pg.welch_anova(data=long, dv="value", between="group")
        assert abs(res.loc[feature, "F"] - float(ref["F"].iloc[0])) < 1e-4
        assert abs(res.loc[feature, "p_value"] - float(ref["p_unc"].iloc[0])) < 1e-4


def test_anova_2way_matches_pingouin() -> None:
    pg = pytest.importorskip("pingouin")
    ft, meta, factors, _ = _build_2way_dataset()
    res = anova_2way(ft, meta, factors=factors)
    f1, f2 = factors
    for feature in ft.index:
        row = ft.loc[feature]
        long = pd.DataFrame(
            {
                "value": row.values,
                f1: meta[f1].loc[row.index].values,
                f2: meta[f2].loc[row.index].values,
            }
        )
        ref = pg.anova(data=long, dv="value", between=[f1, f2], detailed=False)
        by_src = ref.set_index("Source")
        assert abs(res.loc[feature, f"F_{f1}"] - float(by_src.loc[f1, "F"])) < 1e-3
        assert abs(res.loc[feature, f"F_{f2}"] - float(by_src.loc[f2, "F"])) < 1e-3
        assert abs(
            res.loc[feature, "F_interaction"]
            - float(by_src.loc[f"{f1} * {f2}", "F"])
        ) < 1e-3
        assert abs(res.loc[feature, f"p_{f1}"] - float(by_src.loc[f1, "p_unc"])) < 1e-4
        assert abs(res.loc[feature, f"p_{f2}"] - float(by_src.loc[f2, "p_unc"])) < 1e-4
        assert abs(
            res.loc[feature, "p_interaction"]
            - float(by_src.loc[f"{f1} * {f2}", "p_unc"])
        ) < 1e-4
