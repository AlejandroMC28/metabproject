"""Tests for the per-feature normality and variance-homogeneity diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metabproject.stats.diagnostics import levene_per_feature, shapiro_per_feature


SEED = 0


def _make_table(n_features: int, n_samples: int) -> tuple[pd.Index, list[str]]:
    idx = pd.Index([f"f{i + 1}" for i in range(n_features)], name="feature_id")
    cols = [f"s{i + 1}" for i in range(n_samples)]
    return idx, cols


def _metadata(samples: list[str], groups: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"group": groups}, index=pd.Index(samples, name="sample"))


# ---------------------------------------------------------------------------
# Across-all-samples Shapiro


def test_shapiro_across_samples_normal_data_flagged_normal() -> None:
    rng = np.random.default_rng(SEED)
    n_features, n_samples = 40, 30
    idx, cols = _make_table(n_features, n_samples)
    ft = pd.DataFrame(rng.normal(size=(n_features, n_samples)), index=idx, columns=cols)

    res = shapiro_per_feature(ft)

    assert list(res.columns) == [
        "n", "statistic", "p_value", "p_adj", "normal", "skipped_reason",
    ]
    assert (res["n"] == n_samples).all()
    frac_normal = float(res["normal"].mean())
    assert frac_normal >= 0.8, f"only {frac_normal:.2f} flagged normal"


def test_shapiro_across_samples_lognormal_flagged_non_normal() -> None:
    rng = np.random.default_rng(SEED)
    n_features, n_samples = 40, 30
    idx, cols = _make_table(n_features, n_samples)
    ft = pd.DataFrame(
        np.exp(rng.normal(size=(n_features, n_samples))), index=idx, columns=cols
    )

    res = shapiro_per_feature(ft)

    frac_non_normal = float((~res["normal"]).mean())
    assert frac_non_normal >= 0.8, f"only {frac_non_normal:.2f} flagged non-normal"


# ---------------------------------------------------------------------------
# Per-group Shapiro


def test_shapiro_per_group_recovers_normality_under_group_shift() -> None:
    # Within each group the data is normal, but the groups have very
    # different means → across-all-samples Shapiro would falsely reject
    # normality due to bimodality; per-group Shapiro should not.
    rng = np.random.default_rng(SEED)
    n_features = 40
    n_per = 15
    idx, _ = _make_table(n_features, 0)
    cols_a = [f"a{i + 1}" for i in range(n_per)]
    cols_b = [f"b{i + 1}" for i in range(n_per)]
    a = rng.normal(loc=0.0, scale=1.0, size=(n_features, n_per))
    b = rng.normal(loc=10.0, scale=1.0, size=(n_features, n_per))
    ft = pd.DataFrame(np.hstack([a, b]), index=idx, columns=cols_a + cols_b)
    meta = _metadata(cols_a + cols_b, ["A"] * n_per + ["B"] * n_per)

    res = shapiro_per_feature(ft, metadata=meta, group_col="group")

    expected = [
        "A_n", "A_statistic", "A_p_value", "A_p_adj", "A_normal",
        "B_n", "B_statistic", "B_p_value", "B_p_adj", "B_normal",
        "all_groups_normal", "n_groups_normal", "skipped_reason",
    ]
    assert list(res.columns) == expected
    assert (res["A_n"] == n_per).all()
    assert (res["B_n"] == n_per).all()
    assert (res["skipped_reason"] == "").all()
    frac = float(res["all_groups_normal"].mean())
    assert frac >= 0.8, f"only {frac:.2f} flagged normal within both groups"

    # Sanity-check: across-all-samples would reject normality on the same data.
    across = shapiro_per_feature(ft)
    assert float((~across["normal"]).mean()) >= 0.8


def test_shapiro_per_group_skips_undersized_group() -> None:
    rng = np.random.default_rng(SEED)
    n_features = 8
    idx, _ = _make_table(n_features, 0)
    # Group A has only 2 samples → skip; group B has 10.
    cols_a = ["a1", "a2"]
    cols_b = [f"b{i + 1}" for i in range(10)]
    data = np.hstack([
        rng.normal(size=(n_features, 2)),
        rng.normal(size=(n_features, 10)),
    ])
    ft = pd.DataFrame(data, index=idx, columns=cols_a + cols_b)
    meta = _metadata(cols_a + cols_b, ["A", "A"] + ["B"] * 10)

    res = shapiro_per_feature(ft, metadata=meta, group_col="group")

    # Group A skipped for every feature.
    assert res["A_p_value"].isna().all()
    assert res["A_statistic"].isna().all()
    assert (~res["A_normal"]).all()
    # Group B tested.
    assert res["B_p_value"].notna().all()
    # Aggregates and skipped_reason.
    assert (res["skipped_reason"] == "A:n<3").all()
    assert (~res["all_groups_normal"]).all()


# ---------------------------------------------------------------------------
# Levene


def test_levene_engineered_heteroscedasticity_flagged() -> None:
    rng = np.random.default_rng(SEED)
    n_features = 40
    n_per = 15
    idx, _ = _make_table(n_features, 0)
    cols_a = [f"a{i + 1}" for i in range(n_per)]
    cols_b = [f"b{i + 1}" for i in range(n_per)]
    a = rng.normal(loc=0.0, scale=1.0, size=(n_features, n_per))
    b = rng.normal(loc=0.0, scale=5.0, size=(n_features, n_per))  # variance 25
    ft = pd.DataFrame(np.hstack([a, b]), index=idx, columns=cols_a + cols_b)
    meta = _metadata(cols_a + cols_b, ["A"] * n_per + ["B"] * n_per)

    res = levene_per_feature(ft, meta, group_col="group")

    assert list(res.columns) == [
        "statistic", "p_value", "p_adj", "homoscedastic", "skipped_reason",
    ]
    frac_het = float((~res["homoscedastic"]).mean())
    assert frac_het >= 0.8, f"only {frac_het:.2f} flagged heteroscedastic"


def test_levene_homoscedastic_data_flagged_homoscedastic() -> None:
    rng = np.random.default_rng(SEED)
    n_features = 40
    n_per = 15
    idx, _ = _make_table(n_features, 0)
    cols_a = [f"a{i + 1}" for i in range(n_per)]
    cols_b = [f"b{i + 1}" for i in range(n_per)]
    a = rng.normal(loc=0.0, scale=1.0, size=(n_features, n_per))
    b = rng.normal(loc=0.0, scale=1.0, size=(n_features, n_per))
    ft = pd.DataFrame(np.hstack([a, b]), index=idx, columns=cols_a + cols_b)
    meta = _metadata(cols_a + cols_b, ["A"] * n_per + ["B"] * n_per)

    res = levene_per_feature(ft, meta, group_col="group")

    frac_homo = float(res["homoscedastic"].mean())
    assert frac_homo >= 0.8, f"only {frac_homo:.2f} flagged homoscedastic"


def test_levene_skips_undersized_group() -> None:
    rng = np.random.default_rng(SEED)
    n_features = 5
    idx, _ = _make_table(n_features, 0)
    cols_a = ["a1", "a2"]  # n=2 → skip
    cols_b = [f"b{i + 1}" for i in range(10)]
    data = np.hstack([
        rng.normal(size=(n_features, 2)),
        rng.normal(size=(n_features, 10)),
    ])
    ft = pd.DataFrame(data, index=idx, columns=cols_a + cols_b)
    meta = _metadata(cols_a + cols_b, ["A", "A"] + ["B"] * 10)

    res = levene_per_feature(ft, meta, group_col="group")

    assert res["p_value"].isna().all()
    assert res["statistic"].isna().all()
    assert res["p_adj"].isna().all()
    assert (~res["homoscedastic"]).all()
    assert (res["skipped_reason"] == "A:n<3").all()


# ---------------------------------------------------------------------------
# Validation errors


def test_shapiro_per_group_requires_metadata() -> None:
    idx, cols = _make_table(3, 4)
    ft = pd.DataFrame(np.ones((3, 4)), index=idx, columns=cols)
    with pytest.raises(ValueError, match="metadata is required"):
        shapiro_per_feature(ft, metadata=None, group_col="group")


def test_diagnostics_reject_missing_group_col() -> None:
    idx, cols = _make_table(3, 4)
    ft = pd.DataFrame(np.ones((3, 4)), index=idx, columns=cols)
    meta = pd.DataFrame({"other": ["A", "A", "B", "B"]}, index=cols)
    with pytest.raises(ValueError, match="not found in metadata.columns"):
        shapiro_per_feature(ft, metadata=meta, group_col="group")
    with pytest.raises(ValueError, match="not found in metadata.columns"):
        levene_per_feature(ft, meta, group_col="group")


def test_diagnostics_reject_single_level_group() -> None:
    idx, cols = _make_table(3, 4)
    ft = pd.DataFrame(np.ones((3, 4)), index=idx, columns=cols)
    meta = _metadata(cols, ["A", "A", "A", "A"])
    with pytest.raises(ValueError, match="at least 2 distinct"):
        shapiro_per_feature(ft, metadata=meta, group_col="group")
    with pytest.raises(ValueError, match="at least 2 distinct"):
        levene_per_feature(ft, meta, group_col="group")


def test_diagnostics_reject_metadata_samples_absent_from_feature_table() -> None:
    idx, cols = _make_table(3, 4)
    ft = pd.DataFrame(np.ones((3, 4)), index=idx, columns=cols)
    # metadata.index references samples that don't exist in feature_table.
    bad_samples = ["z1", "z2", "z3", "z4"]
    meta = _metadata(bad_samples, ["A", "A", "B", "B"])
    with pytest.raises(ValueError, match="absent from feature_table.columns"):
        shapiro_per_feature(ft, metadata=meta, group_col="group")
    with pytest.raises(ValueError, match="absent from feature_table.columns"):
        levene_per_feature(ft, meta, group_col="group")


def test_levene_rejects_unknown_center() -> None:
    idx, cols = _make_table(3, 6)
    ft = pd.DataFrame(np.ones((3, 6)), index=idx, columns=cols)
    meta = _metadata(cols, ["A"] * 3 + ["B"] * 3)
    with pytest.raises(ValueError, match="center must be one of"):
        levene_per_feature(ft, meta, group_col="group", center="bogus")
