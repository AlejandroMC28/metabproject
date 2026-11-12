"""Tests for the per-feature Kruskal-Wallis routine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metabproject.stats.kruskal import kruskal_wallis


SEED = 0


def _make_table(n_features: int, sample_names: list[str]) -> pd.DataFrame:
    idx = pd.Index([f"f{i + 1}" for i in range(n_features)], name="feature_id")
    return pd.DataFrame(np.zeros((n_features, len(sample_names))), index=idx, columns=sample_names)


def _metadata(samples: list[str], factors: dict[str, list[str]]) -> pd.DataFrame:
    return pd.DataFrame(factors, index=pd.Index(samples, name="sample"))


def test_kruskal_recovers_planted_effect_on_nonnormal_data() -> None:
    rng = np.random.default_rng(SEED)
    n_features = 20
    n_per = 8
    groups = ["A"] * n_per + ["B"] * n_per + ["C"] * n_per
    samples = [f"s{i + 1}" for i in range(len(groups))]
    ft = _make_table(n_features, samples)
    # Exponential (right-skewed, non-normal) data with strong group shifts.
    a = rng.exponential(scale=1.0, size=(n_features, n_per))
    b = rng.exponential(scale=1.0, size=(n_features, n_per)) + 5.0
    c = rng.exponential(scale=1.0, size=(n_features, n_per)) + 10.0
    ft.iloc[:] = np.hstack([a, b, c])
    meta = _metadata(samples, {"group": groups})

    res = kruskal_wallis(ft, meta, factor="group")

    assert list(res.columns) == [
        "H", "p_value", "p_adj", "significant", "group_medians",
    ]
    assert res["significant"].all()
    assert set(res.loc["f1", "group_medians"].keys()) == {"A", "B", "C"}


def test_kruskal_null_data_mostly_ns() -> None:
    rng = np.random.default_rng(SEED)
    n_features = 30
    samples = [f"s{i + 1}" for i in range(18)]
    groups = ["A"] * 6 + ["B"] * 6 + ["C"] * 6
    ft = pd.DataFrame(
        rng.exponential(size=(n_features, len(samples))),
        index=pd.Index([f"f{i + 1}" for i in range(n_features)], name="feature_id"),
        columns=samples,
    )
    meta = _metadata(samples, {"group": groups})

    res = kruskal_wallis(ft, meta, factor="group")
    assert (~res["significant"]).mean() >= 0.8


def test_kruskal_skips_undersized_group() -> None:
    samples = ["a1", "a2"] + [f"b{i + 1}" for i in range(6)] + [f"c{i + 1}" for i in range(6)]
    groups = ["A"] * 2 + ["B"] * 6 + ["C"] * 6
    rng = np.random.default_rng(SEED)
    ft = pd.DataFrame(
        rng.normal(size=(5, len(samples))),
        index=pd.Index([f"f{i + 1}" for i in range(5)], name="feature_id"),
        columns=samples,
    )
    meta = _metadata(samples, {"group": groups})

    res = kruskal_wallis(ft, meta, factor="group", min_samples_per_group=3)

    assert res["p_value"].isna().all()
    assert (~res["significant"]).all()
    assert res.loc["f1", "group_medians"] == {}


def test_kruskal_rejects_single_level() -> None:
    samples = ["s1", "s2", "s3", "s4"]
    ft = _make_table(3, samples)
    meta = _metadata(samples, {"group": ["A", "A", "A", "A"]})
    with pytest.raises(ValueError, match="at least 2 distinct"):
        kruskal_wallis(ft, meta, factor="group")
