"""Tests for the per-feature Welch t-test routine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from metabproject.stats.univariate import welch_ttest


def _matrix(values: list[list[float]]) -> pd.DataFrame:
    cols = [f"s{i+1}" for i in range(len(values[0]))]
    idx = pd.Index([f"f{i+1}" for i in range(len(values))], name="feature_id")
    return pd.DataFrame(values, index=idx, columns=cols)


def test_welch_ttest_known_means_and_direction() -> None:
    # Feature 1: group A clearly higher; Feature 2: group B clearly higher.
    ft = _matrix([
        [10.0, 10.1, 9.9, 1.0, 1.1, 0.9],
        [1.0, 1.1, 0.9, 10.0, 10.1, 9.9],
    ])
    fc = _matrix([
        [1000.0, 1010.0, 990.0, 10.0, 11.0, 9.0],
        [10.0, 11.0, 9.0, 1000.0, 1010.0, 990.0],
    ])
    a_samples = ["s1", "s2", "s3"]
    b_samples = ["s4", "s5", "s6"]

    res = welch_ttest(ft, fc, a_samples, b_samples, min_samples=3, alpha=0.05)

    assert list(res.columns) == [
        "mean_a", "mean_b", "n_a", "n_b", "log2fc",
        "t_stat", "p_value", "p_adj", "direction", "significant",
    ]
    # log2fc follows the log2(B / A) convention → feature 1 (A high) is "down",
    # feature 2 (B high) is "up".
    assert res.loc["f1", "direction"] == "down"
    assert res.loc["f2", "direction"] == "up"
    assert bool(res.loc["f1", "significant"]) is True
    assert bool(res.loc["f2", "significant"]) is True
    assert np.isclose(res.loc["f1", "log2fc"], np.log2(10.0 / 1000.0), atol=0.5)
    assert np.isclose(res.loc["f2", "log2fc"], np.log2(1000.0 / 10.0), atol=0.5)


def test_welch_ttest_log2_transform_uses_difference_of_means() -> None:
    # Processed table already on log2 scale. fc_table is irrelevant and may be None.
    ft = _matrix([
        [1.0, 1.1, 0.9, 5.0, 5.1, 4.9],   # feature 1: B - A ≈ +4 (up in B)
        [5.0, 5.1, 4.9, 1.0, 1.1, 0.9],   # feature 2: B - A ≈ -4 (down in B)
    ])
    res = welch_ttest(
        ft, None, ["s1", "s2", "s3"], ["s4", "s5", "s6"],
        min_samples=3, alpha=0.05, transform="log2",
    )
    assert np.isclose(res.loc["f1", "log2fc"], 4.0, atol=0.2)
    assert np.isclose(res.loc["f2", "log2fc"], -4.0, atol=0.2)
    assert res.loc["f1", "direction"] == "up"
    assert res.loc["f2", "direction"] == "down"


def test_welch_ttest_log10_transform_rescales_to_log2() -> None:
    # On a log10 scale, B - A = 1.0 corresponds to log2fc = log2(10) ≈ 3.32.
    ft = _matrix([
        [1.0, 1.1, 0.9, 2.0, 2.1, 1.9],
    ])
    res = welch_ttest(
        ft, None, ["s1", "s2", "s3"], ["s4", "s5", "s6"],
        min_samples=3, alpha=0.05, transform="log10",
    )
    assert np.isclose(res.loc["f1", "log2fc"], np.log2(10.0), atol=0.2)


def test_welch_ttest_non_log_transform_requires_fc_table() -> None:
    ft = _matrix([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]])
    try:
        welch_ttest(ft, None, ["s1", "s2", "s3"], ["s4", "s5", "s6"], transform="pareto")
    except ValueError as e:
        assert "fc_table is required" in str(e)
    else:
        raise AssertionError("expected ValueError when fc_table is None and transform != log")


def test_welch_ttest_log2_no_scale_takes_log_path() -> None:
    # Group A all 2.0, group B all 6.0 on processed (log2) scale →
    # log2fc = mean_b - mean_a = 4.0 exactly. No fc_table needed.
    ft = _matrix([[2.0, 2.0, 2.0, 6.0, 6.0, 6.0]])
    res = welch_ttest(
        ft, None, ["s1", "s2", "s3"], ["s4", "s5", "s6"],
        min_samples=3, alpha=0.05, transform="log2", scale="none",
    )
    assert res.loc["f1", "log2fc"] == 4.0


def test_welch_ttest_log2_with_scaling_uses_fc_table() -> None:
    # transform=log2 but scale=pareto → log-path disabled; log2fc must
    # come from fc_table (raw-area scale). Processed table values are
    # therefore irrelevant for log2fc and chosen to be misleading.
    ft = _matrix([[0.1, 0.1, 0.1, -0.1, -0.1, -0.1]])  # mean_b - mean_a = -0.2
    fc = _matrix([[100.0, 100.0, 100.0, 400.0, 400.0, 400.0]])  # log2(400/100) = 2
    res = welch_ttest(
        ft, fc, ["s1", "s2", "s3"], ["s4", "s5", "s6"],
        min_samples=3, alpha=0.05, transform="log2", scale="pareto",
    )
    assert res.loc["f1", "log2fc"] == 2.0


def test_welch_ttest_log2_with_scaling_requires_fc_table() -> None:
    ft = _matrix([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]])
    try:
        welch_ttest(
            ft, None, ["s1", "s2", "s3"], ["s4", "s5", "s6"],
            transform="log2", scale="pareto",
        )
    except ValueError as e:
        msg = str(e)
        assert "fc_table is required" in msg
        assert "log2" in msg
        assert "pareto" in msg
    else:
        raise AssertionError(
            "expected ValueError when transform=log2 but scale=pareto and fc_table is None"
        )


def test_welch_ttest_no_transform_no_scale_requires_fc_table() -> None:
    ft = _matrix([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]])
    try:
        welch_ttest(
            ft, None, ["s1", "s2", "s3"], ["s4", "s5", "s6"],
            transform="none", scale="none",
        )
    except ValueError as e:
        assert "fc_table is required" in str(e)
    else:
        raise AssertionError("expected ValueError when transform=none and fc_table is None")


def test_welch_ttest_log10_with_autoscale_uses_fc_table() -> None:
    # Processed table is log10 then autoscaled → log-path disabled.
    # fc_table provides the raw-area means: log2(800/100) = 3.0.
    ft = _matrix([[0.5, 0.5, 0.5, -0.5, -0.5, -0.5]])
    fc = _matrix([[100.0, 100.0, 100.0, 800.0, 800.0, 800.0]])
    res = welch_ttest(
        ft, fc, ["s1", "s2", "s3"], ["s4", "s5", "s6"],
        min_samples=3, alpha=0.05, transform="log10", scale="auto",
    )
    assert res.loc["f1", "log2fc"] == 3.0


def test_welch_ttest_min_samples_gates_to_nan() -> None:
    # Only one non-NaN value in group B for feature 1.
    ft = _matrix([
        [10.0, 11.0, 9.0, 5.0, np.nan, np.nan],
        [10.0, 11.0, 9.0, 5.0, 5.1, 4.9],
    ])
    fc = ft.copy() * 100
    res = welch_ttest(ft, fc, ["s1", "s2", "s3"], ["s4", "s5", "s6"],
                      min_samples=3, alpha=0.05)
    assert np.isnan(res.loc["f1", "p_value"])
    assert np.isnan(res.loc["f1", "p_adj"])
    assert res.loc["f1", "direction"] == "ns"
    assert not bool(res.loc["f1", "significant"])
    assert not np.isnan(res.loc["f2", "p_value"])


def test_welch_ttest_bh_adjustment_monotone() -> None:
    rng = np.random.default_rng(0)
    n_features = 30
    ft = _matrix(rng.normal(size=(n_features, 6)).tolist())
    fc = _matrix((np.abs(rng.normal(size=(n_features, 6))) + 1).tolist())
    res = welch_ttest(ft, fc, ["s1", "s2", "s3"], ["s4", "s5", "s6"])
    valid = res["p_value"].notna()
    p = res.loc[valid, "p_value"].to_numpy()
    padj = res.loc[valid, "p_adj"].to_numpy()
    # BH-adjusted p-values must be >= raw p-values.
    assert np.all(padj + 1e-12 >= p)
