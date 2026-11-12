"""Tests for ``run_random_forest``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metabproject.stats.multivariate.random_forest import (
    RFResult,
    run_random_forest,
)


def test_run_rf_classifies_planted_groups(
    planted_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ft, md = planted_dataset
    res = run_random_forest(
        ft, md, target_col="group", n_estimators=100, seed=123
    )

    assert isinstance(res, RFResult)
    assert res.accuracy > 0.8
    assert set(res.classes_) == {"A", "B"}

    # Feature importance has the expected columns and ordering.
    assert list(res.feature_importance.columns) == [
        "feature", "mdi_importance", "mda_mean", "mda_std",
    ]
    mda = res.feature_importance["mda_mean"].to_numpy()
    assert np.all(np.diff(mda) <= 1e-12)

    # The discriminative planted features (f1..f20) should dominate the top.
    planted = {f"f{i + 1}" for i in range(20)}
    top10 = set(res.feature_importance["feature"].head(10))
    assert len(top10 & planted) >= 7


def test_run_rf_oob_curve_has_expected_shape(
    planted_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ft, md = planted_dataset
    res = run_random_forest(ft, md, target_col="group", n_estimators=80, seed=123)
    assert list(res.oob_curve.columns) == ["n_estimators", "oob_score"]
    assert res.oob_curve["n_estimators"].is_monotonic_increasing
    assert res.oob_curve["n_estimators"].iloc[-1] == 80
    assert ((res.oob_curve["oob_score"] >= 0) & (res.oob_curve["oob_score"] <= 1)).all()


def test_run_rf_is_reproducible(
    planted_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ft, md = planted_dataset
    a = run_random_forest(ft, md, target_col="group", n_estimators=60, seed=42)
    b = run_random_forest(ft, md, target_col="group", n_estimators=60, seed=42)
    assert a.accuracy == b.accuracy
    pd.testing.assert_frame_equal(a.feature_importance, b.feature_importance)


def test_run_rf_mda_top_k_is_reproducible(
    planted_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ft, md = planted_dataset
    a = run_random_forest(
        ft, md, target_col="group", n_estimators=60, seed=42, mda_top_k=5,
    )
    b = run_random_forest(
        ft, md, target_col="group", n_estimators=60, seed=42, mda_top_k=5,
    )
    pd.testing.assert_frame_equal(a.feature_importance, b.feature_importance)


def test_run_rf_rejects_missing_target_col(
    planted_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ft, md = planted_dataset
    with pytest.raises(ValueError, match="metadata has no column"):
        run_random_forest(ft, md, target_col="treatment")


def test_run_rf_rejects_single_class(
    planted_feature_table: pd.DataFrame,
) -> None:
    md = pd.DataFrame(
        {"group": ["A"] * planted_feature_table.shape[1]},
        index=planted_feature_table.columns,
    )
    md.index.name = "sample_name"
    with pytest.raises(ValueError, match=">=2 distinct classes"):
        run_random_forest(planted_feature_table, md, target_col="group")


def test_run_rf_rejects_undersized_class(
    planted_feature_table: pd.DataFrame,
) -> None:
    md = pd.DataFrame(
        {"group": ["A"] * 19 + ["B"]},
        index=planted_feature_table.columns,
    )
    md.index.name = "sample_name"
    with pytest.raises(ValueError, match="<2 samples"):
        run_random_forest(planted_feature_table, md, target_col="group")


def test_run_rf_rejects_nan(
    planted_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ft, md = planted_dataset
    ft = ft.copy()
    ft.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        run_random_forest(ft, md, target_col="group")


def test_run_rf_rejects_bad_params(
    planted_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ft, md = planted_dataset
    with pytest.raises(ValueError, match="n_estimators"):
        run_random_forest(ft, md, target_col="group", n_estimators=1)
    with pytest.raises(ValueError, match="test_size"):
        run_random_forest(ft, md, target_col="group", test_size=0.0)
    with pytest.raises(ValueError, match="n_permutation_repeats"):
        run_random_forest(ft, md, target_col="group", n_permutation_repeats=0)
