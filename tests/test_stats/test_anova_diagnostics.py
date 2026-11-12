"""Tests for stats/anova_diagnostics.py — residual diagnostic figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metabproject.stats.anova_diagnostics import (
    ResidualDiagnostics,
    anova_2way_residual_diagnostics,
)
from metabproject.stats.comparison_runner import run_comparison
from metabproject.stats.comparisons import Comparison


SEED = 0


def _two_way_design() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    samples, l1, l2 = [], [], []
    for v in ("a", "b", "c"):
        for t in ("p", "q", "r"):
            for r in range(4):
                samples.append(f"{v}{t}{r + 1}")
                l1.append(v)
                l2.append(t)
    meta = pd.DataFrame(
        {"variety": l1, "treatment": l2},
        index=pd.Index(samples, name="sample"),
    )
    return meta, np.array(l1), np.array(l2)


def _feature_table(
    meta: pd.DataFrame,
    n_features: int,
    rng: np.random.Generator,
    *,
    skew: bool = False,
) -> pd.DataFrame:
    n = len(meta)
    l1 = meta["variety"].to_numpy()
    l2 = meta["treatment"].to_numpy()
    base = np.where(l1 == "a", 1.0, 0.0) + np.where(l2 == "p", 0.5, 0.0)
    noise = (
        rng.standard_exponential((n_features, n)) - 1.0
        if skew
        else rng.normal(size=(n_features, n))
    )
    rows = base + 0.5 * noise
    return pd.DataFrame(
        rows,
        index=pd.Index([f"f{i}" for i in range(n_features)], name="feature_id"),
        columns=meta.index.tolist(),
    )


# --------------------------------------------------------------------------- #
# Pure function
# --------------------------------------------------------------------------- #

def test_returns_diagnostics_with_expected_fields() -> None:
    meta, _, _ = _two_way_design()
    rng = np.random.default_rng(SEED)
    ft = _feature_table(meta, n_features=40, rng=rng)

    diag = anova_2way_residual_diagnostics(ft, meta, ["variety", "treatment"])

    assert isinstance(diag, ResidualDiagnostics)
    assert diag.n_features == 40
    assert diag.n_samples == 36
    assert np.isfinite(diag.median_r2)
    assert 0.0 <= diag.frac_shapiro_reject <= 1.0
    assert len(diag.figure.axes) >= 4  # 4 panels + colorbar
    plt.close(diag.figure)


def test_normal_residuals_have_low_rejection_rate() -> None:
    meta, _, _ = _two_way_design()
    rng = np.random.default_rng(SEED)
    ft = _feature_table(meta, n_features=200, rng=rng, skew=False)
    diag = anova_2way_residual_diagnostics(ft, meta, ["variety", "treatment"])
    # Under H0 the test rejects ~5% at α=0.05; tolerate up to 15%.
    assert diag.frac_shapiro_reject < 0.15
    plt.close(diag.figure)


def test_skewed_residuals_have_higher_rejection_rate() -> None:
    meta, _, _ = _two_way_design()
    rng = np.random.default_rng(SEED)
    ft_norm = _feature_table(meta, n_features=200, rng=rng, skew=False)
    ft_skew = _feature_table(meta, n_features=200, rng=rng, skew=True)
    d_norm = anova_2way_residual_diagnostics(ft_norm, meta, ["variety", "treatment"])
    d_skew = anova_2way_residual_diagnostics(ft_skew, meta, ["variety", "treatment"])
    assert d_skew.frac_shapiro_reject > d_norm.frac_shapiro_reject + 0.10
    plt.close(d_norm.figure)
    plt.close(d_skew.figure)


def test_raises_on_wrong_factor_count() -> None:
    meta, _, _ = _two_way_design()
    rng = np.random.default_rng(SEED)
    ft = _feature_table(meta, n_features=5, rng=rng)
    with pytest.raises(ValueError, match="2 factors"):
        anova_2way_residual_diagnostics(ft, meta, ["variety"])


def test_raises_when_no_complete_features() -> None:
    meta, _, _ = _two_way_design()
    rng = np.random.default_rng(SEED)
    ft = _feature_table(meta, n_features=5, rng=rng)
    ft.iloc[:, 0] = np.nan  # every feature now has a NaN
    with pytest.raises(ValueError, match="without missing values"):
        anova_2way_residual_diagnostics(ft, meta, ["variety", "treatment"])


# --------------------------------------------------------------------------- #
# Runner wiring
# --------------------------------------------------------------------------- #

def test_run_comparison_writes_residual_figure(tmp_path: Path) -> None:
    meta, _, _ = _two_way_design()
    rng = np.random.default_rng(SEED)
    ft = _feature_table(meta, n_features=30, rng=rng)
    cmp = Comparison(
        name="x", kind="anova_2way",
        factors=["variety", "treatment"], posthoc="tukey",
    )
    out = tmp_path / "comp"
    result = run_comparison(
        cmp, ft, meta,
        output_dir=out,
        residual_plots=True,
    )
    assert result.residuals_path == out / "residuals_diagnostics.png"
    assert result.residuals_path.exists()
    assert (out / "residuals_diagnostics.svg").exists()
    summary = result.export_summary["residuals"]
    assert summary["n_features"] == 30
    assert summary["n_samples"] == 36
    assert "median_r2" in summary
    assert "frac_shapiro_reject" in summary


def test_run_comparison_residual_plots_requires_2way() -> None:
    meta, _, _ = _two_way_design()
    rng = np.random.default_rng(SEED)
    ft = _feature_table(meta, n_features=10, rng=rng)
    cmp1 = Comparison(
        name="x", kind="anova_1way", factors=["variety"], posthoc="tukey",
    )
    with pytest.raises(ValueError, match="anova_2way"):
        run_comparison(
            cmp1, ft, meta, output_dir="x", residual_plots=True,
        )
