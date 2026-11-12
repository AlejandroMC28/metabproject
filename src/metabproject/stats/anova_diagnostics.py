"""Residual-based diagnostics for two-way ANOVA across a feature table.

Two entry points:

- :func:`anova_2way_residual_diagnostics` — aggregate 4-panel figure and a
  per-feature stats DataFrame (R², residual σ, Shapiro p / p_adj, max
  |std residual|, normal flag) for finding troublesome features.
- :func:`feature_residual_diagnostics_figure` — 2-panel figure (QQ +
  residual histogram) for a single feature.

Aggregate panels:

1. **Pooled standardized residual QQ plot** — overall normality of
   residuals. Heavy tails ⇒ ANOVA p-values are anti-conservative.
2. **Per-feature Shapiro-Wilk p-value histogram** — distribution of the
   normality test across features. Under H0 (residuals normal) this is
   uniform on [0, 1]; a spike near 0 quantifies how many features
   violate normality.
3. **Residuals vs fitted (pooled hexbin)** — homoscedasticity check.
   A funnel shape means the log/pareto transform didn't equalize
   variance.
4. **Per-feature R² distribution** — validity of fit. Tells you how
   much variance the full 2-way model explains across the dataset.

Pure: takes DataFrames in, returns dataclasses with figures / tables.
No file I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, norm, shapiro

from metabproject.stats._util import validate_factor_columns
from metabproject.stats.anova import _build_2way_designs


@dataclass
class ResidualDiagnostics:
    """Output of :func:`anova_2way_residual_diagnostics`."""

    figure: plt.Figure
    n_features: int
    n_samples: int
    median_r2: float
    frac_shapiro_reject: float
    per_feature: pd.DataFrame


def _fit_2way_residuals(
    feature_table: pd.DataFrame,
    metadata: pd.DataFrame,
    factors: list[str],
) -> dict:
    """Fit ``y ~ f1 + f2 + f1:f2`` per feature; return residuals + context.

    Only features with no missing values across the valid samples are
    fit. The returned ``complete_mask`` is feature-aligned so callers
    can map results back to ``feature_table.index``.
    """
    if len(factors) != 2:
        raise ValueError(f"requires exactly 2 factors, got {len(factors)}")
    if factors[0] == factors[1]:
        raise ValueError("factors must be distinct")
    validate_factor_columns(feature_table, metadata, factors)
    f1, f2 = factors

    sample_names = list(feature_table.columns)
    meta_aligned = metadata.loc[sample_names, factors]
    valid = (
        ~meta_aligned[f1].isna().to_numpy()
        & ~meta_aligned[f2].isna().to_numpy()
    )
    if not valid.any():
        raise ValueError("no samples valid for both factors")

    a_labels = meta_aligned[f1].to_numpy()[valid]
    b_labels = meta_aligned[f2].to_numpy()[valid]
    a_levels = sorted(pd.unique(a_labels), key=str)
    b_levels = sorted(pd.unique(b_labels), key=str)
    n_a, n_b = len(a_levels), len(b_levels)
    if n_a < 2 or n_b < 2:
        raise ValueError("both factors must have ≥ 2 observed levels")

    a_idx = np.searchsorted(
        np.asarray([str(x) for x in a_levels]),
        np.asarray([str(x) for x in a_labels]),
    )
    b_idx = np.searchsorted(
        np.asarray([str(x) for x in b_levels]),
        np.asarray([str(x) for x in b_labels]),
    )
    X_full, _, _, _ = _build_2way_designs(a_idx, b_idx, n_a, n_b)

    matrix = feature_table.to_numpy(dtype=float, copy=False)
    Y_valid = matrix[:, valid]
    complete = ~np.isnan(Y_valid).any(axis=1)
    Y = Y_valid[complete]
    if Y.shape[0] == 0:
        raise ValueError(
            "no features without missing values across the valid samples"
        )

    Q, _ = np.linalg.qr(X_full, mode="reduced")
    Yt = Y.T  # (n_valid_samples, n_features_complete)
    fitted = Q @ (Q.T @ Yt)
    resid = Yt - fitted

    n_valid_samples = resid.shape[0]
    df_resid = n_valid_samples - X_full.shape[1]
    if df_resid <= 0:
        raise ValueError(
            f"residual df ≤ 0 (n={n_valid_samples}, params={X_full.shape[1]})"
        )

    valid_samples = [s for s, v in zip(sample_names, valid) if v]

    return {
        "resid": resid,
        "fitted": fitted,
        "Y": Y,
        "valid_samples": valid_samples,
        "complete_mask": complete,
        "feature_ids": feature_table.index,
        "n_valid_samples": n_valid_samples,
        "df_resid": df_resid,
        "n_params": X_full.shape[1],
        "factors": (f1, f2),
        "a_labels": a_labels,
        "b_labels": b_labels,
        "a_levels": a_levels,
        "b_levels": b_levels,
    }


def _per_feature_stats(ctx: dict, alpha: float) -> pd.DataFrame:
    """Per-feature R² / σ / Shapiro p / max |std residual| / normal flag.

    Rows are returned for every feature in ``feature_table.index`` —
    features without complete data carry NaN stats and
    ``in_full_model=False``.
    """
    resid = ctx["resid"]
    Y = ctx["Y"]
    df_resid = ctx["df_resid"]
    n_valid_samples = ctx["n_valid_samples"]
    feature_ids = ctx["feature_ids"]
    complete = ctx["complete_mask"]

    n_fit = resid.shape[1]
    rss = (resid * resid).sum(axis=0)
    tss = ((Y - Y.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2_fit = 1.0 - rss / tss
        sigma_fit = np.sqrt(rss / df_resid)
        std_resid_fit = resid / sigma_fit
    max_abs_std_fit = np.nanmax(np.abs(std_resid_fit), axis=0)

    shapiro_p_fit = np.full(n_fit, np.nan)
    for j in range(n_fit):
        col = resid[:, j]
        if not np.all(np.isfinite(col)) or col.std(ddof=1) == 0:
            continue
        try:
            shapiro_p_fit[j] = float(shapiro(col).pvalue)
        except ValueError:
            continue

    valid_p = ~np.isnan(shapiro_p_fit)
    p_adj_fit = np.full(n_fit, np.nan)
    if valid_p.any():
        p_adj_fit[valid_p] = false_discovery_control(
            shapiro_p_fit[valid_p], method="bh",
        )
    normal_fit = (p_adj_fit >= alpha) & ~np.isnan(p_adj_fit)

    n_features = len(feature_ids)
    in_full = complete
    r2 = np.full(n_features, np.nan)
    sigma = np.full(n_features, np.nan)
    shapiro_p = np.full(n_features, np.nan)
    p_adj = np.full(n_features, np.nan)
    normal = np.zeros(n_features, dtype=bool)
    max_abs_std = np.full(n_features, np.nan)

    r2[in_full] = r2_fit
    sigma[in_full] = sigma_fit
    shapiro_p[in_full] = shapiro_p_fit
    p_adj[in_full] = p_adj_fit
    normal[in_full] = normal_fit
    max_abs_std[in_full] = max_abs_std_fit

    return pd.DataFrame(
        {
            "in_full_model": in_full,
            "n_samples": np.where(in_full, n_valid_samples, 0).astype(int),
            "df_resid": np.where(in_full, df_resid, 0).astype(int),
            "r2": r2,
            "residual_sigma": sigma,
            "max_abs_std_resid": max_abs_std,
            "shapiro_p": shapiro_p,
            "shapiro_p_adj": p_adj,
            "normal_residuals": normal,
        },
        index=feature_ids,
    )


def anova_2way_residual_diagnostics(
    feature_table: pd.DataFrame,
    metadata: pd.DataFrame,
    factors: list[str],
    *,
    alpha: float = 0.05,
    qq_max_points: int = 20000,
    rng_seed: int = 0,
) -> ResidualDiagnostics:
    """Build the 4-panel residual diagnostic figure for a 2-way ANOVA.

    Fits the full additive + interaction model
    ``y ~ factor1 + factor2 + factor1:factor2`` once per feature (only
    features with no missing values across the valid samples), extracts
    residuals and fitted values via QR, then summarizes them across the
    whole feature table.

    Args:
        feature_table: Features × samples processed (log-scale) table.
        metadata: Sample metadata indexed by sample name.
        factors: Exactly two metadata column names ``[factor1, factor2]``.
        alpha: Significance threshold for the Shapiro-Wilk rejection
            count, the dashed line on the histogram panel, and the
            per-feature ``normal_residuals`` flag.
        qq_max_points: Subsample cap for the pooled QQ scatter so the
            panel stays readable on large datasets. Sampling uses
            ``rng_seed``.
        rng_seed: Seed for the QQ subsample.

    Returns:
        :class:`ResidualDiagnostics` with the aggregate figure, headline
        numbers, and a per-feature DataFrame (R², residual σ, Shapiro
        p / p_adj, max |std residual|, ``normal_residuals``,
        ``in_full_model``) indexed by ``feature_id``.

    Raises:
        ValueError: If ``factors`` doesn't have length 2, the metadata
            has no samples valid for both factors, either factor has
            fewer than 2 observed levels, or no feature has complete
            data across the valid samples.
    """
    ctx = _fit_2way_residuals(feature_table, metadata, factors)
    per_feature = _per_feature_stats(ctx, alpha=alpha)

    resid = ctx["resid"]
    fitted = ctx["fitted"]
    df_resid = ctx["df_resid"]
    n_valid_samples = ctx["n_valid_samples"]
    f1, f2 = ctx["factors"]

    rss = (resid * resid).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma = np.sqrt(rss / df_resid)
        std_resid = resid / sigma

    fit_idx = per_feature.index[per_feature["in_full_model"].to_numpy()]
    shapiro_p_fit = per_feature.loc[fit_idx, "shapiro_p"].to_numpy()
    r2_fit = per_feature.loc[fit_idx, "r2"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # Panel 1: pooled QQ
    flat_std = std_resid.ravel()
    flat_std = flat_std[np.isfinite(flat_std)]
    if flat_std.size > qq_max_points:
        rng = np.random.default_rng(rng_seed)
        flat_std = rng.choice(flat_std, qq_max_points, replace=False)
    flat_std.sort()
    if flat_std.size:
        theo = norm.ppf((np.arange(flat_std.size) + 0.5) / flat_std.size)
        axes[0, 0].scatter(theo, flat_std, s=4, alpha=0.25, edgecolors="none")
        lo = float(min(theo.min(), flat_std.min()))
        hi = float(max(theo.max(), flat_std.max()))
        axes[0, 0].plot([lo, hi], [lo, hi], color="firebrick", lw=1)
    axes[0, 0].set_xlabel("Theoretical quantiles (N(0,1))")
    axes[0, 0].set_ylabel("Std. residual quantiles")
    axes[0, 0].set_title(f"Pooled QQ ({flat_std.size:,} residuals)")

    # Panel 2: Shapiro p histogram
    sp = shapiro_p_fit[~np.isnan(shapiro_p_fit)]
    if sp.size:
        axes[0, 1].hist(sp, bins=20, range=(0, 1), color="steelblue", edgecolor="white")
    axes[0, 1].axvline(alpha, color="firebrick", lw=1, ls="--", label=f"α = {alpha}")
    frac_reject = float((sp < alpha).mean()) if sp.size else float("nan")
    axes[0, 1].set_xlabel("Per-feature Shapiro-Wilk p")
    axes[0, 1].set_ylabel("Feature count")
    axes[0, 1].set_title(
        f"Shapiro on residuals — {frac_reject:.1%} reject at α={alpha}"
        if sp.size
        else "Shapiro on residuals — no testable features"
    )
    axes[0, 1].legend(loc="upper right", fontsize=9)

    # Panel 3: residuals vs fitted hexbin
    fitted_flat = fitted.ravel()
    resid_flat = resid.ravel()
    mask = np.isfinite(fitted_flat) & np.isfinite(resid_flat)
    if mask.any():
        hb = axes[1, 0].hexbin(
            fitted_flat[mask],
            resid_flat[mask],
            gridsize=60,
            cmap="viridis",
            mincnt=1,
            bins="log",
        )
        fig.colorbar(hb, ax=axes[1, 0], label="log10 count")
    axes[1, 0].axhline(0, color="firebrick", lw=1)
    axes[1, 0].set_xlabel("Fitted values")
    axes[1, 0].set_ylabel("Residuals")
    axes[1, 0].set_title("Residuals vs fitted (pooled)")

    # Panel 4: R² distribution
    r2_clean = r2_fit[np.isfinite(r2_fit)]
    median_r2 = float(np.median(r2_clean)) if r2_clean.size else float("nan")
    if r2_clean.size:
        axes[1, 1].hist(r2_clean, bins=30, color="darkseagreen", edgecolor="white")
        axes[1, 1].axvline(
            median_r2, color="firebrick", lw=1, ls="--",
            label=f"median = {median_r2:.2f}",
        )
    axes[1, 1].set_xlabel("Per-feature R²")
    axes[1, 1].set_ylabel("Feature count")
    axes[1, 1].set_title("Validity of fit (per-feature R²)")
    axes[1, 1].legend(loc="upper right", fontsize=9)

    n_features_fit = int(resid.shape[1])
    fig.suptitle(
        f"Two-way ANOVA residual diagnostics — {f1} × {f2} "
        f"({n_features_fit:,} features, {n_valid_samples} samples)",
        fontsize=12,
    )
    fig.tight_layout()

    return ResidualDiagnostics(
        figure=fig,
        n_features=n_features_fit,
        n_samples=int(n_valid_samples),
        median_r2=median_r2,
        frac_shapiro_reject=frac_reject,
        per_feature=per_feature,
    )


def feature_residual_diagnostics_figure(
    feature_id,
    feature_table: pd.DataFrame,
    metadata: pd.DataFrame,
    factors: list[str],
) -> plt.Figure:
    """Two-panel residual diagnostic for a single feature.

    Left: standardized residual QQ plot against N(0,1).
    Right: residual histogram with a fitted-N(0, σ̂) overlay.

    Args:
        feature_id: Row label in ``feature_table.index``.
        feature_table: Features × samples processed table.
        metadata: Sample metadata indexed by sample name.
        factors: Exactly two metadata column names.

    Returns:
        Matplotlib figure (caller closes it).

    Raises:
        KeyError: If ``feature_id`` isn't in ``feature_table.index``.
        ValueError: If the feature has any missing values across the
            valid samples (cannot be fit), or any condition that
            :func:`anova_2way_residual_diagnostics` would reject.
    """
    if feature_id not in feature_table.index:
        raise KeyError(f"feature_id {feature_id!r} not in feature_table.index")

    sub = feature_table.loc[[feature_id]]
    ctx = _fit_2way_residuals(sub, metadata, factors)

    complete = ctx["complete_mask"]
    if not complete.all():
        raise ValueError(
            f"feature {feature_id!r} has missing values across valid samples — "
            "cannot fit the 2-way model"
        )

    resid = ctx["resid"][:, 0]
    df_resid = ctx["df_resid"]
    sigma = float(np.sqrt((resid * resid).sum() / df_resid))
    std_resid = resid / sigma if sigma > 0 else resid

    try:
        sh_p = float(shapiro(resid).pvalue) if resid.std(ddof=1) > 0 else float("nan")
    except ValueError:
        sh_p = float("nan")

    f1, f2 = ctx["factors"]
    fig, (ax_qq, ax_hist) = plt.subplots(1, 2, figsize=(10, 4.2))

    sr_sorted = np.sort(std_resid)
    theo = norm.ppf((np.arange(sr_sorted.size) + 0.5) / sr_sorted.size)
    ax_qq.scatter(theo, sr_sorted, s=24, alpha=0.85, color="#3b82f6", edgecolors="none")
    lo = float(min(theo.min(), sr_sorted.min()))
    hi = float(max(theo.max(), sr_sorted.max()))
    ax_qq.plot([lo, hi], [lo, hi], color="firebrick", lw=1)
    ax_qq.set_xlabel("Theoretical quantiles (N(0,1))")
    ax_qq.set_ylabel("Std. residual quantiles")
    ax_qq.set_title(f"QQ plot ({resid.size} residuals)")

    bins = max(6, min(20, resid.size // 2))
    ax_hist.hist(
        resid, bins=bins, density=True, color="steelblue",
        edgecolor="white", alpha=0.85,
    )
    if sigma > 0 and np.isfinite(sigma):
        xs = np.linspace(resid.min(), resid.max(), 200)
        ax_hist.plot(xs, norm.pdf(xs, loc=0, scale=sigma), color="firebrick", lw=1.2,
                     label=f"N(0, σ̂={sigma:.3g})")
        ax_hist.legend(loc="upper right", fontsize=9)
    ax_hist.axvline(0, color="black", lw=0.7, ls=":")
    ax_hist.set_xlabel("Residual")
    ax_hist.set_ylabel("Density")
    sh_label = f"Shapiro p = {sh_p:.3g}" if np.isfinite(sh_p) else "Shapiro p = n/a"
    ax_hist.set_title(f"Residual distribution — {sh_label}")

    fig.suptitle(
        f"Feature {feature_id} residual diagnostics — {f1} × {f2}",
        fontsize=11,
    )
    fig.tight_layout()
    return fig
