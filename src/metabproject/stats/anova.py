"""Per-feature ANOVA routines: classical one-way, Welch one-way, two-way.

All three functions are pure: they take a processed feature table and a
metadata frame, validate inputs at the top, and return a per-feature
DataFrame indexed by ``feature_id``. BH-FDR is applied across features
via :func:`metabproject.stats._util.bh_adjust` (one independent
correction per omnibus p-value column — three corrections for the
two-way layout). No I/O, no plotting, no CLI knowledge.

The two-way layout also assigns each feature a bucket-style
``classification`` label (``ns``, ``factor1_only``, ``factor2_only``,
``both_no_interaction``, ``interaction``, ``interaction_only``) — the
:func:`metabproject.stats.comparison_runner.run_comparison` dispatcher
uses this to route post-hoc tests.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from scipy.stats import f_oneway

from metabproject.stats._util import (
    bh_adjust,
    parallel_per_feature,
    validate_factor_columns,
)


def _rss_multi(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Residual sum of squares for OLS ``Y ~ X`` per column of ``Y``.

    Uses reduced QR so ``RSS = ||y||² − ||Qᵀy||²``, avoiding the residual
    materialisation. ``X`` is ``(n, p)``, ``Y`` is ``(n, F)``; returns
    ``(F,)``.
    """
    Q, _ = np.linalg.qr(X, mode="reduced")
    QTY = Q.T @ Y
    return (Y * Y).sum(axis=0) - (QTY * QTY).sum(axis=0)


def _rss_single(X: np.ndarray, y: np.ndarray) -> float:
    Q, _ = np.linalg.qr(X, mode="reduced")
    QTy = Q.T @ y
    return float(y @ y - QTy @ QTy)


def _build_2way_designs(
    a_idx: np.ndarray, b_idx: np.ndarray, a_levels: int, b_levels: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the four design matrices for a Type-II 2-way layout.

    Treatment coding with level 0 as the reference. Returns
    ``(X_full, X_no_int, X_no_A, X_no_B)`` where the reduced models for
    testing A and B follow the Type-II convention (drop the term and the
    interaction, keep the other main effect).
    """
    n = a_idx.size
    intercept = np.ones((n, 1))
    A = np.zeros((n, a_levels - 1))
    for i in range(1, a_levels):
        A[:, i - 1] = a_idx == i
    B = np.zeros((n, b_levels - 1))
    for j in range(1, b_levels):
        B[:, j - 1] = b_idx == j
    # Interaction = element-wise product of every A dummy with every B dummy.
    AB = (A[:, :, None] * B[:, None, :]).reshape(n, -1)
    X_full = np.hstack([intercept, A, B, AB])
    X_no_int = np.hstack([intercept, A, B])
    X_no_A = np.hstack([intercept, B])
    X_no_B = np.hstack([intercept, A])
    return X_full, X_no_int, X_no_A, X_no_B


_CLASS_LABELS = np.array(
    [
        "ns",  # 000
        "factor1_only",  # 001 = s1
        "factor2_only",  # 010 = s2
        "both_no_interaction",  # 011 = s1+s2
        "interaction_only",  # 100 = si
        "interaction",  # 101 = s1+si
        "interaction",  # 110 = s2+si
        "interaction",  # 111 = s1+s2+si
    ],
    dtype=object,
)


def _welch_anova_from_arrays(arrays: list[np.ndarray]) -> tuple[float, float]:
    """One-way Welch's ANOVA from per-group arrays (Welch 1951).

    Returns ``(F, p)``; ``(nan, nan)`` if any group has zero variance.
    """
    k = len(arrays)
    n = np.array([a.size for a in arrays], dtype=float)
    m = np.array([a.mean() for a in arrays])
    v = np.array([a.var(ddof=1) for a in arrays])
    if np.any(v <= 0):
        return (np.nan, np.nan)
    w = n / v
    W = w.sum()
    m_bar = (w * m).sum() / W
    tmp = ((1.0 - w / W) ** 2 / (n - 1.0)).sum()
    num = (w * (m - m_bar) ** 2).sum() / (k - 1)
    denom = 1.0 + 2.0 * (k - 2) / (k**2 - 1.0) * tmp
    F = num / denom
    df1 = k - 1
    df2 = (k**2 - 1.0) / (3.0 * tmp)
    p = float(f_dist.sf(F, df1, df2))
    return float(F), p


def _group_membership(
    metadata: pd.DataFrame, factor: str, feature_table: pd.DataFrame
) -> dict[str, list[str]]:
    """Return {group_label: [sample names in feature_table.columns]}."""
    groups: dict[str, list[str]] = {}
    for g, idx in metadata.groupby(factor, dropna=True).groups.items():
        samples = [s for s in idx if s in feature_table.columns]
        groups[str(g)] = samples
    return groups


def anova_1way(
    feature_table: pd.DataFrame,
    metadata: pd.DataFrame,
    factor: str,
    min_samples_per_group: int = 3,
    alpha: float = 0.05,
    n_jobs: int | None = -1,
) -> pd.DataFrame:
    """Per-feature one-way classical ANOVA via ``scipy.stats.f_oneway``.

    Args:
        feature_table: Features × samples processed table.
        metadata: Sample metadata indexed by sample name.
        factor: Metadata column whose unique non-NaN values define the
            groups (≥2 levels required).
        min_samples_per_group: Minimum non-NaN samples *per group* for
            a feature to be tested. Features falling short are recorded
            as NaN and excluded from FDR.
        alpha: Significance threshold for the ``significant`` flag.

    Returns:
        DataFrame indexed by ``feature_id`` with columns ``F, p_value,
        p_adj, significant, group_means``. ``group_means`` is a dict
        ``{group_label: mean}`` per feature (``{}`` when skipped).
    """
    validate_factor_columns(feature_table, metadata, [factor])
    groups = _group_membership(metadata, factor, feature_table)
    group_labels = sorted(groups, key=str)

    n_features = len(feature_table)
    group_arrays = {
        g: feature_table[s].to_numpy(dtype=float) for g, s in groups.items()
    }

    def _worker(i: int) -> tuple[float, float, dict[str, float]]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            per_group_values: dict[str, np.ndarray] = {}
            for g in group_labels:
                row = group_arrays[g][i]
                row = row[~np.isnan(row)]
                if row.size < min_samples_per_group:
                    return (np.nan, np.nan, {})
                per_group_values[g] = row
            f_stat, p_value = f_oneway(*per_group_values.values())
            return (
                float(f_stat),
                float(p_value),
                {g: float(v.mean()) for g, v in per_group_values.items()},
            )

    results = parallel_per_feature(range(n_features), _worker, n_jobs)
    f_stat = np.array([r[0] for r in results])
    p_value = np.array([r[1] for r in results])
    means_per_row = [r[2] for r in results]

    p_adj = bh_adjust(p_value)
    significant = (p_adj < alpha) & ~np.isnan(p_adj)

    return pd.DataFrame(
        {
            "F": f_stat,
            "p_value": p_value,
            "p_adj": p_adj,
            "significant": significant,
            "group_means": pd.Series(means_per_row, index=feature_table.index),
        },
        index=feature_table.index,
    )


def welch_anova(
    feature_table: pd.DataFrame,
    metadata: pd.DataFrame,
    factor: str,
    min_samples_per_group: int = 3,
    alpha: float = 0.05,
    n_jobs: int | None = -1,
) -> pd.DataFrame:
    """Per-feature one-way Welch's ANOVA (Welch 1951).

    Same shape and return schema as :func:`anova_1way`, but uses Welch's
    correction for unequal variances (and unequal n) — more robust than
    classical ANOVA when Levene's test rejects homoscedasticity. The
    statistic is computed directly from the standard Welch (1951)
    formula; see :func:`_welch_anova_from_arrays`.
    """
    validate_factor_columns(feature_table, metadata, [factor])
    groups = _group_membership(metadata, factor, feature_table)
    group_labels = sorted(groups, key=str)

    n_features = len(feature_table)
    group_arrays = {
        g: feature_table[s].to_numpy(dtype=float) for g, s in groups.items()
    }

    def _worker(i: int) -> tuple[float, float, dict[str, float]]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            per_group_values: dict[str, np.ndarray] = {}
            for g in group_labels:
                row = group_arrays[g][i]
                row = row[~np.isnan(row)]
                if row.size < min_samples_per_group:
                    return (np.nan, np.nan, {})
                per_group_values[g] = row
            f_stat, p_value = _welch_anova_from_arrays(list(per_group_values.values()))
            return (
                f_stat,
                p_value,
                {g: float(v.mean()) for g, v in per_group_values.items()},
            )

    results = parallel_per_feature(range(n_features), _worker, n_jobs)
    f_stat = np.array([r[0] for r in results])
    p_value = np.array([r[1] for r in results])
    means_per_row = [r[2] for r in results]

    p_adj = bh_adjust(p_value)
    significant = (p_adj < alpha) & ~np.isnan(p_adj)

    return pd.DataFrame(
        {
            "F": f_stat,
            "p_value": p_value,
            "p_adj": p_adj,
            "significant": significant,
            "group_means": pd.Series(means_per_row, index=feature_table.index),
        },
        index=feature_table.index,
    )


def anova_2way(
    feature_table: pd.DataFrame,
    metadata: pd.DataFrame,
    factors: list[str],
    min_samples_per_cell: int = 2,
    alpha: float = 0.05,
    n_jobs: int | None = -1,
) -> pd.DataFrame:
    """Per-feature two-way ANOVA with interaction via vectorised QR-OLS.

    Three p-values per feature — factor1 main effect, factor2 main
    effect, and interaction — are BH-corrected **independently** across
    features. Each feature gets a bucket-style ``classification`` label
    based on which of the three corrected p-values fall below ``alpha``.

    Args:
        feature_table: Features × samples processed table.
        metadata: Sample metadata indexed by sample name.
        factors: Exactly two metadata column names ``[factor1, factor2]``.
        min_samples_per_cell: Minimum non-NaN samples in **every**
            factor1 × factor2 cell. Features falling short are recorded
            as NaN and excluded from FDR.
        alpha: Significance threshold for the three ``sig_*`` flags.

    Returns:
        DataFrame indexed by ``feature_id`` with columns:
        ``F_{f1}, p_{f1}, p_{f1}_adj, sig_{f1},
        F_{f2}, p_{f2}, p_{f2}_adj, sig_{f2},
        F_interaction, p_interaction, p_interaction_adj, sig_interaction,
        classification``. The ``{f1}`` and ``{f2}`` tokens are the
        literal factor names passed in ``factors``.
    """
    if len(factors) != 2:
        raise ValueError(f"anova_2way requires exactly 2 factors, got {len(factors)}")
    if factors[0] == factors[1]:
        raise ValueError("anova_2way factors must be distinct")
    validate_factor_columns(feature_table, metadata, factors)

    f1, f2 = factors
    n_features = len(feature_table)
    sample_names = list(feature_table.columns)

    # Restrict to samples valid for BOTH factors (drop rows where either label is NaN).
    meta_aligned = metadata.loc[sample_names, factors]
    valid_sample_mask = (
        ~meta_aligned[f1].isna().to_numpy() & ~meta_aligned[f2].isna().to_numpy()
    )

    f_f1 = np.full(n_features, np.nan)
    f_f2 = np.full(n_features, np.nan)
    f_ix = np.full(n_features, np.nan)
    p_f1 = np.full(n_features, np.nan)
    p_f2 = np.full(n_features, np.nan)
    p_ix = np.full(n_features, np.nan)

    if valid_sample_mask.any():
        a_labels = meta_aligned[f1].to_numpy()[valid_sample_mask]
        b_labels = meta_aligned[f2].to_numpy()[valid_sample_mask]
        # Observed levels only — a level with no samples would yield an
        # all-zero design column and break rank.
        a_levels = sorted(pd.unique(a_labels), key=str)
        b_levels = sorted(pd.unique(b_labels), key=str)
        n_a, n_b = len(a_levels), len(b_levels)

        if n_a >= 2 and n_b >= 2:
            a_idx = np.searchsorted(
                np.asarray([str(x) for x in a_levels]),
                np.asarray([str(x) for x in a_labels]),
            )
            b_idx = np.searchsorted(
                np.asarray([str(x) for x in b_levels]),
                np.asarray([str(x) for x in b_labels]),
            )
            cell_idx = a_idx * n_b + b_idx
            n_cells = n_a * n_b

            matrix = feature_table.to_numpy(dtype=float, copy=False)
            Y_full = matrix[:, valid_sample_mask]

            # Per-(feature, cell) non-NaN counts: a single matmul does it
            # for every feature at once.
            cell_one_hot = np.zeros((valid_sample_mask.sum(), n_cells), dtype=float)
            cell_one_hot[np.arange(cell_one_hot.shape[0]), cell_idx] = 1.0
            non_nan = (~np.isnan(Y_full)).astype(float)
            valid_per_cell = non_nan @ cell_one_hot  # (n_features, n_cells)
            feature_passes = (valid_per_cell >= min_samples_per_cell).all(axis=1)
            has_nan = np.isnan(Y_full).any(axis=1)

            X_full, X_no_int, X_no_A, X_no_B = _build_2way_designs(
                a_idx, b_idx, n_a, n_b
            )
            df_full = X_full.shape[1]
            df_A = n_a - 1
            df_B = n_b - 1
            df_int = (n_a - 1) * (n_b - 1)

            # Vectorised fast path: all features with zero NaN.
            vec_idx = np.where(feature_passes & ~has_nan)[0]
            if vec_idx.size:
                Y = Y_full[vec_idx].T  # (n_valid, n_vec)
                rss_full = _rss_multi(X_full, Y)
                rss_no_int = _rss_multi(X_no_int, Y)
                rss_no_A = _rss_multi(X_no_A, Y)
                rss_no_B = _rss_multi(X_no_B, Y)

                df_resid = Y.shape[0] - df_full
                with np.errstate(divide="ignore", invalid="ignore"):
                    ms_resid = rss_full / df_resid
                    F_A = ((rss_no_A - rss_no_int) / df_A) / ms_resid
                    F_B = ((rss_no_B - rss_no_int) / df_B) / ms_resid
                    F_I = ((rss_no_int - rss_full) / df_int) / ms_resid
                    p_A_v = f_dist.sf(F_A, df_A, df_resid)
                    p_B_v = f_dist.sf(F_B, df_B, df_resid)
                    p_I_v = f_dist.sf(F_I, df_int, df_resid)

                f_f1[vec_idx] = F_A
                f_f2[vec_idx] = F_B
                f_ix[vec_idx] = F_I
                p_f1[vec_idx] = p_A_v
                p_f2[vec_idx] = p_B_v
                p_ix[vec_idx] = p_I_v

            # Per-feature fallback for features with NaN. Same QR machinery,
            # just restricted to each feature's non-NaN mask. Skips features
            # that fail per-cell minimums (left as NaN).
            fb_idx = np.where(feature_passes & has_nan)[0]
            if fb_idx.size:

                def _fallback(
                    fi: int,
                ) -> tuple[float, float, float, float, float, float]:
                    row = Y_full[fi]
                    mask = ~np.isnan(row)
                    y = row[mask]
                    Xf = X_full[mask]
                    Xni = X_no_int[mask]
                    XnA = X_no_A[mask]
                    XnB = X_no_B[mask]
                    n_v = y.size
                    df_r = n_v - df_full
                    if df_r <= 0:
                        return (np.nan,) * 6
                    try:
                        rss_f = _rss_single(Xf, y)
                        rss_ni = _rss_single(Xni, y)
                        rss_nA = _rss_single(XnA, y)
                        rss_nB = _rss_single(XnB, y)
                    except np.linalg.LinAlgError:
                        return (np.nan,) * 6
                    ms_r = rss_f / df_r
                    if not np.isfinite(ms_r) or ms_r <= 0:
                        return (np.nan,) * 6
                    FA = ((rss_nA - rss_ni) / df_A) / ms_r
                    FB = ((rss_nB - rss_ni) / df_B) / ms_r
                    FI = ((rss_ni - rss_f) / df_int) / ms_r
                    return (
                        float(FA),
                        float(FB),
                        float(FI),
                        float(f_dist.sf(FA, df_A, df_r)),
                        float(f_dist.sf(FB, df_B, df_r)),
                        float(f_dist.sf(FI, df_int, df_r)),
                    )

                results = parallel_per_feature(list(fb_idx), _fallback, n_jobs)
                for k, fi in enumerate(fb_idx):
                    f_f1[fi], f_f2[fi], f_ix[fi], p_f1[fi], p_f2[fi], p_ix[fi] = (
                        results[k]
                    )

    p_f1_adj = bh_adjust(p_f1)
    p_f2_adj = bh_adjust(p_f2)
    p_ix_adj = bh_adjust(p_ix)

    sig_f1 = (p_f1_adj < alpha) & ~np.isnan(p_f1_adj)
    sig_f2 = (p_f2_adj < alpha) & ~np.isnan(p_f2_adj)
    sig_ix = (p_ix_adj < alpha) & ~np.isnan(p_ix_adj)

    code = (
        sig_f1.astype(np.uint8)
        | (sig_f2.astype(np.uint8) << 1)
        | (sig_ix.astype(np.uint8) << 2)
    )
    classification = _CLASS_LABELS[code]
    return pd.DataFrame(
        {
            f"F_{f1}": f_f1,
            f"p_{f1}": p_f1,
            f"p_{f1}_adj": p_f1_adj,
            f"sig_{f1}": sig_f1,
            f"F_{f2}": f_f2,
            f"p_{f2}": p_f2,
            f"p_{f2}_adj": p_f2_adj,
            f"sig_{f2}": sig_f2,
            "F_interaction": f_ix,
            "p_interaction": p_ix,
            "p_interaction_adj": p_ix_adj,
            "sig_interaction": sig_ix,
            "classification": classification,
        },
        index=feature_table.index,
    )
