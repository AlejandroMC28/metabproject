"""Per-feature pairwise post-hoc tests.

Each function takes a feature table, a metadata frame, a factor column,
and an explicit list of features to test (typically the ones that passed
the omnibus). All three return a **long-format** DataFrame with one row
per (feature, pairwise comparison) so two-way ANOVA + Tukey results from
different factors can be stacked into a single frame downstream.

The common column ``factor`` echoes the factor name passed in, which is
how the runner distinguishes Tukey rows from factor1 vs factor2 vs a
combined factor in the 2-way layout.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import studentized_range

from metabproject.stats._util import (
    _sf_parallel,
    parallel_per_feature,
    validate_factor_columns,
)


def _hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    """Hedges' g with the standard small-sample bias correction."""
    nx, ny = x.size, y.size
    pooled_var = ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (
        nx + ny - 2
    )
    s_pooled = float(np.sqrt(pooled_var))
    if s_pooled == 0.0:
        return float("nan")
    d = (x.mean() - y.mean()) / s_pooled
    j = 1.0 - 3.0 / (4.0 * (nx + ny) - 9.0)
    return float(j * d)


def _samples_by_group(
    metadata: pd.DataFrame, factor: str, feature_table: pd.DataFrame
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for g, idx in metadata.groupby(factor, dropna=True).groups.items():
        samples = [s for s in idx if s in feature_table.columns]
        out[str(g)] = samples
    return out


def _long_frame(
    feature_row: np.ndarray,
    groups: dict[str, list[str]],
    sample_to_col: dict[str, int],
) -> pd.DataFrame:
    values: list[float] = []
    labels: list[str] = []
    for g, samples in groups.items():
        for s in samples:
            v = feature_row[sample_to_col[s]]
            if not np.isnan(v):
                values.append(float(v))
                labels.append(g)
    return pd.DataFrame({"value": values, "group": labels})


def tukey_hsd(
    feature_table: pd.DataFrame,
    metadata: pd.DataFrame,
    factor: str,
    features_to_test: list[str],
    alpha: float = 0.05,
    n_jobs: int | None = -1,
) -> pd.DataFrame:
    """Per-feature Tukey HSD via ``scipy.stats.studentized_range``.

    Implementation note: per-feature ANOVA sufficient statistics (per-group
    sums / sums-of-squares / counts) are accumulated vectorised across all
    features at once, then the studentized-range p-values are computed in
    a single batched + process-parallel call. ``studentized_range.sf``
    integrates per ``(q, df)`` pair internally, so the
    :func:`metabproject.stats._util._sf_parallel` helper (a) deduplicates
    identical df values across features (one integration instead of
    thousands in balanced designs) and (b) splits across processes for
    unbalanced or NaN-driven cases — quad doesn't reliably release the GIL.

    Effect size is Hedges' g with the standard small-sample correction,
    computed pairwise from the two groups in each comparison (not from
    the pooled MSE).

    Args:
        feature_table: Features × samples table.
        metadata: Sample metadata indexed by sample name.
        factor: Metadata column whose unique non-NaN values define the
            groups for pairwise comparison.
        features_to_test: Feature IDs to run the test on (typically the
            features that passed the omnibus).
        alpha: Threshold for the ``significant`` flag.
        n_jobs: Workers for the studentized-range p-value step.
            ``-1`` = all cores, ``1`` / ``None`` = serial. Per-feature
            math is fully vectorised; parallelism only applies to the
            integration-bound p-value step.

    Returns:
        Long-format DataFrame with columns ``feature, factor, group_A,
        group_B, mean_A, mean_B, diff, p_value, significant, hedges_g``.
        Empty (zero-row) DataFrame with the correct schema when
        ``features_to_test`` is empty.
    """
    columns = [
        "feature", "factor", "group_A", "group_B",
        "mean_A", "mean_B", "diff", "p_value", "significant", "hedges_g",
    ]
    if not features_to_test:
        return pd.DataFrame(columns=columns)

    validate_factor_columns(feature_table, metadata, [factor])

    groups = _samples_by_group(metadata, factor, feature_table)
    # Skip groups with no samples present in the feature table — they can't
    # contribute anyway and would just be dead columns.
    present_groups = sorted([g for g, s in groups.items() if s], key=str)
    if len(present_groups) < 2:
        return pd.DataFrame(columns=columns)

    sample_to_col = {s: i for i, s in enumerate(feature_table.columns)}
    group_cols: dict[str, np.ndarray] = {
        g: np.asarray([sample_to_col[s] for s in groups[g]], dtype=int)
        for g in present_groups
    }
    k = len(present_groups)

    feat_pos = feature_table.index.get_indexer(features_to_test)
    missing = [features_to_test[i] for i in np.where(feat_pos < 0)[0]]
    if missing:
        raise ValueError(
            f"features not in feature_table.index: {missing[:10]}"
            f"{' ...' if len(missing) > 10 else ''}"
        )

    matrix = feature_table.to_numpy(dtype=float, copy=False)
    Y = matrix[feat_pos]  # (n_feat, n_samples)
    n_feat = Y.shape[0]

    # Per-group sums / sum-of-squares / counts vectorised across features.
    group_sum = np.zeros((n_feat, k))
    group_sum_sq = np.zeros((n_feat, k))
    group_n = np.zeros((n_feat, k))
    for gi, g in enumerate(present_groups):
        sub = Y[:, group_cols[g]]
        valid = ~np.isnan(sub)
        group_n[:, gi] = valid.sum(axis=1)
        sub_filled = np.where(valid, sub, 0.0)
        group_sum[:, gi] = sub_filled.sum(axis=1)
        group_sum_sq[:, gi] = (sub_filled * sub_filled).sum(axis=1)

    # Features with <2 non-NaN samples in any group are dropped entirely.
    feature_ok = (group_n >= 2).all(axis=1)
    ok_idx = np.where(feature_ok)[0]
    if ok_idx.size == 0:
        return pd.DataFrame(columns=columns)

    with np.errstate(divide="ignore", invalid="ignore"):
        means = group_sum / group_n  # safe for ok features; junk for skipped.
    # Centred SS per group via the Σx² − n·mean² identity. Cheap because
    # we already have the three sufficient statistics; gives us the
    # pooled MSE for the q denominator and the per-pair pooled variance
    # for Hedges' g in one pass.
    ss_g = group_sum_sq - group_n * means * means

    total_ss = ss_g[ok_idx].sum(axis=1)
    total_n = group_n[ok_idx].sum(axis=1)
    df_resid = total_n - k
    with np.errstate(divide="ignore", invalid="ignore"):
        mse = total_ss / df_resid

    pair_indices = [(i, j) for i in range(k) for j in range(i + 1, k)]
    n_pairs = len(pair_indices)
    n_ok = ok_idx.size
    n_rows = n_ok * n_pairs

    feature_ids = np.asarray(features_to_test, dtype=object)
    feature_rep = np.repeat(feature_ids[ok_idx], n_pairs)
    pair_a = np.asarray([present_groups[i] for i, _ in pair_indices], dtype=object)
    pair_b = np.asarray([present_groups[j] for _, j in pair_indices], dtype=object)
    group_a_rep = np.tile(pair_a, n_ok)
    group_b_rep = np.tile(pair_b, n_ok)

    means_ok = means[ok_idx]
    n_ok_g = group_n[ok_idx]
    ss_ok = ss_g[ok_idx]

    # Stage 1: compute every q-statistic and effect size up front, fully
    # vectorised. No p-values yet — those are the slow part.
    mean_a = np.empty(n_rows)
    mean_b = np.empty(n_rows)
    diff = np.empty(n_rows)
    hedges = np.empty(n_rows)
    q_all = np.empty(n_rows)
    df_all = np.empty(n_rows)

    for pidx, (i, j) in enumerate(pair_indices):
        m_i = means_ok[:, i]
        m_j = means_ok[:, j]
        n_i = n_ok_g[:, i]
        n_j = n_ok_g[:, j]
        # diff is mean_B - mean_A by convention (see posthoc docstring + tukey_pair_to_contrast):
        # positive ⇒ group_B higher than group_A.
        d = m_j - m_i
        with np.errstate(divide="ignore", invalid="ignore"):
            se = np.sqrt(mse / 2.0 * (1.0 / n_i + 1.0 / n_j))
            q = np.abs(d) / se
            pooled_var = (ss_ok[:, i] + ss_ok[:, j]) / (n_i + n_j - 2.0)
            s_pooled = np.sqrt(pooled_var)
            j_corr = 1.0 - 3.0 / (4.0 * (n_i + n_j) - 9.0)
            h = j_corr * d / s_pooled
            h = np.where(s_pooled > 0, h, np.nan)

        # Strided assignment: rows are laid out as
        # [feat0-pair0, feat0-pair1, ..., feat1-pair0, ...], so writing
        # to slice(pidx, n_rows, n_pairs) hits this pair's row in every
        # feature with a single vectorised store.
        slot = slice(pidx, n_rows, n_pairs)
        mean_a[slot] = m_i
        mean_b[slot] = m_j
        diff[slot] = d
        hedges[slot] = h
        q_all[slot] = q
        df_all[slot] = df_resid

    # Stage 2: one batched + parallelised studentized-range call for all
    # rows at once. scipy's studentized_range.sf is continuous down to
    # numerical precision — unlike statsmodels' psturng which floors p
    # at ~1e-3 and used to pile every highly-significant Tukey pair onto
    # the same -log10(p) = 3 ceiling on volcano plots. The finite mask
    # guards against q = inf from degenerate features (e.g. zero MSE).
    finite = np.isfinite(q_all)
    pval = np.full(n_rows, np.nan)
    if finite.any():
        pval[finite] = _sf_parallel(q_all[finite], k, df_all[finite], n_jobs)

    sig = (pval < alpha) & ~np.isnan(pval)
    return pd.DataFrame(
        {
            "feature": feature_rep,
            "factor": factor,
            "group_A": group_a_rep,
            "group_B": group_b_rep,
            "mean_A": mean_a,
            "mean_B": mean_b,
            "diff": diff,
            "p_value": pval,
            "significant": sig,
            "hedges_g": hedges,
        },
        columns=columns,
    )


def games_howell(
    feature_table: pd.DataFrame,
    metadata: pd.DataFrame,
    factor: str,
    features_to_test: list[str],
    alpha: float = 0.05,
    n_jobs: int | None = -1,
) -> pd.DataFrame:
    """Per-feature Games-Howell pairwise post-hoc test.

    Heteroscedasticity-robust analog of :func:`tukey_hsd`. Same return
    schema. Implemented inline against ``scipy.stats.studentized_range``
    using the standard Welch t / Satterthwaite-df pairing (Games & Howell
    1976) — no pingouin dependency.
    """
    columns = [
        "feature", "factor", "group_A", "group_B",
        "mean_A", "mean_B", "diff", "p_value", "significant", "hedges_g",
    ]
    if not features_to_test:
        return pd.DataFrame(columns=columns)

    validate_factor_columns(feature_table, metadata, [factor])
    groups = _samples_by_group(metadata, factor, feature_table)
    sample_to_col = {s: i for i, s in enumerate(feature_table.columns)}

    def _worker(feature: str) -> list[dict]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row = feature_table.loc[feature].to_numpy(dtype=float)
            long = _long_frame(row, groups, sample_to_col)
            if long["group"].nunique() < 2:
                return []
            present = sorted(long["group"].unique().tolist())
            arrs = [
                long.loc[long["group"] == g, "value"].to_numpy() for g in present
            ]
            if any(a.size < 2 for a in arrs):
                return []
            k = len(present)
            means = np.array([a.mean() for a in arrs])
            vars_ = np.array([a.var(ddof=1) for a in arrs])
            ns = np.array([a.size for a in arrs], dtype=float)
            out: list[dict] = []
            for i in range(k):
                for j in range(i + 1, k):
                    vi_ni, vj_nj = vars_[i] / ns[i], vars_[j] / ns[j]
                    se2 = vi_ni + vj_nj
                    if se2 <= 0:
                        p = float("nan")
                    else:
                        se = float(np.sqrt(se2))
                        t = (means[i] - means[j]) / se
                        df = se2**2 / (
                            (vi_ni**2) / (ns[i] - 1) + (vj_nj**2) / (ns[j] - 1)
                        )
                        q = abs(t) * float(np.sqrt(2.0))
                        p = float(studentized_range.sf(q, k, df))
                    out.append(
                        {
                            "feature": feature,
                            "factor": factor,
                            "group_A": present[i],
                            "group_B": present[j],
                            "mean_A": float(means[i]),
                            "mean_B": float(means[j]),
                            "diff": float(means[i] - means[j]),
                            "p_value": p,
                            "significant": bool(p < alpha) if not np.isnan(p) else False,
                            "hedges_g": _hedges_g(arrs[i], arrs[j]),
                        }
                    )
            return out

    rows = [r for chunk in parallel_per_feature(features_to_test, _worker, n_jobs) for r in chunk]
    return pd.DataFrame(rows, columns=columns)


def dunn(
    feature_table: pd.DataFrame,
    metadata: pd.DataFrame,
    factor: str,
    features_to_test: list[str],
    p_adjust: str = "fdr_bh",
    alpha: float = 0.05,
    n_jobs: int | None = -1,
) -> pd.DataFrame:
    """Per-feature Dunn's test via ``scikit_posthocs.posthoc_dunn``.

    Non-parametric pairwise companion to
    :func:`metabproject.stats.kruskal.kruskal_wallis`. ``p_adjust`` is
    passed through to ``scikit_posthocs.posthoc_dunn`` and is applied
    **across pairwise comparisons within each feature** (not across
    features). Use ``"fdr_bh"`` (the default) to mirror the omnibus
    BH-FDR style.

    Returns a long-format DataFrame with columns ``feature, factor,
    group_A, group_B, rank_mean_diff, p_value, significant``. The
    ``rank_mean_diff`` field is ``mean_rank(A) − mean_rank(B)`` after
    ranking all non-NaN observations across the groups in the test —
    matching how Dunn's z-statistic itself is built.

    ``group_A`` / ``group_B`` are assigned in **alphabetical order** of
    the group labels (via ``sorted``), so the sign of ``rank_mean_diff``
    is deterministic across runs regardless of metadata iteration order.
    A positive value means the alphabetically-earlier group has the
    larger mean rank.
    """
    columns = [
        "feature", "factor", "group_A", "group_B",
        "rank_mean_diff", "p_value", "significant",
    ]
    if not features_to_test:
        return pd.DataFrame(columns=columns)

    validate_factor_columns(feature_table, metadata, [factor])
    groups = _samples_by_group(metadata, factor, feature_table)
    sample_to_col = {s: i for i, s in enumerate(feature_table.columns)}

    def _worker(feature: str) -> list[dict]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row = feature_table.loc[feature].to_numpy(dtype=float)
            long = _long_frame(row, groups, sample_to_col)
            present_groups = long["group"].unique().tolist()
            if len(present_groups) < 2:
                return []

            pmat = sp.posthoc_dunn(
                long, val_col="value", group_col="group", p_adjust=p_adjust,
            )
            ranks = long["value"].rank().to_numpy()
            mean_rank = {
                g: float(ranks[long["group"].to_numpy() == g].mean())
                for g in present_groups
            }
            out: list[dict] = []
            seen: set[tuple[str, str]] = set()
            for a in present_groups:
                for b in present_groups:
                    if a == b:
                        continue
                    key = tuple(sorted([str(a), str(b)]))
                    if key in seen:
                        continue
                    seen.add(key)
                    ga, gb = key
                    p = float(pmat.loc[ga, gb])
                    out.append(
                        {
                            "feature": feature,
                            "factor": factor,
                            "group_A": ga,
                            "group_B": gb,
                            "rank_mean_diff": mean_rank[ga] - mean_rank[gb],
                            "p_value": p,
                            "significant": bool(p < alpha),
                        }
                    )
            return out

    rows = [r for chunk in parallel_per_feature(features_to_test, _worker, n_jobs) for r in chunk]
    return pd.DataFrame(rows, columns=columns)
