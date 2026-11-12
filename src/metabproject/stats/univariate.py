"""Welch's t-test per feature for one contrast.

The test runs on the processed feature table. ``log2fc`` uses the convention
``log2(B / A)`` (positive = up in group B). When the project was processed
with a ``log2`` or ``log10`` transform AND no scaling was applied on top, the
difference of group means on the processed table is the fold change (up to a
constant for ``log10``), and the ``fc_table`` is not consulted. For any other
combination — a non-log transform, or a log transform followed by scaling
(e.g. pareto/auto) that breaks the ``log(B) - log(A) = log(B/A)`` identity —
``log2fc`` is computed from a separate imputed-but-untransformed fold-change
matrix so that fold change stays biological regardless of how the table was
processed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, ttest_ind

_LOG_TRANSFORMS = {"log2", "log10"}


def welch_ttest(
    feature_table: pd.DataFrame,
    fc_table: pd.DataFrame | None,
    samples_a: list[str],
    samples_b: list[str],
    min_samples: int = 3,
    alpha: float = 0.05,
    transform: str | None = None,
    scale: str | None = None,
) -> pd.DataFrame:
    """Run a per-feature Welch t-test between two sample sets.

    Args:
        feature_table: Processed feature table (features × samples).
        fc_table: Imputed-but-untransformed feature table on the original
            area scale, used for ``log2fc`` whenever the log-path shortcut
            cannot be taken. May be ``None`` only when ``transform`` is
            ``"log2"`` or ``"log10"`` AND ``scale`` is ``None``/``"none"``.
            When provided, must share an index with ``feature_table``.
        samples_a: Sample names for group A (columns in ``feature_table``).
        samples_b: Sample names for group B.
        min_samples: Per-group minimum of non-NaN values required to test a
            feature. Features below the threshold get NaN p-values and are
            excluded from FDR adjustment.
        alpha: Significance threshold for ``direction`` and ``significant``.
        transform: Transform applied at ``process``. The log-path shortcut
            (``log2fc = mean_b - mean_a`` on the processed table, rescaled
            to log2 for ``"log10"``) is taken only when this is ``"log2"``
            or ``"log10"`` AND ``scale`` indicates no scaling. Any other
            combination requires ``fc_table``.
        scale: Scaling applied at ``process`` after ``transform`` (e.g.
            ``"pareto"``, ``"auto"``, ``"range"``). When this is anything
            other than ``None``/``"none"``, per-feature scaling breaks the
            ``log(B) - log(A) = log(B/A)`` identity, so the log-path
            shortcut is disabled and ``fc_table`` is required even for
            log-family transforms.

    Returns:
        Per-feature DataFrame keyed by ``feature_id`` with columns:
        ``mean_a, mean_b, n_a, n_b, log2fc, t_stat, p_value, p_adj,
        direction, significant``. ``log2fc`` follows the convention
        ``log2(B / A)``; ``direction`` is ``"up"`` when ``log2fc > 0`` and
        significant, ``"down"`` when ``log2fc < 0`` and significant.
    """
    use_log_path = transform in _LOG_TRANSFORMS and scale in (None, "none")

    if not use_log_path:
        if fc_table is None:
            raise ValueError(
                "fc_table is required when transform is not 'log2'/'log10' "
                "or when scaling has been applied on top "
                f"(got transform={transform!r}, scale={scale!r})"
            )
        if not feature_table.index.equals(fc_table.index):
            raise ValueError("feature_table and fc_table must share the same index")

    missing_a = [s for s in samples_a if s not in feature_table.columns]
    missing_b = [s for s in samples_b if s not in feature_table.columns]
    if missing_a or missing_b:
        raise ValueError(
            f"Sample(s) absent from feature_table: A={missing_a}, B={missing_b}"
        )

    a = feature_table[samples_a].to_numpy(dtype=float)
    b = feature_table[samples_b].to_numpy(dtype=float)

    mean_a = np.nanmean(a, axis=1)
    mean_b = np.nanmean(b, axis=1)
    n_a = np.sum(~np.isnan(a), axis=1)
    n_b = np.sum(~np.isnan(b), axis=1)

    testable = (n_a >= min_samples) & (n_b >= min_samples)

    t_stat = np.full(len(feature_table), np.nan)
    p_value = np.full(len(feature_table), np.nan)

    if testable.any():
        res = ttest_ind(
            a[testable], b[testable],
            axis=1, equal_var=False, nan_policy="omit",
        )
        t_stat[testable] = np.asarray(res.statistic, dtype=float)
        p_value[testable] = np.asarray(res.pvalue, dtype=float)

    p_adj = np.full(len(feature_table), np.nan)
    valid = ~np.isnan(p_value)
    if valid.any():
        p_adj[valid] = false_discovery_control(p_value[valid], method="bh")

    if use_log_path:
        # Processed table is already on a log scale: difference of means is
        # the log fold change. Rescale log10 → log2.
        diff = mean_b - mean_a
        log2fc = diff if transform == "log2" else diff * np.log2(10.0)
    else:
        # Raw-scale FC matrix: log2(B / A) on arithmetic means.
        assert fc_table is not None  # narrowed by use_log_path guard above
        fc_a = fc_table[samples_a].to_numpy(dtype=float)
        fc_b = fc_table[samples_b].to_numpy(dtype=float)
        fc_mean_a = np.nanmean(fc_a, axis=1)
        fc_mean_b = np.nanmean(fc_b, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                (fc_mean_a > 0) & (fc_mean_b > 0),
                fc_mean_b / fc_mean_a,
                np.nan,
            )
            log2fc = np.log2(ratio)

    significant = (p_adj < alpha) & ~np.isnan(p_adj)
    direction = np.where(
        ~significant, "ns",
        np.where(log2fc > 0, "up", "down"),
    )

    return pd.DataFrame(
        {
            "mean_a": mean_a,
            "mean_b": mean_b,
            "n_a": n_a.astype(int),
            "n_b": n_b.astype(int),
            "log2fc": log2fc,
            "t_stat": t_stat,
            "p_value": p_value,
            "p_adj": p_adj,
            "direction": direction,
            "significant": significant,
        },
        index=feature_table.index,
    )
