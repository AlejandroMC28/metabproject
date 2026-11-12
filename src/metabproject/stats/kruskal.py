"""Per-feature Kruskal-Wallis rank-sum test via ``scipy.stats.kruskal``.

Non-parametric one-way analog of :func:`metabproject.stats.anova.anova_1way`.
Use when normality (per
:func:`metabproject.stats.diagnostics.shapiro_per_feature`) is rejected
within groups and a parametric ANOVA is therefore inappropriate.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import kruskal

from metabproject.stats._util import (
    bh_adjust,
    parallel_per_feature,
    validate_factor_columns,
)


def kruskal_wallis(
    feature_table: pd.DataFrame,
    metadata: pd.DataFrame,
    factor: str,
    min_samples_per_group: int = 3,
    alpha: float = 0.05,
    n_jobs: int | None = -1,
) -> pd.DataFrame:
    """Per-feature Kruskal-Wallis omnibus test.

    Args:
        feature_table: Features × samples table.
        metadata: Sample metadata indexed by sample name.
        factor: Metadata column whose unique non-NaN values define the
            groups (≥2 levels required).
        min_samples_per_group: Per-group minimum non-NaN values for a
            feature to be tested. Features below the threshold get NaN
            results and are excluded from FDR.
        alpha: Significance threshold for the ``significant`` flag.

    Returns:
        DataFrame indexed by ``feature_id`` with columns ``H, p_value,
        p_adj, significant, group_medians``. ``group_medians`` is a dict
        ``{group_label: median}`` per feature (``{}`` when skipped).

    Raises:
        ValueError: If ``factor`` is missing from metadata, has fewer
            than 2 levels, or metadata.index references samples absent
            from ``feature_table.columns``.
    """
    validate_factor_columns(feature_table, metadata, [factor])

    groups: dict[str, list[str]] = {}
    for g, idx in metadata.groupby(factor, dropna=True).groups.items():
        samples = [s for s in idx if s in feature_table.columns]
        groups[str(g)] = samples
    group_labels = sorted(groups, key=str)
    group_arrays = {g: feature_table[s].to_numpy(dtype=float) for g, s in groups.items()}

    n_features = len(feature_table)

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
            try:
                h_stat, p_value = kruskal(*per_group_values.values())
            except ValueError:
                # All values identical → undefined ranks.
                return (np.nan, np.nan, {})
            return (
                float(h_stat),
                float(p_value),
                {g: float(np.median(v)) for g, v in per_group_values.items()},
            )

    results = parallel_per_feature(range(n_features), _worker, n_jobs)
    h_stat = np.array([r[0] for r in results])
    p_value = np.array([r[1] for r in results])
    medians_per_row = [r[2] for r in results]

    p_adj = bh_adjust(p_value)
    significant = (p_adj < alpha) & ~np.isnan(p_adj)

    return pd.DataFrame(
        {
            "H": h_stat,
            "p_value": p_value,
            "p_adj": p_adj,
            "significant": significant,
            "group_medians": pd.Series(medians_per_row, index=feature_table.index),
        },
        index=feature_table.index,
    )
