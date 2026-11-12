"""Per-feature diagnostic tests for normality and variance homogeneity.

These functions inform the choice between parametric (e.g. Welch t-test)
and non-parametric (e.g. Mann-Whitney U) inference, but they do not
perform inference themselves. They are pure library functions: they take
DataFrames and return DataFrames keyed by ``feature_id``, with no I/O,
no plotting, and no CLI knowledge. BH-FDR adjustment uses the same
``scipy.stats.false_discovery_control`` routine as ``univariate.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, levene, shapiro

_LEVENE_CENTERS = {"median", "mean", "trimmed"}
_MIN_N = 3


def _bh_adjust(p_value: np.ndarray) -> np.ndarray:
    """BH-FDR adjustment that preserves NaNs."""
    p_adj = np.full_like(p_value, np.nan, dtype=float)
    valid = ~np.isnan(p_value)
    if valid.any():
        p_adj[valid] = false_discovery_control(p_value[valid], method="bh")
    return p_adj


def _validate_metadata_against_table(
    feature_table: pd.DataFrame,
    metadata: pd.DataFrame,
    group_col: str,
) -> None:
    if group_col not in metadata.columns:
        raise ValueError(
            f"group_col={group_col!r} not found in metadata.columns "
            f"(available: {list(metadata.columns)})"
        )
    levels = metadata[group_col].dropna().unique()
    if len(levels) < 2:
        raise ValueError(
            f"group_col={group_col!r} must have at least 2 distinct non-NaN "
            f"levels (got {len(levels)}: {list(levels)})"
        )
    missing = [s for s in metadata.index if s not in feature_table.columns]
    if missing:
        raise ValueError(
            "metadata.index contains samples absent from feature_table.columns: "
            f"{missing[:10]}{' ...' if len(missing) > 10 else ''}"
        )


def shapiro_per_feature(
    feature_table: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    group_col: str | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Per-feature Shapiro-Wilk normality test.

    Args:
        feature_table: Feature table (features × samples). Tested values
            are taken row-wise.
        metadata: Sample metadata indexed by sample name. Required when
            ``group_col`` is given; ignored otherwise.
        group_col: If ``None``, run Shapiro on each feature's distribution
            across all ``feature_table`` columns. If provided, run Shapiro
            on each feature's distribution within each group separately
            (the groups are the unique non-NaN values of
            ``metadata[group_col]``).
        alpha: Threshold for the ``normal`` flag — a feature (or
            feature/group cell) is flagged ``normal`` when its
            BH-adjusted p-value is ``>= alpha`` (i.e. we fail to reject
            normality).

    Returns:
        DataFrame indexed by ``feature_id``.

        Across-all-samples mode (``group_col is None``) columns:
        ``n, statistic, p_value, p_adj, normal, skipped_reason``.
        Features with fewer than 3 non-NaN values are skipped (NaN
        statistic / p_value / p_adj, ``normal=False``,
        ``skipped_reason="n<3"``). BH is applied across features.

        Per-group mode columns (groups in sorted order): for each group
        ``g``, ``{g}_n, {g}_statistic, {g}_p_value, {g}_p_adj, {g}_normal``;
        then ``all_groups_normal`` (bool, True iff every group's
        ``{g}_normal`` is True for that feature), ``n_groups_normal``
        (int), and ``skipped_reason`` (semicolon-joined list of
        ``"{g}:n<3"`` entries; empty string when no group was skipped).
        BH is applied within each group across features.

    Raises:
        ValueError: If ``group_col`` is given but missing from metadata,
            has fewer than 2 levels, or metadata.index references samples
            absent from ``feature_table.columns``.
    """
    if group_col is None:
        n_features = len(feature_table)
        data = feature_table.to_numpy(dtype=float)

        n = np.sum(~np.isnan(data), axis=1)
        statistic = np.full(n_features, np.nan)
        p_value = np.full(n_features, np.nan)
        skipped_reason = np.array([""] * n_features, dtype=object)

        for i in range(n_features):
            row = data[i]
            row = row[~np.isnan(row)]
            if row.size < _MIN_N:
                skipped_reason[i] = "n<3"
                continue
            res = shapiro(row)
            statistic[i] = float(res.statistic)
            p_value[i] = float(res.pvalue)

        p_adj = _bh_adjust(p_value)
        normal = (p_adj >= alpha) & ~np.isnan(p_adj)

        return pd.DataFrame(
            {
                "n": n.astype(int),
                "statistic": statistic,
                "p_value": p_value,
                "p_adj": p_adj,
                "normal": normal,
                "skipped_reason": skipped_reason,
            },
            index=feature_table.index,
        )

    if metadata is None:
        raise ValueError("metadata is required when group_col is provided")
    _validate_metadata_against_table(feature_table, metadata, group_col)

    groups = sorted(metadata[group_col].dropna().unique().tolist(), key=str)
    n_features = len(feature_table)

    per_group: dict[str, dict[str, np.ndarray]] = {}
    for g in groups:
        samples = metadata.index[metadata[group_col] == g].tolist()
        data = feature_table[samples].to_numpy(dtype=float)
        n = np.sum(~np.isnan(data), axis=1)
        statistic = np.full(n_features, np.nan)
        p_value = np.full(n_features, np.nan)
        for i in range(n_features):
            row = data[i]
            row = row[~np.isnan(row)]
            if row.size < _MIN_N:
                continue
            res = shapiro(row)
            statistic[i] = float(res.statistic)
            p_value[i] = float(res.pvalue)
        p_adj = _bh_adjust(p_value)
        per_group[str(g)] = {
            "n": n,
            "statistic": statistic,
            "p_value": p_value,
            "p_adj": p_adj,
        }

    columns: dict[str, np.ndarray] = {}
    normal_stack = np.zeros((len(groups), n_features), dtype=bool)
    tested_stack = np.zeros((len(groups), n_features), dtype=bool)
    skipped_lists: list[list[str]] = [[] for _ in range(n_features)]

    for gi, g in enumerate(groups):
        gs = str(g)
        cols = per_group[gs]
        is_normal = (cols["p_adj"] >= alpha) & ~np.isnan(cols["p_adj"])
        tested = ~np.isnan(cols["p_value"])
        columns[f"{gs}_n"] = cols["n"].astype(int)
        columns[f"{gs}_statistic"] = cols["statistic"]
        columns[f"{gs}_p_value"] = cols["p_value"]
        columns[f"{gs}_p_adj"] = cols["p_adj"]
        columns[f"{gs}_normal"] = is_normal
        normal_stack[gi] = is_normal
        tested_stack[gi] = tested
        skipped_idx = np.where(~tested)[0]
        for i in skipped_idx:
            skipped_lists[i].append(f"{gs}:n<3")

    all_tested = tested_stack.all(axis=0)
    all_groups_normal = normal_stack.all(axis=0) & all_tested
    n_groups_normal = normal_stack.sum(axis=0).astype(int)

    skipped_reason = np.array(
        ["; ".join(parts) for parts in skipped_lists], dtype=object
    )

    columns["all_groups_normal"] = all_groups_normal
    columns["n_groups_normal"] = n_groups_normal
    columns["skipped_reason"] = skipped_reason

    return pd.DataFrame(columns, index=feature_table.index)


def levene_per_feature(
    feature_table: pd.DataFrame,
    metadata: pd.DataFrame,
    group_col: str,
    center: str = "median",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Per-feature Levene's test for variance homogeneity across groups.

    Uses ``scipy.stats.levene`` with ``center='median'`` by default (the
    Brown-Forsythe variant, more robust to non-normality than the
    mean-centered version).

    Args:
        feature_table: Feature table (features × samples).
        metadata: Sample metadata indexed by sample name.
        group_col: Column in ``metadata`` whose unique non-NaN values
            define the groups. Must have at least 2 levels.
        center: Passed through to ``scipy.stats.levene``. One of
            ``"median"``, ``"mean"``, ``"trimmed"``.
        alpha: Threshold for the ``homoscedastic`` flag — a feature is
            flagged ``homoscedastic`` when its BH-adjusted p-value is
            ``>= alpha`` (i.e. we fail to reject equal variances).

    Returns:
        DataFrame indexed by ``feature_id`` with columns
        ``statistic, p_value, p_adj, homoscedastic, skipped_reason``. A
        feature is skipped (NaN statistic / p_value / p_adj,
        ``homoscedastic=False``) when any group has fewer than 3 non-NaN
        values; ``skipped_reason`` is a semicolon-joined list of
        ``"{g}:n<3"`` entries.

    Raises:
        ValueError: If ``group_col`` is missing from metadata, has fewer
            than 2 levels, metadata.index references samples absent from
            ``feature_table.columns``, or ``center`` is not one of the
            allowed values.
    """
    if center not in _LEVENE_CENTERS:
        raise ValueError(
            f"center must be one of {sorted(_LEVENE_CENTERS)}, got {center!r}"
        )
    _validate_metadata_against_table(feature_table, metadata, group_col)

    groups = sorted(metadata[group_col].dropna().unique().tolist(), key=str)
    group_arrays: list[tuple[str, np.ndarray]] = []
    for g in groups:
        samples = metadata.index[metadata[group_col] == g].tolist()
        group_arrays.append((str(g), feature_table[samples].to_numpy(dtype=float)))

    n_features = len(feature_table)
    statistic = np.full(n_features, np.nan)
    p_value = np.full(n_features, np.nan)
    skipped_lists: list[list[str]] = [[] for _ in range(n_features)]

    for i in range(n_features):
        per_group_values: list[np.ndarray] = []
        ok = True
        for g, arr in group_arrays:
            row = arr[i]
            row = row[~np.isnan(row)]
            if row.size < _MIN_N:
                skipped_lists[i].append(f"{g}:n<3")
                ok = False
            else:
                per_group_values.append(row)
        if not ok:
            continue
        res = levene(*per_group_values, center=center)
        statistic[i] = float(res.statistic)
        p_value[i] = float(res.pvalue)

    p_adj = _bh_adjust(p_value)
    homoscedastic = (p_adj >= alpha) & ~np.isnan(p_adj)
    skipped_reason = np.array(
        ["; ".join(parts) for parts in skipped_lists], dtype=object
    )

    return pd.DataFrame(
        {
            "statistic": statistic,
            "p_value": p_value,
            "p_adj": p_adj,
            "homoscedastic": homoscedastic,
            "skipped_reason": skipped_reason,
        },
        index=feature_table.index,
    )
