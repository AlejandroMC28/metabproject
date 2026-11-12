"""Principal Coordinates Analysis with PERMANOVA and PERMDISP.

Distances are computed with :func:`scipy.spatial.distance.pdist` and the
ordination, PERMANOVA, and PERMDISP routines come from ``scikit-bio``.
scikit-bio is an optional dependency (see the ``pcoa`` extra in
``pyproject.toml``); importing this module does not require it, but calling
:func:`run_pcoa` will raise :class:`ImportError` with installation guidance
when it is missing.

scikit-bio's :func:`permanova` does not return an R² statistic, so it is
recomputed here from the pseudo-F using

    R² = (F · (k − 1) / (n − k)) / (1 + F · (k − 1) / (n − k))

where ``n`` is the number of samples and ``k`` the number of groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

_ALLOWED_METRICS = frozenset(
    {"euclidean", "cityblock", "canberra", "braycurtis", "jaccard", "minkowski"}
)


@dataclass
class PCoAResult:
    """Result of a PCoA fit with PERMANOVA / PERMDISP tests.

    Attributes:
        scores: Sample coordinates (samples × ``PC1..PCk``).
        explained_variance_ratio: Proportion of variance explained by each
            ordination axis (from skbio's ``proportion_explained``).
        distance_matrix: The :class:`skbio.DistanceMatrix` used for the
            ordination and the tests.
        permanova: When a single ``group_col`` was passed, a dict with
            ``test_statistic``, ``p_value``, ``R2``, ``sample_size``,
            ``number_of_groups``. When a list of grouping columns was
            passed, a dict keyed by column name → that same dict. ``R2``
            is computed manually from the pseudo-F.
        permdisp: PERMDISP equivalent of ``permanova``: either a single
            ``{test_statistic, p_value}`` dict, or one such dict per
            grouping column.
    """

    scores: pd.DataFrame
    explained_variance_ratio: np.ndarray
    distance_matrix: Any  # skbio.DistanceMatrix; quoted to avoid hard import
    permanova: dict
    permdisp: dict


def run_pcoa(
    feature_table: pd.DataFrame,
    metadata: pd.DataFrame,
    group_col: str | list[str],
    metric: str = "euclidean",
    permutations: int = 999,
    seed: int = 0,
) -> PCoAResult:
    """Run PCoA on the processed feature table with group-aware tests.

    Args:
        feature_table: Processed feature table (features × samples). The
            table is transposed internally so distances are computed
            between samples. Must contain no NaN values.
        metadata: Sample metadata indexed by sample name. Must contain
            ``group_col`` and cover every column of ``feature_table``.
        group_col: Name of the column in ``metadata`` carrying the group
            label used by PERMANOVA and PERMDISP. May also be a list of
            column names, in which case PERMANOVA / PERMDISP are run once
            per factor (e.g. ``["treatment", "variety"]``).
        metric: Distance metric passed to :func:`scipy.spatial.distance.pdist`.
            One of ``{euclidean, cityblock, canberra, braycurtis, jaccard,
            minkowski}``.
        permutations: Number of permutations for PERMANOVA and PERMDISP.
        seed: Seed forwarded to the permutation tests.

    Returns:
        :class:`PCoAResult` with sample scores, explained variance ratios,
        the distance matrix, and PERMANOVA / PERMDISP summaries.

    Raises:
        ImportError: If ``scikit-bio`` is not installed.
        ValueError: If ``metric`` is unsupported, ``group_col`` is missing
            from ``metadata``, the feature table contains NaNs, sample
            names cannot be aligned between table and metadata, or any
            group has fewer than 2 samples.
    """
    if metric not in _ALLOWED_METRICS:
        raise ValueError(
            f"Unsupported metric {metric!r} "
            f"(allowed: {sorted(_ALLOWED_METRICS)})"
        )
    if permutations < 1:
        raise ValueError(f"permutations must be >= 1 (got {permutations!r})")

    try:
        from skbio import DistanceMatrix
        from skbio.stats.distance import permanova, permdisp
        from skbio.stats.ordination import pcoa as skbio_pcoa
    except ImportError as e:  # pragma: no cover - exercised when extra missing
        raise ImportError(
            "scikit-bio is required for PCoA; install the optional 'pcoa' extra "
            "(pip install 'metabproject[pcoa]')"
        ) from e

    nan_features = int(feature_table.isna().any(axis=1).sum())
    if nan_features:
        raise ValueError(
            f"feature_table contains NaN values in {nan_features} feature(s); "
            "impute before running PCoA"
        )

    multi = isinstance(group_col, (list, tuple))
    group_cols = list(group_col) if multi else [group_col]
    bad = [c for c in group_cols if c not in metadata.columns]
    if bad:
        raise ValueError(
            f"metadata has no column(s) {bad} "
            f"(available: {metadata.columns.tolist()})"
        )

    sample_names = list(feature_table.columns)
    missing = [s for s in sample_names if s not in metadata.index]
    if missing:
        raise ValueError(
            f"Sample(s) absent from metadata index: {missing}"
        )

    for col in group_cols:
        groups_col = metadata.loc[sample_names, col].tolist()
        counts = pd.Series(groups_col).value_counts()
        if counts.size < 2:
            raise ValueError(
                f"group_col {col!r} must have >=2 distinct levels "
                f"(got {counts.index.tolist()})"
            )
        too_small = counts[counts < 2]
        if not too_small.empty:
            raise ValueError(
                f"group_col {col!r} levels with <2 samples: "
                f"{too_small.to_dict()}"
            )

    # Samples × features for pdist.
    matrix = feature_table.to_numpy(dtype=float).T
    dist = squareform(pdist(matrix, metric=metric))
    dm = DistanceMatrix(dist, ids=sample_names)

    ordination = skbio_pcoa(dm)
    k_components = ordination.samples.shape[1]
    pc_cols = [f"PC{i + 1}" for i in range(k_components)]
    scores = pd.DataFrame(
        ordination.samples.to_numpy(),
        index=sample_names,
        columns=pc_cols,
    )
    evr = np.asarray(ordination.proportion_explained.to_numpy(), dtype=float)

    permanova_by_col: dict[str, dict] = {}
    permdisp_by_col: dict[str, dict] = {}
    for col in group_cols:
        groups_col = metadata.loc[sample_names, col].tolist()
        perm = permanova(dm, groups_col, permutations=permutations, seed=seed)
        f_stat = float(perm["test statistic"])
        n_samples = int(perm["sample size"])
        n_groups = int(perm["number of groups"])
        denom = 1.0 + f_stat * (n_groups - 1) / (n_samples - n_groups)
        r2 = (f_stat * (n_groups - 1) / (n_samples - n_groups)) / denom
        pdisp = permdisp(dm, groups_col, permutations=permutations, seed=seed)
        permanova_by_col[col] = {
            "test_statistic": f_stat,
            "p_value": float(perm["p-value"]),
            "R2": float(r2),
            "sample_size": n_samples,
            "number_of_groups": n_groups,
        }
        permdisp_by_col[col] = {
            "test_statistic": float(pdisp["test statistic"]),
            "p_value": float(pdisp["p-value"]),
        }

    return PCoAResult(
        scores=scores,
        explained_variance_ratio=evr,
        distance_matrix=dm,
        permanova=permanova_by_col if multi else permanova_by_col[group_cols[0]],
        permdisp=permdisp_by_col if multi else permdisp_by_col[group_cols[0]],
    )
