"""Hierarchical clustering analysis on the processed feature table.

Clustering is performed over samples: the feature table (features ×
samples) is transposed so each sample becomes a vector. Distances come
from :func:`scipy.spatial.distance.pdist` and the linkage from
:func:`scipy.cluster.hierarchy.linkage`. When the caller passes
``n_clusters``, a flat assignment is produced via
:func:`scipy.cluster.hierarchy.fcluster` with ``criterion='maxclust'``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

_ALLOWED_METRICS = frozenset(
    {"euclidean", "cityblock", "canberra", "braycurtis", "jaccard", "minkowski",
     "correlation", "hamming", "chebyshev", "mahalanobis", "seuclidean", "sqeuclidean"}
)

_ALLOWED_LINKAGE = frozenset(
    {"single", "complete", "average", "weighted", "centroid", "median", "ward"}
)


@dataclass
class HCAResult:
    """Result of a hierarchical clustering run.

    Attributes:
        linkage_matrix: SciPy linkage matrix, shape ``(n_samples - 1, 4)``.
        cluster_labels: Flat cluster assignment indexed by sample name, or
            ``None`` when ``n_clusters`` was not requested.
        metric: Distance metric used.
        linkage_method: Linkage method used.
    """

    linkage_matrix: np.ndarray
    cluster_labels: pd.Series | None
    metric: str
    linkage_method: str


def run_hca(
    feature_table: pd.DataFrame,
    metric: str = "euclidean",
    linkage_method: str = "complete",
    n_clusters: int | None = None,
) -> HCAResult:
    """Hierarchically cluster samples from the processed feature table.

    Args:
        feature_table: Processed feature table (features × samples). The
            table is transposed internally so clustering acts on samples.
            Must contain no NaN values.
        metric: Distance metric passed to
            :func:`scipy.spatial.distance.pdist`. Common choices include
            ``euclidean``, ``cityblock``, ``canberra``, ``braycurtis``,
            ``jaccard``, ``minkowski``.
        linkage_method: Linkage rule for
            :func:`scipy.cluster.hierarchy.linkage`. One of
            ``{single, complete, average, weighted, centroid, median, ward}``.
            ``ward`` requires ``metric='euclidean'``.
        n_clusters: When given, cut the dendrogram into this many clusters
            and return the flat assignment. When ``None``, only the linkage
            matrix is returned.

    Returns:
        :class:`HCAResult` with the linkage matrix and, optionally, flat
        cluster labels indexed by sample name.

    Raises:
        ValueError: If ``metric`` or ``linkage_method`` is unsupported, the
            feature table contains NaNs, the table has fewer than 2
            samples, or ``n_clusters`` is not in ``[1, n_samples]``. Also
            raised when ``ward`` is paired with a non-euclidean metric.
    """
    if metric not in _ALLOWED_METRICS:
        raise ValueError(
            f"Unsupported metric {metric!r} "
            f"(allowed: {sorted(_ALLOWED_METRICS)})"
        )
    if linkage_method not in _ALLOWED_LINKAGE:
        raise ValueError(
            f"Unsupported linkage_method {linkage_method!r} "
            f"(allowed: {sorted(_ALLOWED_LINKAGE)})"
        )
    if linkage_method == "ward" and metric != "euclidean":
        raise ValueError(
            "linkage_method='ward' requires metric='euclidean' "
            f"(got metric={metric!r})"
        )

    n_samples = feature_table.shape[1]
    if n_samples < 2:
        raise ValueError(
            f"feature_table must have at least 2 samples (got {n_samples})"
        )

    nan_features = int(feature_table.isna().any(axis=1).sum())
    if nan_features:
        raise ValueError(
            f"feature_table contains NaN values in {nan_features} feature(s); "
            "impute before running HCA"
        )

    sample_names = list(feature_table.columns)
    matrix = feature_table.to_numpy(dtype=float).T

    distances = pdist(matrix, metric=metric)
    Z = linkage(distances, method=linkage_method)

    labels: pd.Series | None = None
    if n_clusters is not None:
        if n_clusters < 1 or n_clusters > n_samples:
            raise ValueError(
                f"n_clusters must be in [1, {n_samples}] (got {n_clusters!r})"
            )
        flat = fcluster(Z, t=n_clusters, criterion="maxclust")
        labels = pd.Series(flat.astype(int), index=sample_names, name="cluster")

    return HCAResult(
        linkage_matrix=Z,
        cluster_labels=labels,
        metric=metric,
        linkage_method=linkage_method,
    )
