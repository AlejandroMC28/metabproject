"""Principal Component Analysis on the processed feature table.

PCA runs on samples as observations and features as variables, so the input
feature table (features × samples) is transposed internally. When
``scale=True`` (the default), a :class:`StandardScaler` is applied to the
transposed matrix before fitting; the scaling is internal to this routine
and does not mutate the caller's table. The number of components actually
fitted is clamped to ``min(n_components, n_samples, n_features)`` and
reported back as ``n_components_used``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class PCAResult:
    """Result of a PCA fit.

    Attributes:
        scores: Sample coordinates in PC space (samples × ``PC1..PCk``).
        loadings: Feature loadings (features × ``PC1..PCk``).
        explained_variance_ratio: Fraction of variance explained by each PC,
            in the same order as the score/loading columns.
        n_components_used: Number of components actually fitted, after
            clamping against the matrix shape.
    """

    scores: pd.DataFrame
    loadings: pd.DataFrame
    explained_variance_ratio: np.ndarray
    n_components_used: int


def run_pca(
    feature_table: pd.DataFrame,
    n_components: int = 10,
    scale: bool = True,
) -> PCAResult:
    """Fit a PCA on the processed feature table.

    Args:
        feature_table: Processed feature table (features × samples). The
            table is transposed internally so PCA observes samples as rows.
            Must contain no NaN values.
        n_components: Maximum number of components to fit. Clamped down to
            ``min(n_components, n_samples, n_features)``.
        scale: When ``True``, apply :class:`StandardScaler` to the
            transposed (samples × features) matrix before fitting. This is
            internal to the routine; the input ``feature_table`` is not
            modified. When ``False``, PCA is fit on the raw transposed
            values (still mean-centred by sklearn).

    Returns:
        :class:`PCAResult` with the fitted scores, loadings, explained
        variance ratios, and the number of components actually used.

    Raises:
        ValueError: If ``feature_table`` contains NaNs, has fewer than two
            samples or two features, or ``n_components`` is not positive.
    """
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1 (got {n_components!r})")
    if feature_table.shape[0] < 2 or feature_table.shape[1] < 2:
        raise ValueError(
            "feature_table must have at least 2 features and 2 samples "
            f"(got shape={feature_table.shape!r})"
        )

    nan_features = int(feature_table.isna().any(axis=1).sum())
    if nan_features:
        raise ValueError(
            f"feature_table contains NaN values in {nan_features} feature(s); "
            "impute before running PCA"
        )

    # Samples become observations.
    matrix = feature_table.to_numpy(dtype=float).T
    sample_names = list(feature_table.columns)
    feature_names = list(feature_table.index)

    if scale:
        matrix = StandardScaler().fit_transform(matrix)

    n_samples, n_features = matrix.shape
    k = min(n_components, n_samples, n_features)

    pca = PCA(n_components=k)
    scores_arr = pca.fit_transform(matrix)

    pc_cols = [f"PC{i + 1}" for i in range(k)]
    scores = pd.DataFrame(scores_arr, index=sample_names, columns=pc_cols)
    loadings = pd.DataFrame(pca.components_.T, index=feature_names, columns=pc_cols)

    return PCAResult(
        scores=scores,
        loadings=loadings,
        explained_variance_ratio=np.asarray(pca.explained_variance_ratio_, dtype=float),
        n_components_used=k,
    )
