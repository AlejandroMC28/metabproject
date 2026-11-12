"""Imputation of missing values (zeros / NaN) after prevalence filtering.

Operates on raw, uncorrected areas so the imputation threshold reflects the
actual instrument detection limit. See ``specs/process.md`` Step 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def impute(
    df: pd.DataFrame,
    method: str = "half-min",
    seed: int = 42,
    lod_value: float | None = None,
    knn_k: int = 5,
) -> tuple[pd.DataFrame, int]:
    """Dispatch to the requested imputation method.

    Args:
        df: Filtered feature table (features × samples).
        method: One of ``half-min``, ``second-min``, ``min``, ``lod``,
            ``knn``, ``none``.
        seed: Random seed for ``half-min`` and ``second-min``.
        lod_value: LOD value, required when ``method='lod'``.
        knn_k: Number of neighbors for ``knn``.

    Returns:
        ``(imputed_df, n_values_imputed)``.
    """
    if method == "half-min":
        return impute_half_min(df, seed=seed)
    if method == "second-min":
        return impute_second_min(df, seed=seed)
    if method == "min":
        return impute_min(df)
    if method == "lod":
        if lod_value is None:
            raise ValueError("LOD imputation requires --lod-value")
        return impute_lod(df, lod_value=lod_value)
    if method == "knn":
        return impute_knn(df, k=knn_k)
    if method == "none":
        return df.copy(), 0
    raise ValueError(f"Unknown imputation method: {method!r}")


def impute_half_min(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, int]:
    """Fill missing values with draws from ``Uniform(0.1·min, 0.5·min)``.

    ``min`` is the global minimum positive value across the dataset. Seeded for
    reproducibility.

    Args:
        df: Feature table (features × samples).
        seed: Random seed.

    Returns:
        ``(imputed_df, n_values_imputed)``.
    """
    result = df.copy()
    mask = (result == 0) | result.isna()
    n_missing = int(mask.sum().sum())

    if n_missing == 0:
        return result, 0

    flat = result.values
    positive_vals = flat[np.isfinite(flat) & (flat > 0)]
    global_min = float(positive_vals.min())

    rng = np.random.default_rng(seed)
    imputed_values = rng.uniform(
        low=global_min * 0.1,
        high=global_min * 0.5,
        size=n_missing,
    )

    # Write imputed values into masked positions row-major.
    result_vals = result.values.copy().astype(float)
    mask_vals = mask.values
    result_vals[mask_vals] = imputed_values
    result = pd.DataFrame(result_vals, index=result.index, columns=result.columns)
    return result, n_missing


def impute_second_min(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, int]:
    """Fill missing values with draws from ``Uniform(0, second_min)``.

    ``second_min`` is the second-smallest *unique* positive value across the
    dataset. This matches the imputation scheme used in the reference
    pipeline (Mejia-Aguilera et al.): uniform random noise bounded above by
    the second-smallest observed intensity, clipped to a strict positive
    floor of ``1e-10`` so the result survives a subsequent log transform.

    Args:
        df: Feature table (features × samples).
        seed: Random seed.

    Returns:
        ``(imputed_df, n_values_imputed)``.
    """
    result = df.copy()
    mask = (result == 0) | result.isna()
    n_missing = int(mask.sum().sum())

    if n_missing == 0:
        return result, 0

    flat = result.values
    positive_vals = flat[np.isfinite(flat) & (flat > 0)]
    unique_pos = np.unique(positive_vals)
    second_min = float(unique_pos[1]) if unique_pos.size > 1 else float(unique_pos[0])

    rng = np.random.default_rng(seed)
    imputed_values = np.clip(
        rng.uniform(low=0.0, high=second_min, size=n_missing),
        1e-10,
        None,
    )

    result_vals = result.values.copy().astype(float)
    result_vals[mask.values] = imputed_values
    result = pd.DataFrame(result_vals, index=result.index, columns=result.columns)
    return result, n_missing


def impute_min(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Replace missing values with the global minimum positive value."""
    result = df.copy()
    mask = (result == 0) | result.isna()
    n_missing = int(mask.sum().sum())

    if n_missing == 0:
        return result, 0

    flat = result.values
    positive_vals = flat[np.isfinite(flat) & (flat > 0)]
    global_min = float(positive_vals.min())

    result_vals = result.values.copy().astype(float)
    result_vals[mask.values] = global_min
    result = pd.DataFrame(result_vals, index=result.index, columns=result.columns)
    return result, n_missing


def impute_lod(df: pd.DataFrame, lod_value: float) -> tuple[pd.DataFrame, int]:
    """Replace missing values with a user-specified limit of detection."""
    result = df.copy()
    mask = (result == 0) | result.isna()
    n_missing = int(mask.sum().sum())

    result_vals = result.values.copy().astype(float)
    result_vals[mask.values] = lod_value
    result = pd.DataFrame(result_vals, index=result.index, columns=result.columns)
    return result, n_missing


def impute_knn(df: pd.DataFrame, k: int = 5) -> tuple[pd.DataFrame, int]:
    """K-nearest-neighbors imputation (features as rows, samples as columns)."""
    from sklearn.impute import KNNImputer

    # Replace explicit 0s with NaN so KNN treats them as missing.
    work = df.copy().astype(float)
    work[work == 0] = float("nan")
    n_missing = int(work.isna().sum().sum())

    imputer = KNNImputer(n_neighbors=k)
    imputed_vals = imputer.fit_transform(work.values)
    result = pd.DataFrame(imputed_vals, index=df.index, columns=df.columns)
    return result, n_missing
