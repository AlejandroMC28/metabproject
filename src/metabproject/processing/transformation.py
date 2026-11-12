"""Variance-stabilizing transformations. See ``specs/process.md`` Step 5.

Transforms the data to approximate normality and stabilize variance. Centering
and standardization live in ``metabproject.processing.scaling``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def transform(df: pd.DataFrame, method: str = "log2") -> pd.DataFrame:
    """Dispatch to the requested transformation method.

    Args:
        df: Complete feature table (features × samples).
        method: One of ``log2``, ``log10``, ``sqrt``, ``none``.

    Returns:
        Transformed DataFrame.

    Raises:
        ValueError: If a log transform is requested but zeros/NaN remain, or
            if ``method`` is unknown.
    """
    if method == "none":
        return df.copy()
    if method == "log2":
        return transform_log2(df)
    if method == "log10":
        return transform_log10(df)
    if method == "sqrt":
        return transform_sqrt(df)
    raise ValueError(f"Unknown transformation method: {method!r}")


def _check_log_safe(df: pd.DataFrame, name: str) -> None:
    n_bad = int(((df <= 0) | df.isna()).sum().sum())
    if n_bad > 0:
        raise ValueError(
            f"Cannot apply {name} transform: {n_bad} zero/negative/NaN values remain. "
            "Impute or filter zero/missing values before applying this transform."
        )


def transform_log2(df: pd.DataFrame) -> pd.DataFrame:
    """``log2(x)``. Requires all values > 0."""
    _check_log_safe(df, "log2")
    return np.log2(df)


def transform_log10(df: pd.DataFrame) -> pd.DataFrame:
    """``log10(x)``. Requires all values > 0."""
    _check_log_safe(df, "log10")
    return np.log10(df)


def transform_sqrt(df: pd.DataFrame) -> pd.DataFrame:
    """``sqrt(x)``. Requires all values >= 0."""
    n_neg = int((df < 0).sum().sum())
    if n_neg > 0:
        raise ValueError(
            f"Cannot apply sqrt transform: {n_neg} negative values remain."
        )
    return np.sqrt(df)
