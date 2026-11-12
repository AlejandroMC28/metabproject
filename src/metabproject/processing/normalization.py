"""Sample normalization (optional). See ``specs/process.md`` Step 4.

Corrects for systematic differences in total signal across samples. Operates on
the complete, weight-corrected matrix (no missing values).
"""

from __future__ import annotations

import pandas as pd


def normalize(df: pd.DataFrame, method: str = "none") -> pd.DataFrame:
    """Dispatch to the requested normalization method.

    Args:
        df: Complete feature table (features × samples), no missing values.
        method: One of ``none``, ``tic``, ``pqn``, ``median``.

    Returns:
        Normalized DataFrame.
    """
    if method == "none":
        return df.copy()
    if method == "tic":
        return normalize_tic(df)
    if method == "pqn":
        return normalize_pqn(df)
    if method == "median":
        return normalize_median(df)
    raise ValueError(f"Unknown normalization method: {method!r}")


def normalize_tic(df: pd.DataFrame) -> pd.DataFrame:
    """Total Ion Current normalization, rescaled to the median sample TIC."""
    tic_per_sample = df.sum(axis=0)
    median_tic = tic_per_sample.median()
    return df.div(tic_per_sample, axis=1) * median_tic


def normalize_pqn(df: pd.DataFrame) -> pd.DataFrame:
    """Probabilistic Quotient Normalization against a median reference sample."""
    reference = df.median(axis=1)
    quotients = df.div(reference, axis=0)
    norm_factors = quotients.median(axis=0)
    return df.div(norm_factors, axis=1)


def normalize_median(df: pd.DataFrame) -> pd.DataFrame:
    """Median normalization, rescaled to the global median of sample medians."""
    sample_medians = df.median(axis=0)
    global_median = sample_medians.median()
    return df.div(sample_medians, axis=1) * global_median
