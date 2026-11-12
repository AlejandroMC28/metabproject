"""Centering and standardization (scaling). See ``specs/process.md`` Step 6.

Scaling is applied after :mod:`metabproject.processing.transformation`. It
centers each feature (row) across samples and divides by a measure of spread.
Useful for PCA / clustering / heatmaps; typically harmful when downstream
analysis depends on the original fold-change scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def scale(df: pd.DataFrame, method: str = "none") -> pd.DataFrame:
    """Dispatch to the requested scaling method.

    Args:
        df: Complete feature table (features × samples).
        method: One of ``none``, ``pareto``, ``auto``.

    Returns:
        Scaled DataFrame.

    Raises:
        ValueError: If ``method`` is unknown.
    """
    if method == "none":
        return df.copy()
    if method == "pareto":
        return scale_pareto(df)
    if method == "auto":
        return scale_auto(df)
    raise ValueError(f"Unknown scaling method: {method!r}")


def scale_pareto(df: pd.DataFrame) -> pd.DataFrame:
    """Pareto scaling: center per feature, divide by sqrt of the std."""
    means = df.mean(axis=1)
    stds = df.std(axis=1)
    return df.sub(means, axis=0).div(np.sqrt(stds), axis=0)


def scale_auto(df: pd.DataFrame) -> pd.DataFrame:
    """Autoscaling (z-score): center per feature, divide by the std."""
    means = df.mean(axis=1)
    stds = df.std(axis=1)
    return df.sub(means, axis=0).div(stds, axis=0)
