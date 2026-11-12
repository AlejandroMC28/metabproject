"""Sample weight correction (optional). See ``specs/process.md`` Step 3."""

from __future__ import annotations

import pandas as pd


def weight_correct(df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Divide each sample's areas by its weight from the metadata.

    Args:
        df: Feature table (features × samples).
        metadata: Sample metadata with a ``weight`` column, indexed by
            ``sample_name``. All weights must be > 0.

    Returns:
        Weight-corrected DataFrame.

    Raises:
        ValueError: If the ``weight`` column is missing or any weight is <= 0.
    """
    if "weight" not in metadata.columns:
        raise ValueError(
            "Metadata missing 'weight' column required for weight normalization"
        )

    result = df.copy()
    for sample_name in result.columns:
        weight = float(metadata.loc[sample_name, "weight"])
        if weight <= 0:
            raise ValueError(
                f"Sample '{sample_name}' has weight <= 0 ({weight}). "
                "All weights must be positive."
            )
        result[sample_name] = result[sample_name] / weight

    return result
