"""Prevalence filtering (per-group detection logic)."""

from __future__ import annotations

import pandas as pd


def prevalence_filter(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    min_prevalence: int = 2,
    min_groups_meeting: int = 1,
    mode: str = "any_value",
    detection_mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Filter features by per-group detection prevalence.

    Keeps a feature if it is detected in at least ``min_prevalence`` samples
    within **at least** ``min_groups_meeting`` groups.

    Two detection criteria are supported:

    - ``mode='any_value'`` (default, legacy): treats ``(area > 0) & notna()``
      as detection. Cannot distinguish a true MZmine detection from a
      gap-filled (ESTIMATED) value.
    - ``mode='detected'``: uses ``detection_mask`` (a features × samples
      boolean DataFrame where ``True`` marks samples MZmine flagged as
      ``DETECTED``). Drops features that survive only because of gap-filling.

    Args:
        df: Feature table (features × samples), values are peak areas.
        metadata: Sample metadata with a ``group`` column, indexed by
            ``sample_name``.
        min_prevalence: Minimum detections required within a single group.
        min_groups_meeting: Number of groups that must each meet
            ``min_prevalence``. Default 1 reproduces the legacy
            "passes in any group" semantics.
        mode: Detection criterion (``'any_value'`` or ``'detected'``).
        detection_mask: Required when ``mode='detected'``. A boolean
            DataFrame with the same feature index as ``df`` and a column
            set that overlaps ``df``'s sample columns; ``True`` marks
            detected samples.

    Returns:
        Filtered DataFrame containing only passing features.

    Raises:
        ValueError: If ``mode`` is unknown; if ``mode='detected'`` but
            ``detection_mask`` is None or shares no columns with ``df``;
            or if ``min_groups_meeting`` < 1.
    """
    if mode not in ("any_value", "detected"):
        raise ValueError(
            f"Unknown mode={mode!r}. Expected 'any_value' or 'detected'."
        )
    if min_groups_meeting < 1:
        raise ValueError(
            f"min_groups_meeting must be >= 1 (got {min_groups_meeting})."
        )

    if mode == "detected":
        if detection_mask is None:
            raise ValueError(
                "mode='detected' requires a detection_mask "
                "(features × samples boolean DataFrame)."
            )
        if not any(c in detection_mask.columns for c in df.columns):
            raise ValueError(
                "detection_mask shares no columns with df. "
                "Expected sample-name columns matching the feature table."
            )
        mask = detection_mask.reindex(index=df.index, columns=df.columns)
        detected_full = mask.fillna(False).astype(bool)
    else:
        detected_full = (df > 0) & df.notna()

    groups = metadata.groupby("group").groups  # {group_name: Index[sample_name]}
    n_meeting = pd.Series(0, index=df.index, dtype=int)

    for group_name, sample_names in groups.items():
        group_cols = [s for s in sample_names if s in df.columns]
        if not group_cols:
            continue
        group_meets = detected_full[group_cols].sum(axis=1) >= min_prevalence
        n_meeting += group_meets.astype(int)

    keep = n_meeting >= min_groups_meeting
    return df[keep].copy()
