"""MZmine full feature table parser.

The MZmine CSV has a three-zone column structure (see ``specs/init.md``):

- **Zone 1** — consensus feature metadata (columns before any ``datafile:*``).
- **Zone 2** — per-sample peak areas (``datafile:*.mzML:area``).
- **Zone 3** — other per-sample detail columns (all remaining ``datafile:*``).

This module isolates all MZmine format specifics; it returns plain DataFrames.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd

#: Matches any per-sample datafile column, capturing filename and metric.
DATAFILE_COL = re.compile(r"^datafile:(?P<filename>.+\.mzML):(?P<metric>.+)$", re.IGNORECASE)

#: Matches datafile area columns specifically.
_AREA_COL = re.compile(r"^datafile:(?P<filename>.+\.mzML):area$", re.IGNORECASE)


def parse_feature_table(
    csv_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse an MZmine full feature table into its three zones.

    Args:
        csv_path: Path to the MZmine full feature table CSV.

    Returns:
        A tuple ``(ms1_features, areas, sample_details)`` where:

        - ``ms1_features``: Zone 1 consensus metadata, indexed by ``feature_id``.
        - ``areas``: Zone 2 peak areas (features × datafile-area columns),
          indexed by ``feature_id``. Columns are raw datafile filenames; mapping
          to ``sample_name`` happens during init using the metadata.
        - ``sample_details``: Zone 3 per-sample detail metrics with a
          ``(filename, metric)`` column MultiIndex, indexed by ``feature_id``.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Feature table not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    cols = df.columns.tolist()

    split = detect_zone_boundaries(cols)
    zone1_cols = cols[:split]
    datafile_cols = cols[split:]

    # Zone 1 → ms1_features
    ms1 = df[zone1_cols].copy()
    if "id" in ms1.columns:
        ms1 = ms1.rename(columns={"id": "feature_id"})
    ms1 = ms1.set_index("feature_id")

    # Split remaining into Zone 2 (area) and Zone 3 (everything else)
    area_cols = [c for c in datafile_cols if _AREA_COL.match(c)]
    detail_cols = [c for c in datafile_cols if c not in area_cols]

    # Zone 2 → areas (feature_id index, raw filename columns)
    areas = df[area_cols].copy()
    areas.index = ms1.index
    rename_area = {}
    for c in area_cols:
        m = _AREA_COL.match(c)
        if m:
            rename_area[c] = m.group("filename")
    areas = areas.rename(columns=rename_area)
    # Ensure numeric; missing/undetected values become NaN (not 0) to preserve
    # the distinction between "not detected" and "actually zero"
    for col in areas.columns:
        areas[col] = pd.to_numeric(areas[col], errors="coerce")
    # Replace explicit 0s that MZmine writes for MISSING features with NaN
    areas = areas.replace(0.0, float("nan"))

    # Zone 3 → sample_details (feature_id index, (filename, metric) MultiIndex)
    if detail_cols:
        detail_df = df[detail_cols].copy()
        detail_df.index = ms1.index
        tuples = []
        for c in detail_cols:
            parsed = parse_sample_filename(c)
            if parsed:
                tuples.append(parsed)
            else:
                tuples.append((c, ""))
        detail_df.columns = pd.MultiIndex.from_tuples(tuples, names=["filename", "metric"])
        sample_details = detail_df
    else:
        sample_details = pd.DataFrame(
            index=ms1.index,
            columns=pd.MultiIndex.from_tuples([], names=["filename", "metric"]),
        )

    return ms1, areas, sample_details


def detect_zone_boundaries(columns: list[str]) -> int:
    """Find the index of the first ``datafile:*`` column.

    Everything before this index is Zone 1 (consensus metadata).

    Args:
        columns: Ordered list of column names from the feature table.

    Returns:
        Index of the first per-sample datafile column.

    Raises:
        ValueError: If no ``datafile:*`` columns are found.
    """
    for i, col in enumerate(columns):
        if col.startswith("datafile:"):
            return i
    raise ValueError(
        "No 'datafile:*' columns found in the feature table. "
        "Is this a valid MZmine full feature table CSV?"
    )


def parse_sample_filename(column_name: str) -> tuple[str, str] | None:
    """Extract ``(filename, metric)`` from a ``datafile:...`` column header.

    Args:
        column_name: A column name, e.g.
            ``datafile:Mots_..._S46.mzML:area``.

    Returns:
        ``(filename, metric)`` if the column is a datafile column, else ``None``.
    """
    m = DATAFILE_COL.match(column_name)
    if m:
        return m.group("filename"), m.group("metric")
    return None
