"""Tests for the MZmine feature table parser (Zone 1/2/3 detection)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from metabproject.io.mzmine import (
    DATAFILE_COL,
    detect_zone_boundaries,
    parse_feature_table,
    parse_sample_filename,
)

N_FEATURES = 50
N_SAMPLES = 6


# ---------------------------------------------------------------------------
# detect_zone_boundaries
# ---------------------------------------------------------------------------


def test_zone_boundary_index(feature_table_df):
    cols = feature_table_df.columns.tolist()
    idx = detect_zone_boundaries(cols)
    assert cols[idx].startswith("datafile:")
    assert not cols[idx - 1].startswith("datafile:")


def test_zone_boundary_no_datafile_cols_raises():
    with pytest.raises(ValueError, match="No 'datafile:\\*' columns"):
        detect_zone_boundaries(["id", "mz", "rt"])


# ---------------------------------------------------------------------------
# parse_sample_filename
# ---------------------------------------------------------------------------


def test_parse_sample_filename_area():
    result = parse_sample_filename("datafile:Mots_S46.mzML:area")
    assert result == ("Mots_S46.mzML", "area")


def test_parse_sample_filename_feature_state():
    result = parse_sample_filename("datafile:Mots_S46.mzML:feature_state")
    assert result == ("Mots_S46.mzML", "feature_state")


def test_parse_sample_filename_non_datafile():
    assert parse_sample_filename("mz") is None
    assert parse_sample_filename("id") is None


# ---------------------------------------------------------------------------
# parse_feature_table
# ---------------------------------------------------------------------------


def test_ms1_features_shape(feature_table_csv):
    ms1, _, _ = parse_feature_table(feature_table_csv)
    # Zone 1 cols minus 'id' (renamed to feature_id → index)
    assert ms1.shape[0] == N_FEATURES
    assert ms1.index.name == "feature_id"
    assert "mz" in ms1.columns
    assert "rt" in ms1.columns


def test_areas_shape(feature_table_csv):
    _, areas, _ = parse_feature_table(feature_table_csv)
    assert areas.shape == (N_FEATURES, N_SAMPLES)


def test_areas_columns_are_filenames(feature_table_csv):
    """Zone 2 columns are raw .mzML filenames, not the full datafile:...:area string."""
    _, areas, _ = parse_feature_table(feature_table_csv)
    for col in areas.columns:
        assert col.endswith(".mzML")
        # Must NOT still have the datafile: prefix
        assert not col.startswith("datafile:")


def test_areas_values_numeric(feature_table_csv):
    _, areas, _ = parse_feature_table(feature_table_csv)
    assert areas.dtypes.apply(lambda dt: pd.api.types.is_float_dtype(dt)).all()


def test_areas_zeros_replaced_with_nan(feature_table_csv):
    """Explicit 0.0 values (MISSING) are replaced with NaN in the area table."""
    _, areas, _ = parse_feature_table(feature_table_csv)
    assert (areas == 0.0).sum().sum() == 0


def test_sample_details_multiindex(feature_table_csv):
    _, _, details = parse_feature_table(feature_table_csv)
    assert isinstance(details.columns, pd.MultiIndex)
    assert details.columns.names == ["filename", "metric"]
    # feature_state should be in there
    metrics = {m for _, m in details.columns}
    assert "feature_state" in metrics


def test_sample_details_shape(feature_table_csv):
    _, _, details = parse_feature_table(feature_table_csv)
    assert details.shape[0] == N_FEATURES


def test_feature_id_index_consistent(feature_table_csv):
    """All three outputs share the same feature_id index."""
    ms1, areas, details = parse_feature_table(feature_table_csv)
    assert (ms1.index == areas.index).all()
    assert (ms1.index == details.index).all()


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_feature_table(tmp_path / "nonexistent.csv")
