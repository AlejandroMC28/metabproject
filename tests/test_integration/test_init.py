"""Integration tests for `metab init`."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

from metabproject.project import MetabProject


def _do_init(tmp_path: Path, feature_table_csv: Path, metadata_csv: Path) -> MetabProject:
    """Helper: init a project in a fresh subdirectory."""
    project_dir = tmp_path / "my_project"
    return MetabProject.init(
        path=project_dir,
        feature_table=feature_table_csv,
        metadata=metadata_csv,
    )


def test_init_creates_project_structure(tmp_path, feature_table_csv, metadata_csv):
    """init creates the project dir, features/, manifest, and all required subdirs."""
    project = _do_init(tmp_path, feature_table_csv, metadata_csv)
    project_dir = project.path

    assert (project_dir / "metabproject.json").exists()
    assert (project_dir / "features" / "feature_table_raw.parquet").exists()
    assert (project_dir / "features" / "metadata.parquet").exists()
    assert (project_dir / "features" / "ms1_features.parquet").exists()
    assert (project_dir / "features" / "sample_details.parquet").exists()
    for subdir in ["qc", "annotations", "stats", "enrichment", "exports"]:
        assert (project_dir / subdir).is_dir()


def test_init_raw_feature_table_shape(tmp_path, feature_table_csv, metadata_csv):
    """raw_feature_table has shape (n_features, n_samples)."""
    project = _do_init(tmp_path, feature_table_csv, metadata_csv)
    ft = project.raw_feature_table
    assert ft.shape[0] == 50   # N_FEATURES from conftest
    assert ft.shape[1] == 6    # N_SAMPLES from conftest


def test_init_metadata_indexed_by_sample_name(tmp_path, feature_table_csv, metadata_csv):
    """metadata.parquet is indexed by sample_name."""
    project = _do_init(tmp_path, feature_table_csv, metadata_csv)
    meta = project.metadata
    assert meta.index.name == "sample_name"
    assert "group" in meta.columns


def test_init_ms1_features_shape(tmp_path, feature_table_csv, metadata_csv):
    """ms1_features has n_features rows and includes mz/rt columns."""
    project = _do_init(tmp_path, feature_table_csv, metadata_csv)
    ms1 = project.ms1_features
    assert ms1.shape[0] == 50
    assert "mz" in ms1.columns
    assert "rt" in ms1.columns


def test_init_manifest_content(tmp_path, feature_table_csv, metadata_csv):
    """Manifest records the init operation with counts and parameters."""
    import json

    project = _do_init(tmp_path, feature_table_csv, metadata_csv)
    manifest = json.loads((project.path / "metabproject.json").read_text())

    assert "project_name" in manifest
    assert "created" in manifest
    assert len(manifest["operations"]) == 1

    op = manifest["operations"][0]
    assert op["command"] == "init"
    assert op["results"]["n_features"] == 50
    assert op["results"]["n_samples"] == 6
    assert "control" in op["results"]["groups"]
    assert "fermented" in op["results"]["groups"]


def test_init_errors_on_existing_dir(tmp_path, feature_table_csv, metadata_csv):
    """Re-running init on an existing directory raises FileExistsError."""
    _do_init(tmp_path, feature_table_csv, metadata_csv)
    with pytest.raises(FileExistsError):
        _do_init(tmp_path, feature_table_csv, metadata_csv)


def test_init_drops_unmatched_samples_with_warning(tmp_path, metadata_df, tmp_path_factory):
    """Feature-table samples absent from metadata are dropped with a UserWarning."""
    import numpy as np
    import pandas as pd

    # Build a feature table with an extra sample not in metadata
    rng = np.random.default_rng(0)
    sample_files = list(metadata_df["sample_file"])
    extra_file = "extra_unknown_sample.mzML"
    all_files = sample_files + [extra_file]

    rows = {
        "id": np.arange(1, 11),
        "mz": rng.uniform(100, 500, 10),
        "rt": rng.uniform(1, 10, 10),
    }
    for f in all_files:
        rows[f"datafile:{f}:area"] = rng.lognormal(10, 1, 10)
        rows[f"datafile:{f}:feature_state"] = "DETECTED"

    ft = pd.DataFrame(rows)
    ft_path = tmp_path / "ft_extra.csv"
    ft.to_csv(ft_path, index=False)

    meta_path = tmp_path / "meta.csv"
    metadata_df.to_csv(meta_path, index=False)

    project_dir = tmp_path / "proj_extra"
    with pytest.warns(UserWarning, match="not in metadata"):
        project = MetabProject.init(
            path=project_dir,
            feature_table=ft_path,
            metadata=meta_path,
        )

    # Extra sample should be absent from the raw feature table.
    assert extra_file not in project.raw_feature_table.columns
    assert project.raw_feature_table.shape[1] == 6


def test_init_raises_on_missing_metadata_cols(tmp_path, feature_table_csv, tmp_path_factory):
    """Missing required metadata columns raise ValueError."""
    import pandas as pd

    bad_meta = pd.DataFrame({"sample_file": ["a.mzML"], "group": ["ctrl"]})
    meta_path = tmp_path / "bad_meta.csv"
    bad_meta.to_csv(meta_path, index=False)

    with pytest.raises(ValueError, match="missing required column"):
        MetabProject.init(
            path=tmp_path / "proj_bad",
            feature_table=feature_table_csv,
            metadata=meta_path,
        )


def test_init_raises_on_duplicate_sample_names(tmp_path, feature_table_df, metadata_df):
    """Duplicate sample_name in metadata raises ValueError."""
    import pandas as pd

    bad_meta = metadata_df.copy()
    bad_meta.loc[bad_meta.index[1], "sample_name"] = bad_meta.loc[bad_meta.index[0], "sample_name"]

    ft_path = tmp_path / "ft.csv"
    meta_path = tmp_path / "meta.csv"
    feature_table_df.to_csv(ft_path, index=False)
    bad_meta.to_csv(meta_path, index=False)

    with pytest.raises(ValueError, match="Duplicate sample_name"):
        MetabProject.init(path=tmp_path / "proj_dup", feature_table=ft_path, metadata=meta_path)
