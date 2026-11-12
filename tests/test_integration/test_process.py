"""Integration tests for `metab process`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from metabproject.project import MetabProject


def _init_project(tmp_path: Path, feature_table_csv: Path, metadata_csv: Path) -> MetabProject:
    return MetabProject.init(
        path=tmp_path / "proj",
        feature_table=feature_table_csv,
        metadata=metadata_csv,
    )


# ---------------------------------------------------------------------------
# Basic end-to-end
# ---------------------------------------------------------------------------


def test_process_produces_feature_table(tmp_path, feature_table_csv, metadata_csv):
    """init -> process produces feature_table.parquet with plausible shape."""
    project = _init_project(tmp_path, feature_table_csv, metadata_csv)
    project.process(min_prevalence=1, impute="min", transform="log2")

    ft = project.feature_table
    assert ft.shape[1] == 6  # 6 samples
    assert ft.shape[0] <= 50  # at most all features (filter may remove some)
    assert ft.shape[0] > 0


def test_process_qc_figure_created(tmp_path, feature_table_csv, metadata_csv):
    """process writes qc/processing_qc.png."""
    project = _init_project(tmp_path, feature_table_csv, metadata_csv)
    project.process(min_prevalence=1, impute="min", transform="log2")
    assert (project.path / "qc" / "processing_qc.png").exists()


def test_process_manifest_entry(tmp_path, feature_table_csv, metadata_csv):
    """process appends a manifest entry with before/after feature counts."""
    project = _init_project(tmp_path, feature_table_csv, metadata_csv)
    project.process(min_prevalence=2, impute="half-min", transform="log2")

    manifest = json.loads((project.path / "metabproject.json").read_text())
    assert len(manifest["operations"]) == 2  # init + process

    op = manifest["operations"][1]
    assert op["command"] == "process"
    assert op["parameters"]["impute"] == "half-min"
    assert op["parameters"]["min_prevalence"] == 2
    assert op["results"]["features_before_filter"] == 50
    assert op["results"]["features_after_filter"] > 0
    assert op["results"]["values_imputed"] >= 0


# ---------------------------------------------------------------------------
# Pipeline order / intermediate values
# ---------------------------------------------------------------------------


def test_process_weight_normalization(tmp_path, feature_table_csv, metadata_csv):
    """With weight_normalize=True, processed values differ from non-normalized."""
    p1 = MetabProject.init(
        path=tmp_path / "proj_no_wt",
        feature_table=feature_table_csv,
        metadata=metadata_csv,
    )
    p1.process(min_prevalence=1, impute="min", weight_normalize=False, transform="none")

    p2 = MetabProject.init(
        path=tmp_path / "proj_wt",
        feature_table=feature_table_csv,
        metadata=metadata_csv,
    )
    p2.process(min_prevalence=1, impute="min", weight_normalize=True, transform="none")

    # Control samples have weight 1.5, fermented 2.06 — tables must differ.
    assert not p1.feature_table.equals(p2.feature_table)


def test_process_all_normalization_methods(tmp_path, feature_table_csv, metadata_csv):
    """All four normalization methods succeed without error."""
    for norm in ["none", "tic", "pqn", "median"]:
        proj = MetabProject.init(
            path=tmp_path / f"proj_{norm}",
            feature_table=feature_table_csv,
            metadata=metadata_csv,
        )
        proj.process(min_prevalence=1, impute="min", normalize=norm, transform="none")
        assert proj.feature_table.shape[1] == 6


def test_process_all_transforms(tmp_path, feature_table_csv, metadata_csv):
    """All transform/scale combos (except log on raw data) succeed without error."""
    combos = [
        ("log2", "none"),
        ("sqrt", "none"),
        ("none", "pareto"),
        ("none", "auto"),
        ("none", "none"),
        ("log2", "pareto"),
    ]
    for tr, sc in combos:
        proj = MetabProject.init(
            path=tmp_path / f"proj_{tr}_{sc}",
            feature_table=feature_table_csv,
            metadata=metadata_csv,
        )
        proj.process(min_prevalence=1, impute="min", transform=tr, scale=sc)
        assert proj.feature_table.shape[1] == 6


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_process_idempotency_errors_without_force(tmp_path, feature_table_csv, metadata_csv):
    """Second run errors without --force."""
    project = _init_project(tmp_path, feature_table_csv, metadata_csv)
    project.process(min_prevalence=1, impute="min", transform="log2")
    with pytest.raises(FileExistsError, match="already processed"):
        project.process(min_prevalence=1, impute="min", transform="log2")


def test_process_idempotency_succeeds_with_force(tmp_path, feature_table_csv, metadata_csv):
    """Second run with --force overwrites existing processed table."""
    project = _init_project(tmp_path, feature_table_csv, metadata_csv)
    project.process(min_prevalence=1, impute="min", transform="log2")
    project.process(min_prevalence=1, impute="min", transform="log2", force=True)
    ft = project.feature_table
    assert ft.shape[1] == 6


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_process_lod_requires_value(tmp_path, feature_table_csv, metadata_csv):
    """--impute lod without --lod-value raises ValueError."""
    project = _init_project(tmp_path, feature_table_csv, metadata_csv)
    with pytest.raises(ValueError, match="LOD imputation requires"):
        project.process(impute="lod", lod_value=None)


def test_process_weight_normalize_requires_weight_col(tmp_path, feature_table_df, tmp_path_factory):
    """--weight-normalize errors if metadata has no weight column."""
    import pandas as pd

    # Filenames must match the conftest fixture (CSF1 S46-48, CSF2 S49-51)
    meta_no_weight = pd.DataFrame({
        "sample_file": (
            [f"Mots_18102024_Foodomics_CSF1_S{46+i}.mzML" for i in range(3)] +
            [f"Mots_18102024_Foodomics_CSF2_S{49+i}.mzML" for i in range(3)]
        ),
        "sample_name": [f"s{i}" for i in range(6)],
        "group": ["ctrl"] * 3 + ["ferm"] * 3,
    })
    ft_path = tmp_path / "ft.csv"
    meta_path = tmp_path / "meta.csv"
    feature_table_df.to_csv(ft_path, index=False)
    meta_no_weight.to_csv(meta_path, index=False)

    project = MetabProject.init(path=tmp_path / "proj", feature_table=ft_path, metadata=meta_path)
    with pytest.raises(ValueError, match="'weight' column"):
        project.process(weight_normalize=True, impute="min")
