"""Integration tests for MetabProject.annotate()."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from metabproject.project import MetabProject


# ---------------------------------------------------------------------------
# Synthetic annotation input files
# The feature IDs here must overlap with those produced by the conftest fixture
# (features 1–50), so we use feature IDs 1, 2, 3, 4.
# ---------------------------------------------------------------------------

_STRUCTURES_TSV = textwrap.dedent("""\
    mappingFeatureId\tstructurePerIdRank\tConfidenceScoreApproximate\tInChIkey2D\tInChI\tname\tsmiles\tmolecularFormula\tadduct
    1\t1\t0.80\tAAAAAAAAAAAAAAA\tInChI=1/A\tCompound A\tCC\tC2H6\t[M+H]+
    2\t1\t0.40\tCCCCCCCCCCCCCCC\tInChI=1/C\tCompound C\tO\tH2O\t[M+Na]+
    3\t1\t0.70\tDDDDDDDDDDDDDDD\tInChI=1/D\tCompound D\tCC=O\tC2H4O\t[M+H]+
""")

_CANOPUS_TSV = textwrap.dedent("""\
    mappingFeatureId\tNPC#pathway\tNPC#pathway Probability\tNPC#superclass\tNPC#superclass Probability\tNPC#class\tNPC#class Probability\tClassyFire#most specific class\tClassyFire#most specific class Probability
    1\tLipids\t0.90\tFatty acids\t0.85\tFA esters\t0.80\tWax esters\t0.75
    2\tAlkaloids\t0.80\tPyridine alkaloids\t0.75\tPiperidine\t0.72\tPiperidine alkaloids\t0.70
    3\tFlavonoids\t0.85\tFlavones\t0.80\tFlavone\t0.75\tFlavone C-glycosides\t0.72
    6\tTerpenoids\t0.90\tSesquiterpenes\t0.88\tEudesmane\t0.85\tEudesmane sesquiterpenes\t0.80
""")

_GNPS_LIBRARY_TSV = textwrap.dedent("""\
    #Scan#\tCompound_Name\tSmiles\tINSCHI\tInChIKey-Planar\tMQScore\tnpclassifier_pathway\tnpclassifier_superclass\tnpclassifier_class\tCF_NPC_class
    1\tGNPS Compound A\tCC\tInChI=1/A\tAAAAAAAAAAAAAAA\t0.90\tLipids\tFatty acids\tFA esters\tWax esters
    4\tGNPS Compound X\tCCCC\tInChI=1/X\tXXXXXXXXXXXXXXX\t0.75\tCarbohydrates\tMonosaccharides\tHexoses\tHexoses
""")

_GNPS_SUSPECT_TSV = textwrap.dedent("""\
    #Scan#\tCompound_Name\tLibraryName
    5\tSuspectMolecule | SIRIUS: C5H6\tGNPS-SUSPECTLIST.mgf
""")


@pytest.fixture()
def initialized_project(tmp_path, feature_table_csv, metadata_csv):
    """Return an initialized MetabProject from the conftest fixture data."""
    project_path = tmp_path / "test_project"
    return MetabProject.init(
        path=project_path,
        feature_table=feature_table_csv,
        metadata=metadata_csv,
    )


@pytest.fixture()
def annotation_files(tmp_path):
    structs = tmp_path / "structure_identifications.tsv"
    canopus = tmp_path / "canopus_structure_summary.tsv"
    gnps_lib = tmp_path / "gnps_library.tsv"
    gnps_sus = tmp_path / "gnps_suspect.tsv"
    structs.write_text(_STRUCTURES_TSV)
    canopus.write_text(_CANOPUS_TSV)
    gnps_lib.write_text(_GNPS_LIBRARY_TSV)
    gnps_sus.write_text(_GNPS_SUSPECT_TSV)
    return structs, canopus, gnps_lib, gnps_sus


def test_annotate_writes_parquets(initialized_project, annotation_files):
    structs, canopus, gnps_lib, gnps_sus = annotation_files
    p = initialized_project
    p.annotate(
        sirius_structures=structs,
        canopus=canopus,
        gnps_library=gnps_lib,
        gnps_suspect=gnps_sus,
    )
    ann_dir = p.path / "annotations"
    assert (ann_dir / "sirius.parquet").exists()
    assert (ann_dir / "canopus.parquet").exists()
    assert (ann_dir / "gnps.parquet").exists()
    assert (ann_dir / "gnps_suspect.parquet").exists()
    assert (ann_dir / "merged.parquet").exists()


def test_annotate_manifest_entry(initialized_project, annotation_files):
    from metabproject.manifest import read_manifest

    structs, canopus, gnps_lib, gnps_sus = annotation_files
    initialized_project.annotate(
        sirius_structures=structs,
        canopus=canopus,
        gnps_library=gnps_lib,
        gnps_suspect=gnps_sus,
    )
    manifest = read_manifest(initialized_project.path)
    ops = manifest["operations"]
    annotate_ops = [o for o in ops if o["command"] == "annotate"]
    assert len(annotate_ops) == 1
    op = annotate_ops[0]
    assert op["parameters"]["sirius_min_confidence"] == 0.64
    assert op["parameters"]["canopus_min_probability"] == 0.7
    assert "n_features_annotated" in op["results"]
    assert "n_canopus_features_total" in op["results"]
    assert "n_inchikey_agree" in op["results"]


def test_annotate_merged_shape(initialized_project, annotation_files):
    structs, canopus, gnps_lib, _ = annotation_files
    initialized_project.annotate(
        sirius_structures=structs,
        canopus=canopus,
        gnps_library=gnps_lib,
    )
    merged = initialized_project.annotations
    assert merged.index.name == "feature_id"
    assert "annotation_source" in merged.columns
    assert "inchikey_agreement" in merged.columns
    assert "is_suspect_only" in merged.columns
    assert "is_canopus_only" in merged.columns
    assert "compound_name_gnps" in merged.columns
    assert "compound_name_sirius" in merged.columns
    assert "sirius_high_confidence" in merged.columns
    # Structural *_alt columns removed
    assert "compound_name_alt" not in merged.columns
    assert "inchikey2d_alt" not in merged.columns
    # Feature 2 has low-conf SIRIUS but CANOPUS passes → included as canopus source
    assert 2 in merged.index
    assert merged.loc[2, "annotation_source"] == "canopus"
    # Feature 6 is CANOPUS-only (no SIRIUS, no GNPS for it)
    assert 6 in merged.index
    assert merged.loc[6, "is_canopus_only"] == True


def test_annotate_lazy_properties(initialized_project, annotation_files):
    structs, canopus, gnps_lib, gnps_sus = annotation_files
    initialized_project.annotate(
        sirius_structures=structs,
        canopus=canopus,
        gnps_library=gnps_lib,
        gnps_suspect=gnps_sus,
    )
    assert initialized_project.sirius_annotations is not None
    assert initialized_project.canopus_annotations is not None
    assert initialized_project.gnps_annotations is not None
    assert initialized_project.gnps_suspect_annotations is not None
    assert initialized_project.annotations is not None


def test_annotate_suspect_none_returns_none(initialized_project, annotation_files):
    structs, canopus, gnps_lib, _ = annotation_files
    initialized_project.annotate(
        sirius_structures=structs,
        canopus=canopus,
        gnps_library=gnps_lib,
    )
    assert initialized_project.gnps_suspect_annotations is None
    assert not (initialized_project.path / "annotations" / "gnps_suspect.parquet").exists()


def test_annotate_force_overwrites(initialized_project, annotation_files):
    structs, canopus, gnps_lib, _ = annotation_files
    initialized_project.annotate(
        sirius_structures=structs,
        canopus=canopus,
        gnps_library=gnps_lib,
    )
    # Second run without --force should raise.
    with pytest.raises(FileExistsError):
        initialized_project.annotate(
            sirius_structures=structs,
            canopus=canopus,
            gnps_library=gnps_lib,
        )
    # With force should succeed and write a second manifest entry.
    from metabproject.manifest import read_manifest
    initialized_project.annotate(
        sirius_structures=structs,
        canopus=canopus,
        gnps_library=gnps_lib,
        force=True,
    )
    manifest = read_manifest(initialized_project.path)
    annotate_ops = [o for o in manifest["operations"] if o["command"] == "annotate"]
    assert len(annotate_ops) == 2


def test_annotate_not_found_raises(initialized_project, tmp_path):
    with pytest.raises(FileNotFoundError):
        initialized_project.annotate(
            sirius_structures=tmp_path / "missing.tsv",
            canopus=tmp_path / "also_missing.tsv",
            gnps_library=tmp_path / "and_missing.tsv",
        )
