"""Integration tests: init → process → annotate → stats → enrich end-to-end."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from metabproject.project import MetabProject

# ---------------------------------------------------------------------------
# Synthetic annotation files (same as test_annotate.py but extended to 5+
# features so enrichment has enough class members to actually run).
# ---------------------------------------------------------------------------

_STRUCTURES_TSV = textwrap.dedent("""\
    mappingFeatureId\tstructurePerIdRank\tConfidenceScoreApproximate\tInChIkey2D\tInChI\tname\tsmiles\tmolecularFormula\tadduct
    1\t1\t0.80\tAAAAAAAAAAAAAAA\tInChI=1/A\tCompound A\tCC\tC2H6\t[M+H]+
    2\t1\t0.40\tCCCCCCCCCCCCCCC\tInChI=1/C\tCompound C\tO\tH2O\t[M+Na]+
    3\t1\t0.70\tDDDDDDDDDDDDDDD\tInChI=1/D\tCompound D\tCC=O\tC2H4O\t[M+H]+
""")

# Six features in CANOPUS — three in Lipids, three in Alkaloids — so at least
# one class passes the default min_class_n=3.
_CANOPUS_TSV = textwrap.dedent("""\
    mappingFeatureId\tNPC#pathway\tNPC#pathway Probability\tNPC#superclass\tNPC#superclass Probability\tNPC#class\tNPC#class Probability\tClassyFire#most specific class\tClassyFire#most specific class Probability
    1\tLipids\t0.90\tFatty acids\t0.85\tFA esters\t0.80\tWax esters\t0.75
    2\tLipids\t0.85\tFatty acids\t0.82\tFA esters\t0.78\tWax esters\t0.72
    3\tLipids\t0.88\tFatty acids\t0.83\tFA esters\t0.79\tWax esters\t0.73
    4\tAlkaloids\t0.80\tPyridine alkaloids\t0.75\tPiperidine\t0.72\tPiperidine alkaloids\t0.70
    5\tAlkaloids\t0.82\tPyridine alkaloids\t0.77\tPiperidine\t0.74\tPiperidine alkaloids\t0.71
    6\tAlkaloids\t0.84\tPyridine alkaloids\t0.79\tPiperidine\t0.76\tPiperidine alkaloids\t0.73
""")

_GNPS_LIBRARY_TSV = textwrap.dedent("""\
    #Scan#\tCompound_Name\tSmiles\tINSCHI\tInChIKey-Planar\tMQScore\tnpclassifier_pathway\tnpclassifier_superclass\tnpclassifier_class\tCF_NPC_class
    1\tGNPS Compound A\tCC\tInChI=1/A\tAAAAAAAAAAAAAAA\t0.90\tLipids\tFatty acids\tFA esters\tWax esters
""")


@pytest.fixture()
def annotation_files(tmp_path: Path):
    structs = tmp_path / "structure_identifications.tsv"
    canopus = tmp_path / "canopus_structure_summary.tsv"
    gnps_lib = tmp_path / "gnps_library.tsv"
    structs.write_text(_STRUCTURES_TSV)
    canopus.write_text(_CANOPUS_TSV)
    gnps_lib.write_text(_GNPS_LIBRARY_TSV)
    return structs, canopus, gnps_lib


@pytest.fixture()
def stats_project(tmp_path: Path, feature_table_csv: Path, metadata_csv: Path,
                  annotation_files) -> MetabProject:
    """Init + process + annotate + stats project ready for enrichment."""
    structs, canopus, gnps_lib = annotation_files
    project_path = tmp_path / "demo"
    p = MetabProject.init(path=project_path, feature_table=feature_table_csv,
                           metadata=metadata_csv)
    p.process(transform="log2")
    p.annotate(sirius_structures=structs, canopus=canopus, gnps_library=gnps_lib)
    contrasts_path = tmp_path / "contrasts.yaml"
    contrasts_path.write_text(
        "contrasts:\n  - name: cvf\n    a: { group: control }\n    b: { group: fermented }\n"
    )
    p.stats(contrasts=contrasts_path, min_samples=3)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_enrich_creates_parquet(stats_project: MetabProject) -> None:
    p = stats_project
    p.enrich(min_class_n=3)
    assert (p.path / "enrichment" / "cvf.parquet").exists()


def test_enrich_output_columns(stats_project: MetabProject) -> None:
    p = stats_project
    p.enrich(min_class_n=3)
    df = p.enrichment("cvf")
    expected = {"level", "class_name", "n", "mean_log2fc", "p_value", "p_adj", "significant"}
    assert expected == set(df.columns)


def test_enrich_levels_filter(stats_project: MetabProject) -> None:
    p = stats_project
    p.enrich(levels=["npc_class"], min_class_n=3)
    df = p.enrichment("cvf")
    assert set(df["level"].unique()) <= {"npc_class"}


def test_enrich_contrast_filter(tmp_path: Path, feature_table_csv: Path,
                                 metadata_csv: Path, annotation_files) -> None:
    structs, canopus, gnps_lib = annotation_files
    project_path = tmp_path / "demo2"
    p = MetabProject.init(path=project_path, feature_table=feature_table_csv,
                           metadata=metadata_csv)
    p.process(transform="log2")
    p.annotate(sirius_structures=structs, canopus=canopus, gnps_library=gnps_lib)
    cf = tmp_path / "contrasts.yaml"
    cf.write_text(
        "contrasts:\n"
        "  - name: c1\n    a: { group: control }\n    b: { group: fermented }\n"
        "  - name: c2\n    a: { group: control }\n    b: { group: fermented }\n"
    )
    p.stats(contrasts=cf, min_samples=3)
    # Only enrich c1.
    p.enrich(contrasts=["c1"], min_class_n=3)
    assert (p.path / "enrichment" / "c1.parquet").exists()
    assert not (p.path / "enrichment" / "c2.parquet").exists()


def test_enrich_idempotent_raises_without_force(stats_project: MetabProject) -> None:
    p = stats_project
    p.enrich(min_class_n=3)
    with pytest.raises(FileExistsError):
        p.enrich(min_class_n=3)


def test_enrich_force_succeeds(stats_project: MetabProject) -> None:
    p = stats_project
    p.enrich(min_class_n=3)
    p.enrich(min_class_n=3, force=True)  # should not raise


def test_enrich_manifest_entry(stats_project: MetabProject) -> None:
    p = stats_project
    p.enrich(min_class_n=3)
    manifest = json.loads((p.path / "metabproject.json").read_text())
    last = manifest["operations"][-1]
    assert last["command"] == "enrich"
    assert last["parameters"]["test"] == "mannwhitneyu"
    assert "cvf" in last["results"]
    assert {"n_total_tested", "n_significant", "per_level"} <= set(last["results"]["cvf"])


def test_enrich_requires_stats(tmp_path: Path, feature_table_csv: Path,
                                metadata_csv: Path, annotation_files) -> None:
    structs, canopus, gnps_lib = annotation_files
    project_path = tmp_path / "demo3"
    p = MetabProject.init(path=project_path, feature_table=feature_table_csv,
                           metadata=metadata_csv)
    p.process(transform="log2")
    p.annotate(sirius_structures=structs, canopus=canopus, gnps_library=gnps_lib)
    with pytest.raises(FileNotFoundError):
        p.enrich()


def test_enrich_requires_annotate(tmp_path: Path, feature_table_csv: Path,
                                   metadata_csv: Path) -> None:
    project_path = tmp_path / "demo4"
    p = MetabProject.init(path=project_path, feature_table=feature_table_csv,
                           metadata=metadata_csv)
    p.process(transform="log2")
    with pytest.raises(FileNotFoundError):
        p.enrich()


def test_enrichment_results_property(stats_project: MetabProject) -> None:
    p = stats_project
    assert p.enrichment_results == {}
    p.enrich(min_class_n=3)
    p._enrichment_results = None  # bust cache
    results = p.enrichment_results
    assert "cvf" in results
    assert isinstance(results["cvf"], pd.DataFrame)


def test_enrichment_accessor_missing_raises(stats_project: MetabProject) -> None:
    p = stats_project
    with pytest.raises(FileNotFoundError):
        p.enrichment("nonexistent")
