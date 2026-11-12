"""Integration tests for MetabProject.import_network()."""

from __future__ import annotations

import textwrap

import pytest

from metabproject.project import MetabProject

# Feature ids overlap the conftest fixture (features 1–50). Component 1: 1-2-3;
# component 2: 4-5; one singleton (component -1) exercised via keep-singletons.
_PAIRS_TSV = textwrap.dedent("""\
    CLUSTERID1\tCLUSTERID2\tComponentIndex\tCosine\tDeltaMZ
    1\t2\t1\t0.90\t2.0
    2\t3\t1\t0.85\t14.0
    4\t5\t2\t0.95\t42.0
    6\t7\t-1\t0.80\t1.0
""")


@pytest.fixture()
def initialized_project(tmp_path, feature_table_csv, metadata_csv):
    return MetabProject.init(
        path=tmp_path / "test_project",
        feature_table=feature_table_csv,
        metadata=metadata_csv,
    )


@pytest.fixture()
def pairs_file(tmp_path):
    p = tmp_path / "filtered_pairs.tsv"
    p.write_text(_PAIRS_TSV)
    return p


def test_import_network_writes_parquets(initialized_project, pairs_file):
    p = initialized_project
    p.import_network(pairs=pairs_file)

    assert (p.path / "network" / "edges.parquet").exists()
    assert (p.path / "network" / "nodes.parquet").exists()

    edges = p.network_edges
    nodes = p.network_nodes
    # Singletons dropped by default → components 1 and 2 only, 3 edges.
    assert len(edges) == 3
    assert set(edges["component_id"]) == {1, 2}
    assert nodes.loc[1, "component_id"] == 1
    assert nodes.loc[1, "n_component_members"] == 3


def test_import_network_manifest_entry(initialized_project, pairs_file):
    p = initialized_project
    p.import_network(pairs=pairs_file)

    from metabproject.manifest import read_manifest

    ops = read_manifest(p.path)["operations"]
    entry = next(op for op in ops if op["command"] == "network")
    assert entry["results"]["n_edges"] == 3
    assert entry["results"]["n_components"] == 2
    assert entry["results"]["n_nodes"] == 5


def test_import_network_keep_singletons(initialized_project, pairs_file):
    p = initialized_project
    p.import_network(pairs=pairs_file, drop_singletons=False)
    assert (p.network_edges["component_id"] == -1).any()


def test_import_network_idempotent(initialized_project, pairs_file):
    p = initialized_project
    p.import_network(pairs=pairs_file)
    with pytest.raises(FileExistsError):
        p.import_network(pairs=pairs_file)
    # Force overwrites cleanly.
    p.import_network(pairs=pairs_file, force=True)
    assert len(p.network_edges) == 3


def test_no_network_returns_none(initialized_project):
    assert initialized_project.network_edges is None
    assert initialized_project.network_nodes is None
