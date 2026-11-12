"""Tests for metabproject.network.gnps_pairs."""

import textwrap

import pytest

from metabproject.network.gnps_pairs import build_node_membership, load_gnps_pairs

# ---------------------------------------------------------------------------
# Synthetic fixture data
# ---------------------------------------------------------------------------

# Component 1: 10-20-30 chain. Component 2: 40-50. A self-loop, a singleton
# (component -1), a duplicated undirected pair (20-10 with a lower cosine),
# and a low-cosine edge to exercise the filters.
_PAIRS_TSV = textwrap.dedent("""\
    CLUSTERID1\tCLUSTERID2\tComponentIndex\tCosine\tDeltaMZ
    10\t20\t1\t0.90\t2.005
    20\t30\t1\t0.85\t14.02
    40\t50\t2\t0.95\t42.01
    20\t10\t1\t0.70\t-2.005
    60\t60\t3\t0.99\t0.0
    70\t80\t-1\t0.88\t1.0
    30\t10\t1\t0.40\t16.03
""")


@pytest.fixture
def pairs_file(tmp_path):
    p = tmp_path / "filtered_pairs.tsv"
    p.write_text(_PAIRS_TSV)
    return p


def test_column_rename_and_dtypes(pairs_file):
    edges = load_gnps_pairs(pairs_file)
    assert list(edges.columns) == [
        "feature_id_1",
        "feature_id_2",
        "component_id",
        "cosine",
        "delta_mz",
    ]
    assert edges["feature_id_1"].dtype == int
    assert edges["feature_id_2"].dtype == int
    assert edges["component_id"].dtype == int


def test_self_loops_dropped(pairs_file):
    edges = load_gnps_pairs(pairs_file)
    assert not (edges["feature_id_1"] == edges["feature_id_2"]).any()


def test_singletons_dropped_by_default(pairs_file):
    edges = load_gnps_pairs(pairs_file)
    assert (edges["component_id"] != -1).all()


def test_keep_singletons(pairs_file):
    edges = load_gnps_pairs(pairs_file, drop_singletons=False)
    assert (edges["component_id"] == -1).any()


def test_undirected_dedup_keeps_max_cosine(pairs_file):
    edges = load_gnps_pairs(pairs_file)
    # 10-20 appears twice (0.90 and 0.70); one row, keeping 0.90.
    pair = edges[(edges["feature_id_1"] == 10) & (edges["feature_id_2"] == 20)]
    assert len(pair) == 1
    assert pair["cosine"].iloc[0] == pytest.approx(0.90)


def test_pairs_ordered_low_high(pairs_file):
    edges = load_gnps_pairs(pairs_file)
    assert (edges["feature_id_1"] < edges["feature_id_2"]).all()


def test_min_cosine_filter(pairs_file):
    edges = load_gnps_pairs(pairs_file, min_cosine=0.80)
    assert (edges["cosine"] >= 0.80).all()
    # 30-10 (0.40) dropped.
    assert edges[(edges["feature_id_1"] == 10) & (edges["feature_id_2"] == 30)].empty


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_gnps_pairs(tmp_path / "nope.tsv")


def test_missing_columns_raises(tmp_path):
    p = tmp_path / "bad.tsv"
    p.write_text("CLUSTERID1\tCLUSTERID2\n10\t20\n")
    with pytest.raises(ValueError, match="Missing required columns"):
        load_gnps_pairs(p)


def test_node_membership(pairs_file):
    edges = load_gnps_pairs(pairs_file)
    nodes = build_node_membership(edges)
    # Node 10 is in component 1 with neighbours 20 and 30 → degree 2.
    assert nodes.loc[10, "component_id"] == 1
    assert nodes.loc[10, "degree"] == 2
    # Component 1 has nodes {10, 20, 30}.
    assert nodes.loc[10, "n_component_members"] == 3
    # Component 2 has nodes {40, 50}.
    assert nodes.loc[40, "n_component_members"] == 2
    assert nodes.index.name == "feature_id"
