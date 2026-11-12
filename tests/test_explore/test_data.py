"""Tests for explore/data.py — data assembly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def processed_project(tmp_path, feature_table_csv, metadata_csv):
    """A project with init + process run (minimal viable state)."""
    from metabproject.project import MetabProject

    p = MetabProject.init(
        path=tmp_path / "proj",
        feature_table=feature_table_csv,
        metadata=metadata_csv,
    )
    p.process(force=False)
    return p


@pytest.fixture
def networked_project(processed_project, tmp_path):
    """A processed project with a GNPS molecular network ingested."""
    import textwrap

    pairs = tmp_path / "filtered_pairs.tsv"
    pairs.write_text(textwrap.dedent("""\
        CLUSTERID1\tCLUSTERID2\tComponentIndex\tCosine\tDeltaMZ
        1\t2\t1\t0.90\t2.0
        2\t3\t1\t0.85\t14.0
        4\t5\t2\t0.95\t42.0
    """))
    processed_project.import_network(pairs=pairs)
    return processed_project


@pytest.fixture
def spectra_project(processed_project, tmp_path):
    """A processed project with MS2 spectra ingested for features 1 and 2."""
    pytest.importorskip("pyteomics")
    import textwrap

    mgf = tmp_path / "spectra.mgf"
    mgf.write_text(textwrap.dedent("""\
        BEGIN IONS
        FEATURE_ID=1
        PEPMASS=345.1234
        CHARGE=1+
        MSLEVEL=2
        SCANS=1
        Num peaks=3
        100.05 500
        150.10 1000
        200.20 250
        END IONS
        BEGIN IONS
        FEATURE_ID=2
        PEPMASS=200.9876
        MSLEVEL=2
        SCANS=2
        Num peaks=1
        99.0 42
        END IONS
    """))
    processed_project.import_spectra(mgf=mgf)
    return processed_project


@pytest.fixture
def similarity_project(processed_project, tmp_path):
    """A processed project with network + spectra + matchms similarity computed."""
    pytest.importorskip("pyteomics")
    pytest.importorskip("matchms")
    import textwrap

    pairs = tmp_path / "filtered_pairs.tsv"
    pairs.write_text(textwrap.dedent("""\
        CLUSTERID1\tCLUSTERID2\tComponentIndex\tCosine\tDeltaMZ
        1\t2\t1\t0.90\t15.99
        2\t3\t1\t0.85\t14.0
    """))
    processed_project.import_network(pairs=pairs)

    mgf = tmp_path / "spectra.mgf"
    mgf.write_text(textwrap.dedent("""\
        BEGIN IONS
        FEATURE_ID=1
        PEPMASS=200.10
        CHARGE=1+
        MSLEVEL=2
        SCANS=1
        Num peaks=3
        80.05 100
        110.07 60
        150.09 40
        END IONS
        BEGIN IONS
        FEATURE_ID=2
        PEPMASS=216.09
        CHARGE=1+
        MSLEVEL=2
        SCANS=2
        Num peaks=3
        80.05 90
        110.07 55
        166.08 45
        END IONS
        BEGIN IONS
        FEATURE_ID=3
        PEPMASS=184.11
        CHARGE=1+
        MSLEVEL=2
        SCANS=3
        Num peaks=3
        80.05 80
        110.07 70
        134.10 30
        END IONS
    """))
    processed_project.import_spectra(mgf=mgf)
    processed_project.compute_similarity()
    return processed_project


def test_assemble_returns_required_keys(processed_project):
    from metabproject.explore.data import assemble_explore_data

    data = assemble_explore_data(processed_project, render_structures=False)
    for key in ("project_name", "n_features", "n_samples", "groups",
                "group_members", "features", "contrasts_meta", "iin_groups",
                "network_components", "filter_options"):
        assert key in data, f"Missing key: {key}"


def test_assemble_feature_count_matches(processed_project):
    from metabproject.explore.data import assemble_explore_data

    data = assemble_explore_data(processed_project, render_structures=False)
    ft = processed_project.feature_table
    assert data["n_features"] == len(ft)
    assert len(data["features"]) == len(ft)


def test_assemble_abundances_match_feature_table(processed_project):
    from metabproject.explore.data import assemble_explore_data

    data = assemble_explore_data(processed_project, render_structures=False)
    ft = processed_project.feature_table
    first_fid = str(ft.index[0])
    first_sample = ft.columns[0]
    expected = ft.iloc[0, 0]
    actual = data["features"][first_fid]["abundances"][first_sample]
    if expected is None or (isinstance(expected, float) and np.isnan(expected)):
        assert actual is None
    else:
        assert abs(actual - float(expected)) < 1e-6


def test_assemble_no_annotation_without_annotate(processed_project):
    from metabproject.explore.data import assemble_explore_data

    data = assemble_explore_data(processed_project, render_structures=False)
    # All features should have null annotation when annotate hasn't run.
    annotations = [f["annotation"] for f in data["features"].values()]
    assert all(a is None for a in annotations)


def test_assemble_no_contrasts_without_stats(processed_project):
    from metabproject.explore.data import assemble_explore_data

    data = assemble_explore_data(processed_project, render_structures=False)
    assert data["contrasts_meta"] == {}
    for fd in data["features"].values():
        assert fd["contrasts"] == {}


def test_assemble_groups_match_metadata(processed_project):
    from metabproject.explore.data import assemble_explore_data

    data = assemble_explore_data(processed_project, render_structures=False)
    meta = processed_project.metadata
    expected_groups = sorted(meta["group"].unique().tolist())
    assert data["groups"] == expected_groups


def test_assemble_filter_options_populated(processed_project):
    from metabproject.explore.data import assemble_explore_data

    data = assemble_explore_data(processed_project, render_structures=False)
    opts = data["filter_options"]
    assert "annotation_sources" in opts
    assert "npc_pathways" in opts
    assert "contrast_names" in opts


def test_assemble_no_network_empty(processed_project):
    from metabproject.explore.data import assemble_explore_data

    data = assemble_explore_data(processed_project, render_structures=False)
    assert data["network_components"] == {}
    for fd in data["features"].values():
        assert fd["network"] is None


def test_assemble_network_components(networked_project):
    from metabproject.explore.data import assemble_explore_data

    data = assemble_explore_data(networked_project, render_structures=False)
    comps = data["network_components"]
    # Components 1 (features 1,2,3) and 2 (features 4,5).
    assert set(comps) == {"1", "2"}
    comp1 = comps["1"]
    assert comp1["component_id"] == 1
    assert comp1["feature_ids"] == [1, 2, 3]
    assert comp1["n_features"] == 3
    assert comp1["n_edges"] == 2
    # Compact [id1, id2, cosine] edge triples.
    assert [1, 2, pytest.approx(0.90)] in [
        [e[0], e[1], pytest.approx(e[2])] for e in comp1["edges"]
    ]
    assert {m["feature_id"] for m in comp1["members"]} == {1, 2, 3}


def test_assemble_per_feature_network_membership(networked_project):
    from metabproject.explore.data import assemble_explore_data

    data = assemble_explore_data(networked_project, render_structures=False)
    net = data["features"]["2"]["network"]
    assert net["component_id"] == 1
    assert net["degree"] == 2  # node 2 links to both 1 and 3
    assert net["n_component_members"] == 3
    # A feature outside any component has null membership.
    assert data["features"]["10"]["network"] is None


def test_assemble_no_ms2_null(processed_project):
    from metabproject.explore.data import assemble_explore_data

    data = assemble_explore_data(processed_project, render_structures=False)
    for fd in data["features"].values():
        assert fd["ms2"] is None


def test_assemble_per_feature_ms2(spectra_project):
    from metabproject.explore.data import assemble_explore_data

    data = assemble_explore_data(spectra_project, render_structures=False)
    ms2 = data["features"]["1"]["ms2"]
    assert ms2 is not None
    assert ms2["precursor_mz"] == pytest.approx(345.1234)
    assert ms2["precursor_charge"] == 1
    assert max(ms2["intensity_rel"]) == pytest.approx(1.0)
    assert len(ms2["mz"]) == ms2["n_peaks"]
    # A feature with no stored spectrum has null ms2.
    assert data["features"]["10"]["ms2"] is None


def test_assemble_no_similarity_meta_without_compute(networked_project):
    from metabproject.explore.data import assemble_explore_data

    data = assemble_explore_data(networked_project, render_structures=False)
    assert data["similarity_meta"] is None
    # Components carry an (empty) similarity block when nothing was computed.
    assert data["network_components"]["1"]["similarity"] == {}


def test_assemble_similarity_meta(similarity_project):
    from metabproject.explore.data import assemble_explore_data

    data = assemble_explore_data(similarity_project, render_structures=False)
    meta = data["similarity_meta"]
    assert meta is not None
    assert set(meta["score_types"]) == {"modified", "direct"}
    # Scalings ordered by the canonical preset order.
    assert meta["scalings"] == ["linear", "sqrt", "massbank"]
    assert meta["tolerance"] == 0.1


def test_assemble_similarity_attached_to_component(similarity_project):
    from metabproject.explore.data import assemble_explore_data

    data = assemble_explore_data(similarity_project, render_structures=False)
    comp = data["network_components"]["1"]
    sim = comp["similarity"]
    # Every within-family member pair (1,2,3 -> 3 pairs) is present, low id first.
    assert set(sim) == {"1|2", "1|3", "2|3"}
    rec = sim["1|2"]["modified"]["linear"]
    assert 0.0 <= rec["score"] <= 1.0
    # Matched peaks are [top_idx, bottom_idx] pairs into the displayed spectra.
    assert all(len(m) == 2 for m in rec["matches"])
    # Modified cosine recovers the precursor-shifted third peak (direct does not).
    n_mod = len(sim["1|2"]["modified"]["linear"]["matches"])
    n_dir = len(sim["1|2"]["direct"]["linear"]["matches"])
    assert n_mod == 3
    assert n_dir == 2


def test_feature_relations_links_parent_and_derived(processed_project, tmp_path):
    from metabproject.explore.data import assemble_explore_data

    rel = tmp_path / "feature_relations.tsv"
    rel.write_text(
        "feature_id\tparent\trole\trelation\tr\tms2_support\tisf_confident\n"
        "2\t1\tloss\t-hexose\t0.97\tTrue\tTrue\n"
        "3\t1\tadduct\t[M+Na]+/[M+H]+\t0.92\t\tFalse\n"
        "4\t999\tisotope\t13C\t0.95\t\tFalse\n"
    )
    data = assemble_explore_data(processed_project, render_structures=False, feature_relations=rel)
    f = data["features"]
    assert f["2"]["derived_from"] == {
        "role": "loss", "relation": "-hexose", "r": 0.97, "ms2_support": True,
        "isf_confident": True, "feature_id": 1, "in_table": True,
    }
    assert [d["feature_id"] for d in f["1"]["derived_ions"]] == [2, 3]
    assert f["1"]["derived_from"] is None
    assert f["4"]["derived_from"]["in_table"] is False  # parent not in the feature table
    assert f["5"]["derived_from"] is None and f["5"]["derived_ions"] == []


def test_feature_relations_absent_by_default(processed_project):
    from metabproject.explore.data import assemble_explore_data

    data = assemble_explore_data(processed_project, render_structures=False)
    assert all(f["derived_from"] is None and f["derived_ions"] == [] for f in data["features"].values())
