"""Integration tests for MetabProject.compute_similarity()."""

from __future__ import annotations

import textwrap

import pytest

pytest.importorskip("matchms")
pytest.importorskip("pyteomics")

from metabproject.project import MetabProject

# Feature ids overlap the conftest fixture (features 1–50). Component 1 groups
# features 1-2-3, component 2 groups 4-5. Feature 1 and 2 share peaks; their
# third peaks differ by the precursor shift (216.09-200.10) so modified cosine
# finds an extra match direct cosine does not.
_PAIRS_TSV = textwrap.dedent("""\
    CLUSTERID1\tCLUSTERID2\tComponentIndex\tCosine\tDeltaMZ
    1\t2\t1\t0.90\t15.99
    2\t3\t1\t0.85\t14.0
    4\t5\t2\t0.95\t42.0
""")


def _block(fid, pepmass, peaks):
    lines = ["BEGIN IONS", f"FEATURE_ID={fid}", f"PEPMASS={pepmass}", "CHARGE=1+",
             "MSLEVEL=2", f"SCANS={fid}", f"Num peaks={len(peaks)}"]
    lines += [f"{m} {i}" for m, i in peaks]
    lines.append("END IONS")
    return "\n".join(lines)


_MGF = "\n".join([
    _block(1, 200.10, [(80.05, 100), (110.07, 60), (150.09, 40)]),
    _block(2, 216.09, [(80.05, 90), (110.07, 55), (166.08, 45)]),
    _block(3, 184.11, [(80.05, 80), (110.07, 70), (134.10, 30)]),
    _block(4, 300.00, [(90.05, 100), (120.07, 50)]),
    _block(5, 342.00, [(90.05, 95), (120.07, 55)]),
    "",
])


@pytest.fixture()
def scored_project(tmp_path, feature_table_csv, metadata_csv):
    """init -> process -> spectra -> network (no similarity yet)."""
    p = MetabProject.init(
        path=tmp_path / "test_project",
        feature_table=feature_table_csv,
        metadata=metadata_csv,
    )
    p.process()

    mgf = tmp_path / "spectra.mgf"
    mgf.write_text(_MGF)
    p.import_spectra(mgf=mgf)

    pairs = tmp_path / "filtered_pairs.tsv"
    pairs.write_text(_PAIRS_TSV)
    p.import_network(pairs=pairs)
    return p


def test_compute_similarity_writes_parquet(scored_project):
    p = scored_project
    p.compute_similarity()

    assert (p.path / "similarity" / "scores.parquet").exists()
    sim = p.spectral_similarity
    assert sim is not None
    # Within-family pairs: C(3,2) in component 1 + C(2,2) in component 2 = 4.
    n_pairs = sim[["feature_id_a", "feature_id_b"]].drop_duplicates().shape[0]
    assert n_pairs == 4
    # 2 score types x 3 scaling presets per pair.
    assert len(sim) == 4 * 2 * 3
    assert set(sim["score_type"]) == {"modified", "direct"}
    assert set(sim["scaling"]) == {"linear", "sqrt", "massbank"}


def test_compute_similarity_modified_beats_direct(scored_project):
    """The 1-2 pair gets an extra matched peak under modified cosine."""
    p = scored_project
    p.compute_similarity()
    sim = p.spectral_similarity
    pair = sim[(sim["feature_id_a"] == 1) & (sim["feature_id_b"] == 2)
               & (sim["scaling"] == "linear")]
    n_mod = pair[pair["score_type"] == "modified"]["n_matches"].iloc[0]
    n_dir = pair[pair["score_type"] == "direct"]["n_matches"].iloc[0]
    assert n_mod == 3
    assert n_dir == 2


def test_compute_similarity_manifest_entry(scored_project):
    p = scored_project
    p.compute_similarity()

    from metabproject.manifest import read_manifest

    ops = read_manifest(p.path)["operations"]
    entry = next(op for op in ops if op["command"] == "similarity")
    assert entry["results"]["n_pairs"] == 4
    assert entry["results"]["n_scored"] == 4
    assert entry["results"]["n_rows"] == 24
    assert entry["parameters"]["tolerance"] == 0.1


def test_compute_similarity_subset_score_type(scored_project):
    p = scored_project
    p.compute_similarity(score_types=("direct",), scalings=("linear",))
    sim = p.spectral_similarity
    assert set(sim["score_type"]) == {"direct"}
    assert set(sim["scaling"]) == {"linear"}
    assert len(sim) == 4  # one row per pair


def test_compute_similarity_idempotent(scored_project):
    p = scored_project
    p.compute_similarity()
    with pytest.raises(FileExistsError):
        p.compute_similarity()
    # Force recomputes cleanly.
    p.compute_similarity(force=True)
    assert p.spectral_similarity is not None


def test_compute_similarity_requires_spectra(tmp_path, feature_table_csv, metadata_csv):
    p = MetabProject.init(
        path=tmp_path / "no_spectra",
        feature_table=feature_table_csv,
        metadata=metadata_csv,
    )
    p.process()
    pairs = tmp_path / "filtered_pairs.tsv"
    pairs.write_text(_PAIRS_TSV)
    p.import_network(pairs=pairs)
    with pytest.raises(ValueError, match="spectra"):
        p.compute_similarity()


def test_compute_similarity_requires_network(tmp_path, feature_table_csv, metadata_csv):
    p = MetabProject.init(
        path=tmp_path / "no_network",
        feature_table=feature_table_csv,
        metadata=metadata_csv,
    )
    p.process()
    mgf = tmp_path / "spectra.mgf"
    mgf.write_text(_MGF)
    p.import_spectra(mgf=mgf)
    with pytest.raises(ValueError, match="network"):
        p.compute_similarity()


def test_no_similarity_returns_none(tmp_path, feature_table_csv, metadata_csv):
    p = MetabProject.init(
        path=tmp_path / "fresh",
        feature_table=feature_table_csv,
        metadata=metadata_csv,
    )
    assert p.spectral_similarity is None
