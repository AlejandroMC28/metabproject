"""Integration tests for MetabProject.import_spectra()."""

from __future__ import annotations

import textwrap

import pytest

pytest.importorskip("pyteomics")

from metabproject.project import MetabProject

# Feature ids overlap the conftest fixture (features 1–50). Feature 99 has no
# match in the table and must be dropped + counted.
_MGF = textwrap.dedent("""\
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
    BEGIN IONS
    FEATURE_ID=99
    PEPMASS=410.0
    CHARGE=2+
    MSLEVEL=2
    SCANS=99
    Num peaks=2
    120.0 10
    130.0 20
    END IONS
""")


@pytest.fixture()
def initialized_project(tmp_path, feature_table_csv, metadata_csv):
    return MetabProject.init(
        path=tmp_path / "test_project",
        feature_table=feature_table_csv,
        metadata=metadata_csv,
    )


@pytest.fixture()
def mgf_file(tmp_path):
    p = tmp_path / "spectra.mgf"
    p.write_text(_MGF)
    return p


def test_import_spectra_writes_parquet(initialized_project, mgf_file):
    p = initialized_project
    p.import_spectra(mgf=mgf_file)

    assert (p.path / "spectra" / "ms2_spectra.parquet").exists()
    spectra = p.ms2_spectra
    # Feature 99 is absent from the table → dropped. 1 and 2 remain.
    assert set(spectra.index) == {1, 2}
    assert spectra.loc[1, "n_peaks"] == 3
    assert spectra.loc[1, "mz"].tolist() == [100.05, 150.10, 200.20]


def test_import_spectra_manifest_entry(initialized_project, mgf_file):
    p = initialized_project
    p.import_spectra(mgf=mgf_file)

    from metabproject.manifest import read_manifest

    ops = read_manifest(p.path)["operations"]
    entry = next(op for op in ops if op["command"] == "spectra")
    assert entry["results"]["n_spectra"] == 2
    assert entry["results"]["n_features_matched"] == 2
    assert entry["results"]["n_features_unmatched"] == 1
    assert entry["results"]["total_peaks"] == 4


def test_import_spectra_idempotent(initialized_project, mgf_file):
    p = initialized_project
    p.import_spectra(mgf=mgf_file)
    with pytest.raises(FileExistsError):
        p.import_spectra(mgf=mgf_file)
    # Force overwrites cleanly.
    p.import_spectra(mgf=mgf_file, force=True)
    assert set(p.ms2_spectra.index) == {1, 2}


def test_no_spectra_returns_none(initialized_project):
    assert initialized_project.ms2_spectra is None
