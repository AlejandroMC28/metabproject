"""Integration tests for `metab explore`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

plotly = pytest.importorskip("plotly", reason="plotly not installed")
jinja2 = pytest.importorskip("jinja2", reason="jinja2 not installed")


@pytest.fixture
def processed_project(tmp_path, feature_table_csv, metadata_csv):
    """Project with init + process only (minimal)."""
    from metabproject.project import MetabProject

    p = MetabProject.init(
        path=tmp_path / "proj_explore",
        feature_table=feature_table_csv,
        metadata=metadata_csv,
    )
    p.process(force=False)
    return p


def test_explore_creates_html(processed_project):
    result = processed_project.explore(render_structures=False, embed_plotly=False)
    assert isinstance(result, Path)
    assert result.exists()
    assert result.suffix == ".html"


def test_explore_default_output_path(processed_project):
    result = processed_project.explore(render_structures=False, embed_plotly=False)
    expected = processed_project.path / "exports" / "explore.html"
    assert result == expected


def test_explore_custom_output_path(processed_project, tmp_path):
    out = tmp_path / "custom.html"
    result = processed_project.explore(
        output=out, render_structures=False, embed_plotly=False
    )
    assert result == out
    assert out.exists()


def test_explore_html_contains_feature_ids(processed_project):
    result = processed_project.explore(render_structures=False, embed_plotly=False)
    content = result.read_text(encoding="utf-8")
    # Feature IDs should appear in the embedded DATA JSON
    assert "feature_id" in content


def test_explore_html_contains_project_name(processed_project):
    result = processed_project.explore(render_structures=False, embed_plotly=False)
    content = result.read_text(encoding="utf-8")
    assert processed_project.path.name in content


def test_explore_writes_manifest_entry(processed_project):
    processed_project.explore(render_structures=False, embed_plotly=False)
    manifest = json.loads(
        (processed_project.path / "metabproject.json").read_text()
    )
    ops = manifest["operations"]
    explore_ops = [op for op in ops if op["command"] == "explore"]
    assert len(explore_ops) == 1
    op = explore_ops[0]
    assert "n_features" in op["results"]
    assert "file_size_mb" in op["results"]


def test_explore_file_size_reasonable(processed_project):
    """With CDN mode and no structures, the file should be small (< 5 MB)."""
    result = processed_project.explore(render_structures=False, embed_plotly=False)
    size_mb = result.stat().st_size / 1_048_576
    assert size_mb < 5.0


def test_explore_overwrite_is_allowed(processed_project):
    """Calling explore twice should overwrite silently (no --force needed)."""
    processed_project.explore(render_structures=False, embed_plotly=False)
    processed_project.explore(render_structures=False, embed_plotly=False)
    manifest = json.loads(
        (processed_project.path / "metabproject.json").read_text()
    )
    explore_ops = [op for op in manifest["operations"] if op["command"] == "explore"]
    assert len(explore_ops) == 2


def test_explore_includes_ms2_spectrum(processed_project, tmp_path):
    """After import_spectra, assembled data + HTML carry the MS2 spectrum."""
    pytest.importorskip("pyteomics")
    import textwrap

    from metabproject.explore.data import assemble_explore_data

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
    """))
    processed_project.import_spectra(mgf=mgf)

    data = assemble_explore_data(processed_project, render_structures=False)
    ms2 = data["features"]["1"]["ms2"]
    assert ms2 is not None
    assert len(ms2["mz"]) == ms2["n_peaks"]

    result = processed_project.explore(render_structures=False, embed_plotly=False)
    content = result.read_text(encoding="utf-8")
    assert "ms2-section" in content


def test_explore_requires_process(tmp_path, feature_table_csv, metadata_csv):
    """Calling explore before process raises FileNotFoundError."""
    from metabproject.project import MetabProject

    p = MetabProject.init(
        path=tmp_path / "proj_no_process",
        feature_table=feature_table_csv,
        metadata=metadata_csv,
    )
    with pytest.raises(FileNotFoundError):
        p.explore(render_structures=False, embed_plotly=False)
