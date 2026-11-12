"""Tests for explore/report.py — HTML generation."""

from __future__ import annotations

from pathlib import Path

import pytest

plotly = pytest.importorskip("plotly", reason="plotly not installed")
jinja2 = pytest.importorskip("jinja2", reason="jinja2 not installed")


@pytest.fixture
def minimal_data():
    return {
        "project_name": "test_project",
        "n_features": 2,
        "n_samples": 4,
        "groups": ["A", "B"],
        "group_members": {"A": ["s1", "s2"], "B": ["s3", "s4"]},
        "features": {
            "1": {
                "feature_id": 1, "mz": 200.0, "rt": 1.5, "charge": 1,
                "abundances": {"s1": 1.0, "s2": 2.0, "s3": 3.0, "s4": 4.0},
                "annotation": None, "sources": {}, "structure_svg": None,
                "contrasts": {}, "iin": None,
                "ms2": {
                    "precursor_mz": 200.0, "precursor_charge": 1, "n_peaks": 2,
                    "base_peak_mz": 150.0, "mz": [100.0, 150.0],
                    "intensity_rel": [0.5, 1.0],
                },
            }
        },
        "contrasts_meta": {},
        "iin_groups": {},
        "filter_options": {
            "annotation_sources": [], "npc_pathways": [], "npc_superclasses": [],
            "npc_classes": [], "classyfire_classes": [], "contrast_names": [],
        },
    }


def test_generate_html_creates_file(tmp_path, minimal_data):
    from metabproject.explore.report import generate_explore_html

    out = tmp_path / "viewer.html"
    result = generate_explore_html(minimal_data, out, embed_plotly=False)
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_generate_html_contains_doctype(tmp_path, minimal_data):
    from metabproject.explore.report import generate_explore_html

    out = tmp_path / "viewer.html"
    generate_explore_html(minimal_data, out, embed_plotly=False)
    content = out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content


def test_generate_html_embeds_data_json(tmp_path, minimal_data):
    from metabproject.explore.report import generate_explore_html

    out = tmp_path / "viewer.html"
    generate_explore_html(minimal_data, out, embed_plotly=False)
    content = out.read_text(encoding="utf-8")
    assert "const DATA = " in content
    assert "test_project" in content


def test_generate_html_cdn_uses_script_tag(tmp_path, minimal_data):
    from metabproject.explore.report import generate_explore_html

    out = tmp_path / "viewer.html"
    generate_explore_html(minimal_data, out, embed_plotly=False)
    content = out.read_text(encoding="utf-8")
    assert "cdn.plot.ly" in content


def test_generate_html_embed_plotly_no_cdn(tmp_path, minimal_data):
    from metabproject.explore.report import generate_explore_html

    out_cdn = tmp_path / "viewer_cdn.html"
    out_embed = tmp_path / "viewer_embed.html"
    generate_explore_html(minimal_data, out_cdn, embed_plotly=False)
    generate_explore_html(minimal_data, out_embed, embed_plotly=True)
    # Embedded version should be substantially larger (full plotly.js included)
    assert out_embed.stat().st_size > out_cdn.stat().st_size
    # Both should contain Plotly JS
    assert "Plotly" in out_embed.read_text(encoding="utf-8")


def test_generate_html_creates_parent_dirs(tmp_path, minimal_data):
    from metabproject.explore.report import generate_explore_html

    out = tmp_path / "deep" / "nested" / "viewer.html"
    generate_explore_html(minimal_data, out, embed_plotly=False)
    assert out.exists()
