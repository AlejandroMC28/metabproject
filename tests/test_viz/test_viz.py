"""Tests for the viz module (PCA, volcano, heatmap)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metabproject.project import MetabProject
from metabproject.viz import (
    heatmap_figure,
    pca_figure,
    select_high_confidence_structure_features,
    volcano_figure,
)


def _init_and_process(tmp_path: Path, feature_table_csv, metadata_csv) -> MetabProject:
    p = MetabProject.init(
        path=tmp_path / "proj",
        feature_table=feature_table_csv,
        metadata=metadata_csv,
    )
    p.process(min_prevalence=1, impute="min", transform="log2")
    return p


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


def test_pca_figure_returns_figure(tmp_path, feature_table_csv, metadata_csv):
    p = _init_and_process(tmp_path, feature_table_csv, metadata_csv)
    fig, info = pca_figure(p.feature_table, p.metadata)
    assert len(fig.axes) == 1
    assert len(info["variance_explained"]) >= 2


def test_pca_3d_has_3d_projection(tmp_path, feature_table_csv, metadata_csv):
    p = _init_and_process(tmp_path, feature_table_csv, metadata_csv)
    fig, info = pca_figure(p.feature_table, p.metadata, three_d=True)
    assert fig.axes[0].name == "3d"
    assert len(info["variance_explained"]) >= 3


def test_pca_figure_scale_matches_run_pca(tmp_path, feature_table_csv, metadata_csv):
    from metabproject.stats.multivariate import run_pca

    p = _init_and_process(tmp_path, feature_table_csv, metadata_csv)
    for scale in (True, False):
        _, info = pca_figure(p.feature_table, p.metadata, scale=scale)
        expected = run_pca(p.feature_table, n_components=2, scale=scale).explained_variance_ratio[:2]
        assert np.allclose(info["variance_explained"][:2], expected)


def test_viz_pca_writes_png_svg_and_manifest(tmp_path, feature_table_csv, metadata_csv):
    p = _init_and_process(tmp_path, feature_table_csv, metadata_csv)
    p.viz_pca()
    assert (p.path / "viz" / "pca.png").exists()
    assert (p.path / "viz" / "pca.svg").exists()
    manifest = json.loads((p.path / "metabproject.json").read_text())
    cmds = [op["command"] for op in manifest["operations"]]
    assert "viz_pca" in cmds


# ---------------------------------------------------------------------------
# Volcano
# ---------------------------------------------------------------------------


def _make_synthetic_contrast(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "log2fc": rng.normal(0, 1.5, n),
            "p_adj": rng.uniform(1e-6, 1.0, n),
        },
        index=[f"f{i}" for i in range(n)],
    )


def test_volcano_returns_figure_and_counts():
    df = _make_synthetic_contrast()
    fig, counts = volcano_figure(df, fc_threshold=1.0, q_threshold=0.05)
    assert set(counts.keys()) == {"up", "down", "ns"}
    assert sum(counts.values()) == len(df)
    assert len(fig.axes) == 1


def test_volcano_clips_zero_qadj():
    df = _make_synthetic_contrast()
    df.loc["f0", "p_adj"] = 0.0
    fig, _ = volcano_figure(df)
    # Should not raise / not produce inf — just check the plot ax has data.
    assert fig.axes[0].collections


def test_volcano_threshold_changes_counts():
    df = _make_synthetic_contrast()
    _, loose = volcano_figure(df, fc_threshold=0.1, q_threshold=0.5)
    _, strict = volcano_figure(df, fc_threshold=2.0, q_threshold=0.001)
    assert loose["up"] + loose["down"] >= strict["up"] + strict["down"]


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------


def test_heatmap_full(tmp_path, feature_table_csv, metadata_csv):
    p = _init_and_process(tmp_path, feature_table_csv, metadata_csv)
    fig, info = heatmap_figure(p.feature_table, p.metadata)
    assert info["n_samples"] == 6
    assert info["n_features"] > 0
    assert fig is not None


def test_heatmap_subset(tmp_path, feature_table_csv, metadata_csv):
    p = _init_and_process(tmp_path, feature_table_csv, metadata_csv)
    pick = p.feature_table.index[:5].tolist()
    fig, info = heatmap_figure(p.feature_table, p.metadata, features=pick)
    assert info["n_features"] <= 5  # zero-variance rows may be dropped


def test_viz_heatmap_features_from_contrast_path(tmp_path, feature_table_csv, metadata_csv, monkeypatch):
    p = _init_and_process(tmp_path, feature_table_csv, metadata_csv)
    # Synthesise a stats contrast parquet so viz_heatmap can use it.
    stats_dir = p.path / "stats"
    stats_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(
        {
            "log2fc": np.linspace(-2, 2, len(p.feature_table)),
            "p_adj": np.linspace(0.001, 0.5, len(p.feature_table)),
            "direction": ["up"] * len(p.feature_table),
            "significant": [True] * len(p.feature_table),
        },
        index=p.feature_table.index,
    )
    df.to_parquet(stats_dir / "synthetic.parquet")
    p.viz_heatmap(features_from_contrast="synthetic", q_threshold=0.1)
    assert (p.path / "viz" / "heatmap.png").exists()
    assert (p.path / "viz" / "heatmap.svg").exists()


def _make_annotations(feature_ids: list) -> pd.DataFrame:
    """Mixed merged-annotation table for filter tests.

    Row 0 → GNPS hit (kept)
    Row 1 → SIRIUS high-confidence (kept)
    Row 2 → SIRIUS low-confidence (dropped)
    Row 3 → suspect-only (dropped)
    Row 4 → CANOPUS-only (dropped)
    """
    fids = list(feature_ids[:5])
    return pd.DataFrame(
        {
            "annotation_source": ["gnps", "sirius", "sirius", "suspect", "canopus"],
            "sirius_high_confidence": [False, True, False, False, False],
            "compound_name": ["AcidA", "AlcoholB", "EsterC", "SuspectD", None],
        },
        index=pd.Index(fids, name="feature_id"),
    )


def test_select_high_confidence_picks_gnps_and_sirius_high():
    fids = [10, 11, 12, 13, 14]
    ann = _make_annotations(fids)
    kept, labels = select_high_confidence_structure_features(ann)
    assert kept == [10, 11]
    assert labels == {10: "AcidA", 11: "AlcoholB"}


def test_select_high_confidence_intersects_subset():
    fids = [10, 11, 12, 13, 14]
    ann = _make_annotations(fids)
    kept, labels = select_high_confidence_structure_features(ann, feature_ids=[11, 12, 99])
    assert kept == [11]
    assert labels == {11: "AlcoholB"}


def test_viz_heatmap_annotated_writes_files_and_manifest(
    tmp_path, feature_table_csv, metadata_csv
):
    p = _init_and_process(tmp_path, feature_table_csv, metadata_csv)
    ann_dir = p.path / "annotations"
    ann_dir.mkdir(exist_ok=True)
    fids = p.feature_table.index.tolist()
    ann = _make_annotations(fids)
    ann.to_parquet(ann_dir / "merged.parquet")

    p.viz_heatmap_annotated()

    assert (p.path / "viz" / "heatmap_annotated.png").exists()
    assert (p.path / "viz" / "heatmap_annotated.svg").exists()
    manifest = json.loads((p.path / "metabproject.json").read_text())
    cmds = [op["command"] for op in manifest["operations"]]
    assert "viz_heatmap_annotated" in cmds


def test_viz_heatmap_annotated_errors_without_annotations(
    tmp_path, feature_table_csv, metadata_csv
):
    p = _init_and_process(tmp_path, feature_table_csv, metadata_csv)
    with pytest.raises(FileNotFoundError):
        p.viz_heatmap_annotated()


def test_pca_markers_override_default_cycle(
    tmp_path, feature_table_csv, metadata_csv
):
    p = _init_and_process(tmp_path, feature_table_csv, metadata_csv)
    fig, _ = pca_figure(
        p.feature_table, p.metadata,
        marker_col="variety",
        markers={"varA": "X", "varB": "P"},
    )
    # Second legend is the marker legend; its handle markers should reflect overrides.
    marker_legend = fig.axes[0].get_legend()
    assert marker_legend is not None
    used = {h.get_marker() for h in marker_legend.legend_handles}
    assert "X" in used
    assert "P" in used


def test_load_markers_yaml_reads_markers_block(tmp_path):
    from metabproject.viz._style import load_markers_yaml

    (tmp_path / "viz.yaml").write_text(
        "markers:\n  control: o\n  malted: ^\n"
    )
    assert load_markers_yaml(tmp_path) == {"control": "o", "malted": "^"}


def test_load_markers_yaml_missing_returns_empty(tmp_path):
    from metabproject.viz._style import load_markers_yaml

    assert load_markers_yaml(tmp_path) == {}


def test_viz_pca_uses_markers_from_viz_yaml(
    tmp_path, feature_table_csv, metadata_csv
):
    from click.testing import CliRunner

    from metabproject.cli import main

    p = _init_and_process(tmp_path, feature_table_csv, metadata_csv)
    (p.path / "viz.yaml").write_text(
        "markers:\n  varA: X\n  varB: P\n"
    )
    result = CliRunner().invoke(
        main, ["viz", "pca", str(p.path), "--marker-col", "variety"]
    )
    assert result.exit_code == 0, result.output

    manifest = json.loads((p.path / "metabproject.json").read_text())
    entry = [e for e in manifest["operations"] if e["command"] == "viz_pca"][-1]
    assert entry["parameters"]["markers"] == {"varA": "X", "varB": "P"}


def test_load_palette_yaml_reads_palette_block(tmp_path):
    from metabproject.viz._style import load_palette_yaml

    (tmp_path / "viz.yaml").write_text(
        "palette:\n  red: '#C0392B'\n  cream: '#E8C999'\n"
    )
    assert load_palette_yaml(tmp_path) == {"red": "#C0392B", "cream": "#E8C999"}


def test_load_palette_yaml_missing_returns_empty(tmp_path):
    from metabproject.viz._style import load_palette_yaml

    assert load_palette_yaml(tmp_path) == {}


def test_load_palette_yaml_rejects_non_mapping_palette(tmp_path):
    from metabproject.viz._style import load_palette_yaml

    (tmp_path / "viz.yaml").write_text("palette: not-a-mapping\n")
    with pytest.raises(ValueError):
        load_palette_yaml(tmp_path)


def test_viz_pca_uses_palette_from_viz_yaml(
    tmp_path, feature_table_csv, metadata_csv
):
    from click.testing import CliRunner

    from metabproject.cli import main

    p = _init_and_process(tmp_path, feature_table_csv, metadata_csv)
    groups = p.metadata["group"].unique().tolist()
    yaml_palette = {g: c for g, c in zip(groups, ["#111111", "#222222"])}
    (p.path / "viz.yaml").write_text(
        "palette:\n" + "".join(f"  {k}: '{v}'\n" for k, v in yaml_palette.items())
    )
    result = CliRunner().invoke(main, ["viz", "pca", str(p.path)])
    assert result.exit_code == 0, result.output

    manifest = json.loads((p.path / "metabproject.json").read_text())
    pca_entries = [e for e in manifest["operations"] if e["command"] == "viz_pca"]
    assert pca_entries, "no viz_pca entry written"
    assert pca_entries[-1]["parameters"]["palette"] == yaml_palette


def test_viz_pca_cli_palette_overrides_viz_yaml(
    tmp_path, feature_table_csv, metadata_csv
):
    from click.testing import CliRunner

    from metabproject.cli import main

    p = _init_and_process(tmp_path, feature_table_csv, metadata_csv)
    groups = p.metadata["group"].unique().tolist()
    g0, g1 = groups[0], groups[1]
    (p.path / "viz.yaml").write_text(
        f"palette:\n  {g0}: '#111111'\n  {g1}: '#222222'\n"
    )
    result = CliRunner().invoke(
        main, ["viz", "pca", str(p.path), "--palette", f"{g0}=#ABCDEF"]
    )
    assert result.exit_code == 0, result.output

    manifest = json.loads((p.path / "metabproject.json").read_text())
    pca_entries = [e for e in manifest["operations"] if e["command"] == "viz_pca"]
    applied = pca_entries[-1]["parameters"]["palette"]

    assert applied[g0] == "#ABCDEF"
    assert applied[g1] == "#222222"


def test_viz_volcano_writes_files(tmp_path, feature_table_csv, metadata_csv):
    p = _init_and_process(tmp_path, feature_table_csv, metadata_csv)
    stats_dir = p.path / "stats"
    stats_dir.mkdir(exist_ok=True)
    df = _make_synthetic_contrast(len(p.feature_table))
    df.index = p.feature_table.index
    df.to_parquet(stats_dir / "synthetic.parquet")
    p.viz_volcano("synthetic")
    assert (p.path / "viz" / "volcano_synthetic.png").exists()
    assert (p.path / "viz" / "volcano_synthetic.svg").exists()
