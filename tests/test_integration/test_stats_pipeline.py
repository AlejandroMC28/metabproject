"""Integration tests: init → process → stats end-to-end."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from metabproject.project import MetabProject


def _write_contrasts(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "contrasts.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def _init_and_process(
    tmp_path: Path,
    feature_table_csv: Path,
    metadata_csv: Path,
    transform: str = "log2",
    scale: str = "none",
) -> MetabProject:
    project_path = tmp_path / "demo"
    p = MetabProject.init(
        path=project_path,
        feature_table=feature_table_csv,
        metadata=metadata_csv,
    )
    p.process(transform=transform, scale=scale)
    return p


def test_stats_end_to_end(tmp_path: Path, feature_table_csv: Path, metadata_csv: Path) -> None:
    p = _init_and_process(tmp_path, feature_table_csv, metadata_csv)
    yaml = """
contrasts:
  - name: control_vs_fermented
    a: { group: control }
    b: { group: fermented }
  - name: varA_control_vs_fermented
    a: { variety: varA, group: control }
    b: { variety: varA, group: fermented }
"""
    # The varA subset has only 2 samples per group in the fixture,
    # so drop min_samples to 2 to exercise the factorial contrast too.
    p.stats(contrasts=_write_contrasts(tmp_path, yaml), min_samples=2)

    stats_dir = p.path / "stats"
    assert (stats_dir / "control_vs_fermented.parquet").exists()
    assert (stats_dir / "varA_control_vs_fermented.parquet").exists()

    df = p.contrast("control_vs_fermented")
    assert df.index.name == "feature_id"
    for col in ["mean_a", "mean_b", "n_a", "n_b", "log2fc",
                "t_stat", "p_value", "p_adj", "direction", "significant"]:
        assert col in df.columns
    # log2fc must be finite for most features (raw-scale, imputed).
    assert df["log2fc"].notna().mean() > 0.9

    # Lazy `contrasts` dict.
    assert set(p.contrasts) == {"control_vs_fermented", "varA_control_vs_fermented"}


def test_stats_pareto_scale_still_yields_finite_log2fc(
    tmp_path: Path, feature_table_csv: Path, metadata_csv: Path
) -> None:
    p = _init_and_process(
        tmp_path, feature_table_csv, metadata_csv,
        transform="none", scale="pareto",
    )
    yaml = """
contrasts:
  - name: control_vs_fermented
    a: { group: control }
    b: { group: fermented }
"""
    p.stats(contrasts=_write_contrasts(tmp_path, yaml))
    df = p.contrast("control_vs_fermented")
    # Decoupled from the transform → log2fc is computed from raw-imputed values.
    assert df["log2fc"].notna().mean() > 0.9


def test_stats_idempotent_without_force(
    tmp_path: Path, feature_table_csv: Path, metadata_csv: Path
) -> None:
    p = _init_and_process(tmp_path, feature_table_csv, metadata_csv)
    yaml = """
contrasts:
  - { name: cvf, a: { group: control }, b: { group: fermented } }
"""
    cf = _write_contrasts(tmp_path, yaml)
    p.stats(contrasts=cf)
    with pytest.raises(FileExistsError):
        p.stats(contrasts=cf)
    p.stats(contrasts=cf, force=True)  # succeeds


def test_stats_unknown_metadata_column_errors(
    tmp_path: Path, feature_table_csv: Path, metadata_csv: Path
) -> None:
    p = _init_and_process(tmp_path, feature_table_csv, metadata_csv)
    yaml = """
contrasts:
  - { name: bad, a: { nope: x }, b: { group: fermented } }
"""
    with pytest.raises(ValueError, match="no column"):
        p.stats(contrasts=_write_contrasts(tmp_path, yaml))


def test_stats_manifest_entry(
    tmp_path: Path, feature_table_csv: Path, metadata_csv: Path
) -> None:
    p = _init_and_process(tmp_path, feature_table_csv, metadata_csv)
    yaml = """
contrasts:
  - { name: cvf, a: { group: control }, b: { group: fermented } }
"""
    p.stats(contrasts=_write_contrasts(tmp_path, yaml))
    manifest = json.loads((p.path / "metabproject.json").read_text())
    last = manifest["operations"][-1]
    assert last["command"] == "stats"
    assert last["parameters"]["test"] == "welch_ttest"
    assert "cvf" in last["results"]
    assert {"n_a", "n_b", "n_tested", "n_significant", "n_up", "n_down"} <= set(last["results"]["cvf"])


def test_stats_requires_process_first(
    tmp_path: Path, feature_table_csv: Path, metadata_csv: Path
) -> None:
    # init only, skip process
    project_path = tmp_path / "demo"
    p = MetabProject.init(
        path=project_path,
        feature_table=feature_table_csv,
        metadata=metadata_csv,
    )
    yaml = """
contrasts:
  - { name: cvf, a: { group: control }, b: { group: fermented } }
"""
    with pytest.raises(FileNotFoundError):
        p.stats(contrasts=_write_contrasts(tmp_path, yaml))
