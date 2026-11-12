"""Tests for the contrasts YAML loader and sample resolver."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from metabproject.stats.contrasts import (
    Contrast,
    load_contrasts,
    resolve_samples,
)


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "contrasts.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_load_contrasts_basic(tmp_path: Path) -> None:
    yaml = """
contrasts:
  - name: control_vs_fermented
    a: { group: control }
    b: { group: fermented }
"""
    [c] = load_contrasts(_write(tmp_path, yaml))
    assert c.name == "control_vs_fermented"
    assert c.a == {"group": ["control"]}
    assert c.b == {"group": ["fermented"]}


def test_load_contrasts_list_values_and_multikey(tmp_path: Path) -> None:
    yaml = """
contrasts:
  - name: varA_control_vs_other_treatments
    a: { variety: varA, group: control }
    b: { variety: varA, group: [fermented, roasted] }
"""
    [c] = load_contrasts(_write(tmp_path, yaml))
    assert c.a == {"variety": ["varA"], "group": ["control"]}
    assert c.b == {"variety": ["varA"], "group": ["fermented", "roasted"]}


def test_load_contrasts_duplicate_name(tmp_path: Path) -> None:
    yaml = """
contrasts:
  - { name: dup, a: { group: a }, b: { group: b } }
  - { name: dup, a: { group: a }, b: { group: b } }
"""
    with pytest.raises(ValueError, match="Duplicate"):
        load_contrasts(_write(tmp_path, yaml))


def test_load_contrasts_unsafe_slug(tmp_path: Path) -> None:
    yaml = """
contrasts:
  - { name: "has space", a: { group: a }, b: { group: b } }
"""
    with pytest.raises(ValueError, match="safe filename slug"):
        load_contrasts(_write(tmp_path, yaml))


def test_load_contrasts_missing_key(tmp_path: Path) -> None:
    yaml = """
contrasts:
  - { name: only_a, a: { group: a } }
"""
    with pytest.raises(ValueError, match="missing key"):
        load_contrasts(_write(tmp_path, yaml))


def test_load_contrasts_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_contrasts(tmp_path / "does_not_exist.yaml")


def test_resolve_samples_basic() -> None:
    meta = pd.DataFrame(
        {"group": ["control", "control", "fermented", "fermented"],
         "variety": ["varA", "varB", "varA", "varB"]},
        index=pd.Index(["s1", "s2", "s3", "s4"], name="sample_name"),
    )
    c = Contrast(name="x", a={"group": ["control"]}, b={"group": ["fermented"]})
    a, b = resolve_samples(c, meta)
    assert sorted(a) == ["s1", "s2"]
    assert sorted(b) == ["s3", "s4"]


def test_resolve_samples_multi_key_filter() -> None:
    meta = pd.DataFrame(
        {"group": ["control", "control", "fermented", "fermented"],
         "variety": ["varA", "varB", "varA", "varB"]},
        index=pd.Index(["s1", "s2", "s3", "s4"], name="sample_name"),
    )
    c = Contrast(
        name="varA_only",
        a={"variety": ["varA"], "group": ["control"]},
        b={"variety": ["varA"], "group": ["fermented"]},
    )
    a, b = resolve_samples(c, meta)
    assert a == ["s1"]
    assert b == ["s3"]


def test_resolve_samples_unknown_column() -> None:
    meta = pd.DataFrame(
        {"group": ["control", "fermented"]},
        index=pd.Index(["s1", "s2"], name="sample_name"),
    )
    c = Contrast(name="x", a={"missing": ["foo"]}, b={"group": ["fermented"]})
    with pytest.raises(ValueError, match="no column"):
        resolve_samples(c, meta)


def test_resolve_samples_unknown_value() -> None:
    meta = pd.DataFrame(
        {"group": ["control", "fermented"]},
        index=pd.Index(["s1", "s2"], name="sample_name"),
    )
    c = Contrast(name="x", a={"group": ["nonsense"]}, b={"group": ["fermented"]})
    with pytest.raises(ValueError, match="not found"):
        resolve_samples(c, meta)


def test_resolve_samples_overlap_raises() -> None:
    meta = pd.DataFrame(
        {"group": ["control", "fermented"]},
        index=pd.Index(["s1", "s2"], name="sample_name"),
    )
    # Both filters end up matching the same sample.
    c = Contrast(name="x", a={"group": ["control"]}, b={"group": ["control"]})
    with pytest.raises(ValueError, match="share sample"):
        resolve_samples(c, meta)
