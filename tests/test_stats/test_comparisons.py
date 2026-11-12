"""Tests for the comparisons YAML loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from metabproject.stats.comparisons import Comparison, load_comparisons


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "comparisons.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_load_comparisons_basic(tmp_path: Path) -> None:
    yaml = """
comparisons:
  - name: variety_main
    kind: anova_1way
    factors: [variety]
    posthoc: tukey
  - name: full_factorial
    kind: anova_2way
    factors: [variety, treatment]
    posthoc: tukey
  - name: variety_nonparam
    kind: kruskal
    factors: [variety]
    posthoc: dunn
"""
    cmps = load_comparisons(_write(tmp_path, yaml))
    assert [c.name for c in cmps] == [
        "variety_main", "full_factorial", "variety_nonparam",
    ]
    assert cmps[0].kind == "anova_1way"
    assert cmps[0].factors == ["variety"]
    assert cmps[0].posthoc == "tukey"
    assert cmps[1].kind == "anova_2way"
    assert cmps[1].factors == ["variety", "treatment"]
    assert cmps[2].posthoc == "dunn"
    assert all(c.alpha == 0.05 for c in cmps)


def test_load_comparisons_alpha_override(tmp_path: Path) -> None:
    yaml = """
comparisons:
  - { name: a, kind: anova_1way, factors: [variety], alpha: 0.01 }
"""
    [c] = load_comparisons(_write(tmp_path, yaml))
    assert c.alpha == 0.01
    assert c.posthoc is None


def test_load_comparisons_no_posthoc(tmp_path: Path) -> None:
    yaml = """
comparisons:
  - { name: a, kind: welch_anova, factors: [variety] }
"""
    [c] = load_comparisons(_write(tmp_path, yaml))
    assert c.posthoc is None


def test_load_comparisons_unknown_kind(tmp_path: Path) -> None:
    yaml = """
comparisons:
  - { name: a, kind: anova_3way, factors: [a, b, c] }
"""
    with pytest.raises(ValueError, match="kind 'anova_3way' unknown"):
        load_comparisons(_write(tmp_path, yaml))


def test_load_comparisons_wrong_factor_count(tmp_path: Path) -> None:
    yaml = """
comparisons:
  - { name: a, kind: anova_2way, factors: [variety] }
"""
    with pytest.raises(ValueError, match="requires exactly 2"):
        load_comparisons(_write(tmp_path, yaml))


def test_load_comparisons_invalid_posthoc_for_kind(tmp_path: Path) -> None:
    yaml = """
comparisons:
  - { name: a, kind: kruskal, factors: [variety], posthoc: tukey }
"""
    with pytest.raises(ValueError, match="not allowed for kind 'kruskal'"):
        load_comparisons(_write(tmp_path, yaml))


def test_load_comparisons_duplicate_name(tmp_path: Path) -> None:
    yaml = """
comparisons:
  - { name: dup, kind: anova_1way, factors: [variety] }
  - { name: dup, kind: kruskal, factors: [variety] }
"""
    with pytest.raises(ValueError, match="Duplicate comparison name"):
        load_comparisons(_write(tmp_path, yaml))


def test_load_comparisons_unsafe_slug(tmp_path: Path) -> None:
    yaml = """
comparisons:
  - { name: "has space", kind: anova_1way, factors: [variety] }
"""
    with pytest.raises(ValueError, match="safe filename slug"):
        load_comparisons(_write(tmp_path, yaml))


def test_load_comparisons_missing_key(tmp_path: Path) -> None:
    yaml = """
comparisons:
  - { name: only_kind, kind: anova_1way }
"""
    with pytest.raises(ValueError, match="missing key"):
        load_comparisons(_write(tmp_path, yaml))


def test_load_comparisons_factors_duplicates(tmp_path: Path) -> None:
    yaml = """
comparisons:
  - { name: a, kind: anova_2way, factors: [variety, variety] }
"""
    with pytest.raises(ValueError, match="contains duplicates"):
        load_comparisons(_write(tmp_path, yaml))


def test_load_comparisons_alpha_out_of_range(tmp_path: Path) -> None:
    yaml = """
comparisons:
  - { name: a, kind: anova_1way, factors: [variety], alpha: 1.5 }
"""
    with pytest.raises(ValueError, match="alpha must be a float"):
        load_comparisons(_write(tmp_path, yaml))


def test_load_comparisons_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_comparisons(tmp_path / "missing.yaml")


def test_load_comparisons_coexists_with_contrasts_block(tmp_path: Path) -> None:
    # Same YAML file holds both contrasts and comparisons; the comparisons
    # loader ignores the contrasts block (and vice versa).
    yaml = """
contrasts:
  - { name: c1, a: { group: control }, b: { group: fermented } }

comparisons:
  - { name: variety_main, kind: anova_1way, factors: [variety], posthoc: tukey }
"""
    cmps = load_comparisons(_write(tmp_path, yaml))
    assert [c.name for c in cmps] == ["variety_main"]


def test_comparison_dataclass_default_alpha() -> None:
    c = Comparison(name="x", kind="anova_1way", factors=["variety"])
    assert c.alpha == 0.05
    assert c.posthoc is None
