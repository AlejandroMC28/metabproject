"""Tests for the factorial-comparison panel in explore/data.py.

The explore viewer surfaces persisted Comparison results
(`<project>/stats/<name>/{omnibus,posthoc}.parquet`). To exercise the
assembly path without re-running an ANOVA on the synthetic fixture (which
has unbalanced cells), the parquets are fabricated directly here and a
matching ``run_comparison`` manifest entry is appended.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metabproject.manifest import append_operation, make_operation_entry


@pytest.fixture
def processed_project(tmp_path, feature_table_csv, metadata_csv):
    from metabproject.project import MetabProject

    p = MetabProject.init(
        path=tmp_path / "proj",
        feature_table=feature_table_csv,
        metadata=metadata_csv,
    )
    p.process(force=False)
    return p


@pytest.fixture
def project_with_comparison(processed_project):
    """Fabricate a 2-way ANOVA omnibus + Tukey posthoc + manifest entry."""
    project = processed_project
    ft = project.feature_table
    fids = list(ft.index)
    name = "treatment_x_variety"
    factors = ["group", "variety"]

    # Fabricate omnibus parquet matching anova_2way's schema.
    rng = np.random.default_rng(0)
    n = len(fids)
    omnibus = pd.DataFrame(
        {
            "F_group": rng.uniform(0.5, 10, n),
            "p_group": rng.uniform(0, 1, n),
            "p_group_adj": rng.uniform(0, 1, n),
            "sig_group": rng.random(n) < 0.3,
            "F_variety": rng.uniform(0.5, 10, n),
            "p_variety": rng.uniform(0, 1, n),
            "p_variety_adj": rng.uniform(0, 1, n),
            "sig_variety": rng.random(n) < 0.3,
            "F_interaction": rng.uniform(0.5, 10, n),
            "p_interaction": rng.uniform(0, 1, n),
            "p_interaction_adj": rng.uniform(0, 1, n),
            "sig_interaction": rng.random(n) < 0.1,
            "classification": rng.choice(
                ["ns", "factor1_only", "factor2_only",
                 "both_no_interaction", "interaction"],
                size=n,
            ),
        },
        index=ft.index,
    )

    posthoc = pd.DataFrame({
        "feature": [fids[0], fids[0], fids[1]],
        "factor": ["group", "variety", "group"],
        "group_A": ["control", "varA", "control"],
        "group_B": ["fermented", "varB", "fermented"],
        "mean_A": [1.2, 0.8, 0.3],
        "mean_B": [2.5, 1.9, 1.7],
        "diff": [1.3, 1.1, 1.4],
        "p_value": [1e-4, 0.04, 0.8],
        "significant": [True, True, False],
        "hedges_g": [1.8, 0.9, 0.1],
    })

    cmp_dir = project.path / "stats" / name
    cmp_dir.mkdir(parents=True, exist_ok=True)
    omnibus.to_parquet(cmp_dir / "omnibus.parquet")
    posthoc.to_parquet(cmp_dir / "posthoc.parquet")

    entry = make_operation_entry(
        command="run_comparison",
        parameters={
            "name": name,
            "kind": "anova_2way",
            "factors": factors,
            "posthoc": "tukey",
            "alpha": 0.05,
        },
        results={"n_features": n},
    )
    append_operation(project.path, entry)
    return project, name, fids, factors


def test_project_comparisons_loads_omnibus_and_posthoc(project_with_comparison):
    project, name, fids, factors = project_with_comparison
    comps = project.comparisons
    assert name in comps
    payload = comps[name]
    assert payload["kind"] == "anova_2way"
    assert payload["factors"] == factors
    assert payload["posthoc"] == "tukey"
    assert payload["omnibus"].shape[0] == len(fids)
    assert payload["posthoc_df"] is not None
    assert "p_group" in payload["omnibus"].columns
    assert "p_interaction" in payload["omnibus"].columns


def test_comparisons_meta_carries_levels_and_sample_factors(project_with_comparison):
    from metabproject.explore.data import assemble_explore_data

    project, name, _, factors = project_with_comparison
    data = assemble_explore_data(project, render_structures=False)
    meta = data["comparisons_meta"]
    assert name in meta
    m = meta[name]
    assert m["kind"] == "anova_2way"
    assert m["factors"] == factors
    assert "group" in m["levels"]
    assert set(m["levels"]["group"]) == {"control", "fermented"}
    assert set(m["levels"]["variety"]) == {"varA", "varB"}
    # Sample → factor map must include every metadata sample.
    sf = m["sample_factors"]
    md = project.metadata
    assert set(sf.keys()) == set(map(str, md.index))
    a_sample = next(iter(md.index))
    assert sf[str(a_sample)]["group"] == str(md.at[a_sample, "group"])
    assert m["has_posthoc"] is True


def test_per_feature_comparisons_contain_terms_and_posthoc(project_with_comparison):
    from metabproject.explore.data import assemble_explore_data

    project, name, fids, _ = project_with_comparison
    data = assemble_explore_data(project, render_structures=False)
    fid = fids[0]
    feat = data["features"][str(fid)]
    assert "comparisons" in feat
    cmp_data = feat["comparisons"][name]
    assert cmp_data["kind"] == "anova_2way"
    assert "group" in cmp_data["terms"]
    assert "variety" in cmp_data["terms"]
    assert cmp_data["interaction"] is not None
    assert cmp_data["classification"] is not None
    # Post-hoc filtered to this feature, schema preserved.
    ph = cmp_data["posthoc"]
    assert len(ph) == 2  # two rows had feature == fids[0]
    assert {r["factor"] for r in ph} == {"group", "variety"}
    assert ph[0]["group_A"] is not None
    assert ph[0]["significant"] is True


def test_no_comparisons_when_none_persisted(processed_project):
    from metabproject.explore.data import assemble_explore_data

    data = assemble_explore_data(processed_project, render_structures=False)
    assert data["comparisons_meta"] == {}
    # Features should not carry a 'comparisons' key when there are no results.
    any_fid = next(iter(data["features"]))
    assert "comparisons" not in data["features"][any_fid]
