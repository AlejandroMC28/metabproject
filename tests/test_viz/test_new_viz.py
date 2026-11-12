"""Smoke tests for new viz helpers: boxplot, volcano_grid, pairwise heatmap."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from metabproject.stats.comparison_runner import run_comparison
from metabproject.stats.comparisons import Comparison
from metabproject.stats.exports import pairwise_count_matrix
from metabproject.viz import (
    feature_cell_means_figure,
    pairwise_count_heatmap,
    tukey_pair_to_contrast,
    volcano_figure,
    volcano_grid,
)


SEED = 0


def _two_way_dataset() -> tuple[pd.DataFrame, pd.DataFrame, Comparison]:
    rng = np.random.default_rng(SEED)
    samples, l1_col, l2_col = [], [], []
    for l1 in ("a", "b"):
        for l2 in ("p", "q"):
            for r in range(6):
                samples.append(f"{l1}{l2}{r + 1}")
                l1_col.append(l1)
                l2_col.append(l2)
    meta = pd.DataFrame(
        {"variety": l1_col, "treatment": l2_col},
        index=pd.Index(samples, name="sample"),
    )
    l1 = np.array(l1_col)
    l2 = np.array(l2_col)
    n = len(samples)
    rows = np.vstack([
        np.where(l1 == "a", 5.0, 0.0) + rng.normal(scale=0.3, size=n),
        np.where(l2 == "p", 5.0, 0.0) + rng.normal(scale=0.3, size=n),
        np.where((l1 == "a") & (l2 == "p"), 5.0, 0.0)
        + rng.normal(scale=0.3, size=n),
    ])
    ft = pd.DataFrame(
        rows,
        index=pd.Index(["f1", "f2", "f3"], name="feature_id"),
        columns=samples,
    )
    cmp = Comparison(
        name="x", kind="anova_2way",
        factors=["variety", "treatment"], posthoc="tukey",
    )
    return ft, meta, cmp


def test_feature_cell_means_figure_runs() -> None:
    ft, meta, cmp = _two_way_dataset()
    result = run_comparison(cmp, ft, meta)
    fig = feature_cell_means_figure(
        "f3", ft, meta, ["variety", "treatment"],
        omnibus_row=result.omnibus.loc["f3"],
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_feature_cell_means_with_annotation_row() -> None:
    ft, meta, _ = _two_way_dataset()
    ann_row = pd.Series({"compound_name": "abrine", "source": "GNPS"})
    fig = feature_cell_means_figure(
        "f1", ft, meta, ["variety", "treatment"],
        annotation_row=ann_row,
    )
    assert "abrine" in fig._suptitle.get_text()
    plt.close(fig)


def test_feature_cell_means_unknown_feature_raises() -> None:
    ft, meta, _ = _two_way_dataset()
    with pytest.raises(ValueError, match="not in feature_table"):
        feature_cell_means_figure(
            "nope", ft, meta, ["variety", "treatment"],
        )


def test_volcano_grid_panel_count_matches() -> None:
    ft, meta, cmp = _two_way_dataset()
    result = run_comparison(cmp, ft, meta)
    panels = [
        {"title": "a vs b", "contrast_df":
            tukey_pair_to_contrast(result.posthoc, "variety", "a", "b")},
        {"title": "p vs q", "contrast_df":
            tukey_pair_to_contrast(result.posthoc, "treatment", "p", "q")},
    ]
    fig, counts = volcano_grid(panels, fc_threshold=0.5, q_threshold=0.05)
    assert len(counts) == 2
    plt.close(fig)


def test_volcano_figure_annotated_features() -> None:
    df = pd.DataFrame(
        {"log2fc": [2.0, -2.0, 0.1], "p_adj": [1e-4, 1e-3, 0.5]},
        index=pd.Index(["a", "b", "c"], name="feature_id"),
    )
    fig, counts = volcano_figure(
        df, fc_threshold=1.0, annotated_features={"a"}, label_top_n=2,
    )
    assert counts["up"] == 1 and counts["down"] == 1 and counts["ns"] == 1
    plt.close(fig)


def test_tukey_pair_to_contrast_sign_flip() -> None:
    ph = pd.DataFrame({
        "feature": ["a"], "factor": ["v"], "group_A": ["x"], "group_B": ["y"],
        "diff": [2.0], "p_value": [0.01], "significant": [True],
    })
    # Same direction as canonical.
    cd = tukey_pair_to_contrast(ph, "v", "x", "y")
    assert float(cd.iloc[0]["log2fc"]) == 2.0
    # Flipped.
    cd2 = tukey_pair_to_contrast(ph, "v", "y", "x")
    assert float(cd2.iloc[0]["log2fc"]) == -2.0


def test_pairwise_count_heatmap_runs() -> None:
    ft, meta, cmp = _two_way_dataset()
    result = run_comparison(cmp, ft, meta)
    mat = pairwise_count_matrix(result.posthoc, ["variety", "treatment"])
    fig = pairwise_count_heatmap(mat, group_separators=[2])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_pairwise_count_heatmap_non_square_raises() -> None:
    bad = pd.DataFrame(np.zeros((3, 2), dtype=int))
    with pytest.raises(ValueError, match="square"):
        pairwise_count_heatmap(bad)
