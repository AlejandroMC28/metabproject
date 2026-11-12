"""Visualization: QC figures and downstream plots."""

from metabproject.viz._style import group_palette, save_figure
from metabproject.viz.boxplot import feature_cell_means_figure
from metabproject.viz.canopus_violin import canopus_violin_figure
from metabproject.viz.heatmap import (
    heatmap_figure,
    pairwise_count_heatmap,
    select_high_confidence_structure_features,
)
from metabproject.viz.iin_boxplot import iin_boxplot_figure
from metabproject.viz.pca import pca_figure
from metabproject.viz.volcano import (
    tukey_pair_to_contrast,
    volcano_figure,
    volcano_grid,
)

__all__ = [
    "group_palette",
    "save_figure",
    "pca_figure",
    "volcano_figure",
    "volcano_grid",
    "tukey_pair_to_contrast",
    "heatmap_figure",
    "pairwise_count_heatmap",
    "feature_cell_means_figure",
    "iin_boxplot_figure",
    "canopus_violin_figure",
    "select_high_confidence_structure_features",
]
