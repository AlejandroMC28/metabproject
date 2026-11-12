"""Processing QC figure generation. See ``specs/process.md`` (QC Figure).

Two-row figure, one column per processing stage actually applied:

- Top row: pooled intensity-distribution KDE per stage.
- Bottom row: per-sample boxplots colored by group, per stage.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless/test environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def make_processing_qc_figure(
    stages: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    output_path: str | Path,
    dpi: int = 150,
) -> Path:
    """Render and save the before/after processing QC figure.

    Args:
        stages: Ordered mapping of stage label (e.g. ``"Raw"``, ``"Filtered"``,
            ``"Imputed"``, ``"Weight-corrected"``, ``"Normalized"``,
            ``"Transformed"``, ``"Scaled"``) to the feature table at that
            stage. Only stages present are plotted.
        metadata: Sample metadata with a ``group`` column, for boxplot coloring.
        output_path: Destination PNG path (e.g. ``qc/processing_qc.png``).
        dpi: Resolution for the saved PNG.

    Returns:
        The path the figure was written to.
    """
    output_path = Path(output_path)
    n_stages = len(stages)

    # Build a consistent group → color mapping.
    groups = metadata["group"]
    unique_groups = groups.unique().tolist()
    palette = dict(zip(unique_groups, sns.color_palette("Set2", n_colors=len(unique_groups))))

    fig_width = max(4 * n_stages, 8)
    fig, axes = plt.subplots(2, n_stages, figsize=(fig_width, 7), squeeze=False)
    fig.subplots_adjust(hspace=0.45, wspace=0.3)

    for col_idx, (stage_label, df) in enumerate(stages.items()):
        ax_top = axes[0][col_idx]
        ax_bot = axes[1][col_idx]

        # Top row: KDE of all values pooled.
        flat_vals = df.values.ravel().astype(float)
        finite_vals = flat_vals[np.isfinite(flat_vals)]
        if len(finite_vals) > 0:
            try:
                sns.kdeplot(finite_vals, ax=ax_top, fill=True, color="steelblue", alpha=0.6)
            except Exception:
                ax_top.hist(finite_vals, bins=40, density=True, color="steelblue", alpha=0.6)
        ax_top.set_title(stage_label, fontsize=10, fontweight="bold")
        ax_top.set_xlabel("Intensity", fontsize=8)
        ax_top.set_ylabel("Density", fontsize=8)
        ax_top.tick_params(labelsize=7)

        # Bottom row: per-sample boxplots colored by group.
        sample_names = df.columns.tolist()
        box_data = [df[s].dropna().values for s in sample_names]
        bp = ax_bot.boxplot(
            box_data,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.5},
            flierprops={"marker": "o", "markersize": 2, "alpha": 0.4},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
        )
        for patch, sample in zip(bp["boxes"], sample_names):
            grp = groups.get(sample, unique_groups[0]) if hasattr(groups, "get") else (
                groups[sample] if sample in groups.index else unique_groups[0]
            )
            patch.set_facecolor(palette.get(grp, "grey"))
            patch.set_alpha(0.8)

        ax_bot.set_xticks(range(1, len(sample_names) + 1))
        ax_bot.set_xticklabels(sample_names, rotation=45, ha="right", fontsize=7)
        ax_bot.set_ylabel("Intensity", fontsize=8)
        ax_bot.tick_params(axis="y", labelsize=7)
        ax_bot.set_title(stage_label, fontsize=10, fontweight="bold")

    # Legend for groups (top-right of figure).
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=palette[g], alpha=0.8, label=g)
        for g in unique_groups
    ]
    fig.legend(handles=handles, title="Group", loc="upper right",
               fontsize=8, title_fontsize=8, bbox_to_anchor=(1.0, 1.0))

    fig.suptitle("Processing QC", fontsize=12, fontweight="bold", y=1.01)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path
