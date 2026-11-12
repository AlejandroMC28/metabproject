"""Pre-imputation missingness diagnostics. See ``specs/process.md``.

Visualises which features are missing in which samples on the filtered table,
right before imputation runs. Helps the user see whether missingness is
concentrated in particular samples (suggesting an injection problem) or spread
evenly across the matrix.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def compute_missingness(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the boolean missingness mask and per-sample summary.

    A value is considered missing when it is ``0`` or NaN — the same rule the
    imputers use.

    Args:
        df: Filtered feature table (features × samples).

    Returns:
        ``(mask, per_sample_summary)``:
        ``mask`` is a boolean DataFrame with the same shape as ``df``
        (``True`` = missing). ``per_sample_summary`` has one row per sample
        with columns ``n_missing``, ``n_features``, ``pct_missing``.
    """
    mask = (df == 0) | df.isna()
    n_features = len(df)
    per_sample = pd.DataFrame(
        {
            "n_missing": mask.sum(axis=0).astype(int),
            "n_features": n_features,
            "pct_missing": (mask.sum(axis=0) / max(n_features, 1) * 100.0).round(2),
        }
    )
    per_sample.index.name = "sample_name"
    return mask, per_sample


def make_missingness_heatmap(
    mask: pd.DataFrame,
    metadata: pd.DataFrame,
    output_path: str | Path,
    dpi: int = 150,
) -> Path:
    """Render a features × samples missingness heatmap with a per-sample bar.

    Features are sorted by total missing count (most-missing on top) so the
    dense-missing band is immediately visible. Samples are coloured by group.

    Args:
        mask: Boolean missingness mask (``True`` = missing), features × samples.
        metadata: Sample metadata with a ``group`` column.
        output_path: Destination PNG path.
        dpi: Resolution for the saved PNG.

    Returns:
        The path the figure was written to.
    """
    output_path = Path(output_path)

    n_features, n_samples = mask.shape
    sample_names = mask.columns.tolist()

    # Sort features by per-feature missingness (most-missing on top).
    order = mask.sum(axis=1).sort_values(ascending=False).index
    sorted_mask = mask.loc[order]
    per_sample_missing = mask.sum(axis=0)
    per_sample_pct = per_sample_missing / max(n_features, 1) * 100.0

    groups = metadata["group"] if "group" in metadata.columns else pd.Series(
        ["all"] * n_samples, index=sample_names
    )
    unique_groups = list(dict.fromkeys(groups.loc[sample_names].tolist()))
    palette = dict(
        zip(unique_groups, sns.color_palette("Set2", n_colors=len(unique_groups)))
    )
    sample_colors = [palette[groups[s]] for s in sample_names]

    fig_width = max(0.35 * n_samples + 2.0, 8)
    fig_height = 7.5
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[1, 6],
        hspace=0.05,
    )

    ax_bar = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0], sharex=ax_bar)

    # Top bar: per-sample missing percentage, coloured by group.
    x = np.arange(n_samples)
    ax_bar.bar(x, per_sample_pct.values, color=sample_colors, edgecolor="black", linewidth=0.4)
    for xi, (n, pct) in enumerate(zip(per_sample_missing.values, per_sample_pct.values)):
        ax_bar.text(
            xi,
            pct,
            f"{int(n)}",
            ha="center",
            va="bottom",
            fontsize=6,
        )
    ax_bar.set_ylabel("% missing", fontsize=8)
    ax_bar.tick_params(axis="y", labelsize=7)
    ax_bar.set_xticks([])
    ax_bar.set_xlim(-0.5, n_samples - 0.5)
    ymax = max(per_sample_pct.max() * 1.15, 1.0) if len(per_sample_pct) else 1.0
    ax_bar.set_ylim(0, ymax)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # Bottom: binary heatmap (white = present, dark = missing).
    ax_heat.imshow(
        sorted_mask.values.astype(float),
        aspect="auto",
        cmap="Greys",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax_heat.set_xticks(x)
    ax_heat.set_xticklabels(sample_names, rotation=45, ha="right", fontsize=7)
    for tick_label, s in zip(ax_heat.get_xticklabels(), sample_names):
        tick_label.set_color(palette[groups[s]])
    ax_heat.set_ylabel(f"Features (n={n_features:,}, sorted by missingness)", fontsize=8)
    ax_heat.set_yticks([])

    # Group legend.
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=palette[g], alpha=0.9, label=g)
        for g in unique_groups
    ]
    fig.legend(
        handles=handles,
        title="Group",
        loc="upper right",
        fontsize=8,
        title_fontsize=8,
        bbox_to_anchor=(1.0, 1.0),
    )

    total_missing = int(mask.values.sum())
    total_cells = int(mask.size)
    pct_total = total_missing / max(total_cells, 1) * 100.0
    fig.suptitle(
        f"Pre-imputation missingness — {total_missing:,} / {total_cells:,} "
        f"values missing ({pct_total:.1f}%)",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path
