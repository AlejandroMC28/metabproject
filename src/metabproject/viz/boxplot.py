"""Per-feature box+strip plot for a two-factor design.

Renders a boxplot of one feature's per-sample log-scale abundance grouped
by factor1 on the x-axis, coloured by factor2. Generalises the
notebook-style figure: groups are coloured
using :func:`metabproject.viz._style.group_palette` and the suptitle
carries the omnibus p-values plus optional compound annotation.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from metabproject.viz._style import group_palette


def feature_cell_means_figure(
    feature_id,
    feature_table: pd.DataFrame,
    metadata: pd.DataFrame,
    factors: list[str],
    omnibus_row: pd.Series | None = None,
    annotation_row: pd.Series | None = None,
    palette: dict | None = None,
    figsize: tuple[float, float] = (7, 5),
) -> plt.Figure:
    """Box + strip plot of one feature grouped by factor1 × factor2.

    Args:
        feature_id: ``feature_id`` (matches ``feature_table.index``).
        feature_table: Features × samples processed (log-scale) table.
        metadata: Sample metadata indexed by sample name.
        factors: ``[factor1, factor2]`` — x-axis category and hue category.
        omnibus_row: Optional row from ``anova_2way`` output for this
            feature. When provided, adjusted p-values and classification
            are written into the suptitle.
        annotation_row: Optional row from a merged annotation table. When
            provided and ``compound_name`` is non-null, suptitle gains
            ``— {compound_name} [{source}]``.
        palette: Optional ``{factor2_level: colour}`` override; falls back
            to ``group_palette`` (Set2) when missing.
        figsize: Matplotlib figure size.

    Returns:
        The matplotlib Figure.
    """
    if len(factors) != 2:
        raise ValueError(f"expected 2 factors, got {len(factors)}")
    f1, f2 = factors
    if feature_id not in feature_table.index:
        raise ValueError(f"feature_id {feature_id!r} not in feature_table")

    samples = [s for s in feature_table.columns if s in metadata.index]
    df = metadata.loc[samples, [f1, f2]].copy()
    df["value"] = feature_table.loc[feature_id, samples].to_numpy()
    df = df.dropna(subset=["value"])

    f1_order = list(metadata[f1].drop_duplicates())
    f2_order = list(metadata[f2].drop_duplicates())
    hue_palette = group_palette(metadata, group_col=f2, custom=palette)

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=df, x=f1, y="value", hue=f2,
        order=f1_order, hue_order=f2_order,
        palette=hue_palette, ax=ax, showfliers=False,
    )
    sns.stripplot(
        data=df, x=f1, y="value", hue=f2,
        order=f1_order, hue_order=f2_order,
        dodge=True, size=4, color="black", alpha=0.7,
        ax=ax, legend=False,
    )

    ax.set_xlabel(f1.title())
    ax.set_ylabel("log2 intensity")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles[: len(f2_order)],
            [str(lbl).title() for lbl in labels[: len(f2_order)]],
            title=f2.title(), frameon=False, loc="best",
        )

    title_parts = [str(feature_id)]
    if omnibus_row is not None:
        cls = omnibus_row.get("classification")
        if cls is not None:
            title_parts.append(f"[{cls}]")
        p_bits = []
        for fac, col in ((f1, f"p_{f1}_adj"), (f2, f"p_{f2}_adj"),
                         ("interaction", "p_interaction_adj")):
            val = omnibus_row.get(col)
            if val is not None and pd.notna(val):
                p_bits.append(f"{fac}: {float(val):.2e}")
        if p_bits:
            title_parts.append("\np_adj — " + "   ".join(p_bits))

    if annotation_row is not None:
        name = annotation_row.get("compound_name")
        source = annotation_row.get("source")
        if name is not None and pd.notna(name):
            tag = f" — {name}"
            if source is not None and pd.notna(source):
                tag += f" [{source}]"
            title_parts.append(tag)

    fig.suptitle("   ".join(title_parts), fontsize=10)
    fig.tight_layout()
    return fig
