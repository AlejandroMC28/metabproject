"""Paired CANOPUS class log2fc violins (class vs. background).

For each enriched CANOPUS class at a chosen level, draws a coloured violin
of the class's per-feature log2fc next to a grey violin of the background
log2fc distribution (all other annotated features). This is the visual
companion to ``metab enrich``'s per-class Mann-Whitney U test.

Publication styling: large
fonts, thick black spines, gridlines, ``*/**/***`` significance brackets.
"""

from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


LABEL_FONTSIZE = 20
TICK_FONTSIZE = 20
LEGEND_FONTSIZE = 18
TITLE_FONTSIZE = 22
SIG_FONTSIZE = 16
VIOLIN_ALPHA = 0.85
SPINE_WIDTH = 2.5
GRID_ALPHA = 0.25
SIG_COLOR = "#d62728"


def _sig_label(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def canopus_violin_figure(
    classes: list[str],
    foreground: dict[str, np.ndarray],
    background: np.ndarray,
    *,
    contrast_label: str,
    level: str,
    fg_color: str = "#4A8FA0",
    bg_color: str = "#BFBFBF",
    enrichment_stats: dict[str, dict] | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, dict]:
    """Render paired class-vs-background violins for a chosen CANOPUS level.

    Args:
        classes: Ordered list of class names to plot (x-axis order).
        foreground: ``{class_name: log2fc array}`` for each class.
        background: log2fc values of all annotated features not in any of
            the plotted classes (shared grey violin redrawn per tick).
        contrast_label: Contrast name, used in the y-axis label.
        level: CANOPUS level (``npc_pathway``, ``npc_superclass``,
            ``npc_class``, ``classyfire_class``) — surfaced in the title.
        fg_color: Foreground violin colour.
        bg_color: Background violin colour.
        enrichment_stats: Optional ``{class: {"p_adj": float, "n": int}}``;
            when a class has ``p_adj < 0.05`` a red ``*/**/***`` significance
            bracket is drawn above its violin pair.
        title: Optional suptitle. Defaults to ``"CANOPUS {level} — {contrast}"``.

    Returns:
        ``(fig, info)`` where ``info`` carries ``n_classes``,
        ``n_features_per_class`` (dict), ``n_background``, and
        ``skipped_empty`` (classes dropped for having no foreground values).
    """
    if not classes:
        raise ValueError("classes is empty — nothing to plot.")

    plotted: list[str] = []
    skipped_empty: list[str] = []
    counts: dict[str, int] = {}
    for c in classes:
        vals = np.asarray(foreground.get(c, []), dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            skipped_empty.append(c)
            continue
        plotted.append(c)
        counts[c] = int(vals.size)
        foreground[c] = vals

    if not plotted:
        raise ValueError("All classes had empty foreground arrays.")

    bg = np.asarray(background, dtype=float)
    bg = bg[np.isfinite(bg)]
    if bg.size == 0:
        raise ValueError("Background log2fc array is empty.")

    n = len(plotted)
    centres = np.arange(n) * 3.5
    fig_width = max(12.0, n * 1.6)
    fig, ax = plt.subplots(figsize=(fig_width, 7.5))
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5, zorder=0)

    for i, cname in enumerate(plotted):
        fg_vals = foreground[cname]
        fg_pos = centres[i] - 0.5
        bg_pos = centres[i] + 0.5

        # Background first so the foreground sits visually on top of it.
        for vals, pos, colour in (
            (bg, bg_pos, bg_color),
            (fg_vals, fg_pos, fg_color),
        ):
            if vals.size < 2:
                ax.scatter([pos], [float(vals[0]) if vals.size else np.nan],
                           color=colour, edgecolor="black", linewidth=1.0,
                           s=40, zorder=3)
                continue
            parts = ax.violinplot(
                [vals], positions=[pos], widths=0.9,
                showmeans=False, showmedians=True, showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(colour)
                body.set_edgecolor("black")
                body.set_alpha(VIOLIN_ALPHA)
                body.set_linewidth(1.5)
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(2.5)

        if enrichment_stats and cname in enrichment_stats:
            q = enrichment_stats[cname].get("p_adj")
            label = _sig_label(q)
            if label != "ns":
                y_max = max(float(fg_vals.max()), float(bg.max()))
                y_top = y_max + 0.5
                x_l = centres[i] - 0.5 - 0.45
                x_r = centres[i] + 0.5 + 0.45
                ax.plot(
                    [x_l, x_l, x_r, x_r],
                    [y_top - 0.15, y_top, y_top, y_top - 0.15],
                    lw=1.5, color="black",
                )
                ax.text(
                    centres[i], y_top + 0.05, label,
                    ha="center", va="bottom",
                    fontsize=SIG_FONTSIZE, color=SIG_COLOR, fontweight="bold",
                )

    ax.set_xticks(centres)
    ax.set_xticklabels(
        [textwrap.fill(c, width=18) for c in plotted],
        rotation=90, ha="center", fontsize=TICK_FONTSIZE,
    )
    ax.set_ylabel(
        f"log₂ fold change ({contrast_label})",
        fontsize=LABEL_FONTSIZE, fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE,
                   width=SPINE_WIDTH, length=6)
    ax.tick_params(axis="x", width=SPINE_WIDTH, length=6, pad=10)

    ax.spines[["top", "right"]].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(SPINE_WIDTH)
        ax.spines[spine].set_edgecolor("black")

    ax.grid(axis="y", alpha=GRID_ALPHA, linewidth=0.8,
            linestyle="-", color="gray")
    ax.set_axisbelow(True)

    ax.legend(
        handles=[
            Patch(facecolor=fg_color, alpha=VIOLIN_ALPHA,
                  edgecolor="black", linewidth=1.5, label="class"),
            Patch(facecolor=bg_color, alpha=VIOLIN_ALPHA,
                  edgecolor="black", linewidth=1.5, label="background"),
        ],
        fontsize=LEGEND_FONTSIZE, frameon=True, loc="lower right",
        edgecolor="black", framealpha=1, fancybox=False,
    )

    fig.suptitle(
        title or f"CANOPUS {level} — {contrast_label}",
        fontsize=TITLE_FONTSIZE, fontweight="bold",
    )
    fig.tight_layout()

    return fig, {
        "n_classes": n,
        "n_features_per_class": counts,
        "n_background": int(bg.size),
        "skipped_empty": skipped_empty,
    }
