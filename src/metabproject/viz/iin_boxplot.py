"""Per-feature boxplots for an IIN group across a chosen contrast.

Given a focal feature, finds every member of its ion identity network (IIN)
group and renders one boxplot per member showing abundance in group A vs
group B of a chosen contrast. Features with a high-confidence structural
annotation are placed first (top-left) and labelled with the compound name.

Publication styling: large
fonts, thick black spines, gridlines.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABEL_FONTSIZE = 18
TICK_FONTSIZE = 16
PANEL_TITLE_FONTSIZE = 14
SUPTITLE_FONTSIZE = 22
BOX_ALPHA = 0.85
SPINE_WIDTH = 2.5
GRID_ALPHA = 0.25


def iin_boxplot_figure(
    members: list[dict],
    feature_table: pd.DataFrame,
    samples_a: list[str],
    samples_b: list[str],
    label_a: str,
    label_b: str,
    palette: dict | None = None,
    ncols: int = 4,
    title: str | None = None,
) -> tuple[plt.Figure, dict[str, int]]:
    """Render per-IIN-member boxplots for a single contrast.

    Args:
        members: Ordered list (high-confidence first) of dicts with keys
            ``feature_id`` (int), ``ion_identity`` (str | None),
            ``compound_name`` (str | None).
        feature_table: Processed feature table indexed by ``feature_id``,
            columns are sample names.
        samples_a: Sample names belonging to group A.
        samples_b: Sample names belonging to group B.
        label_a: Group A label (used for the left tick + legend).
        label_b: Group B label (used for the right tick + legend).
        palette: Optional ``{label: colour}`` mapping for the two boxes.
            Missing labels fall back to a fixed default.
        ncols: Number of columns in the subplot grid.
        title: Optional figure suptitle (e.g. IIN id + neutral mass).

    Returns:
        ``(fig, info)`` with ``info = {"n_members": n}``.
    """
    if not members:
        raise ValueError("members is empty — no IIN partners to plot.")

    palette = palette or {}
    color_a = palette.get(label_a, "#4A8FA0")
    color_b = palette.get(label_b, "#D44A4A")

    n = len(members)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False,
    )
    flat = axes.flatten()

    for ax, m in zip(flat, members):
        fid = m["feature_id"]
        if fid not in feature_table.index:
            ax.text(0.5, 0.5, f"feat {fid}\nnot in table",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=TICK_FONTSIZE, color="grey")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        row = feature_table.loc[fid]
        va = row.reindex(samples_a).dropna().to_numpy(dtype=float)
        vb = row.reindex(samples_b).dropna().to_numpy(dtype=float)

        bp = ax.boxplot(
            [va, vb], positions=[1, 2], widths=0.6, patch_artist=True,
            medianprops={"color": "black", "linewidth": 2.5},
            boxprops={"linewidth": 1.5},
            whiskerprops={"linewidth": 1.5, "color": "black"},
            capprops={"linewidth": 1.5, "color": "black"},
            flierprops={"marker": "o", "markersize": 4,
                        "markerfacecolor": "grey", "markeredgecolor": "none"},
        )
        for patch, c in zip(bp["boxes"], (color_a, color_b)):
            patch.set_facecolor(c)
            patch.set_alpha(BOX_ALPHA)
            patch.set_edgecolor("black")

        rng = np.random.default_rng(int(fid))
        for x, v, c in [(1, va, color_a), (2, vb, color_b)]:
            if v.size:
                jitter = (rng.random(v.size) - 0.5) * 0.20
                ax.scatter(np.full_like(v, x) + jitter, v,
                           color=c, edgecolor="black", linewidth=0.6,
                           s=32, alpha=0.9, zorder=3)

        ax.set_xticks([1, 2])
        ax.set_xticklabels([label_a, label_b], fontsize=TICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE,
                       width=SPINE_WIDTH, length=5)
        ax.tick_params(axis="x", width=SPINE_WIDTH, length=5, pad=6)

        ax.spines[["top", "right"]].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_linewidth(SPINE_WIDTH)
            ax.spines[spine].set_edgecolor("black")

        ax.grid(axis="y", alpha=GRID_ALPHA, linewidth=0.8,
                linestyle="-", color="gray")
        ax.set_axisbelow(True)

        parts = [f"feat {fid}"]
        if m.get("ion_identity"):
            parts.append(str(m["ion_identity"]))
        if m.get("compound_name"):
            parts.append(str(m["compound_name"]))
        ax.set_title(" | ".join(parts),
                     fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold")

    for ax in flat[n:]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=SUPTITLE_FONTSIZE, fontweight="bold")
    fig.tight_layout()
    return fig, {"n_members": n}
