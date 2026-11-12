"""Volcano plots — single contrast and multi-panel grids.

Takes a per-contrast stats DataFrame (columns ``log2fc, p_adj``) and renders
``log2fc`` (x) vs ``-log10(p_adj)`` (y), with threshold lines and
per-direction colouring. Features in ``annotated_features`` are drawn with
a thin black edge so structurally annotated hits stand out against the
unannotated background.

``tukey_pair_to_contrast`` adapts a Tukey post-hoc long-format frame to the
``[log2fc, p_adj]`` schema, so multivariate (2-way ANOVA + Tukey) outputs
can be fed straight into the existing single- and multi-panel volcanoes.
"""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Up = enriched in group B (positive log2fc).
_COLORS = {"up": "#d6604d", "down": "#4393c3", "ns": "#bdbdbd"}
_ANN_EDGE_COLOR = "#000000"
_ANN_EDGE_WIDTH = 0.7


def _render_volcano_ax(
    ax: plt.Axes,
    contrast_df: pd.DataFrame,
    fc_threshold: float,
    q_threshold: float,
    label_top_n: int,
    annotated_features: set | None,
    label_lookup: pd.Series | None = None,
) -> dict[str, int]:
    """Render a volcano onto a single axis. Returns count dict."""
    for col in ("log2fc", "p_adj"):
        if col not in contrast_df.columns:
            raise ValueError(f"contrast_df missing required column {col!r}")

    df = contrast_df.dropna(subset=["log2fc", "p_adj"]).copy()

    q = df["p_adj"].to_numpy(dtype=float)
    pos = q[q > 0]
    floor = float(pos.min()) if pos.size else 1e-300
    q_clipped = np.where(q <= 0, floor, q)
    df["neg_log10_q"] = -np.log10(q_clipped)

    up = (df["log2fc"] >= fc_threshold) & (df["p_adj"] < q_threshold)
    down = (df["log2fc"] <= -fc_threshold) & (df["p_adj"] < q_threshold)
    ns = ~(up | down)
    df["plot_direction"] = np.where(up, "up", np.where(down, "down", "ns"))

    if annotated_features is None:
        ann_set: set = set()
    else:
        ann_set = set(annotated_features)
    df["is_annotated"] = df.index.isin(ann_set)

    for label in ("ns", "up", "down"):
        sub = df[df["plot_direction"] == label]
        plain = sub[~sub["is_annotated"]]
        anno = sub[sub["is_annotated"]]
        if not plain.empty:
            ax.scatter(
                plain["log2fc"], plain["neg_log10_q"],
                c=_COLORS[label], s=14, alpha=0.7 if label == "ns" else 0.85,
                edgecolor="none",
            )
        if not anno.empty:
            ax.scatter(
                anno["log2fc"], anno["neg_log10_q"],
                c=_COLORS[label], s=22, alpha=0.95,
                edgecolor=_ANN_EDGE_COLOR, linewidths=_ANN_EDGE_WIDTH,
            )

    ax.axhline(-np.log10(q_threshold), color="grey", linestyle="--", linewidth=0.8)
    ax.axvline(fc_threshold, color="grey", linestyle="--", linewidth=0.8)
    ax.axvline(-fc_threshold, color="grey", linestyle="--", linewidth=0.8)

    ax.set_xlabel("log$_2$ fold change")
    ax.set_ylabel(r"$-\log_{10}$(adjusted p)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if label_top_n > 0:
        sig_df = df[df["plot_direction"] != "ns"].copy()
        if not sig_df.empty:
            sig_df["_score"] = sig_df["log2fc"].abs() * sig_df["neg_log10_q"]
            top = sig_df.nlargest(label_top_n, "_score")
            for fid, row in top.iterrows():
                if label_lookup is not None and fid in label_lookup.index:
                    text = str(label_lookup.loc[fid])
                    if len(text) > 28:
                        text = text[:25] + "..."
                else:
                    text = str(fid)
                ax.annotate(
                    text, (row["log2fc"], row["neg_log10_q"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points",
                )

    counts = {"up": int(up.sum()), "down": int(down.sum()), "ns": int(ns.sum())}
    ax.text(
        0.98, 0.95,
        f"up: {counts['up']}   down: {counts['down']}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#444444",
    )
    return counts


def volcano_figure(
    contrast_df: pd.DataFrame,
    fc_threshold: float = 1.0,
    q_threshold: float = 0.05,
    title: str | None = None,
    label_top_n: int = 0,
    annotated_features: Iterable | None = None,
    label_lookup: pd.Series | None = None,
) -> tuple[plt.Figure, dict[str, int]]:
    """Build a single-panel volcano.

    Args:
        contrast_df: Stats DataFrame for a single contrast. Must have
            ``log2fc`` and ``p_adj`` columns; index is the feature_id.
        fc_threshold: |log2FC| cutoff (vertical guideline + up/down split).
        q_threshold: Adjusted-p cutoff (horizontal guideline).
        title: Optional figure title.
        label_top_n: Number of top-significant features to label. 0 disables.
        annotated_features: Optional iterable of ``feature_id`` to outline
            with a thin black edge.
        label_lookup: Optional ``Series`` mapping ``feature_id`` → label text
            (e.g. compound name). Used for the ``label_top_n`` annotations
            when present; falls back to the feature_id itself.

    Returns:
        ``(fig, counts)`` with ``counts = {"up": …, "down": …, "ns": …}``.
    """
    ann_set = set(annotated_features) if annotated_features is not None else None
    fig, ax = plt.subplots(figsize=(7, 6))
    counts = _render_volcano_ax(
        ax, contrast_df, fc_threshold, q_threshold,
        label_top_n=label_top_n,
        annotated_features=ann_set,
        label_lookup=label_lookup,
    )
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig, counts


def volcano_grid(
    panels: list[dict],
    fc_threshold: float = 1.0,
    q_threshold: float = 0.05,
    label_top_n: int = 0,
    annotated_features: Iterable | None = None,
    label_lookup: pd.Series | None = None,
    ncols: int = 3,
    panel_size: tuple[float, float] = (5.0, 4.5),
    suptitle: str | None = None,
    row_labels: list[str] | None = None,
) -> tuple[plt.Figure, list[dict[str, int]]]:
    """Build a multi-panel volcano grid.

    Args:
        panels: List of dicts with keys:

            * ``"title"`` (str) — panel title.
            * ``"contrast_df"`` (DataFrame) — same schema as
              :func:`volcano_figure` input.
        fc_threshold, q_threshold, label_top_n, annotated_features,
        label_lookup: Forwarded to each panel.
        ncols: Number of columns in the grid (rows are inferred).
        panel_size: ``(width, height)`` per panel in inches.
        suptitle: Optional figure-level title.
        row_labels: Optional ``len(rows)``-long list of left-margin row
            labels.

    Returns:
        ``(fig, counts)`` where ``counts`` is one count-dict per panel.
    """
    if not panels:
        raise ValueError("panels must be non-empty")
    n = len(panels)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        squeeze=False,
    )
    ann_set = set(annotated_features) if annotated_features is not None else None

    counts_list: list[dict[str, int]] = []
    for idx, panel in enumerate(panels):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        counts = _render_volcano_ax(
            ax, panel["contrast_df"], fc_threshold, q_threshold,
            label_top_n=label_top_n,
            annotated_features=ann_set,
            label_lookup=label_lookup,
        )
        ax.set_title(panel.get("title", ""), fontsize=10)
        counts_list.append(counts)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    if row_labels is not None:
        if len(row_labels) != nrows:
            raise ValueError(
                f"row_labels has {len(row_labels)} entries, expected {nrows}"
            )
        fig.subplots_adjust(left=0.08)
        for r, lbl in enumerate(row_labels):
            pos = axes[r, 0].get_position()
            y = (pos.y0 + pos.y1) / 2
            fig.text(
                0.02, y, lbl, ha="left", va="center",
                rotation=90, fontsize=11, fontweight="bold",
            )

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")

    fig.tight_layout(rect=(0.04, 0.0, 1.0, 0.97) if row_labels else None)
    return fig, counts_list


def tukey_pair_to_contrast(
    posthoc: pd.DataFrame,
    factor: str,
    group_A: str,
    group_B: str,
) -> pd.DataFrame:
    """Adapt one Tukey pair to a volcano-ready ``[log2fc, p_adj]`` frame.

    The Tukey output uses canonical ``group_A < group_B`` ordering with
    ``diff = mean_B - mean_A``. When the caller's desired direction does not
    match, the sign of ``diff`` is flipped so the returned ``log2fc`` reads
    as "B vs A" in the caller's terms.

    Args:
        posthoc: Long-format Tukey DataFrame from :func:`run_comparison`.
        factor: Value to match in the ``factor`` column.
        group_A: Reference group (denominator) for the desired ``log2fc``.
        group_B: Numerator — positive ``log2fc`` means up in B.

    Returns:
        DataFrame indexed by ``feature`` with columns ``log2fc`` and
        ``p_adj``. Empty if no matching rows are found.
    """
    sub = posthoc[posthoc["factor"] == factor]
    pair_mask = (
        ((sub["group_A"] == group_A) & (sub["group_B"] == group_B))
        | ((sub["group_A"] == group_B) & (sub["group_B"] == group_A))
    )
    rows = sub[pair_mask].copy()
    if rows.empty:
        return pd.DataFrame(columns=["log2fc", "p_adj"])

    swap = rows["group_A"].to_numpy() != group_A
    diff = rows["diff"].to_numpy(dtype=float)
    log2fc = np.where(swap, -diff, diff)

    return pd.DataFrame(
        {"log2fc": log2fc, "p_adj": rows["p_value"].to_numpy(dtype=float)},
        index=pd.Index(rows["feature"].to_numpy(), name="feature_id"),
    )
