"""Heatmap (clustermap) of a processed feature table.

Wraps ``seaborn.clustermap``. Subset rows via ``features``, z-score by row,
and colour the column bar by a metadata group column.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from metabproject.viz._style import group_palette


def select_high_confidence_structure_features(
    annotations: pd.DataFrame,
    feature_ids: list | None = None,
) -> tuple[list, dict]:
    """Pick features whose merged annotation is a GNPS hit or SIRIUS high-conf match.

    A feature is retained when ``annotation_source`` contains ``"gnps"`` or
    ``sirius_high_confidence`` is ``True``, and ``compound_name`` is non-null.

    Args:
        annotations: Merged annotation table indexed by ``feature_id``.
        feature_ids: Optional list to intersect with (e.g. a contrast subset).

    Returns:
        ``(feature_ids, labels)`` where ``labels`` maps each retained
        ``feature_id`` to its ``compound_name``.
    """
    source = annotations.get("annotation_source")
    if source is None:
        source_mask = pd.Series(False, index=annotations.index)
    else:
        source_mask = source.fillna("").astype(str).str.contains("gnps")

    sirius_high = annotations.get("sirius_high_confidence")
    if sirius_high is None:
        sirius_mask = pd.Series(False, index=annotations.index)
    else:
        sirius_mask = sirius_high.fillna(False).astype(bool)

    name = annotations.get("compound_name")
    if name is None:
        return [], {}
    has_name = name.notna()

    keep = (source_mask | sirius_mask) & has_name
    selected = annotations.index[keep]

    if feature_ids is not None:
        selected = selected.intersection(pd.Index(feature_ids))

    labels = annotations.loc[selected, "compound_name"].to_dict()
    return list(selected), labels


def heatmap_figure(
    feature_table: pd.DataFrame,
    metadata: pd.DataFrame,
    features: list | None = None,
    group_col: str = "group",
    group_col_2: str | None = None,
    palette: dict | None = None,
    z_score: str | None = "rows",
    cluster_rows: bool = True,
    cluster_cols: bool = True,
    cmap: str = "vlag",
    title: str | None = None,
    row_labels: dict | None = None,
) -> tuple[plt.Figure, dict[str, int]]:
    """Build a clustered heatmap of features × samples.

    Args:
        feature_table: Processed feature table indexed by ``feature_id``.
        metadata: Sample metadata indexed by sample name.
        features: Optional list of ``feature_id`` values to subset rows to.
            ``None`` uses the full table.
        group_col: Metadata column used for the (primary) column colour bar.
        group_col_2: Optional second metadata column rendered as an extra
            colour bar above the primary one. Both bars share the same
            ``palette`` mapping (each column looks up its own labels).
        z_score: ``"rows"`` to z-score each feature row, ``None`` to plot
            values as-is.
        cluster_rows: Hierarchical-cluster the feature rows.
        cluster_cols: Hierarchical-cluster the sample columns.
        cmap: Matplotlib colormap.
        title: Optional figure title.
        row_labels: Optional ``feature_id -> label`` mapping. When provided,
            rows are renamed before plotting; missing entries keep their
            ``feature_id``.

    Returns:
        ``(fig, info)`` with ``info = {"n_features": …, "n_samples": …}``.
    """
    samples = [s for s in feature_table.columns if s in metadata.index]
    if not samples:
        raise ValueError(
            "No overlap between feature_table columns and metadata index."
        )

    if features is not None:
        missing = [f for f in features if f not in feature_table.index]
        if missing:
            raise ValueError(
                f"{len(missing)} feature_id(s) not in feature_table "
                f"(first few: {missing[:5]})"
            )
        df = feature_table.loc[list(features), samples]
    else:
        df = feature_table[samples]

    if df.shape[0] == 0:
        raise ValueError("No features to plot.")

    colour_map = group_palette(metadata, group_col=group_col, custom=palette)
    primary = metadata.loc[samples, group_col].map(colour_map)
    primary.name = group_col

    colour_map_2: dict | None = None
    if group_col_2 is not None:
        if group_col_2 not in metadata.columns:
            raise KeyError(
                f"group_col_2 {group_col_2!r} not in metadata "
                f"(have: {metadata.columns.tolist()})"
            )
        colour_map_2 = group_palette(metadata, group_col=group_col_2, custom=palette)
        secondary = metadata.loc[samples, group_col_2].map(colour_map_2)
        secondary.name = group_col_2
        col_colors = pd.concat([secondary, primary], axis=1)
    else:
        col_colors = primary

    z_arg: int | None
    if z_score == "rows":
        z_arg = 0
    elif z_score is None or z_score == "none":
        z_arg = None
    else:
        raise ValueError(f"z_score must be 'rows' or None, got {z_score!r}")

    # Drop zero-variance rows when z-scoring (would produce NaN).
    if z_arg == 0:
        row_std = df.std(axis=1)
        keep = row_std > 0
        if (~keep).any():
            df = df.loc[keep]

    if row_labels:
        df = df.rename(index={k: v for k, v in row_labels.items() if v is not None})

    # clustermap returns a ClusterGrid; .figure is the underlying Figure.
    cg = sns.clustermap(
        df,
        z_score=z_arg,
        row_cluster=cluster_rows,
        col_cluster=cluster_cols,
        cmap=cmap,
        col_colors=col_colors,
        figsize=(max(7, 0.45 * len(samples) + 4), min(14, 0.12 * len(df) + 4)),
        xticklabels=True,
        yticklabels=df.shape[0] <= 60,
        cbar_kws={"label": "z-score" if z_arg == 0 else "value"},
    )
    cg.ax_heatmap.set_xlabel("")
    cg.ax_heatmap.set_ylabel("")

    # Group legend (one patch per group).
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colour_map[g], label=g)
        for g in colour_map
    ]
    leg = cg.ax_col_dendrogram.legend(
        handles=handles, title=group_col, loc="center left",
        bbox_to_anchor=(1.02, 0.5), fontsize=8, title_fontsize=9, frameon=False,
    )
    if colour_map_2 is not None:
        cg.ax_col_dendrogram.add_artist(leg)
        handles2 = [
            plt.Rectangle((0, 0), 1, 1, facecolor=colour_map_2[g], label=g)
            for g in colour_map_2
        ]
        cg.ax_col_dendrogram.legend(
            handles=handles2, title=group_col_2, loc="center left",
            bbox_to_anchor=(1.02, 1.0), fontsize=8, title_fontsize=9, frameon=False,
        )

    if title:
        cg.figure.suptitle(title, fontsize=12, fontweight="bold", y=1.02)

    return cg.figure, {"n_features": int(df.shape[0]), "n_samples": int(df.shape[1])}


def pairwise_count_heatmap(
    count_matrix: pd.DataFrame,
    group_separators: list[int] | None = None,
    figsize: tuple[float, float] = (9, 8),
    cmap: str = "YlOrRd",
    cbar_label: str = "Significant features",
    title: str | None = None,
) -> plt.Figure:
    """Heatmap of a symmetric pairwise count matrix.

    Args:
        count_matrix: Square ``DataFrame`` with identical row/column index
            of cell labels and integer counts. The diagonal is masked
            (self-pairs).
        group_separators: Optional list of index positions at which to
            draw thick black separator lines (e.g. ``[3, 6]`` for three
            blocks of three in a 3 × 3 design).
        figsize: Matplotlib figure size.
        cmap: Colormap name.
        cbar_label: Colorbar label.
        title: Optional figure title.

    Returns:
        The matplotlib Figure.
    """
    if count_matrix.shape[0] != count_matrix.shape[1]:
        raise ValueError(
            f"count_matrix must be square, got shape {count_matrix.shape}"
        )

    mask = np.eye(count_matrix.shape[0], dtype=bool)
    pretty = [str(c).replace("_", " ").title() for c in count_matrix.index]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        count_matrix, mask=mask, annot=True, fmt="d", cmap=cmap,
        cbar_kws={"label": cbar_label},
        linewidths=0.4, linecolor="white",
        xticklabels=pretty, yticklabels=pretty, ax=ax,
    )
    if group_separators:
        for k in group_separators:
            ax.axhline(k, color="black", lw=2.0)
            ax.axvline(k, color="black", lw=2.0)
    ax.set_xticklabels(pretty, rotation=40, ha="right")
    ax.set_yticklabels(pretty, rotation=0)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig
