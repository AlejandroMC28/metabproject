"""Shared styling utilities for the visualization module.

Two helpers used by every figure: a consistent group→color palette and a
PNG+SVG saver. The palette matches the ``Set2`` palette used by
:func:`metabproject.viz.qc.make_processing_qc_figure` so all figures in a
project share the same group colors.
"""

from __future__ import annotations

from pathlib import Path

import yaml

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless/test environments
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def group_palette(
    metadata: pd.DataFrame,
    group_col: str = "group",
    custom: dict | None = None,
) -> dict:
    """Build a ``{group_name: color}`` mapping from a metadata DataFrame.

    Args:
        metadata: Sample metadata containing ``group_col``.
        group_col: Column to read group labels from.
        custom: Optional user-supplied dict overriding (or fully replacing)
            the default Set2 palette. Keys present in ``custom`` use the
            supplied colour; remaining unique values in ``metadata[group_col]``
            fall back to Set2. Colour values can be any matplotlib-compatible
            specification (hex string, RGB tuple, named colour).

    Returns:
        Dict mapping each unique value in ``metadata[group_col]`` to a colour,
        in the order they first appear.

    Raises:
        KeyError: If ``group_col`` is not in ``metadata``.
    """
    if group_col not in metadata.columns:
        raise KeyError(
            f"group column {group_col!r} not in metadata "
            f"(have: {metadata.columns.tolist()})"
        )
    unique = metadata[group_col].unique().tolist()
    fallback = dict(zip(unique, sns.color_palette("Set2", n_colors=len(unique))))
    if custom:
        # Only honour custom entries that match labels in this group column.
        # ``custom`` is often a project-wide palette covering multiple group
        # types (e.g. variety + treatment loaded from viz.yaml); applying it
        # unconditionally would pollute the legend with foreign labels.
        unique_str = {str(u): u for u in unique}
        for k, v in custom.items():
            if k in fallback:
                fallback[k] = v
            elif str(k) in unique_str:
                fallback[unique_str[str(k)]] = v
    return fallback


def parse_palette_string(spec: str) -> dict[str, str]:
    """Parse a ``"k1=#hex,k2=#hex"`` palette string into a dict.

    Args:
        spec: Comma-separated ``key=colour`` pairs, e.g.
            ``"red=#C0392B,cream=#E8C999,brown=#6D4C41"``.

    Returns:
        Dict mapping group label to colour string.

    Raises:
        ValueError: If a pair is malformed.
    """
    result: dict[str, str] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(
                f"Malformed palette entry {pair!r} (expected 'key=color')"
            )
        k, v = pair.split("=", 1)
        result[k.strip()] = v.strip()
    return result


def _read_viz_yaml(project_root: str | Path) -> dict:
    """Return the parsed top-level ``viz.yaml`` mapping, or ``{}`` if absent."""
    path = Path(project_root) / "viz.yaml"
    if not path.exists():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed YAML in {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must be a YAML mapping at the top level")
    return loaded


def load_palette_yaml(project_root: str | Path) -> dict[str, str]:
    """Load a ``{group: colour}`` palette from ``<project_root>/viz.yaml``.

    The file is optional. When present, it may contain a top-level
    ``palette:`` mapping of group label to colour string (any
    matplotlib-compatible value). Other top-level keys are ignored so
    ``viz.yaml`` can carry additional viz config (e.g. ``markers:``).

    Returns:
        Palette dict, or ``{}`` if the file is missing or has no
        ``palette:`` block.

    Raises:
        ValueError: If ``viz.yaml`` is malformed or ``palette:`` is not a
            mapping.
    """
    loaded = _read_viz_yaml(project_root)
    palette = loaded.get("palette")
    if palette is None:
        return {}
    if not isinstance(palette, dict):
        raise ValueError(
            f"{Path(project_root) / 'viz.yaml'}: 'palette' must be a mapping "
            f"of group to colour"
        )
    return {str(k): str(v) for k, v in palette.items()}


def load_markers_yaml(project_root: str | Path) -> dict[str, str]:
    """Load a ``{label: matplotlib_marker}`` mapping from ``viz.yaml``.

    Used by ``metab viz pca`` together with ``--marker-col`` to pin marker
    shapes to specific labels (e.g. ``control: o`` always a circle). Labels
    missing from the mapping fall back to the default marker cycle in
    ``viz/pca.py``.

    Returns:
        Marker dict, or ``{}`` if the file is missing or has no
        ``markers:`` block.

    Raises:
        ValueError: If ``markers:`` is present but not a mapping.
    """
    loaded = _read_viz_yaml(project_root)
    markers = loaded.get("markers")
    if markers is None:
        return {}
    if not isinstance(markers, dict):
        raise ValueError(
            f"{Path(project_root) / 'viz.yaml'}: 'markers' must be a mapping "
            f"of label to matplotlib marker"
        )
    return {str(k): str(v) for k, v in markers.items()}


def save_figure(
    fig: plt.Figure,
    out_dir: str | Path,
    basename: str,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Save a figure as both PNG and SVG.

    Args:
        fig: Matplotlib figure to save.
        out_dir: Destination directory (created if missing).
        basename: Filename without extension (e.g. ``"pca"`` writes
            ``pca.png`` and ``pca.svg``).
        dpi: Resolution for the PNG.

    Returns:
        ``(png_path, svg_path)`` — both written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{basename}.png"
    svg_path = out_dir / f"{basename}.svg"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    return png_path, svg_path
