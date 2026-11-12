"""RDKit SMILES → SVG rendering for the feature viewer.

RDKit is an optional dependency. If not installed, returns an empty dict and
emits a warning.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass


def render_structures(
    smiles_series: pd.Series,
    size: tuple[int, int] = (250, 200),
) -> dict[str, str]:
    """Render unique SMILES strings to SVG using RDKit.

    Args:
        smiles_series: Series of SMILES strings (may contain nulls/duplicates).
        size: (width, height) in pixels for the rendered SVG.

    Returns:
        Dict mapping SMILES string to SVG string. Invalid or null SMILES are
        omitted. Returns empty dict if RDKit is not installed.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        warnings.warn(
            "RDKit not installed — 2D structure rendering disabled. "
            "Install with: conda install -c conda-forge rdkit",
            UserWarning,
            stacklevel=2,
        )
        return {}

    unique_smiles = set(smiles_series.dropna().unique())
    svg_map: dict[str, str] = {}
    w, h = size

    for smi in unique_smiles:
        if not smi or not isinstance(smi, str):
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
            drawer.drawOptions().addStereoAnnotation = True
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            svg_map[smi] = svg
        except Exception:
            continue

    return svg_map
