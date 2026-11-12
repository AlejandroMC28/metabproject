"""Load and clean SIRIUS structure identifications (structural data only)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


# Columns kept from structure_identifications.tsv (before renaming).
_SIRIUS_COLS = [
    "mappingFeatureId",
    "structurePerIdRank",
    "ConfidenceScoreApproximate",
    "InChIkey2D",
    "InChI",
    "name",
    "smiles",
    "molecularFormula",
    "adduct",
]
# Optional columns we parse when present but don't require — older SIRIUS
# exports may omit them. Currently: ``links`` (database hit registry).
_SIRIUS_OPTIONAL_COLS = ["links"]

_RENAME = {
    "mappingFeatureId": "feature_id",
    "ConfidenceScoreApproximate": "sirius_confidence",
    "InChIkey2D": "sirius_InChIkey2D",
    "InChI": "sirius_InChI",
    "name": "sirius_compound_name",
    "smiles": "sirius_smiles",
    "molecularFormula": "sirius_mol_formula",
    "adduct": "sirius_adduct",
}


def _parse_databases(links: object) -> list[str]:
    """Parse a SIRIUS ``links`` cell into the list of database names.

    SIRIUS stores hits as ``DB1:(id1 id2);DB2:(id);DB3:(null)``; we keep
    the bare database name (left of ``:(``), preserving discovery order
    and de-duplicating.
    """
    if not isinstance(links, str) or not links:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for tok in links.split(";"):
        name = tok.split(":(", 1)[0].strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def load_sirius(
    structures_path: str | Path,
    min_confidence: float = 0.64,
) -> pd.DataFrame:
    """Load and clean SIRIUS structure identifications.

    Reads ``structure_identifications.tsv``, keeps only rank-1 predictions,
    sanitises the confidence score (SIRIUS uses ``-inf`` for unscored features),
    computes a ``sirius_high_confidence`` flag, and returns a DataFrame indexed
    by ``feature_id``.

    Taxonomy (NPC/ClassyFire) is **not** loaded here — use
    :func:`load_canopus` for compound-class predictions.

    Args:
        structures_path: Path to ``structure_identifications.tsv``.
        min_confidence: ``ConfidenceScoreApproximate`` threshold for the
            ``sirius_high_confidence`` flag (default 0.64).

    Returns:
        DataFrame indexed by ``feature_id`` with ``sirius_*`` columns.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If required columns are absent.
    """
    structures_path = Path(structures_path)
    if not structures_path.exists():
        raise FileNotFoundError(f"SIRIUS structures file not found: {structures_path}")

    df = pd.read_csv(structures_path, sep="\t", quoting=csv.QUOTE_NONE)
    df.columns = df.columns.str.removeprefix("SIRIUS_")  # newer SIRIUS exports prefix every column

    missing = [c for c in _SIRIUS_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {structures_path.name}: {missing}\n"
            f"  Found: {df.columns.tolist()}"
        )

    # Keep rank-1 predictions only.
    keep_cols = _SIRIUS_COLS + [c for c in _SIRIUS_OPTIONAL_COLS if c in df.columns]
    df = df[df["structurePerIdRank"] == 1][keep_cols].copy()
    df = df.drop(columns=["structurePerIdRank"])
    df = df.rename(columns=_RENAME)

    df["feature_id"] = df["feature_id"].astype(int)
    df = df.set_index("feature_id")

    # SIRIUS uses -inf for unscored features; treat any non-finite value as missing.
    df["sirius_confidence"] = pd.to_numeric(df["sirius_confidence"], errors="coerce")
    df.loc[~np.isfinite(df["sirius_confidence"].astype(float)), "sirius_confidence"] = pd.NA

    df["sirius_high_confidence"] = (
        df["sirius_confidence"].notna() & (df["sirius_confidence"] >= min_confidence)
    )

    if "links" in df.columns:
        df["sirius_databases"] = df["links"].map(_parse_databases)
        df = df.drop(columns=["links"])

    return df


def compute_gnps_sirius_topk_match(
    gnps_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    min_confidence: float = 0.64,
) -> pd.DataFrame:
    """Cross-check GNPS library InChIKey 2D against SIRIUS top-k structures.

    For every feature with a GNPS library match, looks up that feature's
    SIRIUS top-k candidates and finds the lowest rank whose
    ``sirius_InChIkey2D`` equals the GNPS planar key.

    Args:
        gnps_df: Output of :func:`load_gnps_library`, indexed by ``feature_id``.
            Must contain ``gnps_InChIkey2D``.
        topk_df: Output of :func:`load_sirius_topk`, indexed by ``feature_id``
            (non-unique). Must contain ``structurePerIdRank``,
            ``sirius_InChIkey2D``, and ``sirius_confidence``.
        min_confidence: Threshold for ``gnps_sirius_highconf`` — the matched
            candidate's ``sirius_confidence`` must be ≥ this value.

    Returns:
        DataFrame indexed by ``feature_id`` (one row per GNPS-hit feature with
        a non-null InChIkey2D) with columns:

        - ``gnps_sirius_rank`` (Int64): literal rank of the match (1..top_k),
          ``0`` if no candidate matches, ``<NA>`` if the feature has no
          SIRIUS top-k rows at all.
        - ``gnps_sirius_highconf`` (bool): True iff the matched candidate's
          ``sirius_confidence`` is ≥ ``min_confidence``.
    """
    empty = pd.DataFrame(
        {
            "gnps_sirius_rank": pd.Series(dtype="Int64"),
            "gnps_sirius_highconf": pd.Series(dtype=bool),
        }
    )
    empty.index.name = "feature_id"

    if "gnps_InChIkey2D" not in gnps_df.columns:
        return empty

    gnps_keys = gnps_df["gnps_InChIkey2D"].dropna()
    if len(gnps_keys) == 0:
        return empty

    ranks: dict = {}
    highconf: dict = {}

    for fid, gnps_key in gnps_keys.items():
        if fid not in topk_df.index:
            ranks[fid] = pd.NA
            highconf[fid] = False
            continue

        rows = topk_df.loc[[fid]]
        matches = rows[rows["sirius_InChIkey2D"] == gnps_key]
        if matches.empty:
            ranks[fid] = 0
            highconf[fid] = False
            continue

        best = matches.sort_values("structurePerIdRank").iloc[0]
        ranks[fid] = int(best["structurePerIdRank"])
        conf = best["sirius_confidence"]
        highconf[fid] = bool(pd.notna(conf) and float(conf) >= min_confidence)

    out = pd.DataFrame(
        {
            "gnps_sirius_rank": pd.Series(ranks, dtype="Int64"),
            "gnps_sirius_highconf": pd.Series(highconf, dtype=bool),
        }
    )
    out.index.name = "feature_id"
    return out


def load_sirius_topk(
    structures_path: str | Path,
    top_k: int = 5,
) -> pd.DataFrame:
    """Load SIRIUS structure identifications for the top-k ranked candidates.

    Unlike :func:`load_sirius`, this returns multiple rows per feature (one per
    rank up to ``top_k``). The result has a non-unique ``feature_id`` index and
    is intended for cross-adduct annotation comparison, not for the merge
    priority chain.

    Args:
        structures_path: Path to ``structure_identifications.tsv``.
        top_k: Maximum rank to retain (inclusive, default 5).

    Returns:
        DataFrame indexed by ``feature_id`` (non-unique) with columns:
        ``structurePerIdRank``, ``sirius_confidence``, ``sirius_InChIkey2D``,
        ``sirius_compound_name``, ``sirius_smiles``, ``sirius_mol_formula``.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If required columns are absent.
    """
    structures_path = Path(structures_path)
    if not structures_path.exists():
        raise FileNotFoundError(f"SIRIUS structures file not found: {structures_path}")

    df = pd.read_csv(structures_path, sep="\t", quoting=csv.QUOTE_NONE)
    df.columns = df.columns.str.removeprefix("SIRIUS_")  # newer SIRIUS exports prefix every column

    missing = [c for c in _SIRIUS_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {structures_path.name}: {missing}\n"
            f"  Found: {df.columns.tolist()}"
        )

    df = df[df["structurePerIdRank"] <= top_k][_SIRIUS_COLS].copy()
    df = df.rename(columns=_RENAME)

    df.index = pd.Index(df["feature_id"].astype(int), name="feature_id")
    df = df.drop(columns=["feature_id"])

    df["sirius_confidence"] = pd.to_numeric(df["sirius_confidence"], errors="coerce")
    df.loc[~np.isfinite(df["sirius_confidence"].astype(float)), "sirius_confidence"] = pd.NA

    return df
