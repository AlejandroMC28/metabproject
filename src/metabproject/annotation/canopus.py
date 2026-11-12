"""Load and clean CANOPUS compound-class predictions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# Required columns: present in every supported CANOPUS export.
_CANOPUS_REQUIRED_COLS = [
    "mappingFeatureId",
    "NPC#pathway",
    "NPC#pathway Probability",
    "NPC#superclass",
    "NPC#superclass Probability",
    "NPC#class",
    "NPC#class Probability",
    "ClassyFire#most specific class",
    "ClassyFire#most specific class Probability",
]

# Optional ClassyFire hierarchy levels — only emitted by newer SIRIUS/CANOPUS
# versions. Loaded when present, silently skipped when absent.
_CANOPUS_OPTIONAL_COLS = [
    "ClassyFire#subclass",
    "ClassyFire#subclass Probability",
    "ClassyFire#level 5",
    "ClassyFire#level 5 Probability",
]

_RENAME = {
    "mappingFeatureId": "feature_id",
    "NPC#pathway": "canopus_npc_pathway",
    "NPC#pathway Probability": "canopus_npc_pathway_probability",
    "NPC#superclass": "canopus_npc_superclass",
    "NPC#superclass Probability": "canopus_npc_superclass_probability",
    "NPC#class": "canopus_npc_class",
    "NPC#class Probability": "canopus_npc_class_probability",
    "ClassyFire#most specific class": "canopus_classyfire_class",
    "ClassyFire#most specific class Probability": "canopus_classyfire_class_probability",
    "ClassyFire#subclass": "canopus_classyfire_subclass",
    "ClassyFire#subclass Probability": "canopus_classyfire_subclass_probability",
    "ClassyFire#level 5": "canopus_classyfire_level5",
    "ClassyFire#level 5 Probability": "canopus_classyfire_level5_probability",
}

_REQUIRED_LEVELS = [
    "npc_pathway", "npc_superclass", "npc_class", "classyfire_class",
]
_OPTIONAL_LEVELS = ["classyfire_subclass", "classyfire_level5"]


def load_canopus(
    canopus_path: str | Path,
    min_probability: float = 0.7,
) -> pd.DataFrame:
    """Load and clean CANOPUS compound-class predictions.

    Reads ``canopus_structure_summary.tsv``, renames columns to ``canopus_*``,
    and computes per-level ``*_passed`` boolean flags (True iff the class
    probability is >= ``min_probability``). Raw taxonomy values are always
    stored so users can inspect below-threshold predictions directly from
    ``canopus.parquet``.

    Args:
        canopus_path: Path to ``canopus_structure_summary.tsv``.
        min_probability: Minimum class probability for a level to be
            considered a valid annotation (default 0.7).

    Returns:
        DataFrame indexed by ``feature_id`` with ``canopus_*`` columns.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If required columns are absent.
    """
    canopus_path = Path(canopus_path)
    if not canopus_path.exists():
        raise FileNotFoundError(f"CANOPUS file not found: {canopus_path}")

    df = pd.read_csv(canopus_path, sep="\t")
    df.columns = df.columns.str.removeprefix("SIRIUS_")  # newer SIRIUS exports prefix every column

    missing = [c for c in _CANOPUS_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {canopus_path.name}: {missing}\n"
            f"  Found: {df.columns.tolist()}"
        )

    # SIRIUS emits one CANOPUS row per molecular-formula candidate (formulaRank
    # 1..k). Without this filter the merged annotations table duplicates every
    # SIRIUS/GNPS hit across the k formula candidates, inflating per-feature
    # counts. Keep the rank-1 formula only — matches `load_sirius` semantics.
    if "formulaRank" in df.columns:
        df = df[df["formulaRank"] == 1]

    keep_cols = list(_CANOPUS_REQUIRED_COLS) + [
        c for c in _CANOPUS_OPTIONAL_COLS if c in df.columns
    ]
    df = df[keep_cols].copy()
    df = df.rename(columns=_RENAME)

    df["feature_id"] = df["feature_id"].astype(int)
    df = df.set_index("feature_id")

    # Compute per-level passed flags (only for levels whose probability
    # column was actually present).
    levels = list(_REQUIRED_LEVELS) + [
        lvl for lvl in _OPTIONAL_LEVELS
        if f"canopus_{lvl}_probability" in df.columns
    ]
    for level in levels:
        prob_col = f"canopus_{level}_probability"
        passed_col = f"canopus_{level}_passed"
        df[passed_col] = df[prob_col].notna() & (df[prob_col] >= min_probability)

    return df
