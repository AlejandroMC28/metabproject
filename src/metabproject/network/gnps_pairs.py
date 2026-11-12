"""Load and clean a GNPS FBMN molecular-network edge list (filtered_pairs)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# Columns kept from the GNPS FBMN filtered_pairs TSV (before renaming).
_PAIRS_COLS = ["CLUSTERID1", "CLUSTERID2", "ComponentIndex", "Cosine", "DeltaMZ"]

# Map raw GNPS column names → clean names.
_PAIRS_RENAME = {
    "CLUSTERID1": "feature_id_1",
    "CLUSTERID2": "feature_id_2",
    "ComponentIndex": "component_id",
    "Cosine": "cosine",
    "DeltaMZ": "delta_mz",
}


def load_gnps_pairs(
    path: str | Path,
    min_cosine: float = 0.0,
    drop_singletons: bool = True,
) -> pd.DataFrame:
    """Load a GNPS FBMN edge list, undirected and deduplicated.

    Reads a GNPS molecular-networking ``filtered_pairs.tsv`` (CLUSTERID ==
    feature id), renames columns, drops self-loops and singletons, applies a
    cosine threshold, and collapses each undirected edge to a single row
    (``feature_id_1 < feature_id_2``, keeping the max-cosine copy of any
    duplicate pair).

    Args:
        path: Path to the GNPS filtered_pairs TSV.
        min_cosine: Drop edges with ``cosine`` below this threshold.
        drop_singletons: Drop edges with ``component_id == -1`` (GNPS marks
            unclustered nodes ``-1``; keeping them collapses into one giant
            pseudo-component).

    Returns:
        Edge DataFrame with columns ``feature_id_1`` (int), ``feature_id_2``
        (int), ``component_id`` (int), ``cosine`` (float), ``delta_mz`` (float),
        one row per undirected edge.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If required columns are absent.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GNPS pairs file not found: {path}")

    df = pd.read_csv(path, sep="\t")

    missing = [c for c in _PAIRS_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {path.name}: {missing}\n"
            f"  Found: {df.columns.tolist()}"
        )

    df = df[_PAIRS_COLS].rename(columns=_PAIRS_RENAME)
    df["feature_id_1"] = df["feature_id_1"].astype(int)
    df["feature_id_2"] = df["feature_id_2"].astype(int)
    df["component_id"] = df["component_id"].astype(int)
    df["cosine"] = df["cosine"].astype(float)
    df["delta_mz"] = df["delta_mz"].astype(float)

    # Drop self-loops.
    df = df[df["feature_id_1"] != df["feature_id_2"]]

    if drop_singletons:
        df = df[df["component_id"] != -1]

    if min_cosine > 0.0:
        df = df[df["cosine"] >= min_cosine]

    # Undirected dedup: order each pair (lo, hi), keep the max-cosine copy.
    lo = df[["feature_id_1", "feature_id_2"]].min(axis=1)
    hi = df[["feature_id_1", "feature_id_2"]].max(axis=1)
    df = df.assign(feature_id_1=lo, feature_id_2=hi)
    df = df.sort_values("cosine", ascending=False)
    df = df.drop_duplicates(subset=["feature_id_1", "feature_id_2"], keep="first")

    return df.reset_index(drop=True)


def build_node_membership(edges: pd.DataFrame) -> pd.DataFrame:
    """Derive a per-node membership table from an edge list.

    Every node in an edge shares that edge's ``component_id`` (a GNPS
    invariant), so membership is a melt of the two id columns.

    Args:
        edges: Edge DataFrame from :func:`load_gnps_pairs`.

    Returns:
        DataFrame indexed by ``feature_id`` with columns ``component_id``
        (int), ``degree`` (int, edges touching the node), and
        ``n_component_members`` (int, unique nodes in the node's component).
    """
    long = pd.concat(
        [
            edges[["feature_id_1", "component_id"]].rename(
                columns={"feature_id_1": "feature_id"}
            ),
            edges[["feature_id_2", "component_id"]].rename(
                columns={"feature_id_2": "feature_id"}
            ),
        ],
        ignore_index=True,
    )

    degree = long.groupby("feature_id").size()
    component_id = long.groupby("feature_id")["component_id"].first()
    comp_size = long.groupby("component_id")["feature_id"].nunique()

    nodes = pd.DataFrame({"component_id": component_id, "degree": degree})
    nodes["n_component_members"] = nodes["component_id"].map(comp_size)
    nodes.index.name = "feature_id"
    return nodes
