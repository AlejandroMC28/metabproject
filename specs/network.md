# `metab network` Specification

## Overview

Ingests a **GNPS Feature-Based Molecular Networking (FBMN)** edge list into a
project. FBMN links features whose MS/MS spectra are similar (a molecular
"family"), giving the drawable pairwise edges — with cosine similarity — that
underpin the explorer's Network tab.

```
metab network PROJECT --pairs filtered_pairs.tsv [--min-cosine 0.0] [--keep-singletons] [--force]
```

Writes to `<project>/network/`. Appends one manifest entry per run. Idempotent:
re-running errors unless `--force`.

> **Not the same as MZmine networking.** `ms1_features.parquet` may carry
> `molecular_networking:net_cluster_id` from MZmine's own GraphML export — that
> column is *cluster membership only*, with no pairwise edges. This command's
> sole source is the GNPS `ComponentIndex` / cosine edge list; the two networks
> are independent and must not be conflated.

---

## Input — `filtered_pairs.tsv`

Standard GNPS FBMN pairs table. Required columns:

| Column | Meaning |
|--------|---------|
| `CLUSTERID1` | Feature id of one endpoint (`CLUSTERID == feature_id`) |
| `CLUSTERID2` | Feature id of the other endpoint |
| `ComponentIndex` | Molecular-family id (`-1` = GNPS singleton) |
| `Cosine` | Modified-cosine spectral similarity of the edge |
| `DeltaMZ` | Precursor m/z difference across the edge |

Missing any required column raises `ValueError` (listing the columns found); a
missing file raises `FileNotFoundError`.

## Loader — `metabproject.network.gnps_pairs`

```python
load_gnps_pairs(path, min_cosine=0.0, drop_singletons=True) -> pd.DataFrame
build_node_membership(edges) -> pd.DataFrame
```

`load_gnps_pairs` returns one row per **undirected, deduplicated** edge:

| Column | Type | Notes |
|--------|------|-------|
| `feature_id_1` | int | Ordered so `feature_id_1 < feature_id_2` |
| `feature_id_2` | int | |
| `component_id` | int | From `ComponentIndex` |
| `cosine` | float | |
| `delta_mz` | float | |

Rules:
- **Rename** GNPS columns to the schema above.
- **Self-loops** (`feature_id_1 == feature_id_2`) dropped.
- **Singletons** (`component_id == -1`) dropped unless `drop_singletons=False`
  (a giant pseudo-component would otherwise swamp the view).
- **`min_cosine`** — edges with `cosine` below the threshold dropped.
- **Undirected dedup** — each pair ordered low/high, then sorted by cosine
  descending and de-duplicated keeping the **maximum** cosine for any multi-edge.

`build_node_membership` melts the two id columns into a node table indexed by
`feature_id`:

| Column | Type | Notes |
|--------|------|-------|
| `component_id` | int | The node's family |
| `degree` | int | Number of incident edges |
| `n_component_members` | int | Nodes in the node's family |

## Storage — `<project>/network/`

- `edges.parquet` — the `load_gnps_pairs` output.
- `nodes.parquet` — the `build_node_membership` output.

On ingest, endpoints are cross-checked against `ms1_features.index`; edges whose
endpoints are absent from the project are dropped and counted.

## Manifest entry

```json
{
  "command": "network",
  "parameters": {"pairs": "...", "min_cosine": 0.0, "drop_singletons": true},
  "results": {
    "n_edges": 2995,
    "n_nodes": 1720,
    "n_components": 299,
    "n_features_matched": 1720,
    "n_features_unmatched": 0
  }
}
```

## Python API

```python
project.import_network(pairs, min_cosine=0.0, drop_singletons=True, force=False)
project.network_edges   # DataFrame, or None if no network ingested
project.network_nodes   # DataFrame, or None if no network ingested
```

Both properties return `None` when `network/` is absent — this is what lets
`metab explore` skip the Network tab cleanly on projects without a network.

## Explorer integration

`metab explore` reads the stored network and emits `network_components` in the
viewer payload (per family: node ids, compact `[id1, id2, cosine]` edge triples,
and a light member table) plus a per-feature `network` membership block. The
Network tab draws the selected family as a node-link graph and, on node click,
shows that feature's per-treatment abundance boxplots. See `specs/explore.md`.
