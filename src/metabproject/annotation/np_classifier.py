"""Classify SMILES via the GNPS2 NP Classifier web API, with a parquet cache.

Used by ``metab annotate --classify-sirius-structures`` to derive NPC
(NaturalProductClassifier) pathway/superclass/class labels directly from the
rank-1 SIRIUS structure SMILES — bypassing CANOPUS's fingerprint-derived
predictions for high-confidence structures.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd


NP_CLASSIFIER_URL = "https://npclassifier.gnps2.org/classify"

CACHE_COLUMNS = [
    "InChIkey2D",
    "smiles",
    "npc_pathway",
    "npc_superclass",
    "npc_class",
    "queried_at",
]


def _first_or_none(value) -> str | None:
    """Pick the first non-empty entry from an NP Classifier result list."""
    if value is None:
        return None
    if isinstance(value, list):
        for item in value:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                return s
        return None
    s = str(value).strip()
    return s or None


def classify_smiles(
    smiles: str,
    *,
    client: httpx.Client | None = None,
    timeout: float = 10.0,
) -> dict | None:
    """Classify one SMILES via the GNPS2 NP Classifier service.

    Args:
        smiles: A SMILES string.
        client: Optional pre-built ``httpx.Client`` for connection reuse. When
            ``None``, a one-shot client is created.
        timeout: Per-request timeout in seconds.

    Returns:
        ``{"npc_pathway": ..., "npc_superclass": ..., "npc_class": ...}`` on
        success; ``None`` on any failure (network, non-200, malformed JSON,
        unclassifiable SMILES). The caller falls back to CANOPUS for that row.
    """
    if not smiles or not str(smiles).strip():
        return None
    params = {"smiles": str(smiles).strip()}
    try:
        if client is None:
            response = httpx.get(NP_CLASSIFIER_URL, params=params, timeout=timeout)
        else:
            response = client.get(NP_CLASSIFIER_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return None

    if not isinstance(payload, dict):
        return None

    return {
        "npc_pathway": _first_or_none(payload.get("pathway_results")),
        "npc_superclass": _first_or_none(payload.get("superclass_results")),
        "npc_class": _first_or_none(payload.get("class_results")),
    }


def load_cache(path: str | Path) -> pd.DataFrame:
    """Load the NP Classifier cache parquet, or return an empty frame."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=CACHE_COLUMNS).set_index("InChIkey2D")
    df = pd.read_parquet(path)
    if df.index.name != "InChIkey2D":
        if "InChIkey2D" in df.columns:
            df = df.set_index("InChIkey2D")
        else:
            return pd.DataFrame(columns=CACHE_COLUMNS).set_index("InChIkey2D")
    return df


def save_cache(cache: pd.DataFrame, path: str | Path) -> None:
    """Write the cache parquet, creating the parent directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache.to_parquet(path)


def classify_features(
    keys_df: pd.DataFrame,
    cache_path: str | Path | None = None,
    *,
    request_delay: float = 0.1,
    timeout: float = 10.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Classify a set of features via NP Classifier with InChIKey-2D caching.

    Each unique ``InChIkey2D`` in ``keys_df`` is classified at most once per
    call (and reused across calls when ``cache_path`` is provided).

    Args:
        keys_df: DataFrame indexed by ``feature_id`` with columns
            ``InChIkey2D`` and ``smiles``. Rows with a null ``InChIkey2D``
            are skipped.
        cache_path: Optional path to a parquet cache. Reads existing entries
            before querying and writes back any new ones after.
        request_delay: Sleep (seconds) between HTTP requests to be polite to
            the API.
        timeout: Per-request timeout in seconds.
        verbose: Print a one-line progress summary.

    Returns:
        DataFrame indexed by ``feature_id`` with columns:

        - ``sirius_struct_npc_pathway``
        - ``sirius_struct_npc_superclass``
        - ``sirius_struct_npc_class``

        Features whose SMILES could not be classified are still present with
        all three columns null — callers can detect this and fall back.
    """
    cache = (
        load_cache(cache_path)
        if cache_path is not None
        else pd.DataFrame(columns=CACHE_COLUMNS).set_index("InChIkey2D")
    )

    # Build per-InChIkey lookup of representative SMILES.
    sub = keys_df.dropna(subset=["InChIkey2D"]).copy()
    sub = sub[sub["smiles"].notna()]
    if sub.empty:
        out = pd.DataFrame(
            columns=[
                "sirius_struct_npc_pathway",
                "sirius_struct_npc_superclass",
                "sirius_struct_npc_class",
            ],
            index=pd.Index([], name="feature_id"),
        )
        return out

    unique = sub.drop_duplicates(subset=["InChIkey2D"]).set_index("InChIkey2D")

    cached_keys = set(cache.index) if len(cache) else set()
    to_query = [k for k in unique.index if k not in cached_keys]

    n_hit = len(unique) - len(to_query)
    n_query = len(to_query)
    n_success = 0

    if n_query > 0:
        with httpx.Client(timeout=timeout) as client:
            new_rows = []
            for key in to_query:
                smiles = unique.at[key, "smiles"]
                result = classify_smiles(smiles, client=client, timeout=timeout)
                if result is not None:
                    n_success += 1
                new_rows.append({
                    "InChIkey2D": key,
                    "smiles": smiles,
                    "npc_pathway": result["npc_pathway"] if result else None,
                    "npc_superclass": result["npc_superclass"] if result else None,
                    "npc_class": result["npc_class"] if result else None,
                    "queried_at": datetime.now(timezone.utc).isoformat(),
                })
                if request_delay > 0:
                    time.sleep(request_delay)

        new_df = pd.DataFrame(new_rows).set_index("InChIkey2D")
        cache = pd.concat([cache, new_df]) if len(cache) else new_df
        if cache_path is not None:
            save_cache(cache, cache_path)

    if verbose:
        print(
            f"NP Classifier: {n_hit} cache hits, {n_query} new queries "
            f"({n_success} classified)"
        )

    # Map back to features.
    out = sub.join(
        cache[["npc_pathway", "npc_superclass", "npc_class"]],
        on="InChIkey2D",
    )
    out = out[["npc_pathway", "npc_superclass", "npc_class"]].rename(columns={
        "npc_pathway": "sirius_struct_npc_pathway",
        "npc_superclass": "sirius_struct_npc_superclass",
        "npc_class": "sirius_struct_npc_class",
    })
    return out
