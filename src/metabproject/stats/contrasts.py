"""Parse and resolve contrast definitions from a YAML file.

A contrast names a single A-vs-B comparison and defines each side as a
metadata-column filter. Multiple columns in one side are AND-ed; list values
within a column are OR-ed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass
class Contrast:
    """A single A-vs-B comparison defined by metadata filters.

    Attributes:
        name: Unique, filename-safe contrast name.
        a: Filter dict for group A. Keys are metadata columns; values are
            lists (single scalars are normalised to one-element lists).
        b: Filter dict for group B, same shape as ``a``.
    """

    name: str
    a: dict[str, list[Any]] = field(default_factory=dict)
    b: dict[str, list[Any]] = field(default_factory=dict)


_SLUG_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def load_contrasts(path: str | Path) -> list[Contrast]:
    """Load and validate contrasts from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        List of :class:`Contrast` objects.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: On malformed YAML, missing keys, duplicate names, or
            unsafe filename slugs.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Contrasts file not found: {p}")

    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or "contrasts" not in doc:
        raise ValueError(
            f"Contrasts file must be a YAML mapping with a top-level "
            f"'contrasts:' key. Got: {type(doc).__name__}"
        )

    entries = doc["contrasts"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("'contrasts:' must be a non-empty list")

    contrasts: list[Contrast] = []
    seen_names: set[str] = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"contrasts[{i}] must be a mapping, got {type(entry).__name__}")
        missing = {"name", "a", "b"} - set(entry)
        if missing:
            raise ValueError(f"contrasts[{i}] missing key(s): {sorted(missing)}")

        name = entry["name"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"contrasts[{i}].name must be a non-empty string")
        if not _SLUG_RE.match(name):
            raise ValueError(
                f"contrasts[{i}].name {name!r} is not a safe filename slug "
                f"(allowed: letters, digits, '_', '-', '.')"
            )
        if name in seen_names:
            raise ValueError(f"Duplicate contrast name: {name!r}")
        seen_names.add(name)

        a = _normalise_filter(entry["a"], f"contrasts[{i}].a")
        b = _normalise_filter(entry["b"], f"contrasts[{i}].b")
        if not a or not b:
            raise ValueError(f"contrasts[{i}] both 'a' and 'b' must define at least one filter")

        contrasts.append(Contrast(name=name, a=a, b=b))

    return contrasts


def resolve_samples(
    contrast: Contrast,
    metadata: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Resolve a contrast's filters to concrete sample names.

    Args:
        contrast: A :class:`Contrast`.
        metadata: Sample metadata indexed by ``sample_name``.

    Returns:
        ``(samples_a, samples_b)`` — disjoint lists of sample names.

    Raises:
        ValueError: If any filter column is missing, any value matches no
            samples, or the two sample sets overlap.
    """
    samples_a = _apply_filter(contrast.a, metadata, f"{contrast.name}.a")
    samples_b = _apply_filter(contrast.b, metadata, f"{contrast.name}.b")

    overlap = set(samples_a) & set(samples_b)
    if overlap:
        raise ValueError(
            f"Contrast {contrast.name!r}: group A and B share sample(s): "
            f"{sorted(overlap)}"
        )

    return samples_a, samples_b


def _normalise_filter(raw: Any, label: str) -> dict[str, list[Any]]:
    """Normalise a filter mapping; scalar values become single-element lists."""
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a mapping, got {type(raw).__name__}")
    out: dict[str, list[Any]] = {}
    for col, val in raw.items():
        if isinstance(val, list):
            if not val:
                raise ValueError(f"{label}.{col} list is empty")
            out[col] = list(val)
        else:
            out[col] = [val]
    return out


def _apply_filter(
    filt: dict[str, list[Any]],
    metadata: pd.DataFrame,
    label: str,
) -> list[str]:
    """Apply a filter mapping to metadata and return matching sample names."""
    mask = pd.Series(True, index=metadata.index)
    for col, values in filt.items():
        if col not in metadata.columns:
            raise ValueError(
                f"{label}: metadata has no column {col!r} "
                f"(available: {metadata.columns.tolist()})"
            )
        col_mask = metadata[col].isin(values)
        unmatched = [v for v in values if v not in set(metadata[col].dropna())]
        if unmatched:
            raise ValueError(
                f"{label}.{col}: value(s) {unmatched} not found in metadata"
            )
        mask &= col_mask
    return metadata.index[mask].tolist()
