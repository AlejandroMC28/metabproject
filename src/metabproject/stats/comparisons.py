"""Parse factorial comparison definitions from a YAML file.

A :class:`Comparison` names a statistical question parameterised by one or
more factors (metadata columns), with an omnibus test (``kind``) and an
optional post-hoc. This is the multi-level / factorial counterpart to
:class:`metabproject.stats.contrasts.Contrast`, which remains the pairwise
A-vs-B abstraction.

A single YAML file may carry both ``contrasts:`` and ``comparisons:``
top-level keys — the two loaders ignore each other's entries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_SLUG_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

_KIND_RULES: dict[str, dict[str, Any]] = {
    "anova_1way": {
        "n_factors": 1,
        "posthocs": {None, "tukey", "games_howell"},
    },
    "anova_2way": {
        "n_factors": 2,
        "posthocs": {None, "tukey"},
    },
    "welch_anova": {
        "n_factors": 1,
        "posthocs": {None, "games_howell"},
    },
    "kruskal": {
        "n_factors": 1,
        "posthocs": {None, "dunn"},
    },
}


@dataclass
class Comparison:
    """A factorial statistical comparison.

    Attributes:
        name: Unique, filename-safe slug.
        kind: One of ``'anova_1way'``, ``'anova_2way'``, ``'welch_anova'``,
            ``'kruskal'``.
        factors: Metadata column names. Length depends on ``kind``: 1 for
            ``anova_1way``/``welch_anova``/``kruskal``, 2 for ``anova_2way``.
        posthoc: ``None`` or a post-hoc name valid for the chosen ``kind``
            (see :data:`_KIND_RULES`).
        alpha: Significance threshold in ``(0, 1)``.
    """

    name: str
    kind: str
    factors: list[str] = field(default_factory=list)
    posthoc: str | None = None
    alpha: float = 0.05


def load_comparisons(path: str | Path) -> list[Comparison]:
    """Load and validate comparisons from a YAML file.

    The YAML file must have a top-level ``comparisons:`` key pointing to a
    non-empty list. Each entry must have ``name``, ``kind``, and
    ``factors``; ``posthoc`` and ``alpha`` are optional.

    Factor-column existence in metadata is **not** validated here — that
    happens at runtime in
    :func:`metabproject.stats.comparison_runner.run_comparison`, since
    metadata isn't known at load time.

    Args:
        path: Path to the YAML file.

    Returns:
        List of :class:`Comparison` objects in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: On malformed YAML, missing keys, duplicate names,
            unsafe filename slugs, unknown ``kind``, wrong number of
            factors for the kind, or an invalid post-hoc for the kind.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Comparisons file not found: {p}")

    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or "comparisons" not in doc:
        raise ValueError(
            "Comparisons file must be a YAML mapping with a top-level "
            f"'comparisons:' key. Got: {type(doc).__name__}"
        )

    entries = doc["comparisons"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("'comparisons:' must be a non-empty list")

    out: list[Comparison] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(
                f"comparisons[{i}] must be a mapping, got {type(entry).__name__}"
            )
        missing = {"name", "kind", "factors"} - set(entry)
        if missing:
            raise ValueError(f"comparisons[{i}] missing key(s): {sorted(missing)}")

        name = entry["name"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"comparisons[{i}].name must be a non-empty string")
        if not _SLUG_RE.match(name):
            raise ValueError(
                f"comparisons[{i}].name {name!r} is not a safe filename slug "
                "(allowed: letters, digits, '_', '-', '.')"
            )
        if name in seen:
            raise ValueError(f"Duplicate comparison name: {name!r}")
        seen.add(name)

        kind = entry["kind"]
        if kind not in _KIND_RULES:
            raise ValueError(
                f"comparisons[{i}].kind {kind!r} unknown; "
                f"must be one of {sorted(_KIND_RULES)}"
            )
        rule = _KIND_RULES[kind]

        factors = entry["factors"]
        if not isinstance(factors, list) or not all(isinstance(f, str) for f in factors):
            raise ValueError(
                f"comparisons[{i}].factors must be a list of column names"
            )
        if len(factors) != rule["n_factors"]:
            raise ValueError(
                f"comparisons[{i}].factors has {len(factors)} entries; "
                f"kind {kind!r} requires exactly {rule['n_factors']}"
            )
        if len(set(factors)) != len(factors):
            raise ValueError(
                f"comparisons[{i}].factors contains duplicates: {factors}"
            )

        posthoc = entry.get("posthoc")
        if posthoc not in rule["posthocs"]:
            allowed = sorted(x for x in rule["posthocs"] if x is not None)
            raise ValueError(
                f"comparisons[{i}].posthoc {posthoc!r} not allowed for "
                f"kind {kind!r}; allowed: {[None, *allowed]}"
            )

        alpha = entry.get("alpha", 0.05)
        if not isinstance(alpha, (int, float)) or not (0.0 < float(alpha) < 1.0):
            raise ValueError(
                f"comparisons[{i}].alpha must be a float in (0, 1), got {alpha!r}"
            )

        out.append(
            Comparison(
                name=name,
                kind=kind,
                factors=list(factors),
                posthoc=posthoc,
                alpha=float(alpha),
            )
        )

    return out
