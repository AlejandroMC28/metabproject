"""matchms spectral similarity (cosine / modified cosine) with matched peaks.

Computes pairwise MS2 similarity for the explorer's spectrum-compare panel. Two
score families are offered — **direct** (``CosineGreedy``) and **modified**
(``ModifiedCosineGreedy``, precursor-shift aware) — each across a handful of
intensity/m-z **scaling** presets.

matchms' public ``.pair()`` returns only a score and a match *count*, never the
matched peak *indices* needed to highlight peaks in the mirror plot. This module
therefore reuses matchms' own peak-pair collection
(:func:`matchms.similarity.spectrum_similarity_functions.collect_peak_pairs`) and
faithfully replicates its greedy selection, recovering both the score and the
exact matched index pairs in one pass. A test asserts the score matches
``CosineGreedy.pair`` / ``ModifiedCosineGreedy.pair``.

Peaks are consumed in the same order the viewer draws them (the base-peak
normalised ``mz`` / ``intensity_rel`` arrays produced by
``explore.spectra.process_spectrum_for_display``), so the returned indices align
directly with the stems on screen. Input peak lists are assumed sorted by
ascending m/z (true for MZmine MGF exports; ``collect_peak_pairs`` relies on it).

``matchms`` is an optional dependency (the ``similarity`` extra); importing the
compute entry point without it raises ``ImportError`` with an install hint.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# Intensity/m-z scaling presets: name -> (mz_power, intensity_power). These map
# straight onto the matchms cosine parameters of the same name.
SCALINGS: dict[str, tuple[float, float]] = {
    "linear": (0.0, 1.0),      # classic cosine, intensity-weighted
    "sqrt": (0.0, 0.5),        # sqrt intensity scaling, tempers dominant peaks
    "massbank": (3.0, 0.6),    # MassBank/NIST-style m/z weighting
}

# Score families offered.
SCORE_TYPES: tuple[str, ...] = ("modified", "direct")


def _peaks_array(mz: Sequence[float], intensity: Sequence[float]) -> np.ndarray:
    """Stack ``mz`` / ``intensity`` into the matchms ``N x 2`` peak array."""
    return np.column_stack(
        [np.asarray(mz, dtype=float), np.asarray(intensity, dtype=float)]
    )


def _norm(spec: np.ndarray, mz_power: float, intensity_power: float) -> float:
    """L2 norm of the power-weighted peak vector (cosine denominator term)."""
    weighted = (spec[:, 0] ** mz_power) * (spec[:, 1] ** intensity_power)
    return float(np.sqrt(np.sum(weighted ** 2)))


def _score_and_matches(
    spec1: np.ndarray,
    spec2: np.ndarray,
    tolerance: float,
    shifts: Sequence[float],
    mz_power: float,
    intensity_power: float,
) -> tuple[float, list[tuple[int, int]]]:
    """Greedy cosine over one or more m/z shifts, returning score + matches.

    Replicates ``CosineGreedy`` / ``ModifiedCosineGreedy``: collect candidate
    peak pairs for each shift, sort by weighted-intensity product (descending,
    stable), then greedily accept non-conflicting pairs. ``shifts=[0.0]`` is the
    direct cosine; ``[0.0, mass_shift]`` is the modified cosine.

    Returns:
        ``(score, matches)`` where ``matches`` is a list of ``(idx1, idx2)`` peak
        index pairs into ``spec1`` / ``spec2``.
    """
    from matchms.similarity.spectrum_similarity_functions import collect_peak_pairs

    collected: list[np.ndarray] = []
    for shift in shifts:
        pairs = collect_peak_pairs(
            spec1, spec2, tolerance, float(shift), mz_power, intensity_power
        )
        if pairs is not None and len(pairs):
            collected.append(pairs)
    if not collected:
        return 0.0, []

    matching_pairs = np.concatenate(collected, axis=0)
    # Stable descending sort by the weighted-intensity product (column 2), exactly
    # as matchms does before its greedy pass.
    order = np.argsort(matching_pairs[:, 2], kind="mergesort")[::-1]
    matching_pairs = matching_pairs[order, :]

    used1: set[int] = set()
    used2: set[int] = set()
    matches: list[tuple[int, int]] = []
    score_sum = 0.0
    for row in matching_pairs:
        a, b = int(row[0]), int(row[1])
        if a not in used1 and b not in used2:
            used1.add(a)
            used2.add(b)
            matches.append((a, b))
            score_sum += float(row[2])

    denom = _norm(spec1, mz_power, intensity_power) * _norm(
        spec2, mz_power, intensity_power
    )
    score = score_sum / denom if denom > 0 else 0.0
    return score, matches


def compute_pair_scores(
    processed_spectra: Mapping[int, Mapping[str, object]],
    pairs: Iterable[tuple[int, int]],
    tolerance: float = 0.1,
    scalings: Sequence[str] = tuple(SCALINGS),
    score_types: Sequence[str] = SCORE_TYPES,
) -> pd.DataFrame:
    """Score MS2 spectrum pairs with matchms, recovering matched peak indices.

    Args:
        processed_spectra: Maps ``feature_id`` -> a dict with ``mz`` (list of
            floats, ascending), ``intensity_rel`` (list of floats), and
            ``precursor_mz`` (float or ``None``) — i.e. the display-processed
            spectrum from ``process_spectrum_for_display``.
        pairs: Unordered ``(feature_id_a, feature_id_b)`` pairs to score. Pairs
            with a missing spectrum are skipped.
        tolerance: m/z match tolerance in Da.
        scalings: Names from :data:`SCALINGS` to compute.
        score_types: Any of ``"direct"`` / ``"modified"``.

    Returns:
        Long-format DataFrame, one row per ``(pair, score_type, scaling)`` with
        columns ``feature_id_a``, ``feature_id_b`` (a < b), ``score_type``,
        ``scaling``, ``score``, ``n_matches``, ``matched_idx_a`` (list[int]),
        ``matched_idx_b`` (list[int]). Empty DataFrame if nothing was scored.

    Raises:
        ImportError: If ``matchms`` is not installed.
        ValueError: If an unknown scaling or score type is requested.
    """
    try:
        import matchms  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "matchms is required to compute spectral similarity. "
            "Install with: pip install 'metabproject[similarity]'"
        ) from exc

    for name in scalings:
        if name not in SCALINGS:
            raise ValueError(
                f"Unknown scaling '{name}'. Choose from {sorted(SCALINGS)}."
            )
    for st in score_types:
        if st not in SCORE_TYPES:
            raise ValueError(
                f"Unknown score type '{st}'. Choose from {sorted(SCORE_TYPES)}."
            )

    # Pre-build peak arrays once per feature.
    arrays: dict[int, np.ndarray] = {}
    precursors: dict[int, float | None] = {}
    for fid, spec in processed_spectra.items():
        mz = spec.get("mz") or []
        inten = spec.get("intensity_rel") or []
        if not mz or not inten:
            continue
        arrays[int(fid)] = _peaks_array(mz, inten)
        pmz = spec.get("precursor_mz")
        precursors[int(fid)] = None if pmz is None else float(pmz)  # type: ignore[arg-type]

    rows: list[dict] = []
    for raw_a, raw_b in pairs:
        a, b = int(raw_a), int(raw_b)
        if a == b or a not in arrays or b not in arrays:
            continue
        lo, hi = (a, b) if a < b else (b, a)
        spec1, spec2 = arrays[lo], arrays[hi]

        for st in score_types:
            if st == "direct":
                shifts: list[float] = [0.0]
            else:  # modified
                p1, p2 = precursors.get(lo), precursors.get(hi)
                if p1 is None or p2 is None:
                    # No precursor info -> modified cosine reduces to direct.
                    shifts = [0.0]
                else:
                    mass_shift = p1 - p2
                    shifts = (
                        [0.0]
                        if abs(mass_shift) <= tolerance
                        else [0.0, mass_shift]
                    )

            for scaling in scalings:
                mz_power, intensity_power = SCALINGS[scaling]
                score, matches = _score_and_matches(
                    spec1, spec2, tolerance, shifts, mz_power, intensity_power
                )
                rows.append(
                    {
                        "feature_id_a": lo,
                        "feature_id_b": hi,
                        "score_type": st,
                        "scaling": scaling,
                        "score": float(score),
                        "n_matches": len(matches),
                        "matched_idx_a": [m[0] for m in matches],
                        "matched_idx_b": [m[1] for m in matches],
                    }
                )

    return pd.DataFrame(
        rows,
        columns=[
            "feature_id_a",
            "feature_id_b",
            "score_type",
            "scaling",
            "score",
            "n_matches",
            "matched_idx_a",
            "matched_idx_b",
        ],
    )
