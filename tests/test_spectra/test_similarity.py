"""Tests for metabproject.spectra.similarity.

Validate the greedy score/match recovery against matchms' own public
``.pair()`` (score + match count), then check matched-index correctness,
modified-vs-direct behaviour on a precursor-shifted pair, and scaling effects.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matchms")

from metabproject.spectra.similarity import (
    SCALINGS,
    compute_pair_scores,
)

# ---------------------------------------------------------------------------
# Synthetic display-processed spectra: {fid: {mz, intensity_rel, precursor_mz}}.
# Features 1 and 2 share two peaks (80.05, 110.07); their third peaks differ by
# exactly the precursor shift (216.09 - 200.10 = 15.99), so they match only
# under the modified cosine. Feature 3 shares nothing with 1/2 beyond 80.05.
# ---------------------------------------------------------------------------

SPECTRA = {
    1: {"mz": [80.05, 110.07, 150.09], "intensity_rel": [1.0, 0.6, 0.4],
        "precursor_mz": 200.10},
    2: {"mz": [80.05, 110.07, 166.08], "intensity_rel": [0.9, 0.55, 0.45],
        "precursor_mz": 216.09},
    3: {"mz": [80.05, 300.20], "intensity_rel": [1.0, 0.3],
        "precursor_mz": 320.30},
}

TOL = 0.1


def _matchms_ref(spec_a, spec_b, score_type, mz_power, intensity_power):
    """Reference score + match count from matchms' public ``.pair()``."""
    from matchms import Spectrum
    import matchms.similarity as ms_sim

    # matchms renamed ModifiedCosine -> ModifiedCosineGreedy (>=0.28); support both.
    CosineGreedy = ms_sim.CosineGreedy
    ModifiedCosine = getattr(ms_sim, "ModifiedCosineGreedy", None) or ms_sim.ModifiedCosine

    def mk(s):
        return Spectrum(
            mz=np.array(s["mz"], dtype=float),
            intensities=np.array(s["intensity_rel"], dtype=float),
            metadata={"precursor_mz": s["precursor_mz"]},
        )

    if score_type == "direct":
        sim = CosineGreedy(
            tolerance=TOL, mz_power=mz_power, intensity_power=intensity_power
        )
    else:
        sim = ModifiedCosine(
            tolerance=TOL, mz_power=mz_power, intensity_power=intensity_power
        )
    res = sim.pair(mk(spec_a), mk(spec_b))
    return float(res["score"]), int(res["matches"])


@pytest.mark.parametrize("score_type", ["direct", "modified"])
@pytest.mark.parametrize("scaling", list(SCALINGS))
def test_matches_matchms_pair(score_type, scaling):
    """Score + match count agree with matchms across all score/scaling combos."""
    df = compute_pair_scores(
        SPECTRA, [(1, 2)], tolerance=TOL,
        scalings=[scaling], score_types=[score_type],
    )
    assert len(df) == 1
    row = df.iloc[0]
    mz_power, intensity_power = SCALINGS[scaling]
    ref_score, ref_n = _matchms_ref(
        SPECTRA[1], SPECTRA[2], score_type, mz_power, intensity_power
    )
    assert row["score"] == pytest.approx(ref_score, abs=1e-9)
    assert row["n_matches"] == ref_n


def test_matched_indices_direct_vs_modified():
    """Direct matches the two shared peaks; modified adds the shifted third."""
    df = compute_pair_scores(
        SPECTRA, [(1, 2)], tolerance=TOL,
        scalings=["linear"], score_types=["direct", "modified"],
    )
    direct = df[df["score_type"] == "direct"].iloc[0]
    modified = df[df["score_type"] == "modified"].iloc[0]

    assert list(zip(direct["matched_idx_a"], direct["matched_idx_b"])) == [(0, 0), (1, 1)]
    # Modified recovers all three, including the precursor-shifted (2, 2) pair.
    mod_pairs = sorted(zip(modified["matched_idx_a"], modified["matched_idx_b"]))
    assert mod_pairs == [(0, 0), (1, 1), (2, 2)]
    assert modified["n_matches"] == 3
    assert direct["n_matches"] == 2


def test_scaling_changes_score():
    """Different intensity/m-z scaling yields different scores for a pair."""
    df = compute_pair_scores(
        SPECTRA, [(1, 2)], tolerance=TOL, score_types=["direct"],
    )
    by_scaling = {r["scaling"]: r["score"] for _, r in df.iterrows()}
    assert set(by_scaling) == set(SCALINGS)
    # massbank m/z weighting differs markedly from linear.
    assert by_scaling["linear"] != pytest.approx(by_scaling["massbank"])


def test_pair_order_normalised():
    """Feature ids are stored low-first regardless of input order."""
    df = compute_pair_scores(
        SPECTRA, [(2, 1)], tolerance=TOL,
        scalings=["linear"], score_types=["direct"],
    )
    row = df.iloc[0]
    assert row["feature_id_a"] == 1
    assert row["feature_id_b"] == 2


def test_missing_spectrum_skipped():
    """A pair referencing an absent feature is silently skipped."""
    df = compute_pair_scores(
        SPECTRA, [(1, 999)], tolerance=TOL,
        scalings=["linear"], score_types=["direct"],
    )
    assert df.empty


def test_self_pair_skipped():
    df = compute_pair_scores(
        SPECTRA, [(1, 1)], tolerance=TOL,
        scalings=["linear"], score_types=["direct"],
    )
    assert df.empty


def test_unknown_scaling_raises():
    with pytest.raises(ValueError, match="scaling"):
        compute_pair_scores(SPECTRA, [(1, 2)], scalings=["nope"])


def test_unknown_score_type_raises():
    with pytest.raises(ValueError, match="score type"):
        compute_pair_scores(SPECTRA, [(1, 2)], score_types=["nope"])


def test_empty_pairs_returns_empty_frame():
    df = compute_pair_scores(SPECTRA, [], tolerance=TOL)
    assert df.empty
    # Columns are still present so downstream code can rely on the schema.
    for col in ("feature_id_a", "score_type", "scaling", "score",
                "n_matches", "matched_idx_a", "matched_idx_b"):
        assert col in df.columns
