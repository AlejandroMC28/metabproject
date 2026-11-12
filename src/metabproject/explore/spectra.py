"""MS2 spectrum processing for the feature viewer.

``spectrum_utils`` is an optional dependency. When it is installed, each stored
raw peak list is cleaned (precursor removal, low-intensity filter, base-peak
scaling) at display time. When it is absent — or when cleanup would empty a
sparse spectrum (e.g. a single-peak scan) — the raw peaks are used instead,
normalised to base-peak relative intensity with plain numpy.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def process_spectrum_for_display(
    mz: Sequence[float],
    intensity: Sequence[float],
    precursor_mz: float | None,
    precursor_charge: int | None,
) -> dict | None:
    """Clean and normalise one MS2 spectrum for a Plotly stem plot.

    Raw peaks are never modified in storage — this runs only when the viewer
    data is assembled. ``spectrum_utils`` cleanup is applied when available,
    each step falling back to the previous state if it would empty the
    spectrum; without ``spectrum_utils`` the raw peaks are normalised directly.

    Args:
        mz: Fragment m/z values.
        intensity: Fragment intensities (raw, unnormalised).
        precursor_mz: Precursor m/z (may be ``None``).
        precursor_charge: Precursor charge (may be ``None`` → treated as 1).

    Returns:
        A JSON-serialisable dict with ``precursor_mz``, ``precursor_charge``,
        ``n_peaks``, ``base_peak_mz``, ``mz`` (list), and ``intensity_rel``
        (list, base-peak = 1.0). Returns ``None`` if there are no peaks.
    """
    mz_arr = np.asarray(mz, dtype=float)
    int_arr = np.asarray(intensity, dtype=float)
    if mz_arr.size == 0 or int_arr.size == 0:
        return None

    charge = int(precursor_charge) if precursor_charge else 1

    mz_arr, int_arr = _clean_with_spectrum_utils(
        mz_arr, int_arr, precursor_mz, charge
    )

    base = float(int_arr.max())
    rel = int_arr / base if base > 0 else int_arr
    base_peak_mz = float(mz_arr[int(np.argmax(int_arr))])

    return {
        "precursor_mz": float(precursor_mz) if precursor_mz is not None else None,
        "precursor_charge": charge,
        "n_peaks": int(mz_arr.size),
        "base_peak_mz": base_peak_mz,
        "mz": [float(v) for v in mz_arr],
        "intensity_rel": [float(v) for v in rel],
    }


def _clean_with_spectrum_utils(
    mz: np.ndarray,
    intensity: np.ndarray,
    precursor_mz: float | None,
    charge: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply spectrum_utils cleanup, degrading to raw peaks on any failure.

    Each transform is applied to a fresh ``MsmsSpectrum`` and only adopted if
    it leaves at least one peak, so sparse spectra keep their raw peaks.
    """
    try:
        from spectrum_utils.spectrum import MsmsSpectrum
    except ImportError:
        return mz, intensity

    if precursor_mz is None:
        return mz, intensity

    cur_mz, cur_int = mz, intensity

    def _apply(fn) -> None:
        nonlocal cur_mz, cur_int
        try:
            spec = MsmsSpectrum(
                "spectrum", float(precursor_mz), charge, cur_mz, cur_int
            )
            fn(spec)
            if spec.mz.size > 0:
                cur_mz, cur_int = np.asarray(spec.mz, dtype=float), np.asarray(
                    spec.intensity, dtype=float
                )
        except Exception:
            pass

    _apply(lambda s: s.remove_precursor_peak(0.1, "Da"))
    _apply(lambda s: s.filter_intensity(min_intensity=0.01, max_num_peaks=150))
    return cur_mz, cur_int
