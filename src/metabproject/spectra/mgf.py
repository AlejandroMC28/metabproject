"""Load MS2 spectra from an MZmine FBMN / GNPS MGF export."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_mgf_spectra(path: str | Path) -> pd.DataFrame:
    """Load MS2 spectra from an MGF, keyed by canonical ``feature_id``.

    Reads an MZmine IIMN-FBMN / GNPS ``*_iimn_fbmn.mgf`` export. Each spectrum
    block carries a ``FEATURE_ID`` (== the MZmine row id, i.e. ``feature_id``),
    a ``PEPMASS`` precursor m/z, an optional ``CHARGE``, and the fragment peak
    list. Raw peaks are returned unmodified — any denoising / scaling happens at
    display time, never here.

    Args:
        path: Path to the MGF file.

    Returns:
        DataFrame indexed by ``feature_id`` (int) with columns ``precursor_mz``
        (float), ``precursor_charge`` (nullable int), ``scan`` (int),
        ``n_peaks`` (int), ``mz`` (list[float]), and ``intensity``
        (list[float]), one row per spectrum.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the MGF yields no spectra.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MGF file not found: {path}")

    try:
        from pyteomics import mgf
    except ImportError as exc:
        raise ImportError(
            "pyteomics is required to import MS2 spectra. "
            "Install with: pip install 'metabproject[spectra]'"
        ) from exc

    rows: list[dict] = []
    with mgf.read(str(path), use_index=False) as reader:
        for spectrum in reader:
            params = spectrum.get("params", {})

            fid = params.get("feature_id", params.get("scans"))
            if fid is None:
                continue
            try:
                fid = int(fid)
            except (TypeError, ValueError):
                continue

            pepmass = params.get("pepmass")
            precursor_mz = float(pepmass[0]) if pepmass else None

            charge = params.get("charge")
            precursor_charge = int(charge[0]) if charge else None

            scan = params.get("scans")
            scan = int(scan) if scan is not None else fid

            mz = spectrum.get("m/z array")
            intensity = spectrum.get("intensity array")
            mz_list = [float(v) for v in mz] if mz is not None else []
            int_list = [float(v) for v in intensity] if intensity is not None else []

            rows.append(
                {
                    "feature_id": fid,
                    "precursor_mz": precursor_mz,
                    "precursor_charge": precursor_charge,
                    "scan": scan,
                    "n_peaks": len(mz_list),
                    "mz": mz_list,
                    "intensity": int_list,
                }
            )

    if not rows:
        raise ValueError(f"No spectra found in {path.name}")

    df = pd.DataFrame(rows).set_index("feature_id")
    df["precursor_charge"] = df["precursor_charge"].astype("Int64")
    return df
