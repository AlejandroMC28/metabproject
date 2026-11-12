# `metab spectra` Specification

## Overview

Ingests **MS2 fragmentation spectra** into a project from an MZmine
IIMN-FBMN / GNPS MGF export (the same MGF submitted to FBMN). Each spectrum is
keyed by the canonical `feature_id`, giving the explorer the raw peak lists it
needs to draw a feature's MS2 spectrum in the Feature Detail view.

```
metab spectra PROJECT --mgf spectra.mgf [--force]
```

Writes to `<project>/spectra/`. Appends one manifest entry per run. Idempotent:
re-running errors unless `--force`.

Raw peaks are stored **unmodified** — any denoising / scaling happens only at
display time (`metab explore`), never at ingest.

---

## Input — MGF

Standard MZmine IIMN-FBMN `*_iimn_fbmn.mgf` export. Each `BEGIN IONS … END IONS`
block carries:

| Field | Meaning |
|-------|---------|
| `FEATURE_ID` | MZmine row id (`== feature_id`). Falls back to `SCANS` if absent. |
| `PEPMASS` | Precursor m/z (pyteomics returns a `(mz, intensity)` tuple; the m/z is used). |
| `CHARGE` | Precursor charge (**optional** — absent on some spectra → stored null). |
| `SCANS` | Scan number. |
| peak list | `m/z intensity` pairs after `Num peaks=`. |

A missing file raises `FileNotFoundError`; an MGF that yields no spectra raises
`ValueError`.

## Loader — `metabproject.spectra.mgf`

```python
load_mgf_spectra(path) -> pd.DataFrame
```

Parses the MGF with `pyteomics.mgf` (lazy import) and returns one row per
spectrum, **indexed by `feature_id`**:

| Column | Type | Notes |
|--------|------|-------|
| `precursor_mz` | float | From `PEPMASS`. |
| `precursor_charge` | Int64 (nullable) | From `CHARGE`; null when absent. |
| `scan` | int | From `SCANS` (defaults to `feature_id`). |
| `n_peaks` | int | Fragment count. |
| `mz` | list[float] | Fragment m/z (raw, unmodified). |
| `intensity` | list[float] | Fragment intensity (raw, unmodified). |

`pyteomics` is an optional dependency (the `spectra` extra); importing the
loader without it raises `ImportError` with an install hint.

## Storage — `<project>/spectra/`

- `ms2_spectra.parquet` — the `load_mgf_spectra` output.

On ingest, feature ids are cross-checked against `ms1_features.index`; spectra
whose id is absent from the project are dropped and counted.

## Manifest entry

```json
{
  "command": "spectra",
  "parameters": {"mgf": "..."},
  "results": {
    "n_spectra": 5151,
    "n_features_matched": 5151,
    "n_features_unmatched": 0,
    "total_peaks": 82736
  }
}
```

## Python API

```python
project.import_spectra(mgf, force=False)
project.ms2_spectra   # DataFrame, or None if no spectra ingested
```

`ms2_spectra` returns `None` when `spectra/` is absent — this is what lets
`metab explore` hide the MS2 card cleanly on projects without spectra.

## Explorer integration

`metab explore` reads the stored spectra and, for each feature, emits an `ms2`
block in the viewer payload via `explore/spectra.process_spectrum_for_display`:
precursor removal, low-intensity filter, and base-peak scaling (using
`spectrum_utils` when installed, falling back to the raw peaks otherwise). The
Feature Detail view draws it as an interactive Plotly stem plot (hover m/z +
relative intensity, precursor marked). See `specs/explore.md`.

## Dependencies

| Package | Required | Purpose |
|---------|----------|---------|
| `pyteomics>=4.6` | `spectra` extra | MGF parsing at ingest. |
| `spectrum-utils>=0.4` | `spectra` extra | Spectral cleanup at display time (degrades to raw peaks if absent). |

Install with:
```
pip install 'metabproject[spectra]'
```
