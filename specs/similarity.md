# `metab similarity` Specification

## Overview

Computes **MS2 spectral similarity** between features within the same GNPS
molecular family, using [matchms](https://github.com/matchms/matchms) — both the
**direct** cosine (`CosineGreedy`) and the precursor-shift-aware **modified**
cosine (`ModifiedCosineGreedy`) — across a small set of intensity/m-z **scaling**
presets. Each pair's **matched peak indices** are recorded so the explorer can
highlight the matching fragments in its head-to-tail compare plot.

```
metab similarity PROJECT [--tolerance 0.1]
                         [--score-type modified|direct|both]
                         [--scaling linear|sqrt|massbank ]...
                         [--force]
```

Writes to `<project>/similarity/`. Appends one manifest entry per run.
Idempotent: re-running errors unless `--force`.

Requires `metab spectra` (MS2 peak lists) **and** `metab network` (family
membership) to have run first — a missing prerequisite raises `ValueError`.

### Why compute at command time

The exported `metab explore` HTML is fully self-contained and offline, so matchms
(Python) cannot run in the browser. All scoring **and** peak matching are done
here, stored to parquet, and embedded into the export. The compare panel then
only looks values up — no client-side computation.

---

## Coverage — within-family pairs

The scored set is every **unordered pair of features** that (a) share a GNPS
molecular family (same `component_id` in `network/nodes.parquet`) and (b) both
carry an MS2 spectrum (`spectra/ms2_spectra.parquet`). This matches exactly what
the explorer's compare panel lets a user pick: *any two members* of a family, not
only directly-connected nodes. Pairs are enumerated with
`itertools.combinations` over the spectra-bearing members of each component.

Singletons and features without a spectrum contribute no pairs.

## Scoring — `metabproject.spectra.similarity`

```python
compute_pair_scores(processed_spectra, pairs, tolerance=0.1,
                    scalings=("linear", "sqrt", "massbank"),
                    score_types=("modified", "direct")) -> pd.DataFrame
```

`processed_spectra` maps `feature_id -> {mz, intensity_rel, precursor_mz}` — the
**display-processed** peaks from `explore.spectra.process_spectrum_for_display`.
Scoring on the same peaks the viewer draws is deliberate: the returned matched
indices then line up 1:1 with the rendered stems.

### Scaling presets

| Name | `mz_power` | `intensity_power` | Notes |
|------|-----------|-------------------|-------|
| `linear` | 0 | 1 | Classic intensity-weighted cosine. |
| `sqrt` | 0 | 0.5 | Sqrt intensity scaling; tempers dominant peaks. |
| `massbank` | 3 | 0.6 | MassBank/NIST-style m/z weighting. |

These map directly onto the matchms cosine parameters of the same name.

### Recovering matched peak indices

matchms' public `.pair()` returns only a **score** and a match **count**, never
the matched peak *indices*. This module therefore reuses matchms' own
`matchms.similarity.spectrum_similarity_functions.collect_peak_pairs` and
faithfully replicates its greedy selection:

1. Collect candidate peak pairs for each m/z shift
   (`[0.0]` for direct; `[0.0, precursor_shift]` for modified, where
   `precursor_shift = precursor_mz_a - precursor_mz_b`; if
   `|shift| <= tolerance` the modified cosine reduces to direct).
2. Sort candidates by weighted-intensity product, descending (stable mergesort).
3. Greedily accept non-conflicting pairs — each peak used at most once.
4. Score = accepted weighted products / (‖a‖·‖b‖) over the power-weighted vectors.

A unit test asserts the resulting score + count match `CosineGreedy.pair` /
`ModifiedCosineGreedy.pair` exactly across every score-type × scaling combination.

Because the greedy order depends on the weighted product, the **matched set is
stored per `(score_type, scaling)`**, not shared across scalings.

### Output — long-format DataFrame

One row per `(pair, score_type, scaling)`:

| Column | Type | Notes |
|--------|------|-------|
| `feature_id_a` | int | Lower feature id (`a < b`). |
| `feature_id_b` | int | Higher feature id. |
| `score_type` | str | `"modified"` or `"direct"`. |
| `scaling` | str | Preset name. |
| `score` | float | Cosine similarity in `[0, 1]`. |
| `n_matches` | int | Number of matched peaks. |
| `matched_idx_a` | list[int] | Matched peak indices into feature `a`'s displayed spectrum. |
| `matched_idx_b` | list[int] | Matched peak indices into feature `b`'s displayed spectrum. |

Pairs referencing a missing spectrum, and self-pairs, are silently skipped.

## Storage — `<project>/similarity/`

- `scores.parquet` — the `compute_pair_scores` output.

## Manifest entry

```json
{
  "command": "similarity",
  "parameters": {
    "tolerance": 0.1,
    "score_types": ["modified", "direct"],
    "scalings": ["linear", "sqrt", "massbank"]
  },
  "results": {
    "n_pairs": 128,
    "n_scored": 128,
    "n_rows": 768
  }
}
```

`n_pairs` is the number of within-family pairs enumerated, `n_scored` the number
actually scored (distinct pairs written), and `n_rows` the total long-format rows
(`n_scored × |score_types| × |scalings|`).

## Python API

```python
project.compute_similarity(tolerance=0.1,
                           score_types=("modified", "direct"),
                           scalings=None,          # None -> all presets
                           force=False)
project.spectral_similarity   # DataFrame, or None if not computed
```

`spectral_similarity` returns `None` when `similarity/` is absent — this is what
lets `metab explore` hide the score-type / scaling toggles cleanly on projects
without similarity.

## Explorer integration

`explore/data.py` reads `project.spectral_similarity` and:

- Attaches a compact `similarity` block to each network component, keyed by
  `"<lo>|<hi>"`:
  `component.similarity["1|2"][score_type][scaling] = {"score", "matches"}`,
  where `matches` is a list of `[idx_a, idx_b]` peak-index pairs.
- Emits a top-level `similarity_meta = {score_types, scalings, tolerance}` used to
  build the compare-panel dropdowns (and to hide them when `None`).

In the viewer's **Compare MS2 spectra** panel (Network view), score-type and
scaling `<select>`s drive the head-to-tail mirror plot: the matchms score +
matched-peak count are shown, matched peaks are drawn in a highlight colour with
faint connectors across the mirror axis, and unmatched peaks are dimmed. The
pre-ingested GNPS cosine is kept as a secondary note when the two nodes share an
edge. See `specs/explore.md`.

## Dependencies

| Package | Required | Purpose |
|---------|----------|---------|
| `matchms>=0.24` | `similarity` extra | Cosine / modified-cosine scoring + peak-pair collection. |

Install with:
```
pip install 'metabproject[similarity]'
```

matchms is imported lazily; the compute entry point raises `ImportError` with an
install hint when it is absent.
