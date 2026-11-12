"""Exports for two-way ANOVA comparison results.

Three exporters, generalised over the two factor names:

* :func:`export_buckets` — write one TSV per ``classification`` bucket
  (``ns``, ``factor1_only``, ``factor2_only``, ``both_no_interaction``,
  ``interaction_only``, ``interaction``) enriched with per-cell means.
* :func:`export_tukey_pair_tsvs` — split the interaction-bucket Tukey
  results into within-factor1 / within-factor2 / cross-corner pair TSVs,
  with an optional annotation join.
* :func:`pairwise_count_matrix` / :func:`export_pairwise_count_matrix` —
  build the symmetric count matrix of significant features per cell pair.

All functions are pure: DataFrames in, DataFrames or files out. The
combined factor name used internally by
:func:`metabproject.stats.comparison_runner.run_comparison` for
interaction-bucket post-hocs is ``_combined_{f1}_x_{f2}`` and is exposed
here via :func:`combined_factor_name`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PAIR_TSV_COLUMNS = [
    "feature", "group_A", "group_B", "mean_A", "mean_B",
    "diff", "p_value", "significant", "hedges_g", "classification",
    "ann_source", "ann_compound_name", "ann_inchikey_2d",
    "ann_npc_pathway", "ann_npc_superclass", "ann_npc_class",
    "ann_classyfire_class", "ann_classyfire_subclass", "ann_classyfire_level5",
]


def combined_factor_name(f1: str, f2: str) -> str:
    """Combined factor name used internally for interaction post-hocs."""
    return f"_combined_{f1}_x_{f2}"


# --------------------------------------------------------------------------- #
# Bucket export
# --------------------------------------------------------------------------- #

def _cell_means(
    feature_table: pd.DataFrame,
    metadata: pd.DataFrame,
    factors: list[str],
) -> pd.DataFrame:
    """Return features × cells frame of per-cell mean abundance.

    Cell label is ``{level_f1}_{level_f2}`` for every observed combination.
    """
    f1, f2 = factors
    samples = [s for s in feature_table.columns if s in metadata.index]
    cell_label = (
        metadata.loc[samples, f1].astype(str)
        + "_"
        + metadata.loc[samples, f2].astype(str)
    )
    cell_label.name = "cell"
    means = (
        feature_table[samples].T.join(cell_label).groupby("cell").mean().T
    )
    return means.round(4)


def export_buckets(
    omnibus: pd.DataFrame,
    feature_table: pd.DataFrame,
    metadata: pd.DataFrame,
    factors: list[str],
    output_dir: str | Path,
) -> dict[str, int]:
    """Write one bucket TSV per ``classification`` value.

    Each TSV is the omnibus rows for that bucket, left-joined with per-cell
    means and the range across cells. Files are sorted by the most relevant
    adjusted p-value (interaction for interaction buckets, the minimum main
    effect for ``both_no_interaction``, etc.).

    Args:
        omnibus: Output of :func:`anova_2way` indexed by ``feature_id``.
        feature_table: Processed (log-scale) feature table used for cell means.
        metadata: Sample metadata containing both factor columns.
        factors: ``[factor1, factor2]`` matching the omnibus column suffixes.
        output_dir: Directory to write ``bucket_<name>.tsv`` files to.

    Returns:
        ``{bucket_name: row_count}`` for every bucket present in the omnibus.
    """
    if "classification" not in omnibus.columns:
        raise ValueError("omnibus DataFrame missing 'classification' column")
    if len(factors) != 2:
        raise ValueError(f"expected 2 factors, got {len(factors)}")
    f1, f2 = factors
    p_f1, p_f2 = f"p_{f1}_adj", f"p_{f2}_adj"
    p_int = "p_interaction_adj"
    for col in (p_f1, p_f2, p_int):
        if col not in omnibus.columns:
            raise ValueError(f"omnibus missing required column {col!r}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cell_means = _cell_means(feature_table, metadata, factors)
    cell_cols = list(cell_means.columns)

    enriched = omnibus.copy().join(cell_means, how="left")
    if cell_cols:
        enriched["max_cell"] = cell_means[cell_cols].max(axis=1)
        enriched["min_cell"] = cell_means[cell_cols].min(axis=1)
        enriched["range_cell"] = (enriched["max_cell"] - enriched["min_cell"]).round(4)
    enriched["_p_min_main"] = enriched[[p_f1, p_f2]].min(axis=1)

    sort_key = {
        "ns": p_int,
        "factor1_only": p_f1,
        "factor2_only": p_f2,
        "both_no_interaction": "_p_min_main",
        "interaction_only": p_int,
        "interaction": p_int,
    }

    counts: dict[str, int] = {}
    for bucket in sort_key:
        sub = enriched[enriched["classification"] == bucket]
        counts[bucket] = int(len(sub))
        if sub.empty:
            continue
        sub = sub.sort_values(sort_key[bucket], ascending=True, kind="mergesort")
        sub = sub.drop(columns=["_p_min_main"])
        sub.to_csv(out / f"bucket_{bucket}.tsv", sep="\t", index=True)

    return counts


# --------------------------------------------------------------------------- #
# Pair TSVs (within-factor + cross-corner)
# --------------------------------------------------------------------------- #

def _interaction_subset(
    posthoc: pd.DataFrame,
    f1: str,
    f2: str,
) -> pd.DataFrame:
    """Filter Tukey rows to the combined-factor (interaction) rows."""
    combined = combined_factor_name(f1, f2)
    sub = posthoc[posthoc["factor"] == combined]
    if sub.empty:
        raise ValueError(
            f"posthoc has no rows with factor == {combined!r}; "
            "run a 2-way ANOVA + Tukey first"
        )
    return sub.copy()


def _orient_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize each row so ``group_A < group_B`` alphabetically.

    Flips ``diff`` sign and swaps ``mean_A``/``mean_B`` to keep
    ``diff = mean_B - mean_A`` consistent.
    """
    df = df.copy()
    swap = df["group_A"] > df["group_B"]
    a = df["group_A"].where(~swap, df["group_B"])
    b = df["group_B"].where(~swap, df["group_A"])
    mean_a = df["mean_A"].where(~swap, df["mean_B"])
    mean_b = df["mean_B"].where(~swap, df["mean_A"])
    diff = df["diff"].where(~swap, -df["diff"])
    df["group_A"], df["group_B"] = a, b
    df["mean_A"], df["mean_B"] = mean_a, mean_b
    df["diff"] = diff
    return df


_ANNOTATION_COLS = {
    "annotation_source": "ann_source",
    "compound_name": "ann_compound_name",
    "inchikey2d": "ann_inchikey_2d",
    "npc_pathway": "ann_npc_pathway",
    "npc_superclass": "ann_npc_superclass",
    "npc_class": "ann_npc_class",
    "classyfire_class": "ann_classyfire_class",
    "classyfire_subclass": "ann_classyfire_subclass",
    "classyfire_level5": "ann_classyfire_level5",
}


def _attach_annotation(
    df: pd.DataFrame,
    annotation_table: pd.DataFrame | None,
) -> pd.DataFrame:
    """Left-join annotation + taxonomy columns onto a Tukey pair frame.

    Missing source columns (e.g. ``classyfire_level5`` on an older CANOPUS
    export) are filled with ``pd.NA`` so the pair TSV schema stays stable.
    """
    if annotation_table is None:
        for v in _ANNOTATION_COLS.values():
            df[v] = pd.NA
        return df
    present = {k: v for k, v in _ANNOTATION_COLS.items() if k in annotation_table.columns}
    lookup = annotation_table.rename(columns=present)[list(present.values())]
    out = df.merge(lookup, left_on="feature", right_index=True, how="left")
    for v in _ANNOTATION_COLS.values():
        if v not in out.columns:
            out[v] = pd.NA
    return out


def export_tukey_pair_tsvs(
    posthoc: pd.DataFrame,
    omnibus: pd.DataFrame,
    output_dir: str | Path,
    factors: list[str],
    annotation_table: pd.DataFrame | None = None,
) -> dict[str, int]:
    """Split interaction-bucket Tukey results into pair TSVs.

    Splits the combined-factor Tukey rows into three groups based on whether
    the two cells differ in factor1, factor2, or both:

    * ``pair_within_{f1}_{level}_{lvl2A}_vs_{lvl2B}.tsv`` — same factor1
      level, different factor2 levels.
    * ``pair_within_{f2}_{level}_{lvl1A}_vs_{lvl1B}.tsv`` — same factor2
      level, different factor1 levels.
    * ``crosscorner_all.tsv`` — cells differing on both factors.

    Each TSV holds only ``significant`` rows (sorted by ``p_value``) with the
    standard pair schema (see :data:`PAIR_TSV_COLUMNS`), augmented with the
    feature's bucket ``classification`` and three annotation columns.

    Args:
        posthoc: Long-format Tukey output from :func:`run_comparison` for an
            ``anova_2way`` + ``tukey`` comparison.
        omnibus: Omnibus DataFrame indexed by ``feature_id`` (used for
            ``classification`` join).
        output_dir: Destination directory.
        factors: ``[factor1, factor2]`` matching the comparison.
        annotation_table: Optional merged annotation table indexed by
            ``feature_id``. Missing columns or missing rows are filled with
            ``pd.NA``.

    Returns:
        ``{filename: row_count}`` for every TSV written.
    """
    if len(factors) != 2:
        raise ValueError(f"expected 2 factors, got {len(factors)}")
    f1, f2 = factors

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sub = _orient_pairs(_interaction_subset(posthoc, f1, f2))

    cls = omnibus["classification"].rename("classification")
    sub = sub.merge(cls, left_on="feature", right_index=True, how="left")
    sub = _attach_annotation(sub, annotation_table)

    # Cell labels were built as "{f1_level}_{f2_level}" by the runner.
    a_parts = sub["group_A"].str.split("_", n=1, expand=True)
    b_parts = sub["group_B"].str.split("_", n=1, expand=True)
    sub[f"{f1}_A"], sub[f"{f2}_A"] = a_parts[0], a_parts[1]
    sub[f"{f1}_B"], sub[f"{f2}_B"] = b_parts[0], b_parts[1]

    same_f1 = sub[f"{f1}_A"] == sub[f"{f1}_B"]
    same_f2 = sub[f"{f2}_A"] == sub[f"{f2}_B"]

    written: dict[str, int] = {}

    for (level, lvl_a, lvl_b), grp in sub[same_f1 & ~same_f2].groupby(
        [f"{f1}_A", f"{f2}_A", f"{f2}_B"], sort=True,
    ):
        sig = grp[grp["significant"]].sort_values("p_value", kind="mergesort")[PAIR_TSV_COLUMNS]
        fname = f"pair_within_{f1}_{level}_{lvl_a}_vs_{lvl_b}.tsv"
        sig.to_csv(out / fname, sep="\t", index=False)
        written[fname] = int(len(sig))

    for (level, lvl_a, lvl_b), grp in sub[~same_f1 & same_f2].groupby(
        [f"{f2}_A", f"{f1}_A", f"{f1}_B"], sort=True,
    ):
        sig = grp[grp["significant"]].sort_values("p_value", kind="mergesort")[PAIR_TSV_COLUMNS]
        fname = f"pair_within_{f2}_{level}_{lvl_a}_vs_{lvl_b}.tsv"
        sig.to_csv(out / fname, sep="\t", index=False)
        written[fname] = int(len(sig))

    cc = sub[~same_f1 & ~same_f2].copy()
    cc["pair"] = cc["group_A"] + "_vs_" + cc["group_B"]
    cc_cols = PAIR_TSV_COLUMNS + ["pair"]
    cc[cc_cols].to_csv(out / "crosscorner_all.tsv", sep="\t", index=False)
    written["crosscorner_all.tsv"] = int(len(cc))

    return written


# --------------------------------------------------------------------------- #
# Pairwise count matrix
# --------------------------------------------------------------------------- #

def pairwise_count_matrix(
    posthoc: pd.DataFrame,
    factors: list[str],
    cells: list[str] | None = None,
) -> pd.DataFrame:
    """Build a symmetric (cells × cells) count of significant Tukey pairs.

    Args:
        posthoc: Long-format Tukey output from a 2-way ANOVA comparison.
        factors: ``[factor1, factor2]`` — used to find the combined-factor
            rows.
        cells: Optional explicit cell-label ordering. When ``None`` the cells
            observed in the posthoc rows are sorted alphabetically.

    Returns:
        Square ``DataFrame`` with the same row/column index, integer counts,
        and zeros on the diagonal.
    """
    f1, f2 = factors
    sub = _interaction_subset(posthoc, f1, f2)
    sig = sub[sub["significant"]]
    if cells is None:
        cells = sorted(set(sub["group_A"]).union(sub["group_B"]))

    mat = pd.DataFrame(0, index=cells, columns=cells, dtype=int)
    counts = sig.groupby(["group_A", "group_B"]).size()
    for (a, b), n in counts.items():
        if a in mat.index and b in mat.columns:
            mat.loc[a, b] += int(n)
            mat.loc[b, a] += int(n)
    return mat


def export_pairwise_count_matrix(
    posthoc: pd.DataFrame,
    output_path: str | Path,
    factors: list[str],
    cells: list[str] | None = None,
) -> pd.DataFrame:
    """Compute :func:`pairwise_count_matrix` and write it to ``output_path``."""
    mat = pairwise_count_matrix(posthoc, factors, cells=cells)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    mat.to_csv(output_path, sep="\t", index=True)
    return mat
