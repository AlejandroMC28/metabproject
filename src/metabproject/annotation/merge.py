"""Merge SIRIUS, CANOPUS, GNPS library, and GNPS suspect annotations into a skinny table."""

from __future__ import annotations

import pandas as pd


def merge_annotations(
    sirius_df: pd.DataFrame,
    canopus_df: pd.DataFrame,
    gnps_df: pd.DataFrame,
    suspect_df: pd.DataFrame | None = None,
    sirius_topk_df: pd.DataFrame | None = None,
    sirius_min_confidence: float = 0.64,
    sirius_struct_npc_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge per-source annotation DataFrames into a skinny curated table.

    Priority for curated structural columns: GNPS library → SIRIUS
    (high-confidence only) → GNPS suspect.

    Priority for curated taxonomy columns: GNPS library → CANOPUS (per-level,
    probability-gated). SIRIUS no longer contributes taxonomy; use
    ``canopus_df`` (from :func:`load_canopus`) for compound-class predictions.

    When ``sirius_topk_df`` is supplied, the table additionally carries a
    GNPS↔SIRIUS top-k cross-check (``gnps_sirius_rank`` and
    ``gnps_sirius_highconf``) for every feature with a GNPS library match.

    Only features with at least one annotation appear in the output.

    Args:
        sirius_df: Output of :func:`load_sirius`, indexed by ``feature_id``.
        canopus_df: Output of :func:`load_canopus`, indexed by ``feature_id``.
        gnps_df: Output of :func:`load_gnps_library`, indexed by ``feature_id``.
        suspect_df: Optional output of :func:`load_gnps_suspect`, indexed by
            ``feature_id``. Pass ``None`` if no suspect data.
        sirius_topk_df: Optional output of :func:`load_sirius_topk`. When
            provided, enables the GNPS↔SIRIUS top-k cross-check columns.
        sirius_min_confidence: Threshold used by the top-k cross-check to
            decide whether the matched candidate is high-confidence (default
            0.64, matches ``load_sirius`` default).

    Returns:
        Skinny merged DataFrame indexed by ``feature_id``.
    """
    from metabproject.annotation.sirius import compute_gnps_sirius_topk_match
    # Outer-join all sources on feature_id.
    merged = gnps_df.join(sirius_df, how="outer")
    merged = merged.join(canopus_df, how="outer")
    if suspect_df is not None:
        merged = merged.join(suspect_df, how="outer")
    if sirius_struct_npc_df is not None and len(sirius_struct_npc_df) > 0:
        merged = merged.join(sirius_struct_npc_df, how="outer")

    result = pd.DataFrame(index=merged.index)

    # -- Presence masks -------------------------------------------------------

    gnps_has = _col(merged, "gnps_compound_name").notna()
    sirius_high = _col(merged, "sirius_high_confidence").fillna(False).astype(bool)
    sirius_has = _col(merged, "sirius_InChIkey2D").notna() & sirius_high
    suspect_has = (
        _col(merged, "suspect_compound_name").notna()
        if suspect_df is not None
        else pd.Series(False, index=merged.index)
    )

    # Per-level CANOPUS presence (probability-gated via *_passed flags).
    canopus_pathway_has = (
        _col(merged, "canopus_npc_pathway").notna()
        & _col(merged, "canopus_npc_pathway_passed").fillna(False).astype(bool)
    )
    canopus_superclass_has = (
        _col(merged, "canopus_npc_superclass").notna()
        & _col(merged, "canopus_npc_superclass_passed").fillna(False).astype(bool)
    )
    canopus_class_has = (
        _col(merged, "canopus_npc_class").notna()
        & _col(merged, "canopus_npc_class_passed").fillna(False).astype(bool)
    )
    canopus_classyfire_has = (
        _col(merged, "canopus_classyfire_class").notna()
        & _col(merged, "canopus_classyfire_class_passed").fillna(False).astype(bool)
    )
    canopus_has = (
        canopus_pathway_has | canopus_superclass_has
        | canopus_class_has | canopus_classyfire_has
    )

    # -- Structural curated columns (GNPS > SIRIUS high-conf > suspect) -------
    # Per-source columns always show the raw value regardless of confidence.
    # The unsuffixed best-pick applies the priority chain.

    def _per_source(gnps_col: str, sirius_col: str) -> tuple:
        """Return (gnps_series, sirius_series) raw from merged."""
        return _col(merged, gnps_col), _col(merged, sirius_col)

    def _best_pick(gnps_s, sirius_s, suspect_col: str | None = None):
        """GNPS if gnps_has, else SIRIUS if sirius_has (high-conf), else suspect."""
        best = gnps_s.where(gnps_has, sirius_s.where(sirius_has))
        if suspect_col is not None and suspect_df is not None:
            sus = _col(merged, suspect_col)
            best = best.where(best.notna(), sus.where(suspect_has))
        return best

    g_name, s_name = _per_source("gnps_compound_name", "sirius_compound_name")
    result["compound_name_gnps"] = g_name
    result["compound_name_sirius"] = s_name
    result["compound_name"] = _best_pick(g_name, s_name, "suspect_compound_name")

    g_ik, s_ik = _per_source("gnps_InChIkey2D", "sirius_InChIkey2D")
    result["inchikey2d_gnps"] = g_ik
    result["inchikey2d_sirius"] = s_ik
    result["inchikey2d"] = _best_pick(g_ik, s_ik)

    g_smi, s_smi = _per_source("gnps_smiles", "sirius_smiles")
    result["smiles_gnps"] = g_smi
    result["smiles_sirius"] = s_smi
    result["smiles"] = _best_pick(g_smi, s_smi)

    g_inchi, s_inchi = _per_source("gnps_InChI", "sirius_InChI")
    result["InChI_gnps"] = g_inchi
    result["InChI_sirius"] = s_inchi
    result["InChI"] = _best_pick(g_inchi, s_inchi)

    _, s_formula = _per_source("", "sirius_mol_formula")
    result["mol_formula_sirius"] = s_formula
    result["mol_formula"] = _best_pick(
        pd.Series(pd.NA, index=merged.index, dtype=object), s_formula, "suspect_formula_sirius"
    )

    _, s_adduct = _per_source("", "sirius_adduct")
    result["adduct_sirius"] = s_adduct
    result["adduct"] = _best_pick(
        pd.Series(pd.NA, index=merged.index, dtype=object), s_adduct, "suspect_adduct"
    )

    result["sirius_high_confidence"] = sirius_high

    # -- Taxonomy curated columns ---------------------------------------------
    # NPC priority: GNPS library → SIRIUS-structure-derived (when sirius_high)
    #               → CANOPUS (per-level, probability-gated).
    # ClassyFire priority: GNPS library → CANOPUS (no structure-derived source).

    struct_pathway_has = (
        _col(merged, "sirius_struct_npc_pathway").notna() & sirius_high
    )
    struct_superclass_has = (
        _col(merged, "sirius_struct_npc_superclass").notna() & sirius_high
    )
    struct_class_has = (
        _col(merged, "sirius_struct_npc_class").notna() & sirius_high
    )

    result["npc_pathway"] = _coalesce(merged, [
        ("gnps_npc_pathway", gnps_has),
        ("sirius_struct_npc_pathway", struct_pathway_has),
        ("canopus_npc_pathway", canopus_pathway_has),
    ])

    result["npc_superclass"] = _coalesce(merged, [
        ("gnps_npc_superclass", gnps_has),
        ("sirius_struct_npc_superclass", struct_superclass_has),
        ("canopus_npc_superclass", canopus_superclass_has),
    ])

    result["npc_class"] = _coalesce(merged, [
        ("gnps_npc_class", gnps_has),
        ("sirius_struct_npc_class", struct_class_has),
        ("canopus_npc_class", canopus_class_has),
    ])

    result["classyfire_class"] = _coalesce(merged, [
        ("gnps_classyfire_class", gnps_has),
        ("canopus_classyfire_class", canopus_classyfire_has),
    ])

    # ClassyFire subclass and level 5 — CANOPUS only (GNPS doesn't carry them).
    # Gated on the corresponding ``*_passed`` flag emitted by the CANOPUS loader.
    for sub_level in ("classyfire_subclass", "classyfire_level5"):
        col = f"canopus_{sub_level}"
        passed_col = f"canopus_{sub_level}_passed"
        if col in merged.columns and passed_col in merged.columns:
            has = (
                _col(merged, col).notna()
                & _col(merged, passed_col).fillna(False).astype(bool)
            )
            result[sub_level] = _coalesce(merged, [(col, has)])
        else:
            result[sub_level] = pd.NA

    struct_has = struct_pathway_has | struct_superclass_has | struct_class_has

    # -- Confidence -----------------------------------------------------------

    result["confidence"] = pd.NA
    if "gnps_MQScore" in merged.columns:
        result.loc[gnps_has, "confidence"] = merged.loc[gnps_has, "gnps_MQScore"]
    if "sirius_confidence" in merged.columns:
        sirius_only = sirius_has & ~gnps_has
        result.loc[sirius_only, "confidence"] = merged.loc[sirius_only, "sirius_confidence"]

    # -- Annotation source ----------------------------------------------------

    sources = pd.DataFrame({
        "gnps": gnps_has,
        "sirius": sirius_has,
        "sirius_struct_npc": struct_has,
        "canopus": canopus_has,
        "suspect": suspect_has,
    }, index=merged.index)

    result["annotation_source"] = sources.apply(
        lambda row: ":".join(k for k, v in row.items() if v) or "Unannotated",
        axis=1,
    )

    result["is_suspect_only"] = result["annotation_source"] == "suspect"
    result["is_canopus_only"] = result["annotation_source"] == "canopus"

    # -- InChIKey agreement (GNPS vs SIRIUS high-conf only) -------------------

    gnps_ik = _col(merged, "gnps_InChIkey2D")
    # Gate on high confidence — low-conf SIRIUS hits must not influence agreement.
    sirius_ik = _col(merged, "sirius_InChIkey2D").where(sirius_high)

    both_present = gnps_ik.notna() & sirius_ik.notna()
    agree = both_present & (gnps_ik == sirius_ik)
    disagree = both_present & (gnps_ik != sirius_ik)
    single = (gnps_ik.notna() | sirius_ik.notna()) & ~both_present

    result["inchikey_agreement"] = pd.NA
    result.loc[agree, "inchikey_agreement"] = "agree"
    result.loc[disagree, "inchikey_agreement"] = "disagree"
    result.loc[single, "inchikey_agreement"] = "single_source"

    # -- Taxonomy alt columns (GNPS vs CANOPUS, whenever they differ) --------
    # Structural disagreement is visible by comparing *_gnps vs *_sirius directly.
    gnps_npc = _col(merged, "gnps_npc_class")
    canopus_npc = _col(merged, "canopus_npc_class")
    npc_differs = gnps_npc.notna() & canopus_npc.notna() & (gnps_npc != canopus_npc)

    gnps_cf = _col(merged, "gnps_classyfire_class")
    canopus_cf = _col(merged, "canopus_classyfire_class")
    cf_differs = gnps_cf.notna() & canopus_cf.notna() & (gnps_cf != canopus_cf)

    result["npc_class_alt"] = pd.NA
    result["classyfire_class_alt"] = pd.NA

    if "canopus_npc_class" in merged.columns:
        result.loc[npc_differs, "npc_class_alt"] = merged.loc[npc_differs, "canopus_npc_class"]
    if "canopus_classyfire_class" in merged.columns:
        result.loc[cf_differs, "classyfire_class_alt"] = merged.loc[cf_differs, "canopus_classyfire_class"]

    # -- GNPS↔SIRIUS top-k cross-check ----------------------------------------

    if sirius_topk_df is not None:
        cross = compute_gnps_sirius_topk_match(
            gnps_df, sirius_topk_df, min_confidence=sirius_min_confidence
        )
        result = result.join(cross, how="left")
        if "gnps_sirius_rank" not in result.columns:
            result["gnps_sirius_rank"] = pd.array([pd.NA] * len(result), dtype="Int64")
        if "gnps_sirius_highconf" not in result.columns:
            result["gnps_sirius_highconf"] = False
        else:
            result["gnps_sirius_highconf"] = result["gnps_sirius_highconf"].fillna(False).astype(bool)

    # -- Drop fully unannotated rows ------------------------------------------

    result = result[result["annotation_source"] != "Unannotated"]

    return result


def _col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return the named column or a null Series aligned to df's index."""
    if col in df.columns:
        return df[col]
    return pd.Series(pd.NA, index=df.index, dtype=object)


def _coalesce(
    df: pd.DataFrame,
    priority: list[tuple[str, pd.Series]],
) -> pd.Series:
    """Return first non-null value per row from columns in priority order.

    Args:
        df: Source DataFrame.
        priority: List of (column_name, mask) pairs in descending priority.

    Returns:
        Series with the coalesced values (dtype object).
    """
    result = pd.Series(pd.NA, index=df.index, dtype=object)
    for col, mask in reversed(priority):
        if col in df.columns:
            valid = mask & df[col].notna()
            result = result.where(~valid, df[col])
    return result
