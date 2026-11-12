"""Assemble all project data into a single JSON-serializable dict for the viewer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from metabproject.project import MetabProject


def assemble_explore_data(
    project: "MetabProject",
    render_structures: bool = True,
    feature_relations: str | Path | None = None,
) -> dict[str, Any]:
    """Assemble all available project data into a JSON-serializable dict.

    Works at any pipeline stage — sections are populated based on what has
    been run. Requires at minimum that ``process`` has been run.

    Args:
        project: An open :class:`~metabproject.project.MetabProject`.
        render_structures: Whether to render 2D structures from SMILES via
            RDKit. Set ``False`` if RDKit is not installed.
        feature_relations: Optional TSV linking derived ions (isotopes,
            adducts, in-source fragments) to their parent feature: one row per
            derived feature with columns ``feature_id``, ``parent``, ``role``,
            ``relation`` and optionally ``r``, ``ms2_support``, ``r_samples``,
            ``ratio_cv``, ``isf_confident``.

    Returns:
        A JSON-serializable dict containing feature data, contrast metadata,
        IIN groups, and filter options.

    Raises:
        FileNotFoundError: If the processed feature table is not available.
    """
    # -- Core data (always required) ------------------------------------------
    ft = project.feature_table  # raises if process not run
    meta = project.metadata
    ms1 = project.ms1_features

    groups = sorted(meta["group"].unique().tolist())
    group_members: dict[str, list[str]] = {
        g: meta.index[meta["group"] == g].tolist() for g in groups
    }

    # -- Annotations (optional) -----------------------------------------------
    ann_df: pd.DataFrame | None = None
    gnps_df: pd.DataFrame | None = None
    sirius_df: pd.DataFrame | None = None
    canopus_df: pd.DataFrame | None = None
    suspect_df: pd.DataFrame | None = None

    try:
        ann_df = project.annotations
    except FileNotFoundError:
        pass

    if ann_df is not None:
        try:
            gnps_df = project.gnps_annotations
        except FileNotFoundError:
            pass
        try:
            sirius_df = project.sirius_annotations
        except FileNotFoundError:
            pass
        try:
            canopus_df = project.canopus_annotations
        except FileNotFoundError:
            pass
        try:
            suspect_df = project.gnps_suspect_annotations
        except (FileNotFoundError, AttributeError):
            pass

    # -- Project palette (optional) -------------------------------------------
    # Loaded before structure rendering on purpose: this pulls in metabproject.viz,
    # which imports seaborn. Importing seaborn *after* RDKit's drawing module has
    # initialised segfaults on some native-lib builds, so seaborn must load first.
    palette: dict[str, str] = {}
    try:
        from metabproject.viz._style import load_palette_yaml

        palette = load_palette_yaml(project.path) or {}
    except Exception:
        palette = {}

    # -- Render structures (optional) -----------------------------------------
    svg_map: dict[str, str] = {}
    if render_structures and ann_df is not None and "smiles" in ann_df.columns:
        from metabproject.explore.structures import render_structures as _render

        svg_map = _render(ann_df["smiles"])

    # -- Stats (optional) -----------------------------------------------------
    contrasts_dict = project.contrasts  # empty dict if stats not run

    # Recover contrast metadata (group labels + sample lists) from manifest.
    contrasts_meta = _build_contrasts_meta(project, meta)

    # -- Comparisons (optional) -----------------------------------------------
    comparisons_data = project.comparisons  # empty dict if none persisted
    residual_diag = _load_residual_diagnostics(project, comparisons_data)
    comparisons_meta = _build_comparisons_meta(comparisons_data, meta, residual_diag)

    # -- IIN validation (optional) --------------------------------------------
    iin_val_df: pd.DataFrame | None = None
    try:
        iin_val_df = project.iin_validation
    except FileNotFoundError:
        pass

    # -- Molecular network (optional) -----------------------------------------
    net_edges = project.network_edges  # None if no network ingested
    net_nodes = project.network_nodes

    # -- MS2 spectra (optional) -----------------------------------------------
    ms2_df = project.ms2_spectra  # None if no spectra ingested

    # -- Spectral similarity (optional) ---------------------------------------
    try:
        sim_df = project.spectral_similarity  # None if not computed
    except Exception:
        sim_df = None
    sim_lookup, similarity_meta = _build_similarity_lookup(sim_df)
    if similarity_meta is not None:
        try:
            from metabproject.manifest import read_manifest

            for op in reversed(read_manifest(project.path).get("operations", [])):
                if op.get("command") == "similarity":
                    similarity_meta["tolerance"] = op.get("parameters", {}).get(
                        "tolerance"
                    )
                    break
        except Exception:
            pass

    # -- Feature relations (optional, external TSV) ----------------------------
    rel_df: pd.DataFrame | None = None
    if feature_relations is not None:
        rel_df = pd.read_csv(feature_relations, sep="\t").set_index("feature_id")

    # -- Build per-feature data -----------------------------------------------
    features_data: dict[str, Any] = {}

    for fid in ft.index:
        fid_str = str(fid)
        row_ms1 = ms1.loc[fid] if fid in ms1.index else None

        # Basic feature properties.
        feat: dict[str, Any] = {
            "feature_id": int(fid),
            "mz": _safe_float(row_ms1, "mz") if row_ms1 is not None else None,
            "rt": _safe_float(row_ms1, "rt") if row_ms1 is not None else None,
            "charge": _safe_val(row_ms1, "charge") if row_ms1 is not None else None,
        }

        # Processed abundances for boxplots.
        feat["abundances"] = {
            col: _to_json_float(ft.at[fid, col]) for col in ft.columns
        }

        # Merged annotation.
        feat["annotation"] = _build_annotation(fid, ann_df)

        # Per-source breakdown.
        feat["sources"] = _build_sources(fid, gnps_df, sirius_df, canopus_df, suspect_df)

        # 2D structure SVG.
        smiles = feat["annotation"].get("smiles") if feat["annotation"] else None
        feat["structure_svg"] = svg_map.get(smiles) if smiles else None

        # Stats per contrast.
        feat["contrasts"] = _build_feature_contrasts(fid, contrasts_dict)

        # Factorial comparisons (anova_1way / anova_2way / welch_anova / kruskal).
        if comparisons_data:
            feat["comparisons"] = _build_feature_comparisons(
                fid, comparisons_data, residual_diag,
            )

        # IIN membership.
        feat["iin"] = _build_iin_membership(fid, row_ms1)

        # Molecular-network membership.
        feat["network"] = _build_network_membership(fid, net_nodes)

        # Isotope / adduct / in-source-fragment links.
        feat["derived_from"], feat["derived_ions"] = _build_relations(fid, rel_df, ft.index)

        # MS2 spectrum (processed for display).
        feat["ms2"] = _build_ms2(fid, ms2_df)

        features_data[fid_str] = feat

    # -- Build IIN groups data ------------------------------------------------
    iin_groups_data: dict[str, Any] = {}
    if iin_val_df is not None and not iin_val_df.empty:
        iin_groups_data = _build_iin_groups(iin_val_df, features_data, contrasts_dict)
    else:
        # Build minimal IIN groups from ms1_features even without validation.
        iin_groups_data = _build_iin_groups_from_ms1(ms1, features_data)

    # -- Build molecular-network components ------------------------------------
    network_components = _build_network_components(
        net_edges, features_data, sim_lookup
    )

    # -- Filter options -------------------------------------------------------
    filter_options = _build_filter_options(features_data, contrasts_dict, comparisons_data)

    return {
        "project_name": project.path.name,
        "n_features": len(ft),
        "n_samples": len(meta),
        "groups": groups,
        "group_members": group_members,
        "features": features_data,
        "contrasts_meta": contrasts_meta,
        "comparisons_meta": comparisons_meta,
        "iin_groups": iin_groups_data,
        "network_components": network_components,
        "similarity_meta": similarity_meta,
        "filter_options": filter_options,
        "palette": palette,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _safe_float(row: pd.Series, col: str) -> float | None:
    if col not in row.index:
        return None
    v = row[col]
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_val(row: pd.Series, col: str) -> Any:
    if col not in row.index:
        return None
    v = row[col]
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _to_json_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _build_annotation(fid: Any, ann_df: pd.DataFrame | None) -> dict[str, Any] | None:
    if ann_df is None or fid not in ann_df.index:
        return None
    row = ann_df.loc[fid]
    result: dict[str, Any] = {}
    cols = [
        "compound_name", "smiles", "inchikey2d", "mol_formula", "adduct",
        "annotation_source", "confidence",
        "npc_pathway", "npc_superclass", "npc_class", "classyfire_class",
        "npc_class_alt", "classyfire_class_alt",
        "compound_name_gnps", "compound_name_sirius",
        "inchikey2d_gnps", "inchikey2d_sirius",
        "sirius_high_confidence", "is_suspect_only", "is_canopus_only",
        "inchikey_agreement",
    ]
    for col in cols:
        if col in row.index:
            v = row[col]
            if isinstance(v, float) and math.isnan(v):
                result[col] = None
            elif isinstance(v, (np.integer,)):
                result[col] = int(v)
            elif isinstance(v, (np.floating,)):
                result[col] = float(v)
            elif isinstance(v, (np.bool_,)):
                result[col] = bool(v)
            else:
                result[col] = v
        else:
            result[col] = None
    return result


def _build_sources(
    fid: Any,
    gnps_df: pd.DataFrame | None,
    sirius_df: pd.DataFrame | None,
    canopus_df: pd.DataFrame | None,
    suspect_df: pd.DataFrame | None,
) -> dict[str, Any | None]:
    sources: dict[str, Any] = {}

    # GNPS library.
    if gnps_df is not None and fid in gnps_df.index:
        row = gnps_df.loc[fid]
        sources["gnps"] = {
            col: _series_val(row, col)
            for col in ["gnps_compound_name", "gnps_smiles", "gnps_InChIkey2D",
                        "gnps_MQScore", "gnps_npc_pathway", "gnps_npc_superclass",
                        "gnps_npc_class", "gnps_classyfire_class"]
            if col in row.index
        }
    else:
        sources["gnps"] = None

    # SIRIUS.
    if sirius_df is not None and fid in sirius_df.index:
        row = sirius_df.loc[fid]
        sources["sirius"] = {
            col: _series_val(row, col)
            for col in ["sirius_compound_name", "sirius_confidence",
                        "sirius_InChIkey2D", "sirius_smiles", "sirius_mol_formula",
                        "sirius_adduct", "sirius_high_confidence"]
            if col in row.index
        }
        # `sirius_databases` is a list[str] (from the SIRIUS `links` column);
        # _series_val would unwrap it incorrectly, so handle it directly.
        if "sirius_databases" in row.index:
            dbs = row["sirius_databases"]
            if dbs is not None and not (isinstance(dbs, float) and math.isnan(dbs)):
                sources["sirius"]["sirius_databases"] = [str(d) for d in dbs]
    else:
        sources["sirius"] = None

    # CANOPUS.
    if canopus_df is not None and fid in canopus_df.index:
        row = canopus_df.loc[fid]
        sources["canopus"] = {
            col: _series_val(row, col)
            for col in ["canopus_npc_pathway", "canopus_npc_pathway_probability",
                        "canopus_npc_superclass", "canopus_npc_superclass_probability",
                        "canopus_npc_class", "canopus_npc_class_probability",
                        "canopus_classyfire_class", "canopus_classyfire_class_probability"]
            if col in row.index
        }
    else:
        sources["canopus"] = None

    # Suspect list.
    if suspect_df is not None and fid in suspect_df.index:
        row = suspect_df.loc[fid]
        sources["suspect"] = {
            col: _series_val(row, col)
            for col in ["suspect_compound_name", "suspect_formula_sirius",
                        "suspect_delta_mz", "suspect_adduct"]
            if col in row.index
        }
    else:
        sources["suspect"] = None

    return sources


def _series_val(row: pd.Series, col: str) -> Any:
    v = row[col]
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _build_feature_contrasts(
    fid: Any,
    contrasts_dict: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, df in contrasts_dict.items():
        if fid not in df.index:
            continue
        row = df.loc[fid]
        result[name] = {
            "log2fc": _to_json_float(row.get("log2fc")),
            "p_value": _to_json_float(row.get("p_value")),
            "p_adj": _to_json_float(row.get("p_adj")),
            "direction": str(row["direction"]) if "direction" in row.index else None,
            "significant": bool(row["significant"]) if "significant" in row.index else None,
            "mean_a": _to_json_float(row.get("mean_a")),
            "mean_b": _to_json_float(row.get("mean_b")),
        }
    return result


def _build_iin_membership(fid: Any, row_ms1: pd.Series | None) -> dict[str, Any] | None:
    if row_ms1 is None:
        return None
    iin_id_col = "ion_identities:iin_id"
    if iin_id_col not in row_ms1.index:
        return None
    iin_id = row_ms1[iin_id_col]
    if iin_id is None or (isinstance(iin_id, float) and math.isnan(iin_id)):
        return None

    partner_ids_raw = row_ms1.get("ion_identities:partner_row_ids", "")
    ion_identity = row_ms1.get("ion_identities:ion_identities", None)

    partner_ids: list[int] = []
    if partner_ids_raw and isinstance(partner_ids_raw, str):
        for p in partner_ids_raw.split(","):
            p = p.strip()
            if p:
                try:
                    partner_ids.append(int(float(p)))
                except ValueError:
                    pass

    return {
        "iin_id": int(float(iin_id)),
        "ion_identity": str(ion_identity) if ion_identity and not (
            isinstance(ion_identity, float) and math.isnan(ion_identity)
        ) else None,
        "partner_ids": partner_ids,
    }


def _build_iin_groups(
    iin_val_df: pd.DataFrame,
    features_data: dict[str, Any],
    contrasts_dict: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for iin_id, row in iin_val_df.iterrows():
        iin_id_str = str(int(iin_id))

        # Parse feature_ids from the validation table.
        feature_ids_raw = row.get("feature_ids", "")
        feature_ids: list[int] = []
        if isinstance(feature_ids_raw, str):
            for p in feature_ids_raw.split(","):
                p = p.strip()
                if p:
                    try:
                        feature_ids.append(int(float(p)))
                    except ValueError:
                        pass

        ion_identities_raw = row.get("ion_identities", "")
        ion_identities: list[str] = []
        if isinstance(ion_identities_raw, str):
            ion_identities = [s.strip() for s in ion_identities_raw.split(";") if s.strip()]

        # Trend consistency per contrast.
        trend_consistent: dict[str, bool | None] = {}
        for col in iin_val_df.columns:
            if col.startswith("trend_") and col.endswith("_consistent"):
                contrast_name = col[len("trend_"):-len("_consistent")]
                v = row[col]
                if isinstance(v, float) and math.isnan(v):
                    trend_consistent[contrast_name] = None
                else:
                    trend_consistent[contrast_name] = bool(v)

        ann_consistent = row.get("annotation_consistent", None)
        if isinstance(ann_consistent, float) and math.isnan(ann_consistent):
            ann_consistent = None
        elif ann_consistent is not None:
            ann_consistent = bool(ann_consistent)

        # Denormalized members for quick rendering.
        members = []
        for fid in feature_ids:
            fd = features_data.get(str(fid))
            if fd is None:
                continue
            iin_mem = fd.get("iin") or {}
            members.append({
                "feature_id": fid,
                "ion_identity": iin_mem.get("ion_identity"),
                "compound_name": (fd.get("annotation") or {}).get("compound_name"),
                "annotation_source": (fd.get("annotation") or {}).get("annotation_source"),
                "contrasts": fd.get("contrasts", {}),
            })

        groups[iin_id_str] = {
            "iin_id": int(iin_id),
            "feature_ids": feature_ids,
            "neutral_mass": _to_json_float(row.get("neutral_mass")),
            "ion_identities": ion_identities,
            "annotation_consistent": ann_consistent,
            "trend_consistent": trend_consistent,
            "n_features": len(feature_ids),
            "members": members,
        }
    return groups


def _build_iin_groups_from_ms1(
    ms1: pd.DataFrame,
    features_data: dict[str, Any],
) -> dict[str, Any]:
    """Build minimal IIN groups directly from ms1_features (no validation table)."""
    iin_id_col = "ion_identities:iin_id"
    if iin_id_col not in ms1.columns:
        return {}

    # Group features by iin_id.
    iin_col = ms1[iin_id_col]
    valid = iin_col.dropna()
    if valid.empty:
        return {}

    groups: dict[str, Any] = {}
    for iin_id, group_ms1 in ms1[iin_col.notna()].groupby(iin_id_col):
        iin_id_str = str(int(iin_id))
        feature_ids = [int(fid) for fid in group_ms1.index.tolist()]

        # Neutral mass from first member.
        neutral_mass = None
        nm_col = "ion_identities:neutral_mass"
        if nm_col in group_ms1.columns:
            neutral_mass = _to_json_float(group_ms1[nm_col].iloc[0])

        # Ion identities list.
        ion_id_col = "ion_identities:ion_identities"
        ion_identities = []
        if ion_id_col in group_ms1.columns:
            ion_identities = [
                str(v) for v in group_ms1[ion_id_col].dropna().tolist()
            ]

        members = []
        for fid in feature_ids:
            fd = features_data.get(str(fid))
            if fd is None:
                continue
            iin_mem = fd.get("iin") or {}
            members.append({
                "feature_id": fid,
                "ion_identity": iin_mem.get("ion_identity"),
                "compound_name": (fd.get("annotation") or {}).get("compound_name"),
                "annotation_source": (fd.get("annotation") or {}).get("annotation_source"),
                "contrasts": fd.get("contrasts", {}),
            })

        groups[iin_id_str] = {
            "iin_id": int(iin_id),
            "feature_ids": feature_ids,
            "neutral_mass": neutral_mass,
            "ion_identities": ion_identities,
            "annotation_consistent": None,
            "trend_consistent": {},
            "n_features": len(feature_ids),
            "members": members,
        }
    return groups


def _build_network_membership(
    fid: Any, net_nodes: pd.DataFrame | None,
) -> dict[str, Any] | None:
    """Per-feature GNPS molecular-network membership, or None if absent."""
    if net_nodes is None or fid not in net_nodes.index:
        return None
    row = net_nodes.loc[fid]
    return {
        "component_id": int(row["component_id"]),
        "degree": int(row["degree"]),
        "n_component_members": int(row["n_component_members"]),
    }


_RELATION_FIELDS = ["role", "relation", "r", "ms2_support", "r_samples", "ratio_cv", "isf_confident"]


def _build_relations(
    fid: Any, rel_df: pd.DataFrame | None, in_table: pd.Index,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """This feature's parent link (if it is a derived ion) and its own derived ions."""
    if rel_df is None:
        return None, []

    def link(row: pd.Series, other: int) -> dict[str, Any]:
        d = {k: _to_json_scalar(row[k]) for k in _RELATION_FIELDS if k in row.index}
        d["feature_id"] = int(other)
        d["in_table"] = bool(other in in_table)
        return d

    parent = link(rel_df.loc[fid], rel_df.at[fid, "parent"]) if fid in rel_df.index else None
    children = [link(row, child) for child, row in rel_df[rel_df["parent"] == fid].iterrows()]
    return parent, children


def _to_json_scalar(v: Any) -> Any:
    if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA:
        return None
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v)
    return v


def _build_ms2(
    fid: Any, ms2_df: pd.DataFrame | None,
) -> dict[str, Any] | None:
    """Per-feature processed MS2 spectrum, or None if no spectrum stored."""
    if ms2_df is None or fid not in ms2_df.index:
        return None
    from metabproject.explore.spectra import process_spectrum_for_display

    row = ms2_df.loc[fid]
    charge = row["precursor_charge"]
    return process_spectrum_for_display(
        mz=row["mz"],
        intensity=row["intensity"],
        precursor_mz=row["precursor_mz"],
        precursor_charge=None if pd.isna(charge) else int(charge),
    )


def _build_similarity_lookup(
    sim_df: pd.DataFrame | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Nested spectral-similarity lookup + metadata for the viewer.

    Builds ``{"<lo>|<hi>": {score_type: {scaling: {"score", "matches"}}}}`` from
    the long-format ``spectral_similarity`` table, where ``matches`` is a list of
    ``[idx_a, idx_b]`` peak index pairs (into the displayed spectra). Also returns
    a ``{"score_types", "scalings"}`` metadata dict the viewer uses to build the
    compare-panel dropdowns. Returns ``({}, None)`` when no similarity exists.
    """
    if sim_df is None or sim_df.empty:
        return {}, None

    lookup: dict[str, Any] = {}
    for _, r in sim_df.iterrows():
        a, b = int(r["feature_id_a"]), int(r["feature_id_b"])
        key = f"{a}|{b}"
        matches = [
            [int(i), int(j)]
            for i, j in zip(r["matched_idx_a"], r["matched_idx_b"])
        ]
        lookup.setdefault(key, {}).setdefault(str(r["score_type"]), {})[
            str(r["scaling"])
        ] = {"score": _to_json_float(r["score"]), "matches": matches}

    # Order scalings by the canonical preset order, extras appended.
    from metabproject.spectra.similarity import SCALINGS

    order = list(SCALINGS)
    present = set(sim_df["scaling"].unique())
    scalings = [s for s in order if s in present] + sorted(
        present - set(order)
    )
    meta = {
        "score_types": sorted(sim_df["score_type"].unique().tolist()),
        "scalings": scalings,
    }
    return lookup, meta


def _build_network_components(
    net_edges: pd.DataFrame | None,
    features_data: dict[str, Any],
    sim_lookup: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build GNPS molecular-network components (families) from the edge list.

    Each component carries its node ids, compact ``[id1, id2, cosine]`` edge
    triples for graph drawing, a light member table, and (when computed) a
    ``similarity`` block keyed by ``"<lo>|<hi>"`` with matchms cosine scores and
    matched peak indices for member pairs. Abundances are not duplicated — the
    front-end reads ``features[fid].abundances``. Returns an empty dict when no
    network has been ingested.
    """
    if net_edges is None or net_edges.empty:
        return {}
    sim_lookup = sim_lookup or {}

    components: dict[str, Any] = {}
    for component_id, grp in net_edges.groupby("component_id"):
        component_id_str = str(int(component_id))

        edges: list[list[Any]] = []
        node_ids: set[int] = set()
        for _, e in grp.iterrows():
            id1 = int(e["feature_id_1"])
            id2 = int(e["feature_id_2"])
            edges.append([id1, id2, _to_json_float(e["cosine"])])
            node_ids.add(id1)
            node_ids.add(id2)

        feature_ids = sorted(node_ids)
        members = []
        for fid in feature_ids:
            fd = features_data.get(str(fid))
            if fd is None:
                continue
            ann = fd.get("annotation") or {}
            members.append({
                "feature_id": fid,
                "compound_name": ann.get("compound_name"),
                "annotation_source": ann.get("annotation_source"),
                "mz": fd.get("mz"),
                "rt": fd.get("rt"),
            })

        # Attach matchms similarity for member pairs (feature_ids sorted, so the
        # "<lo>|<hi>" key matches the stored ordering).
        comp_sim: dict[str, Any] = {}
        for i, fa in enumerate(feature_ids):
            for fb in feature_ids[i + 1:]:
                key = f"{fa}|{fb}"
                if key in sim_lookup:
                    comp_sim[key] = sim_lookup[key]

        components[component_id_str] = {
            "component_id": int(component_id),
            "feature_ids": feature_ids,
            "n_features": len(feature_ids),
            "n_edges": len(edges),
            "edges": edges,
            "members": members,
            "similarity": comp_sim,
        }
    return components


def _build_contrasts_meta(
    project: "MetabProject",
    meta: pd.DataFrame,
) -> dict[str, Any]:
    """Recover contrast group labels and sample lists from manifest."""
    from metabproject.manifest import read_manifest

    manifest = read_manifest(project.path)
    stats_entries = [op for op in manifest["operations"] if op.get("command") == "stats"]
    if not stats_entries:
        return {}

    last_stats = stats_entries[-1]
    contrasts_file = last_stats.get("parameters", {}).get("contrasts_file")
    if not contrasts_file or not Path(contrasts_file).exists():
        # Fallback: derive sample groups from metadata only.
        return _contrasts_meta_from_stats_files(project, meta)

    try:
        from metabproject.stats.contrasts import load_contrasts, resolve_samples

        contrast_list = load_contrasts(contrasts_file)
        result: dict[str, Any] = {}
        for c in contrast_list:
            try:
                samples_a, samples_b = resolve_samples(c, meta)
            except ValueError:
                continue
            # Derive human-readable group label from the filter dict.
            label_a = _filter_label(c.a, meta)
            label_b = _filter_label(c.b, meta)
            result[c.name] = {
                "name": c.name,
                "group_a": label_a,
                "group_b": label_b,
                "samples_a": samples_a,
                "samples_b": samples_b,
                "filter_a": {k: v for k, v in c.a.items()},
                "filter_b": {k: v for k, v in c.b.items()},
            }
        return result
    except Exception:
        return _contrasts_meta_from_stats_files(project, meta)


def _contrasts_meta_from_stats_files(
    project: "MetabProject",
    meta: pd.DataFrame,
) -> dict[str, Any]:
    """Fallback: build contrast metadata from parquet filenames only."""
    stats_dir = project.path / "stats"
    if not stats_dir.exists():
        return {}
    result: dict[str, Any] = {}
    for p in sorted(stats_dir.glob("*.parquet")):
        name = p.stem
        # Use all samples in each group as a best-effort fallback.
        groups = sorted(meta["group"].unique().tolist())
        if len(groups) >= 2:
            g_a, g_b = groups[0], groups[1]
            samples_a = meta.index[meta["group"] == g_a].tolist()
            samples_b = meta.index[meta["group"] == g_b].tolist()
        else:
            samples_a, samples_b = [], []
        result[name] = {
            "name": name,
            "group_a": samples_a[0].split("_")[0] if samples_a else "A",
            "group_b": samples_b[0].split("_")[0] if samples_b else "B",
            "samples_a": samples_a,
            "samples_b": samples_b,
        }
    return result


def _filter_label(filter_dict: dict[str, list[Any]], meta: pd.DataFrame) -> str:
    """Derive a short human-readable label from a contrast filter dict."""
    parts = []
    for col, vals in filter_dict.items():
        parts.append("+".join(str(v) for v in vals))
    return "/".join(parts)


def _load_residual_diagnostics(
    project: "MetabProject",
    comparisons: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Load ``residual_diagnostics_per_feature.parquet`` for any comparison.

    Returns a dict ``{comparison_name → per_feature_df}`` containing only
    comparisons whose parquet exists on disk.
    """
    result: dict[str, pd.DataFrame] = {}
    for name in comparisons:
        path = project.path / "stats" / name / "residual_diagnostics_per_feature.parquet"
        if not path.exists():
            continue
        try:
            result[name] = pd.read_parquet(path)
        except Exception:
            continue
    return result


def _build_comparisons_meta(
    comparisons: dict[str, Any],
    meta: pd.DataFrame,
    residual_diag: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Compact JSON-serialisable summary of every persisted comparison.

    For each comparison: kind, factors, posthoc kind, alpha, the sorted
    unique level set per factor (from ``meta``), and a ``sample → {factor →
    level}`` map. The JS viewer uses ``sample_factors`` + ``levels`` to
    bucket each sample into the right cell of the grouped boxplot.
    """
    result: dict[str, Any] = {}
    for name, payload in comparisons.items():
        factors: list[str] = list(payload.get("factors") or [])
        present = [f for f in factors if f in meta.columns]
        levels: dict[str, list[str]] = {}
        sample_factors: dict[str, dict[str, str]] = {}
        for f in present:
            levels[f] = sorted(meta[f].dropna().astype(str).unique().tolist())
        if present:
            for sample, row in meta[present].iterrows():
                sample_factors[str(sample)] = {
                    f: (str(row[f]) if pd.notna(row[f]) else None) for f in present
                }
        result[name] = {
            "name": name,
            "kind": payload.get("kind"),
            "factors": factors,
            "posthoc": payload.get("posthoc"),
            "alpha": _to_json_float(payload.get("alpha")),
            "levels": levels,
            "sample_factors": sample_factors,
            "has_posthoc": payload.get("posthoc_df") is not None,
            "has_residual_diag": bool(residual_diag and name in residual_diag),
        }
    return result


def _build_feature_comparisons(
    fid: Any,
    comparisons: dict[str, Any],
    residual_diag: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Per-feature slice of every persisted comparison.

    Echoes ``kind``, ``classification`` (2-way only), and dynamic per-term
    p-values (``p_<factor>``, ``p_<factor>_adj``, ``sig_<factor>``,
    plus ``p_interaction*`` for 2-way). Filters the post-hoc long-format
    DataFrame down to rows for this feature.
    """
    result: dict[str, Any] = {}
    for name, payload in comparisons.items():
        omnibus: pd.DataFrame = payload["omnibus"]
        if fid not in omnibus.index:
            continue
        row = omnibus.loc[fid]

        factors: list[str] = list(payload.get("factors") or [])

        terms: dict[str, dict[str, Any]] = {}
        for f in factors:
            entry: dict[str, Any] = {}
            for src, key in (("p_", "p"), ("p_", "p_adj"), ("sig_", "sig")):
                col = f"sig_{f}" if key == "sig" else (
                    f"p_{f}_adj" if key == "p_adj" else f"p_{f}"
                )
                if col in row.index:
                    v = row[col]
                    if key == "sig":
                        entry[key] = (
                            None if (isinstance(v, float) and math.isnan(v))
                            else bool(v)
                        )
                    else:
                        entry[key] = _to_json_float(v)
            if entry:
                terms[f] = entry

        interaction: dict[str, Any] | None = None
        if "p_interaction" in row.index:
            interaction = {
                "p": _to_json_float(row.get("p_interaction")),
                "p_adj": _to_json_float(row.get("p_interaction_adj")),
                "sig": (
                    None if "sig_interaction" not in row.index
                    or (isinstance(row["sig_interaction"], float)
                        and math.isnan(row["sig_interaction"]))
                    else bool(row["sig_interaction"])
                ),
            }

        classification = None
        if "classification" in row.index:
            v = row["classification"]
            if not (isinstance(v, float) and math.isnan(v)):
                classification = str(v)

        # Per-feature post-hoc rows.
        ph_rows: list[dict[str, Any]] = []
        posthoc_df: pd.DataFrame | None = payload.get("posthoc_df")
        if posthoc_df is not None and not posthoc_df.empty and "feature" in posthoc_df.columns:
            sub = posthoc_df[posthoc_df["feature"] == fid]
            for _, r in sub.iterrows():
                ph_rows.append({
                    "factor": str(r["factor"]) if "factor" in r.index and pd.notna(r["factor"]) else None,
                    "group_A": str(r["group_A"]) if "group_A" in r.index and pd.notna(r["group_A"]) else None,
                    "group_B": str(r["group_B"]) if "group_B" in r.index and pd.notna(r["group_B"]) else None,
                    "mean_A": _to_json_float(r.get("mean_A")),
                    "mean_B": _to_json_float(r.get("mean_B")),
                    "diff": _to_json_float(r.get("diff")),
                    "p_value": _to_json_float(r.get("p_value")),
                    "p_adj": _to_json_float(r.get("p_adj")) if "p_adj" in r.index else None,
                    "significant": (
                        bool(r["significant"]) if "significant" in r.index
                        and not (isinstance(r["significant"], float) and math.isnan(r["significant"]))
                        else None
                    ),
                    "hedges_g": _to_json_float(r.get("hedges_g")) if "hedges_g" in r.index else None,
                })

        residual_entry: dict[str, Any] | None = None
        if residual_diag and name in residual_diag:
            rd = residual_diag[name]
            if fid in rd.index:
                rr = rd.loc[fid]
                in_full = rr.get("in_full_model", False)
                if isinstance(in_full, (np.bool_,)):
                    in_full = bool(in_full)
                normal = rr.get("normal_residuals", False)
                if isinstance(normal, (np.bool_,)):
                    normal = bool(normal)
                residual_entry = {
                    "in_full_model": bool(in_full),
                    "r2": _to_json_float(rr.get("r2")),
                    "residual_sigma": _to_json_float(rr.get("residual_sigma")),
                    "max_abs_std_resid": _to_json_float(rr.get("max_abs_std_resid")),
                    "shapiro_p": _to_json_float(rr.get("shapiro_p")),
                    "shapiro_p_adj": _to_json_float(rr.get("shapiro_p_adj")),
                    "normal_residuals": bool(normal) if in_full else None,
                }

        result[name] = {
            "kind": payload.get("kind"),
            "factors": factors,
            "classification": classification,
            "terms": terms,
            "interaction": interaction,
            "posthoc": ph_rows,
            "residual_diag": residual_entry,
        }
    return result


def _build_filter_options(
    features_data: dict[str, Any],
    contrasts_dict: dict[str, pd.DataFrame],
    comparisons_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sources: set[str] = set()
    pathways: set[str] = set()
    superclasses: set[str] = set()
    classes: set[str] = set()
    cf_classes: set[str] = set()

    for fd in features_data.values():
        ann = fd.get("annotation") or {}
        src = ann.get("annotation_source")
        if src:
            for s in str(src).split(":"):
                if s:
                    sources.add(s)
        if not src:
            sources.add("unannotated")
        for key, bucket in [
            ("npc_pathway", pathways),
            ("npc_superclass", superclasses),
            ("npc_class", classes),
            ("classyfire_class", cf_classes),
        ]:
            v = ann.get(key)
            if v:
                # Multi-label values ("Chromanes|Coumarins") offer each label.
                bucket.update(str(v).split("|"))

    comparisons_filt: dict[str, Any] = {}
    if comparisons_data:
        for name, payload in comparisons_data.items():
            omnibus = payload.get("omnibus")
            cls_set: set[str] = set()
            if omnibus is not None and "classification" in omnibus.columns:
                for v in omnibus["classification"].dropna().unique():
                    cls_set.add(str(v))
            comparisons_filt[name] = {
                "kind": payload.get("kind"),
                "factors": list(payload.get("factors") or []),
                "classifications": sorted(cls_set),
            }

    return {
        "annotation_sources": sorted(sources),
        "npc_pathways": sorted(pathways),
        "npc_superclasses": sorted(superclasses),
        "npc_classes": sorted(classes),
        "classyfire_classes": sorted(cf_classes),
        "contrast_names": sorted(contrasts_dict.keys()),
        "comparisons": comparisons_filt,
    }
