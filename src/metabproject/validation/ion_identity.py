"""Ion identity network (IIN) consistency validation.

Checks whether features grouped by MZmine as the same compound (different
adducts) are internally consistent in terms of fold-change direction across
contrasts and structural annotations from SIRIUS/GNPS.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


_IIN_ID_COL = "ion_identities:iin_id"
_IIN_PARTNER_COL = "ion_identities:partner_row_ids"
_IIN_IDENTITY_COL = "ion_identities:ion_identities"
_IIN_NEUTRAL_MASS_COL = "ion_identities:neutral_mass"

def parse_iin_groups(ms1_features: pd.DataFrame) -> pd.DataFrame:
    """Extract IIN group membership from ms1_features.

    Args:
        ms1_features: DataFrame indexed by ``feature_id`` with ion_identities
            columns from MZmine.

    Returns:
        DataFrame indexed by ``feature_id`` with columns:
        ``iin_id`` (int), ``ion_identity`` (str), ``partner_ids`` (list[int]).
        Only features that belong to an IIN group are included.

    Raises:
        ValueError: If ``ion_identities:iin_id`` is absent (IIN not enabled
            in MZmine).
    """
    if _IIN_ID_COL not in ms1_features.columns:
        raise ValueError(
            f"Ion identity network column not found in ms1_features: "
            f"['{_IIN_ID_COL}']\n"
            "Enable 'Ion Identity Networking' in MZmine before exporting."
        )

    mask = ms1_features[_IIN_ID_COL].notna()
    if not mask.any():
        return pd.DataFrame(
            columns=["iin_id", "ion_identity", "partner_ids"],
            index=pd.Index([], name="feature_id"),
        )

    df = ms1_features.loc[mask, [_IIN_ID_COL]].copy()
    df["iin_id"] = pd.to_numeric(df[_IIN_ID_COL], errors="coerce").astype("Int64")

    # Include ion_identity label when available.
    if _IIN_IDENTITY_COL in ms1_features.columns:
        df["ion_identity"] = ms1_features.loc[mask, _IIN_IDENTITY_COL].astype(str)
    else:
        df["ion_identity"] = pd.NA

    if _IIN_PARTNER_COL in ms1_features.columns:
        df["partner_ids"] = ms1_features.loc[mask, _IIN_PARTNER_COL].apply(_parse_id_list)
    else:
        # Derive partner_ids by grouping on iin_id: every other feature in
        # the same network is a partner. Some MZmine exports omit the
        # explicit partner_row_ids column.
        members_by_group = df.groupby("iin_id").apply(lambda g: g.index.tolist())
        df["partner_ids"] = [
            [pid for pid in members_by_group.loc[gid] if pid != fid]
            for fid, gid in zip(df.index, df["iin_id"])
        ]

    df = df.drop(columns=[_IIN_ID_COL])

    return df


def _parse_id_list(value: Any) -> list[int]:
    """Parse a comma-separated string of feature IDs to a list of ints."""
    if pd.isna(value) or str(value).strip() in ("", "nan", "NaN"):
        return []
    raw = re.sub(r"[\[\]{}]", "", str(value))
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    result = []
    for p in parts:
        try:
            result.append(int(float(p)))
        except (ValueError, TypeError):
            pass
    return result


def check_trend_consistency(
    iin_groups: pd.DataFrame,
    contrast_results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Check fold-change direction agreement within IIN groups per contrast.

    For each IIN group × contrast, counts the number of members that are "up",
    "down", or "ns". A group is consistent when all tested members that have a
    direction agree (all up, all down, or all ns). Features absent from the
    stats table (e.g., filtered out during processing) are treated as untested
    and excluded from the consistency check.

    Args:
        iin_groups: Output of :func:`parse_iin_groups`.
        contrast_results: Dict mapping contrast name to a stats DataFrame
            indexed by ``feature_id`` with a ``direction`` column
            (``"up"``, ``"down"``, ``"ns"``).

    Returns:
        DataFrame with one row per (iin_id, contrast). Columns:
        ``iin_id``, ``contrast``, ``n_features``, ``n_tested``,
        ``n_up``, ``n_down``, ``n_ns``, ``directions_observed``,
        ``trend_consistent``.
    """
    rows: list[dict] = []

    group_members = iin_groups.reset_index().groupby("iin_id")["feature_id"].apply(list)

    for contrast_name, stats_df in contrast_results.items():
        if "direction" not in stats_df.columns:
            continue
        direction_series = stats_df["direction"]

        for iin_id, fids in group_members.items():
            tested_fids = [f for f in fids if f in direction_series.index]
            directions = direction_series.loc[tested_fids] if tested_fids else pd.Series(dtype=str)

            n_up = int((directions == "up").sum())
            n_down = int((directions == "down").sum())
            n_ns = int((directions == "ns").sum())
            n_tested = len(tested_fids)

            # Consistent = all tested non-ns members agree, or all are ns/untested.
            non_ns = directions[directions != "ns"]
            trend_consistent: bool
            if len(non_ns) == 0:
                trend_consistent = True
            else:
                trend_consistent = non_ns.nunique() == 1

            rows.append({
                "iin_id": iin_id,
                "contrast": contrast_name,
                "n_features": len(fids),
                "n_tested": n_tested,
                "n_up": n_up,
                "n_down": n_down,
                "n_ns": n_ns,
                "directions_observed": ",".join(sorted(directions.unique())) if n_tested else "",
                "trend_consistent": trend_consistent,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "iin_id", "contrast", "n_features", "n_tested",
            "n_up", "n_down", "n_ns", "directions_observed", "trend_consistent",
        ])

    return pd.DataFrame(rows)


def check_annotation_consistency(
    iin_groups: pd.DataFrame,
    merged_annotations: pd.DataFrame,
    sirius_topk: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Check structural annotation agreement within IIN groups.

    Only features with a **high-confidence structural annotation** (non-null
    ``inchikey2d`` in merged_annotations, i.e. GNPS library match or high-
    confidence SIRIUS) are compared. CANOPUS-only, suspect-only, and
    unannotated features are excluded from the comparison — they do not
    reduce the comparison pool or trigger an inconsistency flag.

    When ``sirius_topk`` is provided, the check is relaxed: for each
    structurally-annotated feature, the rank-1 InChIKey2D is extended with
    its top-k SIRIUS candidates. A shared key anywhere in those sets counts
    as consistent (InChIKey2D is adduct-independent).

    Args:
        iin_groups: Output of :func:`parse_iin_groups`.
        merged_annotations: Merged annotation table indexed by ``feature_id``
            with an ``inchikey2d`` column. Only features with a non-null
            ``inchikey2d`` are considered structurally annotated.
        sirius_topk: Optional SIRIUS top-k DataFrame indexed by ``feature_id``
            (non-unique) with a ``sirius_InChIkey2D`` column.

    Returns:
        DataFrame with one row per iin_id. Columns:
        ``iin_id``, ``n_features``, ``n_hc_annotated``,
        ``annotation_sources``, ``unique_inchikeys_rank1``,
        ``shared_inchikey_rank1``, ``shared_inchikey_topk``,
        ``annotation_consistent``.
    """
    rows: list[dict] = []
    group_members = iin_groups.reset_index().groupby("iin_id")["feature_id"].apply(list)

    has_topk = sirius_topk is not None and "sirius_InChIkey2D" in sirius_topk.columns
    has_ann = "inchikey2d" in merged_annotations.columns
    has_source = "annotation_source" in merged_annotations.columns

    for iin_id, fids in group_members.items():
        n_features = len(fids)

        # High-confidence structurally annotated: non-null inchikey2d only.
        # This means GNPS library or high-confidence SIRIUS — not CANOPUS-only
        # or suspect-only, which carry no InChIKey in the merged table.
        hc_fids: list[int] = []
        sources: list[str] = []
        rank1_keys: list[str] = []

        if has_ann:
            for f in fids:
                if f not in merged_annotations.index:
                    continue
                key = merged_annotations.at[f, "inchikey2d"]
                if pd.isna(key):
                    continue
                hc_fids.append(f)
                rank1_keys.append(str(key))
                if has_source:
                    src = merged_annotations.at[f, "annotation_source"]
                    if pd.notna(src):
                        sources.append(str(src))

        n_hc_annotated = len(hc_fids)
        unique_rank1 = len(set(rank1_keys))
        shared_rank1: str | None = rank1_keys[0] if unique_rank1 == 1 and rank1_keys else None

        # Top-k overlap check: extend each HC-annotated feature's candidate
        # set with its SIRIUS top-k predictions, then intersect across members.
        shared_topk: str | None = None
        if has_topk and n_hc_annotated >= 2:
            candidate_sets: list[set[str]] = []
            for f in hc_fids:
                base_key = rank1_keys[hc_fids.index(f)]
                candidates: set[str] = {base_key}
                if f in sirius_topk.index:
                    topk_keys = sirius_topk.loc[[f], "sirius_InChIkey2D"].dropna()
                    candidates |= set(topk_keys.astype(str))
                candidate_sets.append(candidates)
            shared_keys = set.intersection(*candidate_sets)
            if shared_keys:
                shared_topk = next(iter(shared_keys))

        # Determine consistency.
        annotation_consistent: bool | None
        if n_hc_annotated < 2:
            annotation_consistent = None
        elif shared_rank1 is not None:
            annotation_consistent = True
        elif shared_topk is not None:
            annotation_consistent = True
        else:
            annotation_consistent = False

        rows.append({
            "iin_id": iin_id,
            "n_features": n_features,
            "n_hc_annotated": n_hc_annotated,
            "annotation_sources": ";".join(sorted(set(sources))) if sources else "",
            "unique_inchikeys_rank1": unique_rank1,
            "shared_inchikey_rank1": shared_rank1,
            "shared_inchikey_topk": shared_topk,
            "annotation_consistent": annotation_consistent,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "iin_id", "n_features", "n_hc_annotated", "annotation_sources",
            "unique_inchikeys_rank1", "shared_inchikey_rank1",
            "shared_inchikey_topk", "annotation_consistent",
        ])

    return pd.DataFrame(rows)


def build_iin_validation_table(
    iin_groups: pd.DataFrame,
    ms1_features: pd.DataFrame,
    trend_results: pd.DataFrame,
    annotation_results: pd.DataFrame,
) -> pd.DataFrame:
    """Combine trend and annotation checks into the final validation table.

    Pivots trend results to wide format (one ``trend_<contrast>_consistent``
    column per contrast) and merges with annotation results and IIN metadata.

    Args:
        iin_groups: Output of :func:`parse_iin_groups`.
        ms1_features: Full ms1_features DataFrame for neutral_mass / ion
            identity context.
        trend_results: Output of :func:`check_trend_consistency`.
        annotation_results: Output of :func:`check_annotation_consistency`.

    Returns:
        DataFrame indexed by ``iin_id`` with one row per IIN group.
    """
    group_members = iin_groups.reset_index().groupby("iin_id")["feature_id"].apply(list)

    # Build group-level metadata from ms1_features.
    meta_rows: list[dict] = []
    for iin_id, fids in group_members.items():
        row: dict = {"iin_id": iin_id, "feature_ids": ",".join(str(f) for f in sorted(fids))}

        neutral_mass: float | None = None
        ion_identities_str = ""
        if _IIN_NEUTRAL_MASS_COL in ms1_features.columns:
            masses = ms1_features.loc[ms1_features.index.isin(fids), _IIN_NEUTRAL_MASS_COL].dropna()
            if len(masses) > 0:
                neutral_mass = float(masses.iloc[0])
        if _IIN_IDENTITY_COL in ms1_features.columns:
            labels = ms1_features.loc[ms1_features.index.isin(fids), _IIN_IDENTITY_COL].dropna().astype(str)
            ion_identities_str = ";".join(labels.tolist())

        row["neutral_mass"] = neutral_mass
        row["ion_identities"] = ion_identities_str
        meta_rows.append(row)

    meta_df = pd.DataFrame(meta_rows).set_index("iin_id")

    # Pivot trend results to wide format.
    if not trend_results.empty:
        trend_wide = trend_results.pivot(index="iin_id", columns="contrast")
        # Multi-level columns are (metric, contrast); flatten to trend_<contrast>_<metric>.
        # metric names: trend_consistent → consistent; directions_observed → directions
        def _col_name(metric: str, contrast: str) -> str:
            metric = metric.replace("trend_consistent", "consistent")
            metric = metric.replace("directions_observed", "directions")
            return f"trend_{contrast}_{metric}"

        trend_wide.columns = [_col_name(col[0], col[1]) for col in trend_wide.columns]
        keep = [
            c for c in trend_wide.columns
            if c.endswith("_consistent") or c.endswith("_directions")
        ]
        trend_wide = trend_wide[keep]
    else:
        trend_wide = pd.DataFrame(index=meta_df.index)

    # Merge everything.
    ann = annotation_results.set_index("iin_id") if not annotation_results.empty else pd.DataFrame(index=meta_df.index)
    result = meta_df.join(ann, how="left").join(trend_wide, how="left")

    # all_consistent: annotation + all trend columns pass.
    trend_consistent_cols = [c for c in result.columns if c.endswith("_consistent") and c.startswith("trend_")]
    ann_consistent = result.get("annotation_consistent")

    if trend_consistent_cols:
        trend_all = result[trend_consistent_cols].all(axis=1)
        if ann_consistent is not None:
            result["all_consistent"] = trend_all & result["annotation_consistent"].fillna(True)
        else:
            result["all_consistent"] = trend_all
    elif ann_consistent is not None:
        result["all_consistent"] = result["annotation_consistent"].fillna(True)
    else:
        result["all_consistent"] = pd.NA

    return result
