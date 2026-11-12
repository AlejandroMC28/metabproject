"""The MetabProject class — central object managing a project directory.

Lazy-loads parquet tables on access, tracks every operation in the manifest,
and provides the Python API that the CLI delegates to.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from metabproject.io.mzmine import parse_feature_table
from metabproject.manifest import (
    append_operation,
    create_manifest,
    make_operation_entry,
    read_manifest,
)


class MetabProject:
    """A single MZmine-run project on disk.

    Wraps a project directory containing a manifest (``metabproject.json``) and
    parquet feature tables. Tables are lazy-loaded on first property access.

    Attributes:
        path: Path to the project directory.
    """

    MANIFEST_NAME = "metabproject.json"

    def __init__(self, path: str | Path) -> None:
        """Open an existing project directory.

        Args:
            path: Path to an initialized project directory.

        Raises:
            FileNotFoundError: If ``path`` is not a metabproject directory.
        """
        self.path = Path(path)
        # Verify it is a valid project directory.
        read_manifest(self.path)
        # Caches for lazy-loaded tables.
        self._raw_feature_table: pd.DataFrame | None = None
        self._feature_table: pd.DataFrame | None = None
        self._metadata: pd.DataFrame | None = None
        self._ms1_features: pd.DataFrame | None = None
        self._sample_details: pd.DataFrame | None = None
        self._detection_mask: pd.DataFrame | None = None
        self._sirius_annotations: pd.DataFrame | None = None
        self._sirius_topk: pd.DataFrame | None = None
        self._canopus_annotations: pd.DataFrame | None = None
        self._gnps_annotations: pd.DataFrame | None = None
        self._gnps_suspect_annotations: pd.DataFrame | None = None
        self._annotations: pd.DataFrame | None = None
        self._contrasts: dict[str, pd.DataFrame] | None = None
        self._comparisons: dict[str, dict[str, Any]] | None = None
        self._enrichment_results: dict[str, pd.DataFrame] | None = None
        self._iin_validation: pd.DataFrame | None = None
        self._network_edges: pd.DataFrame | None = None
        self._network_nodes: pd.DataFrame | None = None
        self._ms2_spectra: pd.DataFrame | None = None
        self._spectral_similarity: pd.DataFrame | None = None

    # -- Construction ---------------------------------------------------------

    @classmethod
    def init(
        cls,
        path: str | Path,
        feature_table: str | Path,
        metadata: str | Path,
    ) -> "MetabProject":
        """Create a new project from an MZmine feature table and metadata CSV.

        See ``specs/init.md`` for the full specification.

        Args:
            path: Project directory to create (must not already exist).
            feature_table: Path to the MZmine full feature table CSV.
            metadata: Path to the sample metadata CSV.

        Returns:
            The initialized MetabProject.

        Raises:
            FileExistsError: If ``path`` already exists.
            FileNotFoundError: If input files are missing.
            ValueError: On metadata validation failures.
        """
        project_path = Path(path)
        feature_table = Path(feature_table)
        metadata = Path(metadata)

        # Guard: project must not already exist.
        if project_path.exists():
            raise FileExistsError(
                f"Project directory already exists: {project_path}\n"
                "Use a different name or delete the existing project first."
            )

        # Guard: input files must exist.
        if not feature_table.exists():
            raise FileNotFoundError(f"Feature table not found: {feature_table}")
        if not metadata.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata}")

        # Step 1: Parse the feature table.
        ms1, areas, sample_details = parse_feature_table(feature_table)

        # Step 2: Load and validate metadata.
        meta_df = _load_and_validate_metadata(metadata, areas)

        # Map area columns (raw filenames → sample_name).
        ft_raw = _map_area_columns(areas, meta_df)

        # Step 3–4: Create directory tree and write files.
        features_dir = project_path / "features"
        for d in [features_dir, project_path / "qc", project_path / "annotations",
                  project_path / "stats", project_path / "enrichment",
                  project_path / "validation", project_path / "exports"]:
            d.mkdir(parents=True)

        ft_raw.to_parquet(features_dir / "feature_table_raw.parquet")
        meta_df.to_parquet(features_dir / "metadata.parquet")
        ms1.to_parquet(features_dir / "ms1_features.parquet")
        sample_details.to_parquet(features_dir / "sample_details.parquet")

        # Persist the MZmine DETECTED-state mask with sample_name columns so
        # downstream callers (e.g. prevalence_filter mode='detected') can
        # distinguish truly observed peaks from gap-filled values.
        detection_mask = _build_detection_mask(sample_details, meta_df)
        if detection_mask is not None:
            detection_mask.to_parquet(features_dir / "detection_mask.parquet")

        # Build group counts summary.
        group_counts = meta_df["group"].value_counts().to_dict()

        # Step 4: Create manifest.
        entry = make_operation_entry(
            command="init",
            parameters={
                "feature_table": str(feature_table),
                "metadata": str(metadata),
            },
            results={
                "n_features": int(len(ft_raw)),
                "n_samples": int(len(meta_df)),
                "groups": {k: int(v) for k, v in group_counts.items()},
            },
        )
        create_manifest(project_path, project_path.name, entry)

        # Step 5: Print summary.
        groups_str = ", ".join(f"{g} ({n})" for g, n in group_counts.items())
        print(f"Project '{project_path.name}' initialized")
        print(f"Features: {len(ft_raw):,}")
        print(f"Samples:  {len(meta_df)}")
        print(f"Groups:   {groups_str}")

        return cls(project_path)

    # -- Commands -------------------------------------------------------------

    def process(
        self,
        min_prevalence: int = 2,
        impute: str = "half-min",
        seed: int = 42,
        lod_value: float | None = None,
        knn_k: int = 5,
        weight_normalize: bool = False,
        normalize: str = "none",
        transform: str = "log2",
        scale: str = "none",
        prevalence_mode: str = "any_value",
        min_groups_meeting: int = 1,
        force: bool = False,
    ) -> None:
        """Run the processing pipeline on the raw feature table.

        Pipeline order (fixed): prevalence filter → imputation → weight
        correction → normalization → transformation → scaling. See
        ``specs/process.md``.

        Args:
            min_prevalence: Min detections per group for the prevalence filter.
            impute: Imputation method (``half-min``, ``min``, ``lod``, ``knn``,
                ``none``).
            seed: Random seed for ``half-min`` imputation.
            lod_value: LOD value, required when ``impute='lod'``.
            knn_k: Number of neighbors for KNN imputation.
            weight_normalize: Whether to divide each sample by its weight.
            normalize: Sample normalization (``none``, ``tic``, ``pqn``,
                ``median``).
            transform: Variance-stabilizing transformation (``log2``,
                ``log10``, ``sqrt``, ``none``).
            scale: Centering/standardization applied after transformation
                (``none``, ``pareto``, ``auto``).
            force: Overwrite an existing processed table.
        """
        from metabproject.processing.correction import weight_correct
        from metabproject.processing.filtering import prevalence_filter
        from metabproject.processing.imputation import impute as impute_fn
        from metabproject.processing.normalization import normalize as normalize_fn
        from metabproject.processing.scaling import scale as scale_fn
        from metabproject.processing.transformation import transform as transform_fn
        from metabproject.viz.missingness import (
            compute_missingness,
            make_missingness_heatmap,
        )
        from metabproject.viz.qc import make_processing_qc_figure

        ft_path = self.path / "features" / "feature_table.parquet"

        if ft_path.exists() and not force:
            raise FileExistsError(
                "Project already processed. Use --force to reprocess."
            )
        if ft_path.exists() and force:
            ft_path.unlink()
            qc_png = self.path / "qc" / "processing_qc.png"
            if qc_png.exists():
                qc_png.unlink()
            miss_png = self.path / "qc" / "missingness_heatmap.png"
            if miss_png.exists():
                miss_png.unlink()
            miss_csv = self.path / "qc" / "missingness_per_sample.csv"
            if miss_csv.exists():
                miss_csv.unlink()
            self._feature_table = None

        # Validate parameters up front.
        if impute == "lod" and lod_value is None:
            raise ValueError("LOD imputation requires --lod-value")
        if weight_normalize and "weight" not in self.metadata.columns:
            raise ValueError(
                "Metadata missing 'weight' column required for weight normalization"
            )

        # Resolve detection mask up front so any error fails before mutation.
        detection_mask = None
        if prevalence_mode == "detected":
            detection_mask = self._resolve_detection_mask()

        df = self.raw_feature_table.copy()
        stages: dict[str, pd.DataFrame] = {"Raw": df}

        # Step 1: Prevalence filter.
        df = prevalence_filter(
            df,
            self.metadata,
            min_prevalence=min_prevalence,
            min_groups_meeting=min_groups_meeting,
            mode=prevalence_mode,
            detection_mask=detection_mask,
        )
        if len(df) == 0:
            raise ValueError(
                f"No features passed prevalence filter (mode={prevalence_mode}, "
                f"min_prevalence={min_prevalence}, "
                f"min_groups_meeting={min_groups_meeting}). "
                "Check your data or lower the threshold."
            )
        n_before = len(self.raw_feature_table)
        n_after = len(df)
        stages["Filtered"] = df

        # Step 2: Imputation.
        # Compute pre-imputation missingness diagnostics on the filtered table
        # so the user can see where imputation is filling values before it runs.
        qc_dir = self.path / "qc"
        qc_dir.mkdir(exist_ok=True)
        miss_mask, per_sample_missing = compute_missingness(df)
        per_sample_missing.to_csv(qc_dir / "missingness_per_sample.csv")
        make_missingness_heatmap(
            miss_mask, self.metadata, qc_dir / "missingness_heatmap.png"
        )

        df, n_imputed = impute_fn(df, method=impute, seed=seed,
                                  lod_value=lod_value, knn_k=knn_k)
        stages["Imputed"] = df

        # Step 3: Weight correction (optional).
        if weight_normalize:
            df = weight_correct(df, self.metadata)
            stages["Weight-corrected"] = df

        # Step 4: Normalization.
        if normalize != "none":
            df = normalize_fn(df, method=normalize)
            stages["Normalized"] = df

        # Step 5: Transformation.
        if transform != "none":
            # Pre-check for zeros/NaN before log transforms.
            if transform in ("log2", "log10"):
                n_zeros = ((df <= 0) | df.isna()).sum().sum()
                if n_zeros > 0:
                    raise ValueError(
                        f"Cannot apply {transform} transform: {n_zeros} zero/NaN values remain. "
                        "Impute or filter zero/missing values before applying this transform."
                    )
            df = transform_fn(df, method=transform)
            stages["Transformed"] = df

        # Step 6: Scaling (optional).
        if scale != "none":
            df = scale_fn(df, method=scale)
            stages["Scaled"] = df

        # Write processed table.
        df.to_parquet(ft_path)

        # Generate QC figure.
        make_processing_qc_figure(stages, self.metadata, qc_dir / "processing_qc.png")

        # Append manifest entry.
        pct_removed = (
            (n_before - n_after) / n_before * 100.0 if n_before else 0.0
        )
        entry = make_operation_entry(
            command="process",
            parameters={
                "min_prevalence": min_prevalence,
                "prevalence_scope": "per_group",
                "prevalence_mode": prevalence_mode,
                "min_groups_meeting": min_groups_meeting,
                "impute": impute,
                "seed": seed,
                "weight_normalize": weight_normalize,
                "normalize": normalize,
                "transform": transform,
                "scale": scale,
            },
            results={
                "features_before_filter": int(n_before),
                "features_after_filter": int(n_after),
                "features_removed": int(n_before - n_after),
                "n_features_before_prevalence": int(n_before),
                "n_features_after_prevalence": int(n_after),
                "prevalence_mode_used": prevalence_mode,
                "min_prevalence_used": int(min_prevalence),
                "min_groups_meeting_used": int(min_groups_meeting),
                "values_imputed": int(n_imputed),
                "values_total": int(n_after * len(self.metadata)),
                "missing_per_sample": {
                    str(s): int(n) for s, n in per_sample_missing["n_missing"].items()
                },
            },
        )
        append_operation(self.path, entry)

        n_total = n_after * len(self.metadata)
        print(f"Processed '{self.path.name}'")
        print(
            f"Prevalence filter (mode={prevalence_mode}, min={min_prevalence} "
            f"in ≥{min_groups_meeting} group{'s' if min_groups_meeting != 1 else ''}): "
            f"{n_before:,} → {n_after:,} features (-{pct_removed:.1f}%)"
        )
        print(f"Imputed:  {n_imputed:,} / {n_total:,} values")
        print("Missing per sample (pre-imputation):")
        for sample, row in per_sample_missing.iterrows():
            print(
                f"  {sample}: {int(row['n_missing']):,} / {int(row['n_features']):,} "
                f"({row['pct_missing']:.2f}%)"
            )
        print(f"Missingness heatmap: {qc_dir / 'missingness_heatmap.png'}")
        print(f"QC figure: {qc_dir / 'processing_qc.png'}")

    def _resolve_detection_mask(self) -> pd.DataFrame:
        """Return the persisted detection mask, rebuilding from the source CSV if needed.

        Projects initialized before ``detection_mask.parquet`` existed are
        recovered by re-parsing the MZmine feature table recorded in the
        ``init`` manifest entry. Raises a clear error if neither is available.
        """
        mask = self.detection_mask
        if mask is not None:
            return mask

        manifest = read_manifest(self.path)
        init_entries = [
            op for op in manifest["operations"] if op.get("command") == "init"
        ]
        feature_table_path: str | None = None
        if init_entries:
            feature_table_path = init_entries[0].get("parameters", {}).get(
                "feature_table"
            )

        if not feature_table_path or not Path(feature_table_path).exists():
            raise FileNotFoundError(
                "Detection mask not available for this project and the "
                "original MZmine feature table could not be located"
                + (f" (looked for {feature_table_path})" if feature_table_path else "")
                + ". Re-run 'metab init' to regenerate "
                "features/detection_mask.parquet, or use "
                "--prevalence-mode any_value."
            )

        _, _, sample_details = parse_feature_table(feature_table_path)
        rebuilt = _build_detection_mask(sample_details, self.metadata)
        if rebuilt is None:
            raise ValueError(
                "MZmine feature table has no 'feature_state' column; cannot "
                "build a detection mask. Use --prevalence-mode any_value."
            )
        # Cache + persist for future runs.
        out_path = self.path / "features" / "detection_mask.parquet"
        rebuilt.to_parquet(out_path)
        self._detection_mask = rebuilt
        return rebuilt

    # -- Commands (continued) -------------------------------------------------

    def annotate(
        self,
        sirius_structures: str | Path,
        canopus: str | Path,
        gnps_library: str | Path,
        gnps_suspect: str | Path | None = None,
        sirius_min_confidence: float = 0.64,
        canopus_min_probability: float = 0.7,
        sirius_top_k: int = 1,
        gnps_polarity: str | None = None,
        classify_sirius_structures: bool = False,
        np_cache: str | Path | None = None,
        force: bool = False,
    ) -> None:
        """Merge SIRIUS, CANOPUS, and GNPS annotations into the project.

        Writes per-source parquets and a skinny merged table to
        ``<project>/annotations/``. See ``specs/annotate.md``.

        Priority for structural columns: GNPS library → SIRIUS (high-confidence
        only) → GNPS suspect. Priority for taxonomy columns: GNPS library →
        CANOPUS (per-level, probability-gated).

        Args:
            sirius_structures: Path to ``structure_identifications.tsv``.
            canopus: Path to ``canopus_structure_summary.tsv``.
            gnps_library: Path to GNPS library TSV.
            gnps_suspect: Optional path to GNPS suspect-list TSV.
            sirius_min_confidence: Confidence threshold for
                ``sirius_high_confidence`` flag (default 0.64).
            canopus_min_probability: Per-level probability threshold for
                CANOPUS class predictions (default 0.7).
            sirius_top_k: Number of SIRIUS candidates to store for cross-adduct
                comparison. When > 1, writes ``annotations/sirius_topk.parquet``.
                Default 1 (rank-1 only, no extra file).
            gnps_polarity: If ``"positive"`` or ``"negative"``, drop GNPS
                library matches whose ``IonMode`` is the opposite polarity.
                Rows with missing IonMode are kept. ``None`` (default) keeps
                all library matches.
            force: Overwrite existing annotation files.

        Raises:
            FileExistsError: If annotations already exist and ``force`` is False.
            FileNotFoundError: If any required input file is missing.
        """
        from metabproject.annotation.canopus import load_canopus
        from metabproject.annotation.gnps import load_gnps_library, load_gnps_suspect
        from metabproject.annotation.merge import merge_annotations
        from metabproject.annotation.sirius import load_sirius, load_sirius_topk

        annotations_dir = self.path / "annotations"
        merged_path = annotations_dir / "merged.parquet"

        if merged_path.exists() and not force:
            raise FileExistsError(
                "Annotations already exist. Use --force to overwrite."
            )
        if merged_path.exists() and force:
            for name in ("sirius.parquet", "sirius_topk.parquet", "canopus.parquet",
                         "gnps.parquet", "gnps_suspect.parquet", "merged.parquet"):
                p = annotations_dir / name
                if p.exists():
                    p.unlink()
            self._sirius_annotations = None
            self._sirius_topk = None
            self._canopus_annotations = None
            self._gnps_annotations = None
            self._gnps_suspect_annotations = None
            self._annotations = None

        # Validate input files.
        for label, p in [
            ("sirius_structures", sirius_structures),
            ("canopus", canopus),
            ("gnps_library", gnps_library),
        ]:
            if not Path(p).exists():
                raise FileNotFoundError(f"{label} file not found: {p}")
        if gnps_suspect is not None and not Path(gnps_suspect).exists():
            raise FileNotFoundError(f"gnps_suspect file not found: {gnps_suspect}")

        # Load sources.
        sirius_df = load_sirius(sirius_structures, min_confidence=sirius_min_confidence)
        canopus_df = load_canopus(canopus, min_probability=canopus_min_probability)
        gnps_df = load_gnps_library(gnps_library, polarity=gnps_polarity)

        suspect_df = None
        if gnps_suspect is not None:
            suspect_df = load_gnps_suspect(gnps_suspect, exclude_feature_ids=gnps_df.index)

        sirius_topk_df = None
        if sirius_top_k > 1:
            sirius_topk_df = load_sirius_topk(sirius_structures, top_k=sirius_top_k)

        sirius_struct_npc_df = None
        if classify_sirius_structures:
            from metabproject.annotation.np_classifier import classify_features
            hc_mask = sirius_df["sirius_high_confidence"].fillna(False).astype(bool)
            candidates = sirius_df.loc[hc_mask, ["sirius_InChIkey2D", "sirius_smiles"]].copy()
            candidates = candidates.rename(columns={
                "sirius_InChIkey2D": "InChIkey2D",
                "sirius_smiles": "smiles",
            })
            # Skip features that already have a GNPS library match — GNPS NPC
            # is preferred and computed offline from the curated structure.
            candidates = candidates.loc[~candidates.index.isin(gnps_df.index)]
            if not candidates.empty:
                cache_path = (
                    Path(np_cache) if np_cache is not None
                    else annotations_dir / "np_classifier_cache.parquet"
                )
                sirius_struct_npc_df = classify_features(candidates, cache_path=cache_path)

        # Merge.
        merged_df = merge_annotations(
            sirius_df,
            canopus_df,
            gnps_df,
            suspect_df,
            sirius_topk_df=sirius_topk_df,
            sirius_min_confidence=sirius_min_confidence,
            sirius_struct_npc_df=sirius_struct_npc_df,
        )

        # Write per-source parquets.
        sirius_df.to_parquet(annotations_dir / "sirius.parquet")
        if sirius_topk_df is not None:
            sirius_topk_df.to_parquet(annotations_dir / "sirius_topk.parquet")
        canopus_df.to_parquet(annotations_dir / "canopus.parquet")
        gnps_df.to_parquet(annotations_dir / "gnps.parquet")
        if suspect_df is not None:
            suspect_df.to_parquet(annotations_dir / "gnps_suspect.parquet")
        merged_df.to_parquet(merged_path)

        # Build result counts.
        n_sirius_rank1 = int(len(sirius_df))
        n_sirius_hc = int(sirius_df["sirius_high_confidence"].sum()) if "sirius_high_confidence" in sirius_df.columns else 0
        n_gnps = int(len(gnps_df))
        n_suspect = int(len(suspect_df)) if suspect_df is not None else 0
        n_annotated = int(len(merged_df))
        n_suspect_only = int(merged_df["is_suspect_only"].sum())
        n_canopus_only = int(merged_df["is_canopus_only"].sum())

        passed_cols = {
            level: f"canopus_{level}_passed"
            for level in ("npc_pathway", "npc_superclass", "npc_class", "classyfire_class")
        }
        n_canopus_pathway = int(canopus_df[passed_cols["npc_pathway"]].sum()) if passed_cols["npc_pathway"] in canopus_df.columns else 0
        n_canopus_superclass = int(canopus_df[passed_cols["npc_superclass"]].sum()) if passed_cols["npc_superclass"] in canopus_df.columns else 0
        n_canopus_class = int(canopus_df[passed_cols["npc_class"]].sum()) if passed_cols["npc_class"] in canopus_df.columns else 0
        n_canopus_classyfire = int(canopus_df[passed_cols["classyfire_class"]].sum()) if passed_cols["classyfire_class"] in canopus_df.columns else 0

        inchikey_agree_col = merged_df.get("inchikey_agreement", pd.Series(dtype=object))
        n_agree = int((inchikey_agree_col == "agree").sum())
        n_disagree = int((inchikey_agree_col == "disagree").sum())

        npc_alt_col = merged_df.get("npc_class_alt", pd.Series(dtype=object))
        cf_alt_col = merged_df.get("classyfire_class_alt", pd.Series(dtype=object))
        n_taxonomy_disagree = int(npc_alt_col.notna().sum() | cf_alt_col.notna().sum())

        # GNPS↔SIRIUS top-k cross-check counts.
        n_gnps_sirius_rank1 = 0
        n_gnps_sirius_topk_match = 0
        n_gnps_sirius_highconf = 0
        n_gnps_sirius_no_match = 0
        if "gnps_sirius_rank" in merged_df.columns:
            rank_col = merged_df["gnps_sirius_rank"]
            n_gnps_sirius_rank1 = int((rank_col == 1).sum())
            n_gnps_sirius_topk_match = int((rank_col >= 1).fillna(False).sum())
            n_gnps_sirius_no_match = int((rank_col == 0).sum())
            n_gnps_sirius_highconf = int(
                merged_df["gnps_sirius_highconf"].fillna(False).astype(bool).sum()
            )

        entry = make_operation_entry(
            command="annotate",
            parameters={
                "sirius_structures": str(sirius_structures),
                "canopus": str(canopus),
                "gnps_library": str(gnps_library),
                "gnps_suspect": str(gnps_suspect) if gnps_suspect else None,
                "sirius_min_confidence": sirius_min_confidence,
                "canopus_min_probability": canopus_min_probability,
                "sirius_top_k": sirius_top_k,
                "gnps_polarity": gnps_polarity,
                "classify_sirius_structures": classify_sirius_structures,
                "np_cache": str(np_cache) if np_cache else None,
                "priority_chain": [
                    "gnps", "sirius_high_confidence",
                    "sirius_struct_npc", "canopus", "suspect",
                ],
            },
            results={
                "n_sirius_rank1": n_sirius_rank1,
                "n_sirius_high_confidence": n_sirius_hc,
                "n_gnps_library": n_gnps,
                "n_gnps_suspect_unique": n_suspect,
                "n_canopus_features_total": int(len(canopus_df)),
                "n_canopus_pathway_passed": n_canopus_pathway,
                "n_canopus_superclass_passed": n_canopus_superclass,
                "n_canopus_class_passed": n_canopus_class,
                "n_canopus_classyfire_passed": n_canopus_classyfire,
                "n_features_annotated": n_annotated,
                "n_features_canopus_only": n_canopus_only,
                "n_features_suspect_only": n_suspect_only,
                "n_inchikey_agree": n_agree,
                "n_inchikey_disagree": n_disagree,
                "n_taxonomy_disagree": n_taxonomy_disagree,
                "n_gnps_sirius_rank1": n_gnps_sirius_rank1,
                "n_gnps_sirius_topk_match": n_gnps_sirius_topk_match,
                "n_gnps_sirius_no_match": n_gnps_sirius_no_match,
                "n_gnps_sirius_highconf": n_gnps_sirius_highconf,
                "n_sirius_struct_npc_classified": (
                    int(sirius_struct_npc_df.notna().any(axis=1).sum())
                    if sirius_struct_npc_df is not None else 0
                ),
            },
        )
        append_operation(self.path, entry)

        print(f"Annotated '{self.path.name}'")
        print(f"SIRIUS rank-1:  {n_sirius_rank1:,}  ({n_sirius_hc:,} high-confidence)")
        print(f"CANOPUS:        {len(canopus_df):,} features total  "
              f"(pathway {n_canopus_pathway:,} / class {n_canopus_class:,} passed @ {canopus_min_probability})")
        print(f"GNPS library:   {n_gnps:,}")
        if gnps_suspect is not None:
            print(f"GNPS suspect:   {n_suspect:,} (unique, deduped vs library)")
        if sirius_topk_df is not None and n_gnps > 0:
            print(
                f"GNPS↔SIRIUS:    {n_gnps_sirius_topk_match:,} of {n_gnps:,} GNPS hits match in top-{sirius_top_k}  "
                f"(rank-1 {n_gnps_sirius_rank1:,}, highconf {n_gnps_sirius_highconf:,})"
            )
        print(f"Features annotated: {n_annotated:,}  "
              f"({n_canopus_only:,} canopus-only, {n_suspect_only:,} suspect-only)")
        if n_agree or n_disagree:
            print(f"InChIKey: {n_agree:,} agree, {n_disagree:,} disagree")

    def import_network(
        self,
        pairs: str | Path,
        min_cosine: float = 0.0,
        drop_singletons: bool = True,
        force: bool = False,
    ) -> None:
        """Import a GNPS FBMN molecular network (filtered_pairs edge list).

        Writes ``network/edges.parquet`` (undirected, deduplicated edges) and
        ``network/nodes.parquet`` (per-node component membership). GNPS
        ``CLUSTERID`` values are the canonical ``feature_id``. See
        ``specs/network.md``.

        Note this is the GNPS ``ComponentIndex`` network, distinct from the
        MZmine ``molecular_networking:net_cluster_id`` columns retained in
        ``ms1_features``.

        Args:
            pairs: Path to the GNPS filtered_pairs TSV.
            min_cosine: Drop edges with cosine below this threshold.
            drop_singletons: Drop ``ComponentIndex == -1`` singleton edges.
            force: Overwrite an existing imported network.

        Raises:
            FileExistsError: If a network is already imported and ``force`` is
                False.
            FileNotFoundError: If the pairs file is missing.
            ValueError: If required columns are absent.
        """
        from metabproject.network.gnps_pairs import (
            build_node_membership,
            load_gnps_pairs,
        )

        network_dir = self.path / "network"
        edges_path = network_dir / "edges.parquet"
        nodes_path = network_dir / "nodes.parquet"

        if edges_path.exists() and not force:
            raise FileExistsError(
                "Network already imported. Use --force to overwrite."
            )
        if edges_path.exists() and force:
            for p in (edges_path, nodes_path):
                if p.exists():
                    p.unlink()
            self._network_edges = None
            self._network_nodes = None

        if not Path(pairs).exists():
            raise FileNotFoundError(f"GNPS pairs file not found: {pairs}")

        edges = load_gnps_pairs(
            pairs, min_cosine=min_cosine, drop_singletons=drop_singletons
        )

        # Keep only edges whose endpoints exist in the project (guards against
        # a mismatched networking job); count what was dropped.
        feature_ids = set(self.ms1_features.index)
        endpoints = set(edges["feature_id_1"]) | set(edges["feature_id_2"])
        unmatched = endpoints - feature_ids
        if unmatched:
            keep = edges["feature_id_1"].isin(feature_ids) & edges[
                "feature_id_2"
            ].isin(feature_ids)
            edges = edges[keep].reset_index(drop=True)

        nodes = build_node_membership(edges)

        network_dir.mkdir(parents=True, exist_ok=True)
        edges.to_parquet(edges_path)
        nodes.to_parquet(nodes_path)

        n_edges = int(len(edges))
        n_nodes = int(len(nodes))
        n_components = int(edges["component_id"].nunique())

        entry = make_operation_entry(
            command="network",
            parameters={
                "pairs": str(pairs),
                "min_cosine": min_cosine,
                "drop_singletons": drop_singletons,
            },
            results={
                "n_edges": n_edges,
                "n_nodes": n_nodes,
                "n_components": n_components,
                "n_features_matched": n_nodes,
                "n_features_unmatched": int(len(unmatched)),
            },
        )
        append_operation(self.path, entry)

        print(f"Imported network for '{self.path.name}'")
        print(f"Edges: {n_edges:,}  |  Nodes: {n_nodes:,}  |  Components: {n_components:,}")
        if unmatched:
            print(f"Dropped {len(unmatched):,} node id(s) absent from the feature table")

    def import_spectra(self, mgf: str | Path, force: bool = False) -> None:
        """Import MS2 spectra from an MZmine FBMN / GNPS MGF export.

        Writes ``spectra/ms2_spectra.parquet``, one row per spectrum keyed by
        canonical ``feature_id``. Raw peak lists are stored unmodified — any
        denoising / scaling happens only at display time. See
        ``specs/spectra.md``.

        Args:
            mgf: Path to the MGF file (keyed by feature id).
            force: Overwrite an already-imported spectra table.

        Raises:
            FileExistsError: If spectra are already imported and ``force`` is
                False.
            FileNotFoundError: If the MGF file is missing.
            ValueError: If the MGF yields no spectra.
        """
        from metabproject.spectra.mgf import load_mgf_spectra

        spectra_dir = self.path / "spectra"
        spectra_path = spectra_dir / "ms2_spectra.parquet"

        if spectra_path.exists() and not force:
            raise FileExistsError(
                "Spectra already imported. Use --force to overwrite."
            )
        if spectra_path.exists() and force:
            spectra_path.unlink()
            self._ms2_spectra = None

        if not Path(mgf).exists():
            raise FileNotFoundError(f"MGF file not found: {mgf}")

        spectra = load_mgf_spectra(mgf)

        # Keep only spectra whose feature id exists in the project (guards
        # against a mismatched MGF); count what was dropped.
        feature_ids = set(self.ms1_features.index)
        matched = spectra.index.isin(feature_ids)
        n_unmatched = int((~matched).sum())
        if n_unmatched:
            spectra = spectra[matched]

        spectra_dir.mkdir(parents=True, exist_ok=True)
        spectra.to_parquet(spectra_path)

        n_spectra = int(len(spectra))
        total_peaks = int(spectra["n_peaks"].sum())

        entry = make_operation_entry(
            command="spectra",
            parameters={"mgf": str(mgf)},
            results={
                "n_spectra": n_spectra,
                "n_features_matched": n_spectra,
                "n_features_unmatched": n_unmatched,
                "total_peaks": total_peaks,
            },
        )
        append_operation(self.path, entry)

        print(f"Imported MS2 spectra for '{self.path.name}'")
        print(f"Spectra: {n_spectra:,}  |  Peaks: {total_peaks:,}")
        if n_unmatched:
            print(f"Dropped {n_unmatched:,} spectrum id(s) absent from the feature table")

    def compute_similarity(
        self,
        tolerance: float = 0.1,
        score_types: Sequence[str] = ("modified", "direct"),
        scalings: Sequence[str] | None = None,
        force: bool = False,
    ) -> None:
        """Compute matchms spectral similarity for within-family spectrum pairs.

        Scores every unordered pair of features that (a) share a GNPS molecular
        family (``import_network``) and (b) both carry an MS2 spectrum
        (``import_spectra``), using matchms **direct** (``CosineGreedy``) and/or
        **modified** (``ModifiedCosineGreedy``) cosine across the requested
        intensity/m-z **scaling** presets. Matched peak indices are recorded so
        the explorer can highlight the matching peaks. Writes
        ``similarity/scores.parquet``. See ``specs/similarity.md``.

        Spectra are scored on the same display-processed peaks the explorer draws
        (``process_spectrum_for_display``), so matched indices align with the
        rendered stems.

        Args:
            tolerance: m/z match tolerance in Da.
            score_types: Any of ``"modified"`` / ``"direct"``.
            scalings: Names from ``spectra.similarity.SCALINGS``; defaults to all.
            force: Overwrite an already-computed similarity table.

        Raises:
            FileExistsError: If similarity is already computed and ``force`` is
                False.
            ValueError: If spectra or a network have not been imported, or an
                unknown score type / scaling is requested.
            ImportError: If ``matchms`` is not installed.
        """
        from metabproject.explore.spectra import process_spectrum_for_display
        from metabproject.spectra.similarity import SCALINGS, compute_pair_scores

        if scalings is None:
            scalings = tuple(SCALINGS)

        sim_dir = self.path / "similarity"
        sim_path = sim_dir / "scores.parquet"

        if sim_path.exists() and not force:
            raise FileExistsError(
                "Similarity already computed. Use --force to overwrite."
            )
        if sim_path.exists() and force:
            sim_path.unlink()
            self._spectral_similarity = None

        spectra = self.ms2_spectra
        if spectra is None:
            raise ValueError(
                "No MS2 spectra found. Run `metab spectra` first."
            )
        nodes = self.network_nodes
        if nodes is None:
            raise ValueError(
                "No molecular network found. Run `metab network` first."
            )

        # Build display-processed spectra (same peaks the explorer renders).
        processed: dict[int, dict] = {}
        for fid, row in spectra.iterrows():
            charge = row["precursor_charge"]
            spec = process_spectrum_for_display(
                mz=row["mz"],
                intensity=row["intensity"],
                precursor_mz=row["precursor_mz"],
                precursor_charge=None if pd.isna(charge) else int(charge),
            )
            if spec is not None:
                processed[int(fid)] = spec

        # Within-family pairs: all unordered pairs of spectra-bearing nodes that
        # share a component (the compare panel lets any two family members be
        # picked, not only directly-connected ones).
        from itertools import combinations

        pairs: list[tuple[int, int]] = []
        members = nodes.index[nodes.index.isin(processed)]
        comp_of = nodes.loc[members, "component_id"]
        for _, grp in comp_of.groupby(comp_of):
            ids = sorted(int(i) for i in grp.index)
            pairs.extend(combinations(ids, 2))

        scores = compute_pair_scores(
            processed,
            pairs,
            tolerance=tolerance,
            scalings=scalings,
            score_types=score_types,
        )

        sim_dir.mkdir(parents=True, exist_ok=True)
        scores.to_parquet(sim_path)
        self._spectral_similarity = None

        n_pairs = len(pairs)
        n_scored = int(scores[["feature_id_a", "feature_id_b"]].drop_duplicates().shape[0])
        entry = make_operation_entry(
            command="similarity",
            parameters={
                "tolerance": tolerance,
                "score_types": list(score_types),
                "scalings": list(scalings),
            },
            results={
                "n_pairs": n_pairs,
                "n_scored": n_scored,
                "n_rows": int(len(scores)),
            },
        )
        append_operation(self.path, entry)

        print(f"Computed spectral similarity for '{self.path.name}'")
        print(
            f"Pairs: {n_scored:,}  |  Score types: {', '.join(score_types)}  |  "
            f"Scalings: {', '.join(scalings)}"
        )

    def stats(
        self,
        contrasts: str | Path,
        alpha: float = 0.05,
        min_samples: int = 3,
        force: bool = False,
    ) -> None:
        """Run per-feature Welch t-tests for the contrasts in a YAML file.

        Writes one parquet per contrast to ``<project>/stats/`` and appends a
        single manifest entry summarising every contrast. See
        ``specs/stats.md``.

        Tests run on the processed feature table; ``log2fc`` is computed from
        a freshly-imputed copy of the raw feature table so that fold change
        is biologically interpretable regardless of the transform used at
        ``process``.

        Args:
            contrasts: Path to the YAML contrasts file.
            alpha: Significance threshold for ``direction`` / ``significant``.
            min_samples: Per-group minimum of non-NaN values required to test
                a feature.
            force: Overwrite existing per-contrast parquets.

        Raises:
            FileExistsError: If any target parquet exists and ``force`` is False.
            FileNotFoundError: If the contrasts file or the processed table is missing.
            ValueError: On malformed contrasts, missing metadata columns, or
                non-disjoint sample sets.
        """
        from metabproject.processing.filtering import prevalence_filter
        from metabproject.processing.imputation import impute as impute_fn
        from metabproject.stats.contrasts import load_contrasts, resolve_samples
        from metabproject.stats.univariate import welch_ttest

        stats_dir = self.path / "stats"
        stats_dir.mkdir(exist_ok=True)

        # Parse contrasts and validate up front.
        contrast_list = load_contrasts(contrasts)
        resolved: list[tuple[Any, list[str], list[str]]] = []  # type: ignore[name-defined]
        for c in contrast_list:
            samples_a, samples_b = resolve_samples(c, self.metadata)
            resolved.append((c, samples_a, samples_b))

        # Existence check before any write.
        target_paths = [stats_dir / f"{c.name}.parquet" for c, _, _ in resolved]
        existing = [p for p in target_paths if p.exists()]
        if existing and not force:
            names = ", ".join(p.name for p in existing)
            raise FileExistsError(
                f"Stats output(s) already exist: {names}. Use --force to overwrite."
            )
        if existing and force:
            for p in existing:
                p.unlink()
            self._contrasts = None

        # Locate latest process entry to reproduce imputation for the FC matrix.
        manifest = read_manifest(self.path)
        process_entries = [
            op for op in manifest["operations"] if op.get("command") == "process"
        ]
        if not process_entries:
            raise FileNotFoundError(
                "No 'process' entry in manifest. Run 'metab process' first."
            )
        proc_params = process_entries[-1]["parameters"]
        transform = str(proc_params.get("transform", "log2"))
        scale = str(proc_params.get("scale", "none"))
        log_path = transform in ("log2", "log10") and scale == "none"

        # For log-family transforms (without scaling on top) the processed
        # table already encodes the fold change as a difference of means —
        # no FC matrix needed. Any scaling step breaks that identity.
        ft = self.feature_table
        if log_path:
            fc_table: pd.DataFrame | None = None
        else:
            raw = self.raw_feature_table.copy()
            raw = raw.where(raw != 0, other=pd.NA)
            raw_filtered = prevalence_filter(
                raw, self.metadata, min_prevalence=int(proc_params.get("min_prevalence", 2))
            )
            fc_table, _ = impute_fn(
                raw_filtered,
                method=str(proc_params.get("impute", "half-min")),
                seed=int(proc_params.get("seed", 42)),
                lod_value=proc_params.get("lod_value"),
                knn_k=int(proc_params.get("knn_k", 5)),
            )
            # Align processed and FC matrices on the intersection of features
            # (prevalence filter is deterministic; in practice these are identical).
            shared = ft.index.intersection(fc_table.index)
            ft = ft.loc[shared]
            fc_table = fc_table.loc[shared]

        # Run each contrast.
        per_contrast: dict[str, dict[str, Any]] = {}
        for (c, samples_a, samples_b), out_path in zip(resolved, target_paths):
            if len(samples_a) < min_samples or len(samples_b) < min_samples:
                per_contrast[c.name] = {
                    "skipped": True,
                    "reason": f"insufficient samples (a={len(samples_a)}, b={len(samples_b)}, min={min_samples})",
                    "n_a": len(samples_a),
                    "n_b": len(samples_b),
                }
                print(f"Skipped contrast '{c.name}': {per_contrast[c.name]['reason']}")
                continue

            result = welch_ttest(
                ft, fc_table, samples_a, samples_b,
                min_samples=min_samples, alpha=alpha,
                transform=transform, scale=scale,
            )
            result.to_parquet(out_path)

            n_tested = int(result["p_value"].notna().sum())
            n_sig = int(result["significant"].sum())
            n_up = int((result["direction"] == "up").sum())
            n_down = int((result["direction"] == "down").sum())
            per_contrast[c.name] = {
                "n_a": len(samples_a),
                "n_b": len(samples_b),
                "n_tested": n_tested,
                "n_significant": n_sig,
                "n_up": n_up,
                "n_down": n_down,
            }
            print(f"Contrast '{c.name}': tested {n_tested:,} features, "
                  f"{n_sig:,} significant ({n_up:,} up, {n_down:,} down)")

        entry = make_operation_entry(
            command="stats",
            parameters={
                "contrasts_file": str(contrasts),
                "alpha": alpha,
                "min_samples": min_samples,
                "test": "welch_ttest",
                "fdr_method": "bh",
                "log2fc_source": (
                    "processed_difference" if log_path else "raw_fc_matrix"
                ),
            },
            results=per_contrast,
        )
        append_operation(self.path, entry)

        self._contrasts = None  # invalidate cache so next access reloads

    # -- Class enrichment -----------------------------------------------------

    def enrich(
        self,
        contrasts: list[str] | None = None,
        levels: list[str] | None = None,
        background: str = "all",
        min_class_n: int = 3,
        alpha: float = 0.05,
        labels: str = "canopus",
        feature_groups: str | Path | None = None,
        force: bool = False,
    ) -> None:
        """Run Mann-Whitney U class enrichment for one or more contrasts.

        For each contrast × CANOPUS level, tests whether each class's
        per-feature ``log2fc`` distribution differs from the background using a
        two-sided Mann-Whitney U test with BH-FDR correction. See
        ``specs/enrich.md``.

        Args:
            contrasts: Contrast names to run. Defaults to all parquets in
                ``<project>/stats/``.
            levels: CANOPUS levels to test. Defaults to all four levels.
            background: Universe used as background for each class test
                (``"all"``, ``"level"``, or ``"any"``). Default ``"all"``
                reproduces the reference pipeline.
            min_class_n: Minimum foreground size to test a class.
            alpha: Significance threshold for ``significant``.
            labels: Class labels to test. ``"canopus"`` (default) uses the
                CANOPUS predictions; ``"curated"`` uses the merged annotation
                classes (GNPS library → SIRIUS-structure NPClassifier →
                CANOPUS), i.e. the ``npc_*`` / ``classyfire_class`` columns of
                ``annotations/merged.parquet``.
            feature_groups: Optional TSV with ``feature_id`` and
                ``representative`` columns (e.g. MZmine correlation groups from
                ``feature_groups.py``). When given, only representative
                features enter the test, so each group counts once; features
                absent from the TSV are kept.
            force: Overwrite existing enrichment parquets.

        Raises:
            FileNotFoundError: If ``canopus_annotations`` or a requested
                contrast parquet is missing.
            FileExistsError: If any target parquet exists and ``force`` is
                False.
            ValueError: If a contrast has no finite ``log2fc`` values.
        """
        from metabproject.enrichment.class_test import CANOPUS_LEVELS, class_enrichment

        enrich_dir = self.path / "enrichment"
        enrich_dir.mkdir(exist_ok=True)

        # Resolve contrast names.
        stats_dir = self.path / "stats"
        if not stats_dir.exists() or not any(stats_dir.glob("*.parquet")):
            raise FileNotFoundError(
                "No stats parquets found. Run 'metab stats' first."
            )
        available = {p.stem for p in stats_dir.glob("*.parquet")}
        if contrasts is None:
            contrast_names = sorted(available)
        else:
            missing = [c for c in contrasts if c not in available]
            if missing:
                raise FileNotFoundError(
                    f"Contrast parquet(s) not found in {stats_dir}: {missing}"
                )
            contrast_names = list(contrasts)

        # Resolve levels.
        if levels is None:
            run_levels = list(CANOPUS_LEVELS)
        else:
            bad = [lv for lv in levels if lv not in CANOPUS_LEVELS]
            if bad:
                raise ValueError(f"Unknown level(s): {bad}. Must be one of {CANOPUS_LEVELS}.")
            run_levels = list(levels)

        # Existence check before any write.
        target_paths = {c: enrich_dir / f"{c}.parquet" for c in contrast_names}
        existing = [p for p in target_paths.values() if p.exists()]
        if existing and not force:
            names = ", ".join(p.name for p in existing)
            raise FileExistsError(
                f"Enrichment output(s) already exist: {names}. Use --force to overwrite."
            )
        if existing and force:
            for p in existing:
                p.unlink()
            self._enrichment_results = None

        if labels not in ("canopus", "curated"):
            raise ValueError(f"labels must be 'canopus' or 'curated'; got {labels!r}")

        # Load class labels once.
        if labels == "canopus":
            ann = self.canopus_annotations
        else:
            # Curated classes are already gated at annotate time; present them
            # in the canopus_<level> / canopus_<level>_passed shape the test expects.
            merged = self.annotations
            ann = pd.DataFrame(index=merged.index)
            for level in run_levels:
                if level in merged.columns:
                    ann[f"canopus_{level}"] = merged[level]
                    ann[f"canopus_{level}_passed"] = merged[level].notna()

        drop_ids: set = set()
        if feature_groups is not None:
            fg = pd.read_csv(feature_groups, sep="\t", usecols=["feature_id", "representative"])
            drop_ids = set(fg.loc[~fg["representative"].astype(bool), "feature_id"])

        # Run each contrast.
        per_contrast: dict[str, Any] = {}
        for name in contrast_names:
            contrast_df = self.contrast(name)
            if drop_ids:
                contrast_df = contrast_df.loc[~contrast_df.index.isin(drop_ids)]

            level_frames: list[pd.DataFrame] = []
            per_level: dict[str, dict[str, int]] = {}
            skipped_levels: list[str] = []

            for level in run_levels:
                import warnings as _w
                with _w.catch_warnings(record=True) as caught:
                    _w.simplefilter("always")
                    frame = class_enrichment(
                        contrast_df=contrast_df,
                        annotations=ann,
                        level=level,
                        background=background,
                        min_class_n=min_class_n,
                        alpha=alpha,
                    )
                for w in caught:
                    print(str(w.message))
                    if "Skipping level" in str(w.message):
                        skipped_levels.append(level)

                n_tested = len(frame)
                n_sig = int(frame["significant"].sum()) if n_tested > 0 else 0
                per_level[level] = {"n_tested": n_tested, "n_significant": n_sig}
                if n_tested > 0:
                    level_frames.append(frame)

            if level_frames:
                result = pd.concat(level_frames, ignore_index=True)
            else:
                result = pd.DataFrame(
                    columns=["level", "class_name", "n", "mean_log2fc",
                             "p_value", "p_adj", "significant"]
                )
            result.to_parquet(target_paths[name], index=False)

            n_total = sum(v["n_tested"] for v in per_level.values())
            n_total_sig = sum(v["n_significant"] for v in per_level.values())
            per_contrast[name] = {
                "n_total_tested": n_total,
                "n_significant": n_total_sig,
                "n_features_used": len(contrast_df),
                "per_level": per_level,
            }
            if skipped_levels:
                per_contrast[name]["skipped_levels"] = skipped_levels
            print(
                f"Enrichment '{name}': tested {n_total:,} classes, "
                f"{n_total_sig:,} significant"
            )

        entry = make_operation_entry(
            command="enrich",
            parameters={
                "contrasts": contrast_names,
                "levels": run_levels,
                "background": background,
                "min_class_n": min_class_n,
                "alpha": alpha,
                "labels": labels,
                "feature_groups": str(feature_groups) if feature_groups is not None else None,
                "test": "mannwhitneyu",
                "fdr_method": "bh",
            },
            results=per_contrast,
        )
        append_operation(self.path, entry)
        self._enrichment_results = None

    # -- Validation -----------------------------------------------------------

    def validate(
        self,
        checks: list[str] | None = None,
        sirius_top_k: int = 5,
        force: bool = False,
    ) -> None:
        """Run validation checks on the project.

        Currently supported checks:

        - ``'iin'``: Ion identity network consistency — checks whether
          features grouped by MZmine as the same compound (different adducts)
          agree on fold-change direction and structural annotations.

        Writes ``<project>/validation/iin_consistency.parquet``.

        Args:
            checks: List of check names to run. Defaults to all supported
                checks (currently ``['iin']``).
            sirius_top_k: Number of SIRIUS candidates to use for annotation
                consistency. Requires ``annotate --sirius-top-k > 1`` to have
                been run. Ignored (falls back to rank-1) if
                ``sirius_topk.parquet`` is absent.
            force: Overwrite existing validation outputs.

        Raises:
            FileExistsError: If output already exists and ``force`` is False.
            FileNotFoundError: If required inputs are missing.
            ValueError: If IIN columns are absent from ms1_features.
        """
        from metabproject.validation.ion_identity import (
            build_iin_validation_table,
            check_annotation_consistency,
            check_trend_consistency,
            parse_iin_groups,
        )

        run_checks = checks if checks is not None else ["iin"]
        valid_checks = {"iin"}
        bad = [c for c in run_checks if c not in valid_checks]
        if bad:
            raise ValueError(f"Unknown check(s): {bad}. Supported: {sorted(valid_checks)}")

        validation_dir = self.path / "validation"
        validation_dir.mkdir(exist_ok=True)

        results: dict[str, Any] = {}

        if "iin" in run_checks:
            out_path = validation_dir / "iin_consistency.parquet"
            if out_path.exists() and not force:
                raise FileExistsError(
                    "IIN validation already exists. Use --force to overwrite."
                )
            if out_path.exists() and force:
                out_path.unlink()
                self._iin_validation = None

            # Prerequisites check.
            stats_dir = self.path / "stats"
            if not stats_dir.exists() or not any(stats_dir.glob("*.parquet")):
                raise FileNotFoundError(
                    "No stats parquets found. Run 'metab stats' first."
                )
            ann_path = self.path / "annotations" / "merged.parquet"
            if not ann_path.exists():
                raise FileNotFoundError(
                    "Merged annotations not found. Run 'metab annotate' first."
                )

            iin_groups = parse_iin_groups(self.ms1_features)

            if iin_groups.empty:
                import warnings as _w
                _w.warn(
                    "No IIN groups found in ms1_features (all iin_id values are NaN). "
                    "Enable Ion Identity Networking in MZmine.",
                    UserWarning,
                    stacklevel=2,
                )
                results["iin"] = {"n_iin_groups": 0, "n_features_in_iins": 0}
            else:
                contrast_results = {
                    p.stem: pd.read_parquet(p)
                    for p in sorted(stats_dir.glob("*.parquet"))
                }
                ann_df = self.annotations
                topk_df = self.sirius_topk
                if topk_df is None and sirius_top_k > 1:
                    print(
                        "Note: sirius_topk.parquet not found. "
                        "Re-run 'metab annotate --force --sirius-top-k "
                        f"{sirius_top_k}' for top-k annotation comparison. "
                        "Proceeding with rank-1 only."
                    )

                trend_df = check_trend_consistency(iin_groups, contrast_results)
                ann_consistency_df = check_annotation_consistency(
                    iin_groups, ann_df, sirius_topk=topk_df
                )
                table = build_iin_validation_table(
                    iin_groups, self.ms1_features, trend_df, ann_consistency_df
                )
                table.to_parquet(out_path)

                n_groups = len(table)
                n_in_iins = iin_groups.shape[0]

                trend_consistent_cols = [
                    c for c in table.columns
                    if c.startswith("trend_") and c.endswith("_consistent")
                ]
                n_trend_consistent = (
                    int(table[trend_consistent_cols].all(axis=1).sum())
                    if trend_consistent_cols else n_groups
                )
                n_trend_inconsistent = n_groups - n_trend_consistent

                ann_col = table.get("annotation_consistent")
                n_ann_consistent = int((ann_col == True).sum()) if ann_col is not None else 0  # noqa: E712
                n_ann_inconsistent = int((ann_col == False).sum()) if ann_col is not None else 0  # noqa: E712
                n_ann_insufficient = int(ann_col.isna().sum()) if ann_col is not None else n_groups  # <2 HC-annotated members
                n_all = int(table.get("all_consistent", pd.Series()).fillna(False).sum())

                results["iin"] = {
                    "n_iin_groups": n_groups,
                    "n_features_in_iins": n_in_iins,
                    "n_trend_consistent": n_trend_consistent,
                    "n_trend_inconsistent": n_trend_inconsistent,
                    "n_annotation_consistent": n_ann_consistent,
                    "n_annotation_inconsistent": n_ann_inconsistent,
                    "n_annotation_insufficient": n_ann_insufficient,
                    "n_all_consistent": n_all,
                }

                print(f"Validated '{self.path.name}' (IIN check)")
                print(f"IIN groups:   {n_groups:,}  ({n_in_iins:,} features)")
                print(f"Trend:        {n_trend_consistent:,} consistent, "
                      f"{n_trend_inconsistent:,} inconsistent")
                print(f"Annotation:   {n_ann_consistent:,} consistent, "
                      f"{n_ann_inconsistent:,} inconsistent, "
                      f"{n_ann_insufficient:,} insufficient")
                print(f"All checks:   {n_all:,} fully consistent")

        entry = make_operation_entry(
            command="validate",
            parameters={"checks": run_checks, "sirius_top_k": sirius_top_k},
            results=results,
        )
        append_operation(self.path, entry)

    # -- Explorer -------------------------------------------------------------

    def explore(
        self,
        output: str | Path | None = None,
        render_structures: bool = True,
        embed_plotly: bool = True,
        feature_relations: str | Path | None = None,
    ) -> Path:
        """Generate an interactive HTML viewer for browsing annotated features.

        Works at any pipeline stage — sections populate based on what data is
        available. Requires at minimum that ``process`` has been run.

        Args:
            output: Output HTML path. Defaults to
                ``<project>/exports/explore.html``.
            render_structures: Whether to render 2D structures from SMILES via
                RDKit. Set ``False`` if RDKit is not installed.
            embed_plotly: Embed plotly.js inline for full self-containment.
                Set ``False`` to use CDN (smaller file, requires internet).
            feature_relations: Optional TSV linking derived ions (isotopes,
                adducts, in-source fragments) to their parent feature; shown
                per feature and as a list filter. See
                :func:`~metabproject.explore.data.assemble_explore_data`.

        Returns:
            Path to the generated HTML file.

        Raises:
            FileNotFoundError: If the processed feature table is not available.
        """
        from metabproject.explore.data import assemble_explore_data
        from metabproject.explore.report import generate_explore_html

        if output is None:
            output = self.path / "exports" / "explore.html"
        output = Path(output)

        data = assemble_explore_data(
            self, render_structures=render_structures, feature_relations=feature_relations,
        )
        result = generate_explore_html(data, output, embed_plotly=embed_plotly)

        n_annotated = sum(
            1 for f in data["features"].values() if f.get("annotation") is not None
        )

        entry = make_operation_entry(
            command="explore",
            parameters={
                "output": str(output),
                "render_structures": render_structures,
                "embed_plotly": embed_plotly,
                "feature_relations": str(feature_relations) if feature_relations else None,
            },
            results={
                "n_features": data["n_features"],
                "n_annotated": n_annotated,
                "n_contrasts": len(data.get("contrasts_meta", {})),
                "n_iin_groups": len(data.get("iin_groups", {})),
                "n_network_components": len(data.get("network_components", {})),
                "file_size_mb": round(output.stat().st_size / 1_048_576, 2),
            },
        )
        append_operation(self.path, entry)

        print(f"Explorer generated: {output}")
        print(f"Features: {data['n_features']:,}  ({n_annotated:,} annotated)")
        print(f"File size: {output.stat().st_size / 1_048_576:.1f} MB")

        return result

    # -- Visualization --------------------------------------------------------

    def viz_pca(
        self,
        group_col: str = "group",
        components: tuple[int, int] = (1, 2),
        three_d: bool = False,
        include_3d: bool = True,
        palette: dict | None = None,
        marker_col: str | None = None,
        markers: dict | None = None,
        basename: str | None = None,
        scale: bool = True,
    ) -> "plt.Figure":
        """Render a PCA scatter and save it as PNG+SVG under ``viz/``.

        When ``three_d`` is False and ``include_3d`` is True (the default), a
        companion 3D plot (``pca_3d``) is generated alongside the 2D figure.
        The returned ``Figure`` is the 2D one in that case.

        Args:
            group_col: Metadata column to colour points by.
            components: Two 1-indexed PCs to plot (ignored when ``three_d``).
            three_d: If True, plot PC1/PC2/PC3 in 3D as the primary output.
            include_3d: If True and ``three_d`` is False, also write a 3D
                companion figure (``pca_3d.{png,svg}``).
            palette: Optional ``{group_label: colour}`` override.
            marker_col: Optional second metadata column → marker shape.
            markers: Optional ``{label: matplotlib_marker}`` override pinning
                specific shapes to labels in ``marker_col``.
            basename: Filename stem for the primary figure.
            scale: Autoscale features before PCA (default True, same as
                :meth:`run_pca`) so the figure matches the reported PCA.

        Returns:
            The primary matplotlib Figure.
        """
        from metabproject.viz import pca_figure, save_figure
        from metabproject.viz._style import load_markers_yaml, load_palette_yaml

        if basename is None:
            basename = "pca_3d" if three_d else (
                "pca" if components == (1, 2) else f"pca_pc{components[0]}_pc{components[1]}"
            )

        # Merge viz.yaml defaults under any caller overrides so the Python
        # API matches the CLI's behavior.
        merged_palette = dict(load_palette_yaml(self.path))
        if palette:
            merged_palette.update(palette)
        palette = merged_palette or None

        merged_markers = dict(load_markers_yaml(self.path))
        if markers:
            merged_markers.update(markers)
        markers = merged_markers or None

        fig, info = pca_figure(
            self.feature_table, self.metadata,
            group_col=group_col, components=components, three_d=three_d,
            palette=palette, marker_col=marker_col, markers=markers,
            scale=scale,
        )
        viz_dir = self.path / "viz"
        png, svg = save_figure(fig, viz_dir, basename)
        outputs = [str(png.relative_to(self.path)), str(svg.relative_to(self.path))]
        variance = list(info["variance_explained"])

        companion_outputs: list[str] = []
        companion_variance: list[float] = []
        if (not three_d) and include_3d:
            fig_3d, info_3d = pca_figure(
                self.feature_table, self.metadata,
                group_col=group_col, three_d=True,
                palette=palette, marker_col=marker_col, markers=markers,
                scale=scale,
            )
            png3, svg3 = save_figure(fig_3d, viz_dir, "pca_3d")
            companion_outputs = [
                str(png3.relative_to(self.path)),
                str(svg3.relative_to(self.path)),
            ]
            companion_variance = list(info_3d["variance_explained"])
            import matplotlib.pyplot as _plt
            _plt.close(fig_3d)

        entry = make_operation_entry(
            command="viz_pca",
            parameters={
                "group_col": group_col,
                "components": list(components),
                "three_d": three_d,
                "include_3d": include_3d,
                "marker_col": marker_col,
                "markers": markers,
                "palette": palette,
                "basename": basename,
                "scale": scale,
            },
            results={
                "outputs": outputs + companion_outputs,
                "variance_explained": [round(v, 6) for v in variance],
                "variance_explained_3d": [round(v, 6) for v in companion_variance] if companion_variance else None,
            },
        )
        append_operation(self.path, entry)
        print(f"PCA figure: {png}")
        if companion_outputs:
            print(f"PCA 3D:     {viz_dir / 'pca_3d.png'}")
        return fig

    def viz_volcano(
        self,
        contrast: str,
        fc_threshold: float = 1.0,
        q_threshold: float = 0.05,
        label_top_n: int = 0,
        basename: str | None = None,
    ) -> "plt.Figure":
        """Render a volcano plot for one contrast and save it as PNG+SVG.

        Args:
            contrast: Contrast name (as written to ``stats/<name>.parquet``).
            fc_threshold: |log2FC| threshold for guideline and up/down split.
            q_threshold: BH-adjusted p-value threshold.
            label_top_n: Annotate the N most-significant feature IDs.
            basename: Filename stem under ``<project>/viz/``. Defaults to
                ``"volcano_<contrast>"``.

        Returns:
            The matplotlib Figure.
        """
        from metabproject.viz import save_figure, volcano_figure

        if basename is None:
            basename = f"volcano_{contrast}"

        contrast_df = self.contrast(contrast)
        fig, counts = volcano_figure(
            contrast_df,
            fc_threshold=fc_threshold, q_threshold=q_threshold,
            title=contrast, label_top_n=label_top_n,
        )
        viz_dir = self.path / "viz"
        png, svg = save_figure(fig, viz_dir, basename)

        entry = make_operation_entry(
            command="viz_volcano",
            parameters={
                "contrast": contrast,
                "fc_threshold": fc_threshold,
                "q_threshold": q_threshold,
                "label_top_n": label_top_n,
                "basename": basename,
            },
            results={
                "outputs": [str(png.relative_to(self.path)), str(svg.relative_to(self.path))],
                "counts": counts,
            },
        )
        append_operation(self.path, entry)
        print(f"Volcano figure: {png} (up {counts['up']}, down {counts['down']}, ns {counts['ns']})")
        return fig

    def viz_iin_boxplot(
        self,
        feature_id: int,
        contrast: str,
        palette: dict | None = None,
        ncols: int = 4,
        basename: str | None = None,
    ) -> "plt.Figure":
        """Render per-IIN-member boxplots for one contrast and save PNG+SVG.

        Finds every feature in the same ion identity network (IIN) group as
        ``feature_id`` and plots its abundance across the two sides of
        ``contrast``. High-confidence-annotated features (GNPS hit or
        ``sirius_high_confidence``) appear first (top-left) and their subplot
        title includes the compound name.

        Args:
            feature_id: Focal feature whose IIN group to plot.
            contrast: Contrast name from the project's contrasts YAML.
            palette: Optional ``{group_label: colour}`` override for the two
                boxes. Falls back to ``viz.yaml`` and then to defaults.
            ncols: Number of columns in the subplot grid.
            basename: Filename stem under ``<project>/viz/``. Defaults to
                ``"iin_boxplot_<contrast>_feat<feature_id>"``.

        Returns:
            The matplotlib Figure.

        Raises:
            ValueError: If the focal feature has no IIN group or the contrast
                cannot be resolved.
        """
        from metabproject.stats.contrasts import load_contrasts, resolve_samples
        from metabproject.validation.ion_identity import parse_iin_groups
        from metabproject.viz import iin_boxplot_figure, save_figure
        from metabproject.viz._style import load_palette_yaml

        manifest = read_manifest(self.path)
        stats_ops = [op for op in manifest["operations"] if op.get("command") == "stats"]
        if not stats_ops:
            raise FileNotFoundError(
                "No 'stats' entry in manifest. Run 'metab stats' first."
            )
        contrasts_file = stats_ops[-1]["parameters"].get("contrasts_file")
        if not contrasts_file:
            raise FileNotFoundError("Latest stats entry has no 'contrasts_file' parameter.")

        # The manifest stores the path as given at the CLI/API call site,
        # which is usually relative to that CWD. Try it verbatim, then
        # relative to the project parent (the typical notebook CWD).
        candidate = Path(contrasts_file)
        if not candidate.exists():
            alt = self.path.parent / contrasts_file
            if alt.exists():
                candidate = alt
        contrast_objs = {c.name: c for c in load_contrasts(candidate)}
        if contrast not in contrast_objs:
            raise ValueError(
                f"Contrast {contrast!r} not found in {contrasts_file} "
                f"(have: {list(contrast_objs)})"
            )
        c = contrast_objs[contrast]
        samples_a, samples_b = resolve_samples(c, self.metadata)

        def _filter_label(filt: dict) -> str:
            return "_".join(
                str(v) for vals in filt.values() for v in vals
            ) or "a"
        label_a = _filter_label(c.a)
        label_b = _filter_label(c.b)

        iin = parse_iin_groups(self.ms1_features)
        if feature_id not in iin.index:
            raise ValueError(
                f"Feature {feature_id} has no IIN group "
                "(not in any ion_identities:iin_id row)."
            )
        iin_id = iin.loc[feature_id, "iin_id"]
        partners = iin.index[iin["iin_id"] == iin_id].tolist()

        ann = self.annotations
        hc_source = ann.get("annotation_source", pd.Series("", index=ann.index))
        hc_sirius = ann.get("sirius_high_confidence", pd.Series(False, index=ann.index))
        hc_name = ann.get("compound_name", pd.Series(pd.NA, index=ann.index))
        is_hc = (
            (hc_source.fillna("").astype(str).str.contains("gnps") | hc_sirius.fillna(False).astype(bool))
            & hc_name.notna()
        )

        members: list[dict] = []
        for fid in partners:
            compound_name = None
            if fid in ann.index and bool(is_hc.get(fid, False)):
                compound_name = ann.at[fid, "compound_name"]
            members.append({
                "feature_id": int(fid),
                "ion_identity": iin.at[fid, "ion_identity"] if pd.notna(iin.at[fid, "ion_identity"]) else None,
                "compound_name": compound_name,
            })
        members.sort(key=lambda m: (m["compound_name"] is None, m["feature_id"]))

        palette_dict = dict(load_palette_yaml(self.path))
        if palette:
            palette_dict.update(palette)

        fig, info = iin_boxplot_figure(
            members=members,
            feature_table=self.feature_table,
            samples_a=samples_a,
            samples_b=samples_b,
            label_a=label_a,
            label_b=label_b,
            palette=palette_dict,
            ncols=ncols,
            title=f"IIN {int(iin_id)} — {contrast} (focal feat {feature_id})",
        )

        if basename is None:
            basename = f"iin_boxplot_{contrast}_feat{feature_id}"
        viz_dir = self.path / "viz"
        png, svg = save_figure(fig, viz_dir, basename)

        entry = make_operation_entry(
            command="viz_iin_boxplot",
            parameters={
                "feature_id": int(feature_id),
                "contrast": contrast,
                "ncols": ncols,
                "basename": basename,
            },
            results={
                "outputs": [str(png.relative_to(self.path)), str(svg.relative_to(self.path))],
                "iin_id": int(iin_id),
                "n_members": info["n_members"],
            },
        )
        append_operation(self.path, entry)
        print(
            f"IIN boxplot: {png} (IIN {int(iin_id)}, {info['n_members']} members)"
        )
        return fig

    def viz_heatmap(
        self,
        features: list | None = None,
        features_from_contrast: str | None = None,
        q_threshold: float = 0.05,
        group_col: str = "group",
        group_col_2: str | None = None,
        palette: dict | None = None,
        z_score: str | None = "rows",
        cluster_rows: bool = True,
        cluster_cols: bool = True,
        cmap: str = "vlag",
        basename: str = "heatmap",
    ) -> "plt.Figure":
        """Render a clustered heatmap and save it as PNG+SVG.

        Args:
            features: Explicit list of ``feature_id`` values to plot.
                Mutually exclusive with ``features_from_contrast``.
            features_from_contrast: Name of a stats contrast — features with
                ``p_adj < q_threshold`` will be selected automatically.
            q_threshold: Significance threshold used with
                ``features_from_contrast``.
            group_col: Metadata column for the column colour bar.
            z_score: ``"rows"`` to z-score each feature row, or ``None``.
            cluster_rows: Hierarchical-cluster rows.
            cluster_cols: Hierarchical-cluster columns.
            cmap: Colormap name.
            basename: Filename stem under ``<project>/viz/``.

        Returns:
            The matplotlib Figure.
        """
        from metabproject.viz import heatmap_figure, save_figure
        from metabproject.viz._style import load_palette_yaml

        if features is not None and features_from_contrast is not None:
            raise ValueError(
                "Pass either features or features_from_contrast, not both."
            )

        selection_source = "all"
        if features_from_contrast is not None:
            cdf = self.contrast(features_from_contrast)
            sig = cdf[cdf["p_adj"] < q_threshold]
            if sig.empty:
                raise ValueError(
                    f"No features with p_adj < {q_threshold} in contrast "
                    f"{features_from_contrast!r}."
                )
            features = sig.index.tolist()
            selection_source = f"sig:{features_from_contrast}"

        merged_palette = dict(load_palette_yaml(self.path))
        if palette:
            merged_palette.update(palette)
        palette = merged_palette or None

        fig, info = heatmap_figure(
            self.feature_table, self.metadata,
            features=features, group_col=group_col, group_col_2=group_col_2,
            palette=palette,
            z_score=z_score, cluster_rows=cluster_rows, cluster_cols=cluster_cols,
            cmap=cmap, title=basename,
        )
        viz_dir = self.path / "viz"
        png, svg = save_figure(fig, viz_dir, basename)

        entry = make_operation_entry(
            command="viz_heatmap",
            parameters={
                "features_from_contrast": features_from_contrast,
                "q_threshold": q_threshold if features_from_contrast else None,
                "n_features_explicit": len(features) if (features is not None and features_from_contrast is None) else None,
                "group_col": group_col,
                "group_col_2": group_col_2,
                "z_score": z_score,
                "cluster_rows": cluster_rows,
                "cluster_cols": cluster_cols,
                "cmap": cmap,
                "basename": basename,
                "selection_source": selection_source,
            },
            results={
                "outputs": [str(png.relative_to(self.path)), str(svg.relative_to(self.path))],
                **info,
            },
        )
        append_operation(self.path, entry)
        print(f"Heatmap figure: {png} ({info['n_features']} features × {info['n_samples']} samples)")
        return fig

    def viz_heatmap_annotated(
        self,
        features_from_contrast: str | None = None,
        q_threshold: float = 0.05,
        group_col: str = "group",
        group_col_2: str | None = None,
        palette: dict | None = None,
        z_score: str | None = "rows",
        cluster_rows: bool = True,
        cluster_cols: bool = True,
        cmap: str = "vlag",
        basename: str = "heatmap_annotated",
    ) -> "plt.Figure":
        """Render a clustered heatmap restricted to high-confidence structure annotations.

        A feature is included when its merged annotation came from GNPS or has
        ``sirius_high_confidence=True`` (and ``compound_name`` is non-null).
        Row labels are the compound name; the underlying ``feature_id`` is
        kept as a fallback when missing.

        Args:
            features_from_contrast: Optional contrast name; restricts the
                annotated subset further to features with ``p_adj <
                q_threshold`` in that contrast.
            q_threshold: Significance threshold used with
                ``features_from_contrast``.
            group_col: Metadata column for the column colour bar.
            z_score: ``"rows"`` to z-score each feature row, or ``None``.
            cluster_rows: Hierarchical-cluster rows.
            cluster_cols: Hierarchical-cluster columns.
            cmap: Colormap name.
            basename: Filename stem under ``<project>/viz/``.

        Returns:
            The matplotlib Figure.
        """
        from metabproject.viz import (
            heatmap_figure,
            save_figure,
            select_high_confidence_structure_features,
        )
        from metabproject.viz._style import load_palette_yaml

        merged_palette = dict(load_palette_yaml(self.path))
        if palette:
            merged_palette.update(palette)
        palette = merged_palette or None

        annotations = self.annotations

        contrast_features: list | None = None
        selection_source = "annotated"
        if features_from_contrast is not None:
            cdf = self.contrast(features_from_contrast)
            sig = cdf[cdf["p_adj"] < q_threshold]
            if sig.empty:
                raise ValueError(
                    f"No features with p_adj < {q_threshold} in contrast "
                    f"{features_from_contrast!r}."
                )
            contrast_features = sig.index.tolist()
            selection_source = f"annotated+sig:{features_from_contrast}"

        feature_ids, labels = select_high_confidence_structure_features(
            annotations, feature_ids=contrast_features,
        )
        if not feature_ids:
            raise ValueError(
                "No features with high-confidence structure annotations "
                "(GNPS hit or sirius_high_confidence=True)."
            )

        # Keep only features that are actually present in the processed table.
        ft = self.feature_table
        feature_ids = [f for f in feature_ids if f in ft.index]
        if not feature_ids:
            raise ValueError(
                "No high-confidence annotated features overlap the processed "
                "feature table — was 'metab process' run?"
            )
        labels = {f: labels[f] for f in feature_ids}

        fig, info = heatmap_figure(
            ft, self.metadata,
            features=feature_ids, group_col=group_col, group_col_2=group_col_2,
            palette=palette,
            z_score=z_score, cluster_rows=cluster_rows, cluster_cols=cluster_cols,
            cmap=cmap, title=basename, row_labels=labels,
        )
        viz_dir = self.path / "viz"
        png, svg = save_figure(fig, viz_dir, basename)

        entry = make_operation_entry(
            command="viz_heatmap_annotated",
            parameters={
                "features_from_contrast": features_from_contrast,
                "q_threshold": q_threshold if features_from_contrast else None,
                "group_col": group_col,
                "group_col_2": group_col_2,
                "z_score": z_score,
                "cluster_rows": cluster_rows,
                "cluster_cols": cluster_cols,
                "cmap": cmap,
                "basename": basename,
                "selection_source": selection_source,
                "filter": "gnps_or_sirius_high_confidence",
            },
            results={
                "outputs": [str(png.relative_to(self.path)), str(svg.relative_to(self.path))],
                **info,
            },
        )
        append_operation(self.path, entry)
        print(f"Annotated heatmap: {png} ({info['n_features']} features × {info['n_samples']} samples)")
        return fig

    def viz_canopus_violin(
        self,
        contrast: str,
        *,
        level: str = "npc_class",
        top_n: int = 15,
        direction: str = "both",
        q_threshold: float | None = 0.05,
        classes: list[str] | None = None,
        min_class_size: int = 3,
        fg_color: str | None = None,
        basename: str | None = None,
    ) -> "plt.Figure":
        """Render paired class-vs-background log2fc violins for one contrast.

        For each selected CANOPUS class at ``level``, draws the class's
        per-feature log2fc next to a shared grey background of all other
        annotated features at that level. Companion visual to
        ``metab enrich``.

        Args:
            contrast: Stats contrast name (parquet under ``stats/``).
            level: CANOPUS level — one of ``npc_pathway``, ``npc_superclass``,
                ``npc_class``, ``classyfire_class``.
            top_n: Max classes to render.
            direction: ``"up"``, ``"down"``, or ``"both"`` (filters classes
                by the sign of ``mean_log2fc`` when enrichment is available).
            q_threshold: If set, restrict to enriched classes with
                ``p_adj < q_threshold``. Pass ``None`` to ignore.
            classes: Explicit class list; overrides the selection logic.
            min_class_size: Drop classes with fewer foreground features.
            fg_color: Foreground violin colour. Defaults to the first colour
                in ``group_palette(metadata)``.
            basename: Filename stem under ``<project>/viz/``. Defaults to
                ``"canopus_violin_<level>_<contrast>"``.

        Returns:
            The matplotlib Figure.

        Raises:
            ValueError: Unknown ``level`` / ``direction``, or no classes
                survive selection + size filter.
            FileNotFoundError: ``stats`` or ``annotate`` haven't been run.
        """
        from metabproject.enrichment.class_test import CANOPUS_LEVELS
        from metabproject.viz import canopus_violin_figure, save_figure
        from metabproject.viz._style import group_palette

        if level not in CANOPUS_LEVELS:
            raise ValueError(
                f"Unknown level {level!r}. Must be one of {CANOPUS_LEVELS}."
            )
        if direction not in ("up", "down", "both"):
            raise ValueError(
                f"Unknown direction {direction!r}. Must be 'up', 'down', or 'both'."
            )

        contrast_df = self.contrast(contrast)
        if "log2fc" not in contrast_df.columns:
            raise ValueError(
                f"Contrast {contrast!r} parquet has no 'log2fc' column."
            )
        log2fc = contrast_df["log2fc"]

        ann = self.canopus_annotations
        class_col = f"canopus_{level}"
        passed_col = f"canopus_{level}_passed"
        if class_col not in ann.columns or passed_col not in ann.columns:
            raise ValueError(
                f"CANOPUS annotations missing {class_col!r} or {passed_col!r}."
            )
        ann = ann[ann[passed_col].fillna(False).astype(bool)]
        ann = ann[ann[class_col].notna() & (ann[class_col] != "")]

        # Align on the contrast's feature index.
        fid_col = "feature_id" if "feature_id" in ann.columns else None
        if fid_col is not None:
            ann = ann.set_index("feature_id")
        ann = ann[ann.index.isin(log2fc.index)]
        if ann.empty:
            raise ValueError(
                f"No annotated features at level {level!r} overlap contrast {contrast!r}."
            )

        # Class selection.
        selection_source: str
        enrichment_stats: dict[str, dict] = {}
        if classes is not None:
            ordered = list(classes)
            selection_source = "explicit"
        else:
            try:
                enr = self.enrichment(contrast)
            except FileNotFoundError:
                enr = None

            if enr is not None and not enr.empty:
                sub = enr[enr["level"] == level].copy()
                if q_threshold is not None:
                    sub = sub[sub["p_adj"] < q_threshold]
                if direction == "up":
                    sub = sub[sub["mean_log2fc"] > 0]
                elif direction == "down":
                    sub = sub[sub["mean_log2fc"] < 0]
                if sub.empty:
                    raise ValueError(
                        f"No enriched classes for contrast {contrast!r} at "
                        f"level {level!r} (direction={direction}, q<"
                        f"{q_threshold})."
                    )
                sub = sub.sort_values("p_adj").head(top_n)
                sub = sub.sort_values("mean_log2fc")
                ordered = sub["class_name"].tolist()
                enrichment_stats = {
                    row["class_name"]: {
                        "p_adj": float(row["p_adj"]),
                        "n": int(row["n"]),
                    }
                    for _, row in sub.iterrows()
                }
                selection_source = "enrichment"
            else:
                # Fallback: rank classes by |mean log2fc| of their features.
                grouped = ann.join(log2fc.rename("log2fc"), how="inner")
                ranked = (
                    grouped.groupby(class_col)["log2fc"]
                    .mean()
                    .abs()
                    .sort_values(ascending=False)
                )
                ordered = ranked.head(top_n).index.tolist()
                # Plot order: ascending mean for visual flow.
                signed = grouped.groupby(class_col)["log2fc"].mean().loc[ordered]
                ordered = signed.sort_values().index.tolist()
                selection_source = "fallback_mean"
                print(
                    "Warning: enrichment for contrast not found — ranking "
                    f"classes by mean |log2fc| of their features. "
                    "Run 'metab enrich' for a stats-driven selection."
                )

        # Build foreground arrays + shared background.
        shown_features: set[int] = set()
        foreground: dict[str, np.ndarray] = {}
        for c in list(ordered):
            members = ann.index[ann[class_col] == c]
            vals = log2fc.reindex(members).dropna().to_numpy(dtype=float)
            if vals.size < min_class_size:
                continue
            foreground[c] = vals
            shown_features.update(int(f) for f in members)

        if not foreground:
            raise ValueError(
                f"No classes survived min_class_size={min_class_size} filter."
            )

        # Background = all annotated features NOT in any plotted class.
        bg_features = ann.index[~ann.index.isin(shown_features)]
        background = log2fc.reindex(bg_features).dropna().to_numpy(dtype=float)
        if background.size == 0:
            raise ValueError(
                "Background log2fc array is empty — no non-class annotated features."
            )

        # Default colour from the project palette.
        if fg_color is None:
            palette = group_palette(self.metadata)
            if palette:
                first = next(iter(palette.values()))
                fg_color = first if isinstance(first, str) else "#4A8FA0"
            else:
                fg_color = "#4A8FA0"

        if basename is None:
            basename = f"canopus_violin_{level}_{contrast}"

        plot_classes = [c for c in ordered if c in foreground]
        fig, info = canopus_violin_figure(
            classes=plot_classes,
            foreground=foreground,
            background=background,
            contrast_label=contrast,
            level=level,
            fg_color=fg_color,
            enrichment_stats=enrichment_stats or None,
        )

        viz_dir = self.path / "viz"
        png, svg = save_figure(fig, viz_dir, basename)

        entry = make_operation_entry(
            command="viz_canopus_violin",
            parameters={
                "contrast": contrast,
                "level": level,
                "top_n": top_n,
                "direction": direction,
                "q_threshold": q_threshold,
                "min_class_size": min_class_size,
                "classes": classes,
                "fg_color": fg_color,
                "basename": basename,
                "selection_source": selection_source,
            },
            results={
                "outputs": [
                    str(png.relative_to(self.path)),
                    str(svg.relative_to(self.path)),
                ],
                "n_classes": info["n_classes"],
                "n_background": info["n_background"],
                "skipped_empty": info["skipped_empty"],
            },
        )
        append_operation(self.path, entry)
        print(
            f"CANOPUS violin: {png} "
            f"({info['n_classes']} classes vs {info['n_background']} background features)"
        )
        return fig

    # -- Lazy-loaded tables ---------------------------------------------------

    @property
    def raw_feature_table(self) -> pd.DataFrame:
        """Original peak areas from MZmine (features × samples), never modified."""
        if self._raw_feature_table is None:
            self._raw_feature_table = pd.read_parquet(
                self.path / "features" / "feature_table_raw.parquet"
            )
        return self._raw_feature_table

    @property
    def feature_table(self) -> pd.DataFrame:
        """Processed feature table (available after ``process``)."""
        if self._feature_table is None:
            ft_path = self.path / "features" / "feature_table.parquet"
            if not ft_path.exists():
                raise FileNotFoundError(
                    "Processed feature table not found. Run 'metab process' first."
                )
            self._feature_table = pd.read_parquet(ft_path)
        return self._feature_table

    @property
    def metadata(self) -> pd.DataFrame:
        """Sample metadata, indexed by ``sample_name``."""
        if self._metadata is None:
            self._metadata = pd.read_parquet(
                self.path / "features" / "metadata.parquet"
            )
        return self._metadata

    @property
    def ms1_features(self) -> pd.DataFrame:
        """Consensus feature-level info (mz, rt, charge, networking, …)."""
        if self._ms1_features is None:
            self._ms1_features = pd.read_parquet(
                self.path / "features" / "ms1_features.parquet"
            )
        return self._ms1_features

    @property
    def sample_details(self) -> pd.DataFrame:
        """Per-sample feature quality metrics (fwhm, tailing, feature_state, …)."""
        if self._sample_details is None:
            self._sample_details = pd.read_parquet(
                self.path / "features" / "sample_details.parquet"
            )
        return self._sample_details

    @property
    def detection_mask(self) -> pd.DataFrame | None:
        """Boolean features × samples mask where True = MZmine ``DETECTED``.

        Persisted by ``init`` as ``features/detection_mask.parquet``. Returns
        ``None`` for projects initialized before this file existed.
        """
        if self._detection_mask is None:
            p = self.path / "features" / "detection_mask.parquet"
            if not p.exists():
                return None
            self._detection_mask = pd.read_parquet(p)
        return self._detection_mask

    @property
    def network_edges(self) -> pd.DataFrame | None:
        """GNPS FBMN molecular-network edges (available after ``import_network``).

        Returns ``None`` for projects with no imported network.
        """
        if self._network_edges is None:
            p = self.path / "network" / "edges.parquet"
            if not p.exists():
                return None
            self._network_edges = pd.read_parquet(p)
        return self._network_edges

    @property
    def network_nodes(self) -> pd.DataFrame | None:
        """Per-node component membership (available after ``import_network``).

        Returns ``None`` for projects with no imported network.
        """
        if self._network_nodes is None:
            p = self.path / "network" / "nodes.parquet"
            if not p.exists():
                return None
            self._network_nodes = pd.read_parquet(p)
        return self._network_nodes

    @property
    def ms2_spectra(self) -> pd.DataFrame | None:
        """MS2 spectra keyed by ``feature_id`` (available after ``import_spectra``).

        Returns ``None`` for projects with no imported spectra.
        """
        if self._ms2_spectra is None:
            p = self.path / "spectra" / "ms2_spectra.parquet"
            if not p.exists():
                return None
            self._ms2_spectra = pd.read_parquet(p)
        return self._ms2_spectra

    @property
    def spectral_similarity(self) -> pd.DataFrame | None:
        """matchms spectral-similarity scores (available after ``compute_similarity``).

        Long-format table (one row per pair × score type × scaling) with matched
        peak indices. Returns ``None`` for projects with no computed similarity.
        """
        if self._spectral_similarity is None:
            p = self.path / "similarity" / "scores.parquet"
            if not p.exists():
                return None
            self._spectral_similarity = pd.read_parquet(p)
        return self._spectral_similarity


    @property
    def sirius_annotations(self) -> pd.DataFrame:
        """Cleaned SIRIUS + CANOPUS annotations (available after ``annotate``)."""
        if self._sirius_annotations is None:
            p = self.path / "annotations" / "sirius.parquet"
            if not p.exists():
                raise FileNotFoundError(
                    "SIRIUS annotations not found. Run 'metab annotate' first."
                )
            self._sirius_annotations = pd.read_parquet(p)
        return self._sirius_annotations

    @property
    def sirius_topk(self) -> pd.DataFrame | None:
        """SIRIUS top-k candidates (available after ``annotate --sirius-top-k > 1``).

        Returns None if not available.
        """
        if self._sirius_topk is None:
            p = self.path / "annotations" / "sirius_topk.parquet"
            if not p.exists():
                return None
            self._sirius_topk = pd.read_parquet(p)
        return self._sirius_topk

    @property
    def canopus_annotations(self) -> pd.DataFrame:
        """CANOPUS compound-class predictions with per-level probabilities (available after ``annotate``)."""
        if self._canopus_annotations is None:
            p = self.path / "annotations" / "canopus.parquet"
            if not p.exists():
                raise FileNotFoundError(
                    "CANOPUS annotations not found. Run 'metab annotate' first."
                )
            self._canopus_annotations = pd.read_parquet(p)
        return self._canopus_annotations

    @property
    def gnps_annotations(self) -> pd.DataFrame:
        """Cleaned GNPS library annotations (available after ``annotate``)."""
        if self._gnps_annotations is None:
            p = self.path / "annotations" / "gnps.parquet"
            if not p.exists():
                raise FileNotFoundError(
                    "GNPS annotations not found. Run 'metab annotate' first."
                )
            self._gnps_annotations = pd.read_parquet(p)
        return self._gnps_annotations

    @property
    def gnps_suspect_annotations(self) -> pd.DataFrame | None:
        """Cleaned GNPS suspect-list annotations, or None if not provided."""
        if self._gnps_suspect_annotations is None:
            p = self.path / "annotations" / "gnps_suspect.parquet"
            if not p.exists():
                return None
            self._gnps_suspect_annotations = pd.read_parquet(p)
        return self._gnps_suspect_annotations

    @property
    def contrasts(self) -> dict[str, pd.DataFrame]:
        """All per-contrast result tables in ``stats/``, keyed by contrast name.

        Lazy-loads every parquet under ``<project>/stats/`` on first access.
        Empty dict if ``stats`` has not been run.
        """
        if self._contrasts is None:
            stats_dir = self.path / "stats"
            if not stats_dir.exists():
                self._contrasts = {}
            else:
                self._contrasts = {
                    p.stem: pd.read_parquet(p)
                    for p in sorted(stats_dir.glob("*.parquet"))
                }
        return self._contrasts

    def contrast(self, name: str) -> pd.DataFrame:
        """Load a single per-contrast result table by name.

        Args:
            name: Contrast name (as defined in the contrasts YAML).

        Returns:
            DataFrame indexed by ``feature_id``.

        Raises:
            FileNotFoundError: If no parquet for ``name`` exists.
        """
        p = self.path / "stats" / f"{name}.parquet"
        if not p.exists():
            raise FileNotFoundError(
                f"Contrast {name!r} not found in {self.path / 'stats'}"
            )
        return pd.read_parquet(p)

    @property
    def enrichment_results(self) -> dict[str, pd.DataFrame]:
        """All per-contrast enrichment tables in ``enrichment/``, keyed by name.

        Lazy-loads every parquet under ``<project>/enrichment/`` on first
        access. Empty dict if ``enrich`` has not been run.
        """
        if self._enrichment_results is None:
            enrich_dir = self.path / "enrichment"
            if not enrich_dir.exists():
                self._enrichment_results = {}
            else:
                self._enrichment_results = {
                    p.stem: pd.read_parquet(p)
                    for p in sorted(enrich_dir.glob("*.parquet"))
                }
        return self._enrichment_results

    def enrichment(self, name: str) -> pd.DataFrame:
        """Load a single per-contrast enrichment table by name.

        Args:
            name: Contrast name.

        Returns:
            DataFrame with columns ``level, class_name, n, mean_log2fc,
            p_value, p_adj, significant``.

        Raises:
            FileNotFoundError: If no enrichment parquet for ``name`` exists.
        """
        p = self.path / "enrichment" / f"{name}.parquet"
        if not p.exists():
            raise FileNotFoundError(
                f"Enrichment results for {name!r} not found in "
                f"{self.path / 'enrichment'}"
            )
        return pd.read_parquet(p)

    @property
    def annotations(self) -> pd.DataFrame:
        """Skinny merged annotation table (available after ``annotate``)."""
        if self._annotations is None:
            p = self.path / "annotations" / "merged.parquet"
            if not p.exists():
                raise FileNotFoundError(
                    "Merged annotations not found. Run 'metab annotate' first."
                )
            self._annotations = pd.read_parquet(p)
        return self._annotations

    @property
    def comparisons(self) -> dict[str, dict[str, Any]]:
        """All persisted :class:`Comparison` results, keyed by name.

        Scans ``<project>/stats/*/omnibus.parquet`` for results written by
        :meth:`run_comparison`. Each entry carries the omnibus DataFrame,
        the post-hoc DataFrame (when present), and the metadata recovered
        from the matching ``run_comparison`` manifest entry (``kind``,
        ``factors``, ``posthoc``, ``alpha``).

        Returns:
            Dict ``{name → {kind, factors, posthoc, alpha, omnibus,
            posthoc_df}}``. Empty dict if no comparison results are on
            disk.
        """
        if self._comparisons is not None:
            return self._comparisons

        results: dict[str, dict[str, Any]] = {}
        stats_dir = self.path / "stats"
        if not stats_dir.exists():
            self._comparisons = results
            return results

        # Recover per-comparison metadata from the most recent manifest
        # `run_comparison` entry whose parameters.name matches.
        meta_by_name: dict[str, dict[str, Any]] = {}
        try:
            manifest = read_manifest(self.path)
            for op in manifest.get("operations", []):
                if op.get("command") != "run_comparison":
                    continue
                params = op.get("parameters", {}) or {}
                name = params.get("name")
                if name:
                    meta_by_name[name] = params
        except FileNotFoundError:
            pass

        for sub in sorted(stats_dir.iterdir()):
            if not sub.is_dir():
                continue
            om_path = sub / "omnibus.parquet"
            if not om_path.exists():
                continue
            omnibus = pd.read_parquet(om_path)
            ph_path = sub / "posthoc.parquet"
            posthoc_df = pd.read_parquet(ph_path) if ph_path.exists() else None

            meta = meta_by_name.get(sub.name, {})
            kind = meta.get("kind")
            factors = list(meta.get("factors", []) or [])
            posthoc = meta.get("posthoc")
            alpha = meta.get("alpha", 0.05)

            # Fallback inference when the manifest entry is missing.
            if not kind or not factors:
                inferred_factors: list[str] = []
                for col in omnibus.columns:
                    if col.startswith("p_") and not col.endswith("_adj") and col != "p_interaction":
                        inferred_factors.append(col[len("p_"):])
                if not factors:
                    factors = inferred_factors
                if not kind:
                    if "p_interaction" in omnibus.columns:
                        kind = "anova_2way"
                    elif len(factors) == 1:
                        kind = "anova_1way"
                    else:
                        kind = "unknown"

            results[sub.name] = {
                "kind": kind,
                "factors": factors,
                "posthoc": posthoc,
                "alpha": alpha,
                "omnibus": omnibus,
                "posthoc_df": posthoc_df,
            }

        self._comparisons = results
        return results

    @property
    def iin_validation(self) -> pd.DataFrame:
        """IIN consistency validation table (available after ``validate``)."""
        if self._iin_validation is None:
            p = self.path / "validation" / "iin_consistency.parquet"
            if not p.exists():
                raise FileNotFoundError(
                    "IIN validation not found. Run 'metab validate' first."
                )
            self._iin_validation = pd.read_parquet(p)
        return self._iin_validation

    # ------------------------------------------------------------------ #
    # Multivariate (library wrappers)
    # ------------------------------------------------------------------ #

    def run_pca(self, n_components: int = 10, scale: bool = True):
        """Wrap :func:`stats.multivariate.run_pca` on the processed table."""
        from metabproject.stats.multivariate import run_pca

        result = run_pca(self.feature_table, n_components=n_components, scale=scale)
        entry = make_operation_entry(
            command="run_pca",
            parameters={"n_components": n_components, "scale": scale},
            results={
                "n_components_used": result.n_components_used,
                "explained_variance_ratio": [
                    round(float(v), 6) for v in result.explained_variance_ratio
                ],
            },
        )
        append_operation(self.path, entry)
        return result

    def run_pcoa(
        self,
        group_col: str | list[str],
        metric: str = "euclidean",
        permutations: int = 999,
        seed: int = 0,
    ):
        """Wrap :func:`stats.multivariate.run_pcoa` (requires the ``pcoa`` extra).

        ``group_col`` may be a single column name or a list of column names.
        With a list, PERMANOVA / PERMDISP are run once per factor and the
        result's ``permanova`` / ``permdisp`` attributes become dicts keyed
        by column name.
        """
        from metabproject.stats.multivariate import run_pcoa

        result = run_pcoa(
            self.feature_table, self.metadata,
            group_col=group_col, metric=metric,
            permutations=permutations, seed=seed,
        )
        entry = make_operation_entry(
            command="run_pcoa",
            parameters={
                "group_col": group_col, "metric": metric,
                "permutations": permutations, "seed": seed,
            },
            results={"permanova": result.permanova, "permdisp": result.permdisp},
        )
        append_operation(self.path, entry)
        return result

    def run_hca(
        self,
        metric: str = "euclidean",
        linkage_method: str = "complete",
        n_clusters: int | None = None,
    ):
        """Wrap :func:`stats.multivariate.run_hca` on the processed table."""
        from metabproject.stats.multivariate import run_hca

        result = run_hca(
            self.feature_table, metric=metric,
            linkage_method=linkage_method, n_clusters=n_clusters,
        )
        entry = make_operation_entry(
            command="run_hca",
            parameters={
                "metric": metric, "linkage_method": linkage_method,
                "n_clusters": n_clusters,
            },
            results={"n_clusters": n_clusters},
        )
        append_operation(self.path, entry)
        return result

    def run_random_forest(
        self,
        target_col: str,
        subset: dict | None = None,
        n_estimators: int = 500,
        test_size: float = 0.25,
        n_permutation_repeats: int = 10,
        mda_top_k: int | None = None,
        seed: int = 123,
        n_jobs: int | None = -1,
        name: str | None = None,
    ):
        """Wrap :func:`stats.multivariate.run_random_forest` and persist outputs.

        Writes ``stats/rf/<name>/``:
            * ``feature_importance.tsv`` — MDI + MDA per feature.
            * ``oob_curve.tsv`` — OOB score vs. tree count.
            * ``classification_report.txt`` — per-class precision/recall/F1.
            * ``summary.json`` — accuracy, classes, run parameters.

        ``name`` defaults to ``rf_<target_col>`` augmented with any
        ``subset`` keys (e.g. ``rf_treatment_variety-red``).
        """
        import json

        from metabproject.stats.multivariate import run_random_forest

        result = run_random_forest(
            self.feature_table, self.metadata,
            target_col=target_col, subset=subset,
            n_estimators=n_estimators,
            test_size=test_size, n_permutation_repeats=n_permutation_repeats,
            mda_top_k=mda_top_k, seed=seed, n_jobs=n_jobs,
        )

        if name is None:
            parts = [f"rf_{target_col}"]
            if subset:
                parts.extend(f"{k}-{v}" for k, v in subset.items())
            name = "_".join(parts)

        out_dir = self.path / "stats" / "rf" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        result.feature_importance.to_csv(
            out_dir / "feature_importance.tsv", sep="\t", index=False,
        )
        result.oob_curve.to_csv(
            out_dir / "oob_curve.tsv", sep="\t", index=False,
        )
        (out_dir / "classification_report.txt").write_text(
            result.classification_report
        )
        (out_dir / "summary.json").write_text(json.dumps({
            "name": name,
            "target_col": target_col,
            "subset": subset,
            "n_estimators": n_estimators,
            "test_size": test_size,
            "n_permutation_repeats": n_permutation_repeats,
            "mda_top_k": mda_top_k,
            "seed": seed,
            "accuracy": float(result.accuracy),
            "classes": [str(c) for c in result.classes_],
        }, indent=2))

        entry = make_operation_entry(
            command="run_random_forest",
            parameters={
                "name": name,
                "target_col": target_col, "subset": subset,
                "n_estimators": n_estimators,
                "test_size": test_size,
                "n_permutation_repeats": n_permutation_repeats,
                "mda_top_k": mda_top_k, "seed": seed,
            },
            results={
                "accuracy": float(result.accuracy),
                "output_dir": str(out_dir.relative_to(self.path)),
            },
        )
        append_operation(self.path, entry)
        print(f"RF outputs: {out_dir}  (accuracy={result.accuracy:.3f})")
        return result

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def run_shapiro(
        self, group_col: str | None = None, alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Wrap :func:`stats.diagnostics.shapiro_per_feature`."""
        from metabproject.stats.diagnostics import shapiro_per_feature

        df = shapiro_per_feature(
            self.feature_table,
            metadata=self.metadata if group_col else None,
            group_col=group_col, alpha=alpha,
        )
        entry = make_operation_entry(
            command="run_shapiro",
            parameters={"group_col": group_col, "alpha": alpha},
            results={"n_features": int(len(df))},
        )
        append_operation(self.path, entry)
        return df

    def run_levene(
        self, group_col: str, center: str = "median", alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Wrap :func:`stats.diagnostics.levene_per_feature`."""
        from metabproject.stats.diagnostics import levene_per_feature

        df = levene_per_feature(
            self.feature_table, self.metadata,
            group_col=group_col, center=center, alpha=alpha,
        )
        entry = make_operation_entry(
            command="run_levene",
            parameters={"group_col": group_col, "center": center, "alpha": alpha},
            results={"n_features": int(len(df))},
        )
        append_operation(self.path, entry)
        return df

    # ------------------------------------------------------------------ #
    # Comparison runner (omnibus + post-hoc + optional exports)
    # ------------------------------------------------------------------ #

    def run_comparison(
        self,
        comparison,
        *,
        export_buckets: bool = True,
        export_pairs: bool = True,
        plot_features: bool = False,
        count_matrix: bool = True,
        residual_plots: bool = False,
        annotation_table: pd.DataFrame | None = None,
        max_feature_plots: int | None = None,
        output_dir: str | Path | None = None,
        exclude_cross_corner: bool = True,
    ):
        """Run a :class:`Comparison`, save tables and (optionally) exports.

        For ``anova_2way`` comparisons, results are written to
        ``stats/<name>/`` and side-effects (bucket TSVs, pair TSVs,
        per-feature boxplots, count matrix) are dispatched by
        :func:`stats.comparison_runner.run_comparison`.

        Args:
            comparison: A :class:`Comparison` instance.
            export_buckets, export_pairs, plot_features, count_matrix,
            residual_plots:
                Forwarded to the runner. Defaults match the typical
                two-way workflow (everything but the heavy feature plots
                and the residual-diagnostics figure).
            annotation_table: Defaults to ``self.annotations`` if the
                project has been annotated; otherwise ``None``.
            max_feature_plots: Cap on per-feature PNGs when
                ``plot_features`` is True.
            output_dir: Custom output root. Defaults to
                ``<project>/stats/<comparison.name>/``.

        Returns:
            :class:`ComparisonResult` from the runner.
        """
        from metabproject.stats.comparison_runner import (
            run_comparison as _run_comparison,
        )

        if annotation_table is None:
            try:
                annotation_table = self.annotations
            except FileNotFoundError:
                annotation_table = None

        if output_dir is None:
            output_dir = self.path / "stats" / comparison.name

        result = _run_comparison(
            comparison, self.feature_table, self.metadata,
            output_dir=output_dir,
            export_buckets=export_buckets,
            export_pairs=export_pairs,
            plot_features=plot_features,
            count_matrix=count_matrix,
            residual_plots=residual_plots,
            annotation_table=annotation_table,
            max_feature_plots=max_feature_plots,
            exclude_cross_corner=exclude_cross_corner,
        )

        # Persist omnibus + posthoc so they're recoverable across sessions.
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        result.omnibus.to_parquet(out_root / "omnibus.parquet")
        if result.posthoc is not None:
            result.posthoc.to_parquet(out_root / "posthoc.parquet")

        rel_outputs = {}
        for label, p in (
            ("bucket_dir", result.bucket_dir),
            ("pair_dir", result.pair_dir),
            ("feature_plots_dir", result.feature_plots_dir),
            ("count_matrix_path", result.count_matrix_path),
            ("residuals_path", result.residuals_path),
            ("residual_per_feature_path", result.residual_per_feature_path),
            ("output_dir", result.output_dir),
        ):
            if p is not None:
                try:
                    rel_outputs[label] = str(Path(p).relative_to(self.path))
                except ValueError:
                    rel_outputs[label] = str(p)

        entry = make_operation_entry(
            command="run_comparison",
            parameters={
                "name": comparison.name,
                "kind": comparison.kind,
                "factors": list(comparison.factors),
                "posthoc": comparison.posthoc,
                "alpha": comparison.alpha,
                "export_buckets": export_buckets,
                "export_pairs": export_pairs,
                "plot_features": plot_features,
                "count_matrix": count_matrix,
                "residual_plots": residual_plots,
                "max_feature_plots": max_feature_plots,
                "exclude_cross_corner": exclude_cross_corner,
            },
            results={
                "n_features": int(len(result.omnibus)),
                "outputs": rel_outputs,
                "export_summary": result.export_summary,
            },
        )
        append_operation(self.path, entry)
        return result

    # ------------------------------------------------------------------ #
    # New viz wrappers
    # ------------------------------------------------------------------ #

    def viz_feature_boxplot(
        self,
        feature_id,
        factors: list[str],
        comparison_name: str | None = None,
        with_annotation: bool = True,
        palette: dict | None = None,
        basename: str | None = None,
    ):
        """Per-feature box+strip plot for a two-factor design."""
        from metabproject.viz import feature_cell_means_figure, save_figure
        from metabproject.viz._style import load_palette_yaml

        merged_palette = dict(load_palette_yaml(self.path))
        if palette:
            merged_palette.update(palette)
        palette = merged_palette or None

        omnibus_row = None
        if comparison_name is not None:
            om_path = self.path / "stats" / comparison_name / "omnibus.parquet"
            if om_path.exists():
                omnibus = pd.read_parquet(om_path)
                if feature_id in omnibus.index:
                    omnibus_row = omnibus.loc[feature_id]

        annotation_row = None
        if with_annotation:
            try:
                ann = self.annotations
                if feature_id in ann.index:
                    annotation_row = ann.loc[feature_id]
            except FileNotFoundError:
                pass

        fig = feature_cell_means_figure(
            feature_id, self.feature_table, self.metadata, factors,
            omnibus_row=omnibus_row, annotation_row=annotation_row,
            palette=palette,
        )
        if basename is None:
            basename = f"feature_{feature_id}"
        viz_dir = self.path / "viz"
        png, svg = save_figure(fig, viz_dir, basename)
        entry = make_operation_entry(
            command="viz_feature_boxplot",
            parameters={
                "feature_id": int(feature_id), "factors": factors,
                "comparison_name": comparison_name,
                "with_annotation": with_annotation, "basename": basename,
            },
            results={"outputs": [str(png.relative_to(self.path)),
                                 str(svg.relative_to(self.path))]},
        )
        append_operation(self.path, entry)
        return fig

    def viz_feature_residuals(
        self,
        feature_id,
        factors: list[str],
        basename: str | None = None,
    ):
        """Single-feature 2-way ANOVA residual QQ + distribution plot.

        Uses the same model as :func:`run_comparison` for
        ``kind='anova_2way'``. Saves PNG + SVG to ``viz/`` and returns
        the figure.

        Args:
            feature_id: Row label in ``feature_table.index``.
            factors: Two metadata column names. Must match the design
                used when the comparison was run (otherwise the
                residuals here won't correspond to those in
                ``stats/<name>/residual_diagnostics_per_feature.parquet``).
            basename: Output basename. Defaults to
                ``feature_<id>_residuals``.
        """
        from metabproject.stats.anova_diagnostics import (
            feature_residual_diagnostics_figure,
        )
        from metabproject.viz._style import save_figure

        fig = feature_residual_diagnostics_figure(
            feature_id, self.feature_table, self.metadata, factors,
        )
        if basename is None:
            basename = f"feature_{feature_id}_residuals"
        viz_dir = self.path / "viz"
        png, svg = save_figure(fig, viz_dir, basename)
        entry = make_operation_entry(
            command="viz_feature_residuals",
            parameters={
                "feature_id": int(feature_id) if hasattr(feature_id, "__int__")
                else str(feature_id),
                "factors": list(factors),
                "basename": basename,
            },
            results={"outputs": [str(png.relative_to(self.path)),
                                 str(svg.relative_to(self.path))]},
        )
        append_operation(self.path, entry)
        return fig

    def viz_volcano_pairs(
        self,
        comparison_name: str,
        factor: str,
        pairs: list[tuple[str, str]],
        annotated: bool = True,
        fc_threshold: float = 1.0,
        q_threshold: float = 0.05,
        label_top_n: int = 5,
        restrict_to_buckets: set[str] | None = None,
        ncols: int = 3,
        row_labels: list[str] | None = None,
        suptitle: str | None = None,
        basename: str | None = None,
    ):
        """Render an annotation-aware multi-panel volcano grid.

        ``factor`` may be either of the comparison's input factors or the
        combined ``_combined_<f1>_x_<f2>`` factor (for interaction-bucket
        pair lookups).
        """
        from metabproject.viz import save_figure, tukey_pair_to_contrast, volcano_grid

        posthoc_path = self.path / "stats" / comparison_name / "posthoc.parquet"
        omnibus_path = self.path / "stats" / comparison_name / "omnibus.parquet"
        if not posthoc_path.exists():
            raise FileNotFoundError(
                f"posthoc not found for {comparison_name!r} (run_comparison first)"
            )
        posthoc = pd.read_parquet(posthoc_path)

        if restrict_to_buckets is not None:
            if not omnibus_path.exists():
                raise FileNotFoundError(
                    "omnibus not found, can't restrict by classification"
                )
            omnibus = pd.read_parquet(omnibus_path)
            cls = omnibus["classification"]
            keep = posthoc["feature"].map(cls).isin(restrict_to_buckets)
            posthoc = posthoc[keep]

        annotated_features: set | None = None
        label_lookup: pd.Series | None = None
        if annotated:
            try:
                ann = self.annotations
                annotated_features = set(ann.index)
                if "compound_name" in ann.columns:
                    label_lookup = ann["compound_name"].dropna()
            except FileNotFoundError:
                annotated_features = set()

        panels = []
        for a, b in pairs:
            contrast_df = tukey_pair_to_contrast(posthoc, factor, a, b)
            panels.append({
                "title": f"{a} vs {b}", "contrast_df": contrast_df,
            })

        fig, counts = volcano_grid(
            panels,
            fc_threshold=fc_threshold, q_threshold=q_threshold,
            label_top_n=label_top_n,
            annotated_features=annotated_features,
            label_lookup=label_lookup,
            ncols=ncols, row_labels=row_labels, suptitle=suptitle,
        )

        if basename is None:
            basename = f"volcano_pairs_{comparison_name}_{factor}"
        viz_dir = self.path / "viz"
        png, svg = save_figure(fig, viz_dir, basename)
        entry = make_operation_entry(
            command="viz_volcano_pairs",
            parameters={
                "comparison_name": comparison_name, "factor": factor,
                "pairs": [list(p) for p in pairs],
                "annotated": annotated, "fc_threshold": fc_threshold,
                "q_threshold": q_threshold, "label_top_n": label_top_n,
                "restrict_to_buckets": (
                    sorted(restrict_to_buckets) if restrict_to_buckets else None
                ),
                "basename": basename,
            },
            results={"outputs": [str(png.relative_to(self.path)),
                                 str(svg.relative_to(self.path))],
                     "counts_per_panel": counts},
        )
        append_operation(self.path, entry)
        return fig

    def viz_pairwise_count_heatmap(
        self,
        comparison_name: str,
        factors: list[str] | None = None,
        group_separators: list[int] | None = None,
        cells: list[str] | None = None,
        basename: str | None = None,
    ):
        """Heatmap of the pairwise significant-feature count matrix."""
        from metabproject.stats.exports import pairwise_count_matrix
        from metabproject.viz import pairwise_count_heatmap, save_figure

        posthoc_path = self.path / "stats" / comparison_name / "posthoc.parquet"
        if not posthoc_path.exists():
            raise FileNotFoundError(
                f"posthoc not found for {comparison_name!r}"
            )
        posthoc = pd.read_parquet(posthoc_path)
        if factors is None:
            # Read from manifest if we can; otherwise fail loud.
            raise ValueError("factors must be provided")

        mat = pairwise_count_matrix(posthoc, factors, cells=cells)
        fig = pairwise_count_heatmap(mat, group_separators=group_separators)

        if basename is None:
            basename = f"pairwise_count_{comparison_name}"
        viz_dir = self.path / "viz"
        png, svg = save_figure(fig, viz_dir, basename)
        entry = make_operation_entry(
            command="viz_pairwise_count_heatmap",
            parameters={
                "comparison_name": comparison_name, "factors": list(factors),
                "group_separators": group_separators,
                "cells": cells, "basename": basename,
            },
            results={"outputs": [str(png.relative_to(self.path)),
                                 str(svg.relative_to(self.path))]},
        )
        append_operation(self.path, entry)
        return fig

    def viz_rf_importance(
        self, rf_result, top_n: int = 30, basename: str = "rf_importance",
    ):
        """Bar plot of top-N random-forest feature importances (MDA)."""
        import matplotlib.pyplot as plt
        from metabproject.viz._style import save_figure

        df = rf_result.feature_importance.head(top_n).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, max(3, 0.25 * len(df) + 1)))
        ax.barh(df["feature"].astype(str), df["mda_mean"],
                xerr=df["mda_std"], color="#4393c3")
        ax.set_xlabel("Permutation importance (MDA)")
        ax.set_ylabel("Feature")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        save_figure(fig, self.path / "viz", basename)
        return fig

    def viz_rf_oob_curve(self, rf_result, basename: str = "rf_oob"):
        """Line plot of OOB score vs n_estimators."""
        import matplotlib.pyplot as plt
        from metabproject.viz._style import save_figure

        oob = rf_result.oob_curve
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(oob["n_estimators"], oob["oob_score"], color="#a42727")
        ax.set_xlabel("n_estimators")
        ax.set_ylabel("OOB score")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        save_figure(fig, self.path / "viz", basename)
        return fig

    def viz_hca_dendrogram(
        self,
        hca_result,
        label_col: str | None = None,
        basename: str = "hca_dendrogram",
    ):
        """Dendrogram of an :class:`HCAResult` linkage matrix."""
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram
        from metabproject.viz._style import save_figure

        samples = list(self.feature_table.columns)
        labels = samples
        if label_col is not None and label_col in self.metadata.columns:
            labels = [
                f"{self.metadata.loc[s, label_col]} ({s})" for s in samples
            ]

        fig, ax = plt.subplots(figsize=(max(6, 0.25 * len(labels) + 2), 5))
        dendrogram(
            hca_result.linkage_matrix, labels=labels, ax=ax,
            color_threshold=0, above_threshold_color="#666666",
        )
        ax.set_ylabel("Distance")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=8)
        fig.tight_layout()
        save_figure(fig, self.path / "viz", basename)
        return fig

    # -- Feature QC -----------------------------------------------------------

    def feature_qc(
        self,
        *,
        check: str,
        name: str,
        mz_tol_ppm: float = 10.0,
        rt_tol: float = 0.1,
        min_correlation: float = 0.7,
        use_fwhm: bool = True,
        group_by: str | None = None,
        min_samples_per_group: int = 3,
        low_alignment_rate_threshold: float = 0.10,
        low_detection_sample_zscore: float = -2.0,
        poor_peak_shape_threshold: float = 2.0,
        mzmine_raw: str | Path | None = None,
        force: bool = False,
    ) -> dict:
        """Run a feature-table QC check on the raw MZmine output.

        Two checks are available:

        - ``"alignment"`` — detect duplicate features from MZmine
          alignment/deconvolution failures.
        - ``"overview"`` — profile the full feature table (shape, detection,
          alignment scores, chromatographic quality, MS2 coverage, ...).

        Args:
            check: Which check to run (``"alignment"`` or ``"overview"``).
            name: Run name. Output files are written under
                ``<project>/feature_qc/<name>__*``.
            mz_tol_ppm: (alignment) m/z tolerance (ppm) for candidate pairs.
            rt_tol: (alignment) RT tolerance (minutes) for candidate pairs.
            min_correlation: (alignment) Pearson cutoff for split-like.
            use_fwhm: (alignment) Use Zone 3 FWHM to compute ΔRT/FWHM.
            group_by: (alignment) Metadata column to slice misalignment rates
                by. When set, duplicate-pair detection is re-run inside each
                level of the column and a per-group TSV + summary section are
                emitted alongside the global outputs.
            min_samples_per_group: (alignment) Skip groups with fewer than
                this many samples (default 3, matches the shared-detection
                threshold used by Pearson). Only used when ``group_by`` is set.
            low_alignment_rate_threshold: (overview) ``alignment_scores:rate``
                cutoff below which features count as low-alignment.
            low_detection_sample_zscore: (overview) z-score cutoff for flagging
                samples with low detection counts (use a negative number).
            poor_peak_shape_threshold: (overview) tailing/asymmetry cutoff
                above which a DETECTED cell counts as poor shape.
            mzmine_raw: Path to the raw MZmine CSV. Defaults to the one
                stored in the ``init`` manifest entry.
            force: Overwrite existing run with the same ``name``.

        Returns:
            The summary/metrics dict for the selected check.
        """
        if check not in ("alignment", "overview"):
            raise ValueError(
                f"Unknown check {check!r}. Available: ['alignment', 'overview']"
            )

        resolved_raw = self._resolve_mzmine_raw(mzmine_raw)
        out_dir = self.path / "feature_qc"
        out_dir.mkdir(parents=True, exist_ok=True)

        ms1, areas, sample_details = parse_feature_table(resolved_raw)
        n_features = len(ms1)

        if check == "alignment":
            return self._feature_qc_alignment(
                ms1=ms1, areas=areas, sample_details=sample_details,
                name=name, out_dir=out_dir, resolved_raw=resolved_raw,
                mz_tol_ppm=mz_tol_ppm, rt_tol=rt_tol,
                min_correlation=min_correlation, use_fwhm=use_fwhm,
                group_by=group_by,
                min_samples_per_group=min_samples_per_group,
                n_features=n_features, force=force,
            )
        return self._feature_qc_overview(
            ms1=ms1, areas=areas, sample_details=sample_details,
            name=name, out_dir=out_dir, resolved_raw=resolved_raw,
            low_alignment_rate_threshold=low_alignment_rate_threshold,
            low_detection_sample_zscore=low_detection_sample_zscore,
            poor_peak_shape_threshold=poor_peak_shape_threshold,
            force=force,
        )

    def _feature_qc_alignment(
        self,
        *,
        ms1: pd.DataFrame,
        areas: pd.DataFrame,
        sample_details: pd.DataFrame,
        name: str,
        out_dir: Path,
        resolved_raw: Path,
        mz_tol_ppm: float,
        rt_tol: float,
        min_correlation: float,
        use_fwhm: bool,
        group_by: str | None,
        min_samples_per_group: int,
        n_features: int,
        force: bool,
    ) -> dict:
        pairs_path = out_dir / f"{name}__pairs.tsv"
        flags_path = out_dir / f"{name}__feature_flags.tsv"
        summary_path = out_dir / f"{name}__summary.txt"
        if pairs_path.exists() and not force:
            raise FileExistsError(
                f"Feature-QC run {name!r} already exists at {pairs_path}. "
                f"Use force=True to overwrite."
            )

        pairs_df, flags_df, summary = _run_alignment_check(
            ms1, areas, sample_details,
            mz_tol_ppm=mz_tol_ppm,
            rt_tol=rt_tol,
            min_correlation=min_correlation,
            use_fwhm=use_fwhm,
        )

        pairs_df.to_csv(pairs_path, sep="\t", index=False)
        flags_df.to_csv(flags_path, sep="\t", index=False)

        per_group_rows: list[dict] = []
        skipped_groups: list[dict] = []
        by_group_path: Path | None = None
        if group_by is not None:
            per_group_rows, skipped_groups = _run_alignment_by_group(
                ms1=ms1, areas=areas, sample_details=sample_details,
                metadata=self.metadata,
                group_by=group_by,
                min_samples_per_group=min_samples_per_group,
                mz_tol_ppm=mz_tol_ppm,
                rt_tol=rt_tol,
                min_correlation=min_correlation,
                use_fwhm=use_fwhm,
            )
            by_group_path = out_dir / f"{name}__by_group.tsv"
            pd.DataFrame(per_group_rows).to_csv(
                by_group_path, sep="\t", index=False,
            )

        summary_path.write_text(
            _format_summary_text(
                name, summary, n_features,
                group_by=group_by,
                per_group_rows=per_group_rows,
                skipped_groups=skipped_groups,
            ),
            encoding="utf-8",
        )

        counts = summary["counts"]
        output_paths = {
            "pairs": str(pairs_path.relative_to(self.path)),
            "flags": str(flags_path.relative_to(self.path)),
            "summary": str(summary_path.relative_to(self.path)),
        }
        if by_group_path is not None:
            output_paths["by_group"] = str(by_group_path.relative_to(self.path))
        results = {
            "check": "alignment",
            "n_features_input": int(n_features),
            "n_suspect_pairs": int(len(pairs_df)),
            "n_alignment_like": int(counts["alignment_like"]),
            "n_split_like": int(counts["split_like"]),
            "n_isomer_like": int(counts["isomer_like"]),
            "n_unclassified": int(counts["unclassified"]),
            "n_features_flagged": int(len(flags_df)),
            "output_paths": output_paths,
        }
        if group_by is not None:
            worst = None
            if per_group_rows:
                worst_row = max(
                    per_group_rows,
                    key=lambda r: r.get("alignment_like_rate", 0.0) or 0.0,
                )
                worst = {
                    "group": worst_row["group_value"],
                    "alignment_like_rate": float(
                        worst_row["alignment_like_rate"]
                    ),
                }
            results["by_group"] = {
                "column": group_by,
                "min_samples_per_group": int(min_samples_per_group),
                "worst_group_alignment_like": worst,
                "rows": per_group_rows,
                "skipped_groups": skipped_groups,
            }
        entry = make_operation_entry(
            command="feature_qc",
            parameters={
                "check": "alignment",
                "name": name,
                "mz_tol_ppm": mz_tol_ppm,
                "rt_tol": rt_tol,
                "min_correlation": min_correlation,
                "use_fwhm": use_fwhm,
                "group_by": group_by,
                "min_samples_per_group": int(min_samples_per_group),
                "mzmine_raw": str(resolved_raw),
                "force": force,
            },
            results=results,
        )
        append_operation(self.path, entry)

        print(
            f"Feature-QC {name!r}: {len(pairs_df)} suspect pair(s) "
            f"({counts['alignment_like']} alignment, "
            f"{counts['split_like']} split, "
            f"{counts['isomer_like']} isomer, "
            f"{counts['unclassified']} other); "
            f"{len(flags_df)} flagged feature(s)."
        )
        if group_by is not None and per_group_rows:
            worst_row = max(
                per_group_rows,
                key=lambda r: r.get("alignment_like_rate", 0.0) or 0.0,
            )
            print(
                f"Worst-aligned group by {group_by}: "
                f"{worst_row['group_value']!r} "
                f"(alignment_like_rate="
                f"{worst_row['alignment_like_rate']:.4f}, "
                f"n_samples={worst_row['n_samples']})."
            )
        return summary

    def _feature_qc_overview(
        self,
        *,
        ms1: pd.DataFrame,
        areas: pd.DataFrame,
        sample_details: pd.DataFrame,
        name: str,
        out_dir: Path,
        resolved_raw: Path,
        low_alignment_rate_threshold: float,
        low_detection_sample_zscore: float,
        poor_peak_shape_threshold: float,
        force: bool,
    ) -> dict:
        import json

        from metabproject.qc.overview import compute_overview
        from metabproject.viz.missingness import make_missingness_heatmap

        json_path = out_dir / f"{name}__overview.json"
        txt_path = out_dir / f"{name}__overview.txt"
        heatmap_path = out_dir / f"{name}__detection_heatmap.png"
        per_sample_csv = out_dir / f"{name}__missing_per_sample.csv"
        if json_path.exists() and not force:
            raise FileExistsError(
                f"Feature-QC overview run {name!r} already exists at "
                f"{json_path}. Use force=True to overwrite."
            )
        if force:
            for p in (heatmap_path, per_sample_csv):
                if p.exists():
                    p.unlink()

        # Map raw datafile filenames to user-facing sample_name via metadata.
        meta_path = self.path / "features" / "metadata.parquet"
        sample_name_map: dict[str, str] | None = None
        meta: pd.DataFrame | None = None
        if meta_path.exists():
            meta = pd.read_parquet(meta_path)
            if "sample_file" in meta.columns:
                sample_name_map = {
                    str(row["sample_file"]): str(name_)
                    for name_, row in meta.iterrows()
                }

        metrics, issues = compute_overview(
            ms1, areas, sample_details,
            low_alignment_rate_threshold=low_alignment_rate_threshold,
            low_detection_sample_zscore=low_detection_sample_zscore,
            poor_peak_shape_threshold=poor_peak_shape_threshold,
            sample_name_map=sample_name_map,
        )

        overview_doc = {"metrics": metrics, "issues": issues}
        json_path.write_text(
            json.dumps(overview_doc, indent=2, default=_json_default),
            encoding="utf-8",
        )
        txt_path.write_text(
            _format_overview_text(name, metrics, issues),
            encoding="utf-8",
        )

        # Build a not-detected mask (features × samples) and render the
        # per-sample missingness heatmap so the user can see where MZmine
        # failed to detect features on a per-file basis.
        detected_mask = (
            sample_details.xs("feature_state", axis=1, level="metric") == "DETECTED"
        )
        not_detected = ~detected_mask
        # Rename mask columns from raw datafile filenames to sample_name
        # where a mapping is available, so heatmap labels match the metadata.
        if sample_name_map:
            not_detected = not_detected.rename(columns=sample_name_map)
        # Build a metadata DataFrame indexed by the heatmap's column labels.
        sample_labels = list(not_detected.columns)
        if meta is not None and "group" in meta.columns:
            heat_metadata = meta.reindex(sample_labels)[["group"]]
            heat_metadata["group"] = heat_metadata["group"].fillna("unknown")
        else:
            heat_metadata = pd.DataFrame(
                {"group": ["all"] * len(sample_labels)}, index=sample_labels
            )
        make_missingness_heatmap(not_detected, heat_metadata, heatmap_path)

        n_not_detected_per_sample = not_detected.sum(axis=0).astype(int)
        n_features_total = int(len(not_detected))
        per_sample_summary = pd.DataFrame(
            {
                "n_missing": n_not_detected_per_sample,
                "n_features": n_features_total,
                "pct_missing": (
                    n_not_detected_per_sample / max(n_features_total, 1) * 100.0
                ).round(2),
            }
        )
        per_sample_summary.index.name = "sample_name"
        per_sample_summary.to_csv(per_sample_csv)

        n_issues_total = len(issues)
        n_issues_warning = sum(1 for i in issues if i.get("severity") == "warning")
        shape = metrics.get("table_shape", {})
        results = {
            "check": "overview",
            "n_features": int(shape.get("n_features", 0)),
            "n_samples": int(shape.get("n_samples", 0)),
            "density": _to_jsonable(shape.get("density")),
            "n_issues_total": int(n_issues_total),
            "n_issues_warning": int(n_issues_warning),
            "output_paths": {
                "overview_json": str(json_path.relative_to(self.path)),
                "overview_txt": str(txt_path.relative_to(self.path)),
                "detection_heatmap": str(heatmap_path.relative_to(self.path)),
                "missing_per_sample_csv": str(per_sample_csv.relative_to(self.path)),
            },
        }
        entry = make_operation_entry(
            command="feature_qc",
            parameters={
                "check": "overview",
                "name": name,
                "low_alignment_rate_threshold": low_alignment_rate_threshold,
                "low_detection_sample_zscore": low_detection_sample_zscore,
                "poor_peak_shape_threshold": poor_peak_shape_threshold,
                "mzmine_raw": str(resolved_raw),
                "force": force,
            },
            results=results,
        )
        append_operation(self.path, entry)

        print(
            f"Feature-QC overview {name!r}: "
            f"{shape.get('n_features', 0)} features × "
            f"{shape.get('n_samples', 0)} samples; "
            f"{n_issues_warning} warning(s), "
            f"{n_issues_total - n_issues_warning} info."
        )
        print("Not-detected per sample:")
        for sample, row in per_sample_summary.iterrows():
            print(
                f"  {sample}: {int(row['n_missing']):,} / "
                f"{int(row['n_features']):,} ({row['pct_missing']:.2f}%)"
            )
        print(f"Detection heatmap: {heatmap_path}")
        return overview_doc

    def viz_feature_qc(
        self,
        *,
        name: str,
        basename: str | None = None,
    ):
        """Render the diagnostic figures for a ``feature_qc`` run.

        Dispatches by run type:

        - alignment runs (``<name>__pairs.tsv`` exists) → scatter + histogram.
        - overview runs (``<name>__overview.json`` exists) → 2 × 3 panel grid.

        Args:
            name: Run name used by :meth:`feature_qc`.
            basename: Stem for the output files. Defaults to ``name``.

        Returns:
            Either ``(scatter_fig, hist_fig)`` for an alignment run or
            ``overview_fig`` for an overview run.
        """
        import matplotlib.pyplot as _plt

        from metabproject.viz._style import save_figure

        feature_qc_dir = self.path / "feature_qc"
        pairs_path = feature_qc_dir / f"{name}__pairs.tsv"
        overview_path = feature_qc_dir / f"{name}__overview.json"

        if pairs_path.exists():
            from metabproject.viz.feature_qc import feature_qc_figures
            pairs_df = pd.read_csv(pairs_path, sep="\t")
            stem = basename or name
            scatter_fig, hist_fig = feature_qc_figures(pairs_df)
            viz_dir = self.path / "viz"
            scatter_png, scatter_svg = save_figure(
                scatter_fig, viz_dir, f"{stem}__feature_qc_scatter"
            )
            hist_png, hist_svg = save_figure(
                hist_fig, viz_dir, f"{stem}__feature_qc_fwhm_hist"
            )
            outputs = [
                str(scatter_png.relative_to(self.path)),
                str(scatter_svg.relative_to(self.path)),
                str(hist_png.relative_to(self.path)),
                str(hist_svg.relative_to(self.path)),
            ]
            entry = make_operation_entry(
                command="viz_feature_qc",
                parameters={"name": name, "basename": basename},
                results={"outputs": outputs, "n_pairs": int(len(pairs_df))},
            )
            append_operation(self.path, entry)
            print(f"Feature-QC viz: {scatter_png}, {hist_png}")
            _plt.close(hist_fig)
            return scatter_fig, hist_fig

        if overview_path.exists():
            import json

            from metabproject.viz.feature_qc import feature_qc_overview_figure

            doc = json.loads(overview_path.read_text(encoding="utf-8"))
            stem = basename or name
            fig = feature_qc_overview_figure(
                doc["metrics"], doc.get("issues", [])
            )
            viz_dir = self.path / "viz"
            png_path, svg_path = save_figure(
                fig, viz_dir, f"{stem}__feature_qc_overview"
            )
            outputs = [
                str(png_path.relative_to(self.path)),
                str(svg_path.relative_to(self.path)),
            ]
            shape = doc["metrics"].get("table_shape", {})
            entry = make_operation_entry(
                command="viz_feature_qc",
                parameters={"name": name, "basename": basename},
                results={
                    "outputs": outputs,
                    "n_features": int(shape.get("n_features", 0)),
                    "n_samples": int(shape.get("n_samples", 0)),
                },
            )
            append_operation(self.path, entry)
            print(f"Feature-QC overview viz: {png_path}")
            return fig

        raise FileNotFoundError(
            f"No feature_qc run named {name!r} under {feature_qc_dir}. "
            f"Run `metab feature-qc` first."
        )

    def _resolve_mzmine_raw(self, override: str | Path | None) -> Path:
        if override is not None:
            path = Path(override)
            if not path.exists():
                raise FileNotFoundError(f"MZmine raw CSV not found: {path}")
            return path
        manifest = read_manifest(self.path)
        for op in manifest.get("operations", []):
            if op.get("command") == "init":
                ft = op.get("parameters", {}).get("feature_table")
                if ft:
                    path = Path(ft)
                    if not path.exists():
                        raise FileNotFoundError(
                            f"MZmine raw CSV from init op not found: {path}. "
                            f"Pass --mzmine-raw to override."
                        )
                    return path
        raise FileNotFoundError(
            "Could not determine MZmine raw CSV path from manifest; "
            "pass --mzmine-raw explicitly."
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _run_alignment_check(
    ms1_features: pd.DataFrame,
    areas: pd.DataFrame,
    sample_details: pd.DataFrame,
    *,
    mz_tol_ppm: float,
    rt_tol: float,
    min_correlation: float,
    use_fwhm: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from metabproject.qc import detect_duplicates

    return detect_duplicates(
        ms1_features, areas, sample_details,
        mz_tol_ppm=mz_tol_ppm,
        rt_tol=rt_tol,
        min_correlation=min_correlation,
        use_fwhm=use_fwhm,
    )


def _run_alignment_by_group(
    *,
    ms1: pd.DataFrame,
    areas: pd.DataFrame,
    sample_details: pd.DataFrame,
    metadata: pd.DataFrame,
    group_by: str,
    min_samples_per_group: int,
    mz_tol_ppm: float,
    rt_tol: float,
    min_correlation: float,
    use_fwhm: bool,
) -> tuple[list[dict], list[dict]]:
    """Run duplicate detection inside each level of a metadata column.

    Returns ``(per_group_rows, skipped_groups)``. Each per-group row carries
    the same per-class pair counts as the global summary plus three
    normalized rates: ``alignment_like_rate``, ``any_suspect_rate``,
    ``flagged_feature_rate``. Rates are normalized by the number of features
    DETECTED in at least one sample of the group, so groups with sparse
    detection are not double-penalized.
    """
    if group_by not in metadata.columns:
        raise ValueError(
            f"--group-by column {group_by!r} not in metadata. "
            f"Available: {sorted(metadata.columns)}"
        )

    if "sample_file" not in metadata.columns:
        raise ValueError(
            "metadata is missing the 'sample_file' column; cannot map "
            "samples to Zone 2/3 columns for per-group analysis."
        )

    file_col_set = set(areas.columns)
    detail_files = set(sample_details.columns.get_level_values(0))

    per_group_rows: list[dict] = []
    skipped_groups: list[dict] = []

    levels = metadata[group_by].dropna().unique().tolist()
    levels.sort(key=lambda v: str(v))

    for level in levels:
        sub = metadata[metadata[group_by] == level]
        files = [
            f for f in sub["sample_file"].tolist()
            if f in file_col_set and f in detail_files
        ]
        n_samples = len(files)
        if n_samples < min_samples_per_group:
            skipped_groups.append({
                "group_value": str(level),
                "n_samples": int(n_samples),
                "reason": (
                    f"fewer than --min-samples-per-group "
                    f"({min_samples_per_group}) usable samples"
                ),
            })
            continue

        areas_sub = areas.loc[:, files]
        details_sub = sample_details.loc[:, pd.IndexSlice[files, :]]

        state_sub = details_sub.xs("feature_state", axis=1, level="metric")
        detected_any = (state_sub == "DETECTED").any(axis=1)
        n_detected = int(detected_any.sum())

        pairs_df, flags_df, summary = _run_alignment_check(
            ms1, areas_sub, details_sub,
            mz_tol_ppm=mz_tol_ppm,
            rt_tol=rt_tol,
            min_correlation=min_correlation,
            use_fwhm=use_fwhm,
        )
        counts = summary["counts"]
        n_pairs = int(len(pairs_df))
        n_flagged = int(len(flags_df))

        def _rate(numer: int) -> float:
            return float(numer) / n_detected if n_detected > 0 else float("nan")

        per_group_rows.append({
            "group_value": str(level),
            "n_samples": n_samples,
            "n_features_detected": n_detected,
            "n_pairs": n_pairs,
            "n_alignment_like": int(counts["alignment_like"]),
            "n_split_like": int(counts["split_like"]),
            "n_isomer_like": int(counts["isomer_like"]),
            "n_unclassified": int(counts["unclassified"]),
            "n_features_flagged": n_flagged,
            "alignment_like_rate": _rate(int(counts["alignment_like"])),
            "any_suspect_rate": _rate(n_pairs),
            "flagged_feature_rate": _rate(n_flagged),
        })

    per_group_rows.sort(
        key=lambda r: r.get("alignment_like_rate", 0.0) or 0.0,
        reverse=True,
    )
    return per_group_rows, skipped_groups


def _format_summary_text(
    name: str,
    summary: dict,
    n_features: int,
    *,
    group_by: str | None = None,
    per_group_rows: list[dict] | None = None,
    skipped_groups: list[dict] | None = None,
) -> str:
    counts = summary["counts"]
    lines = [
        f"Feature-QC run: {name}",
        f"Features in input: {n_features}",
        "",
        "Pair counts by classification:",
        f"  alignment_like : {counts['alignment_like']}",
        f"  split_like     : {counts['split_like']}",
        f"  isomer_like    : {counts['isomer_like']}",
        f"  unclassified   : {counts['unclassified']}",
        "",
        "Top 10 most-flagged features:",
    ]
    if summary["top_flagged"]:
        for row in summary["top_flagged"]:
            lines.append(
                f"  feature_id={row['feature_id']}  "
                f"n_pairs={row['n_pairs']}  "
                f"classes={row['classifications']}  "
                f"action={row['recommended_action']}"
            )
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("Suggested parameter adjustments:")
    if summary["suggestions"]:
        for s in summary["suggestions"]:
            lines.append(f"  - {s}")
    else:
        lines.append("  (no actionable signal)")

    if group_by is not None:
        lines.append("")
        lines.append(f"Per-group misalignment (grouped by {group_by!r}):")
        rows = per_group_rows or []
        if rows:
            worst = rows[0]
            lines.append(
                f"  Worst-aligned group: {worst['group_value']!r}  "
                f"alignment_like_rate={worst['alignment_like_rate']:.4f}  "
                f"(n_samples={worst['n_samples']}, "
                f"n_features_detected={worst['n_features_detected']})"
            )
            lines.append("")
            lines.append(
                "  group  n_samples  n_detected  n_align  n_split  "
                "n_isomer  n_unclass  align_rate  any_rate  flag_rate"
            )
            for r in rows:
                lines.append(
                    f"  {r['group_value']}  "
                    f"{r['n_samples']}  "
                    f"{r['n_features_detected']}  "
                    f"{r['n_alignment_like']}  "
                    f"{r['n_split_like']}  "
                    f"{r['n_isomer_like']}  "
                    f"{r['n_unclassified']}  "
                    f"{r['alignment_like_rate']:.4f}  "
                    f"{r['any_suspect_rate']:.4f}  "
                    f"{r['flagged_feature_rate']:.4f}"
                )
        else:
            lines.append("  (no scoreable groups)")
        if skipped_groups:
            lines.append("")
            lines.append("  Skipped groups (insufficient samples):")
            for s in skipped_groups:
                lines.append(
                    f"    {s['group_value']}  n_samples={s['n_samples']}  "
                    f"({s['reason']})"
                )
    return "\n".join(lines) + "\n"


def _to_jsonable(value):
    """Convert numpy/pandas scalars to plain Python (NaN → None)."""
    import math

    if value is None:
        return None
    if isinstance(value, float):
        return None if math.isnan(value) else value
    try:
        import numpy as _np
        if isinstance(value, _np.floating):
            v = float(value)
            return None if math.isnan(v) else v
        if isinstance(value, _np.integer):
            return int(value)
        if isinstance(value, _np.bool_):
            return bool(value)
    except ImportError:
        pass
    return value


def _json_default(o):
    """``json.dumps`` ``default=`` for numpy / pandas types."""
    import math

    try:
        import numpy as _np
        if isinstance(o, _np.floating):
            v = float(o)
            return None if math.isnan(v) else v
        if isinstance(o, _np.integer):
            return int(o)
        if isinstance(o, _np.bool_):
            return bool(o)
        if isinstance(o, _np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, float) and math.isnan(o):
        return None
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _fmt(v, fmt: str = ".3f") -> str:
    """Pretty-print a numeric value (NaN/None → '-')."""
    import math

    if v is None:
        return "-"
    if isinstance(v, float) and math.isnan(v):
        return "-"
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return str(v)


def _format_overview_text(name: str, metrics: dict, issues: list) -> str:
    lines = [f"Feature-QC overview: {name}", ""]

    shape = metrics.get("table_shape", {})
    lines += [
        "Table shape",
        f"  n_features : {shape.get('n_features', 0)}",
        f"  n_samples  : {shape.get('n_samples', 0)}",
        f"  density    : {_fmt(shape.get('density'))}",
    ]
    fps = shape.get("features_per_sample", {})
    if fps:
        lines.append(
            f"  features/sample: median={_fmt(fps.get('median'), '.0f')} "
            f"mean={_fmt(fps.get('mean'), '.1f')} "
            f"p25={_fmt(fps.get('p25'), '.0f')} "
            f"p75={_fmt(fps.get('p75'), '.0f')} "
            f"min={_fmt(fps.get('min'), '.0f')} "
            f"max={_fmt(fps.get('max'), '.0f')}"
        )
    lines.append("")

    pf = metrics.get("per_feature_detection", {})
    if pf:
        lines.append("Per-feature detection")
        hist = pf.get("detection_histogram", {})
        for label, entry in hist.items():
            lines.append(
                f"  {label:>10s} : {entry.get('count', 0):>6d}  "
                f"({_fmt(entry.get('fraction'))})"
            )
        nds = pf.get("n_detected_stats", {})
        if nds:
            lines.append(
                f"  n_detected: median={_fmt(nds.get('median'), '.1f')} "
                f"mean={_fmt(nds.get('mean'), '.1f')} "
                f"p25={_fmt(nds.get('p25'), '.1f')} "
                f"p75={_fmt(nds.get('p75'), '.1f')}"
            )
        lines.append("")

    ps = metrics.get("per_sample_detection", {})
    if ps:
        lines.append("Per-sample detection (ascending)")
        for row in ps.get("per_sample_detected", []):
            lines.append(
                f"  {row['sample']:<32s} n={row['n_detected']:>6d} "
                f"frac={_fmt(row['fraction_of_features'])}"
            )
        flagged = ps.get("flagged_samples", [])
        if flagged:
            lines.append(f"  flagged: {', '.join(flagged)}")
        lines.append("")

    aq = metrics.get("alignment_quality", {})
    if aq:
        lines.append("Alignment quality")
        for metric_name, stats in aq.get("stats", {}).items():
            short = metric_name.split(":", 1)[-1]
            lines.append(
                f"  {short:<28s} median={_fmt(stats.get('median'))} "
                f"p25={_fmt(stats.get('p25'))} p75={_fmt(stats.get('p75'))} "
                f"p95={_fmt(stats.get('p95'))}"
            )
        lines.append(
            f"  n_low_alignment_rate (<{_fmt(aq.get('low_alignment_rate_threshold'))})"
            f": {aq.get('n_low_alignment_rate', 0)} "
            f"({_fmt(aq.get('fraction_low_alignment_rate'))})"
        )
        lines.append("")

    cq = metrics.get("chromatographic_quality", {})
    if cq:
        lines.append("Chromatographic quality")
        fg = cq.get("fwhm_global", {})
        lines.append(
            f"  fwhm   : median={_fmt(fg.get('median'))} "
            f"p25={_fmt(fg.get('p25'))} p75={_fmt(fg.get('p75'))}"
        )
        tf = cq.get("tailing_factor", {})
        lines.append(
            f"  tailing: median={_fmt(tf.get('median'))} "
            f"p25={_fmt(tf.get('p25'))} p75={_fmt(tf.get('p75'))} "
            f"p95={_fmt(tf.get('p95'))}"
        )
        af = cq.get("asymmetry_factor", {})
        lines.append(
            f"  asym   : median={_fmt(af.get('median'))} "
            f"p25={_fmt(af.get('p25'))} p75={_fmt(af.get('p75'))} "
            f"p95={_fmt(af.get('p95'))}"
        )
        lines.append(
            f"  fraction tailing>{_fmt(cq.get('poor_peak_shape_threshold'))}"
            f": {_fmt(cq.get('fraction_poor_tailing'))}"
        )
        lines.append(
            f"  fraction asym>{_fmt(cq.get('poor_peak_shape_threshold'))}"
            f": {_fmt(cq.get('fraction_poor_asymmetry'))}"
        )
        lines.append("")

    ms2 = metrics.get("ms2_coverage", {})
    if ms2:
        lines.append("MS2 coverage")
        lines.append(
            f"  fraction_with_ms2: {_fmt(ms2.get('fraction_with_ms2'))}"
        )
        deciles = ms2.get("ms2_by_intensity_decile", {})
        if deciles:
            row = " ".join(
                f"d{k}={_fmt(v, '.2f')}" for k, v in sorted(
                    deciles.items(), key=lambda kv: int(kv[0])
                )
            )
            lines.append(f"  by_intensity_decile: {row}")
        lines.append("")

    mr = metrics.get("mass_rt_coverage", {})
    if mr:
        mz_r = mr.get("mz_range", {})
        rt_r = mr.get("rt_range", {})
        lines += [
            "Mass / RT coverage",
            f"  m/z: {_fmt(mz_r.get('min'))} – {_fmt(mz_r.get('max'))}",
            f"  rt : {_fmt(rt_r.get('min'))} – {_fmt(rt_r.get('max'))}",
            "",
        ]

    ann = metrics.get("annotation_coverage")
    if ann:
        lines.append("Annotation coverage")
        if "fraction_preferred_annotated" in ann:
            lines.append(
                f"  preferred: {_fmt(ann['fraction_preferred_annotated'])}"
            )
        if "fraction_lipid_annotated" in ann:
            lines.append(
                f"  lipid    : {_fmt(ann['fraction_lipid_annotated'])}"
            )
        for k, v in ann.get("by_method", {}).items():
            lines.append(f"  method[{k}]: {v}")
        lines.append("")

    cs = metrics.get("charge_state")
    if cs:
        lines.append("Charge state")
        counts = cs.get("counts", {})
        for k in ("1", "2", "3+", "missing"):
            lines.append(
                f"  {k:>7s}: {counts.get(k, 0):>6d}  "
                f"({_fmt(cs.get('fractions', {}).get(k))})"
            )
        lines.append("")

    warnings = [i for i in issues if i.get("severity") == "warning"]
    infos = [i for i in issues if i.get("severity") != "warning"]
    lines.append("Issues")
    if not issues:
        lines.append("  (none)")
    for grp_label, grp in (("warning", warnings), ("info", infos)):
        for issue in grp:
            details = issue.get("details", {})
            detail_str = ", ".join(
                f"{k}={_fmt(v) if isinstance(v, (int, float)) else v}"
                for k, v in details.items()
            )
            lines.append(
                f"  [{grp_label}] {issue.get('type', '?')}: {detail_str}"
            )
    return "\n".join(lines) + "\n"


def _load_and_validate_metadata(
    metadata_path: Path,
    areas: pd.DataFrame,
) -> pd.DataFrame:
    """Load metadata CSV and validate it against the area column filenames.

    Args:
        metadata_path: Path to the metadata CSV.
        areas: Zone 2 area DataFrame with raw filenames as columns.

    Returns:
        Validated metadata DataFrame indexed by ``sample_name``.

    Raises:
        ValueError: On any validation failure.
    """
    meta = pd.read_csv(metadata_path)

    # Required columns.
    required = {"sample_file", "sample_name", "group"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(
            f"Metadata missing required column(s): {sorted(missing)}\n"
            f"  Found columns: {meta.columns.tolist()}"
        )

    # Unique sample names.
    dupes = meta["sample_name"][meta["sample_name"].duplicated()].tolist()
    if dupes:
        raise ValueError(f"Duplicate sample_name values in metadata: {dupes}")

    # Every metadata sample_file must have a matching area column.
    ft_files = set(areas.columns)
    unmatched = [f for f in meta["sample_file"] if f not in ft_files]
    if unmatched:
        raise ValueError(
            f"Metadata sample_file(s) not found in feature table: {unmatched}\n"
            f"  Feature table files: {sorted(ft_files)}"
        )

    # Feature table files not in metadata → warn and drop.
    extra_files = ft_files - set(meta["sample_file"])
    if extra_files:
        warnings.warn(
            f"Feature table has {len(extra_files)} sample(s) not in metadata; "
            f"they will be dropped: {sorted(extra_files)}",
            UserWarning,
            stacklevel=3,
        )

    meta = meta.set_index("sample_name")
    return meta


def _map_area_columns(
    areas: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    """Rename area DataFrame columns from raw filenames to sample names.

    Args:
        areas: Zone 2 DataFrame with raw ``.mzML`` filenames as columns.
        meta_df: Validated metadata indexed by ``sample_name``.

    Returns:
        Area DataFrame with ``sample_name`` columns, only for samples in metadata.
    """
    file_to_name = {row["sample_file"]: name for name, row in meta_df.iterrows()}
    # Keep only columns that appear in metadata (drop extras).
    keep_cols = [c for c in areas.columns if c in file_to_name]
    result = areas[keep_cols].rename(columns=file_to_name)
    return result


def _build_detection_mask(
    sample_details: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> pd.DataFrame | None:
    """Build a features × samples boolean mask of MZmine ``DETECTED`` peaks.

    Extracts the ``feature_state`` slice from Zone 3 sample_details and
    renames raw datafile filenames to ``sample_name`` using ``meta_df``.

    Args:
        sample_details: Zone 3 frame with ``(filename, metric)`` MultiIndex
            columns, as returned by ``parse_feature_table``.
        meta_df: Validated metadata indexed by ``sample_name``.

    Returns:
        Boolean DataFrame indexed by ``feature_id`` with ``sample_name``
        columns. ``None`` if ``feature_state`` is absent from the MZmine
        export (older versions / non-detection workflows).
    """
    if not isinstance(sample_details.columns, pd.MultiIndex):
        return None
    metrics = set(sample_details.columns.get_level_values("metric"))
    if "feature_state" not in metrics:
        return None

    state = sample_details.xs("feature_state", level="metric", axis=1)
    mask = state == "DETECTED"

    file_to_name = {row["sample_file"]: name for name, row in meta_df.iterrows()}
    keep_cols = [c for c in mask.columns if c in file_to_name]
    mask = mask[keep_cols].rename(columns=file_to_name)
    return mask
