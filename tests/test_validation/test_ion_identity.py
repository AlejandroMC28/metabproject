"""Tests for metabproject.validation.ion_identity."""

from __future__ import annotations

import pandas as pd
import pytest

from metabproject.validation.ion_identity import (
    build_iin_validation_table,
    check_annotation_consistency,
    check_trend_consistency,
    parse_iin_groups,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ms1_with_iin() -> pd.DataFrame:
    """Synthetic ms1_features with IIN columns. Three IIN groups:
    - IIN 1: features 1 (M+H) and 2 (M+Na) — consistent
    - IIN 2: features 3 (M+H) and 4 (M+2Na) — inconsistent trend
    - IIN 3: features 5 (M+H) and 6 (M+K) — annotation inconsistent
    Feature 7 has no IIN (NaN iin_id).
    """
    return pd.DataFrame(
        {
            "mz": [150.1, 172.1, 200.2, 221.2, 300.3, 316.3, 400.4],
            "rt": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0],
            "ion_identities:iin_id": [1, 1, 2, 2, 3, 3, float("nan")],
            "ion_identities:partner_row_ids": [
                "2", "1", "4", "3", "6", "5", float("nan")
            ],
            "ion_identities:ion_identities": [
                "[M+H]+", "[M+Na]+",
                "[M+H]+", "[M+2Na]+",
                "[M+H]+", "[M+K]+",
                float("nan"),
            ],
            "ion_identities:neutral_mass": [
                149.1, 149.1, 199.2, 199.2, 299.3, 299.3, float("nan")
            ],
        },
        index=pd.Index([1, 2, 3, 4, 5, 6, 7], name="feature_id"),
    )


@pytest.fixture
def ms1_no_iin(ms1_with_iin) -> pd.DataFrame:
    """ms1_features without IIN columns."""
    return ms1_with_iin.drop(
        columns=[
            "ion_identities:iin_id",
            "ion_identities:partner_row_ids",
            "ion_identities:ion_identities",
            "ion_identities:neutral_mass",
        ]
    )


@pytest.fixture
def ms1_all_nan_iin(ms1_with_iin) -> pd.DataFrame:
    """ms1_features where all iin_id are NaN."""
    df = ms1_with_iin.copy()
    df["ion_identities:iin_id"] = float("nan")
    return df


@pytest.fixture
def iin_groups(ms1_with_iin) -> pd.DataFrame:
    return parse_iin_groups(ms1_with_iin)


@pytest.fixture
def contrast_consistent() -> dict[str, pd.DataFrame]:
    """IIN 1 (features 1,2) both up; IIN 2 (features 3,4) both down; IIN 3 both ns."""
    return {
        "A_vs_B": pd.DataFrame(
            {"direction": ["up", "up", "down", "down", "ns", "ns", "up"]},
            index=pd.Index([1, 2, 3, 4, 5, 6, 7], name="feature_id"),
        )
    }


@pytest.fixture
def contrast_inconsistent() -> dict[str, pd.DataFrame]:
    """IIN 2 (features 3,4) has up/down — inconsistent."""
    return {
        "A_vs_B": pd.DataFrame(
            {"direction": ["up", "up", "up", "down", "ns", "ns", "up"]},
            index=pd.Index([1, 2, 3, 4, 5, 6, 7], name="feature_id"),
        )
    }


@pytest.fixture
def merged_annotations_same_key() -> pd.DataFrame:
    """IIN 1 (features 1,2) and IIN 2 (3,4) share an InChIKey; IIN 3 disagrees."""
    return pd.DataFrame(
        {
            "inchikey2d": [
                "AAAAAAAAAAAAA", "AAAAAAAAAAAAA",  # IIN 1 — agree
                "BBBBBBBBBBBBB", "BBBBBBBBBBBBB",  # IIN 2 — agree
                "CCCCCCCCCCCCC", "DDDDDDDDDDDDD",  # IIN 3 — disagree
            ],
            "annotation_source": ["gnps", "gnps", "sirius", "sirius", "gnps", "sirius"],
        },
        index=pd.Index([1, 2, 3, 4, 5, 6], name="feature_id"),
    )


@pytest.fixture
def sirius_topk_overlap() -> pd.DataFrame:
    """Top-k SIRIUS data where IIN 3 features share a key at rank 2."""
    return pd.DataFrame(
        {
            "structurePerIdRank": [1, 2, 1, 2],
            "sirius_InChIkey2D": [
                "CCCCCCCCCCCCC", "EEEEEEEEEEEEE",  # feature 5: rank-1=C, rank-2=E
                "DDDDDDDDDDDDD", "EEEEEEEEEEEEE",  # feature 6: rank-1=D, rank-2=E
            ],
            "sirius_confidence": [0.8, 0.5, 0.75, 0.4],
        },
        index=pd.Index([5, 5, 6, 6], name="feature_id"),
    )


# ---------------------------------------------------------------------------
# parse_iin_groups
# ---------------------------------------------------------------------------

class TestParseIinGroups:
    def test_returns_only_iin_features(self, ms1_with_iin):
        df = parse_iin_groups(ms1_with_iin)
        assert set(df.index) == {1, 2, 3, 4, 5, 6}
        assert 7 not in df.index

    def test_iin_id_column(self, ms1_with_iin):
        df = parse_iin_groups(ms1_with_iin)
        assert df.loc[1, "iin_id"] == 1
        assert df.loc[3, "iin_id"] == 2
        assert df.loc[5, "iin_id"] == 3

    def test_partner_ids_parsed(self, ms1_with_iin):
        df = parse_iin_groups(ms1_with_iin)
        assert df.loc[1, "partner_ids"] == [2]
        assert df.loc[3, "partner_ids"] == [4]

    def test_ion_identity_column(self, ms1_with_iin):
        df = parse_iin_groups(ms1_with_iin)
        assert df.loc[1, "ion_identity"] == "[M+H]+"

    def test_missing_columns_raises(self, ms1_no_iin):
        with pytest.raises(ValueError, match="Ion identity network column not found"):
            parse_iin_groups(ms1_no_iin)

    def test_partner_ids_derived_from_iin_id(self, ms1_with_iin):
        # When partner_row_ids is absent, partner_ids should be derived from
        # iin_id grouping (every other feature in the same network).
        df_in = ms1_with_iin.drop(columns=["ion_identities:partner_row_ids"])
        df = parse_iin_groups(df_in)
        ref = parse_iin_groups(ms1_with_iin)
        for fid in df.index:
            assert sorted(df.loc[fid, "partner_ids"]) == sorted(ref.loc[fid, "partner_ids"])

    def test_all_nan_returns_empty(self, ms1_all_nan_iin):
        df = parse_iin_groups(ms1_all_nan_iin)
        assert df.empty

    def test_index_name(self, ms1_with_iin):
        df = parse_iin_groups(ms1_with_iin)
        assert df.index.name == "feature_id"


# ---------------------------------------------------------------------------
# check_trend_consistency
# ---------------------------------------------------------------------------

class TestCheckTrendConsistency:
    def test_consistent_groups(self, iin_groups, contrast_consistent):
        result = check_trend_consistency(iin_groups, contrast_consistent)
        assert result is not None
        row_iin1 = result[(result["iin_id"] == 1) & (result["contrast"] == "A_vs_B")].iloc[0]
        assert row_iin1["trend_consistent"] == True  # noqa: E712
        assert row_iin1["n_up"] == 2

    def test_inconsistent_group(self, iin_groups, contrast_inconsistent):
        result = check_trend_consistency(iin_groups, contrast_inconsistent)
        row_iin2 = result[(result["iin_id"] == 2) & (result["contrast"] == "A_vs_B")].iloc[0]
        assert row_iin2["trend_consistent"] == False  # noqa: E712

    def test_all_ns_is_consistent(self, iin_groups, contrast_consistent):
        result = check_trend_consistency(iin_groups, contrast_consistent)
        row_iin3 = result[(result["iin_id"] == 3) & (result["contrast"] == "A_vs_B")].iloc[0]
        assert row_iin3["trend_consistent"] == True  # noqa: E712
        assert row_iin3["n_ns"] == 2

    def test_missing_features_excluded(self, iin_groups):
        # Stats only has feature 1, feature 2 missing → untested, not inconsistent.
        partial_stats = {
            "A_vs_B": pd.DataFrame(
                {"direction": ["up"]},
                index=pd.Index([1], name="feature_id"),
            )
        }
        result = check_trend_consistency(iin_groups, partial_stats)
        row = result[(result["iin_id"] == 1)].iloc[0]
        assert row["n_tested"] == 1
        assert row["trend_consistent"] == True  # noqa: E712

    def test_empty_contrast_returns_empty(self, iin_groups):
        result = check_trend_consistency(iin_groups, {})
        assert result.empty

    def test_multiple_contrasts(self, iin_groups):
        contrasts = {
            "A_vs_B": pd.DataFrame(
                {"direction": ["up"] * 6 + ["up"]},
                index=pd.Index([1, 2, 3, 4, 5, 6, 7], name="feature_id"),
            ),
            "A_vs_C": pd.DataFrame(
                {"direction": ["down"] * 6 + ["down"]},
                index=pd.Index([1, 2, 3, 4, 5, 6, 7], name="feature_id"),
            ),
        }
        result = check_trend_consistency(iin_groups, contrasts)
        assert set(result["contrast"].unique()) == {"A_vs_B", "A_vs_C"}


# ---------------------------------------------------------------------------
# check_annotation_consistency
# ---------------------------------------------------------------------------

class TestCheckAnnotationConsistency:
    def test_consistent_same_key(self, iin_groups, merged_annotations_same_key):
        result = check_annotation_consistency(iin_groups, merged_annotations_same_key)
        row_iin1 = result[result["iin_id"] == 1].iloc[0]
        assert row_iin1["annotation_consistent"] == True  # noqa: E712
        assert row_iin1["shared_inchikey_rank1"] == "AAAAAAAAAAAAA"

    def test_inconsistent_different_keys(self, iin_groups, merged_annotations_same_key):
        result = check_annotation_consistency(iin_groups, merged_annotations_same_key)
        row_iin3 = result[result["iin_id"] == 3].iloc[0]
        assert row_iin3["annotation_consistent"] == False  # noqa: E712
        assert row_iin3["shared_inchikey_rank1"] is None or pd.isna(row_iin3["shared_inchikey_rank1"])

    def test_topk_rescue_inconsistent(
        self, iin_groups, merged_annotations_same_key, sirius_topk_overlap
    ):
        result = check_annotation_consistency(
            iin_groups, merged_annotations_same_key, sirius_topk=sirius_topk_overlap
        )
        row_iin3 = result[result["iin_id"] == 3].iloc[0]
        # Rank-1 disagree but rank-2 share EEEEEEEEEEEEE
        assert row_iin3["annotation_consistent"] == True  # noqa: E712
        assert row_iin3["shared_inchikey_topk"] == "EEEEEEEEEEEEE"

    def test_canopus_only_excluded(self, iin_groups):
        # Features present in merged_annotations but with null inchikey2d
        # (CANOPUS-only) should not count as HC-annotated.
        canopus_only_ann = pd.DataFrame(
            {
                "inchikey2d": [None, None, None, None, None, None],
                "annotation_source": ["canopus"] * 6,
            },
            index=pd.Index([1, 2, 3, 4, 5, 6], name="feature_id"),
        )
        result = check_annotation_consistency(iin_groups, canopus_only_ann)
        for _, row in result.iterrows():
            assert row["n_hc_annotated"] == 0
            assert row["annotation_consistent"] is None

    def test_insufficient_annotation_is_na(self, iin_groups):
        # No annotations at all → n_hc_annotated < 2 → NA.
        empty_ann = pd.DataFrame(
            columns=["inchikey2d", "annotation_source"],
            index=pd.Index([], name="feature_id"),
        )
        result = check_annotation_consistency(iin_groups, empty_ann)
        for _, row in result.iterrows():
            assert row["annotation_consistent"] is None


# ---------------------------------------------------------------------------
# build_iin_validation_table
# ---------------------------------------------------------------------------

class TestBuildIinValidationTable:
    def test_one_row_per_iin(
        self, iin_groups, ms1_with_iin, contrast_consistent, merged_annotations_same_key
    ):
        trend = check_trend_consistency(iin_groups, contrast_consistent)
        ann = check_annotation_consistency(iin_groups, merged_annotations_same_key)
        table = build_iin_validation_table(iin_groups, ms1_with_iin, trend, ann)
        assert len(table) == 3
        assert set(table.index) == {1, 2, 3}

    def test_trend_columns_present(
        self, iin_groups, ms1_with_iin, contrast_consistent, merged_annotations_same_key
    ):
        trend = check_trend_consistency(iin_groups, contrast_consistent)
        ann = check_annotation_consistency(iin_groups, merged_annotations_same_key)
        table = build_iin_validation_table(iin_groups, ms1_with_iin, trend, ann)
        assert "trend_A_vs_B_consistent" in table.columns

    def test_all_consistent_column(
        self, iin_groups, ms1_with_iin, contrast_consistent, merged_annotations_same_key
    ):
        trend = check_trend_consistency(iin_groups, contrast_consistent)
        ann = check_annotation_consistency(iin_groups, merged_annotations_same_key)
        table = build_iin_validation_table(iin_groups, ms1_with_iin, trend, ann)
        assert "all_consistent" in table.columns
        # IIN 1: trend consistent, annotation consistent → all_consistent True
        assert table.loc[1, "all_consistent"] == True  # noqa: E712 or table.loc[1, "all_consistent"] == True

    def test_feature_ids_column(
        self, iin_groups, ms1_with_iin, contrast_consistent, merged_annotations_same_key
    ):
        trend = check_trend_consistency(iin_groups, contrast_consistent)
        ann = check_annotation_consistency(iin_groups, merged_annotations_same_key)
        table = build_iin_validation_table(iin_groups, ms1_with_iin, trend, ann)
        assert "feature_ids" in table.columns
        assert "1" in str(table.loc[1, "feature_ids"])

    def test_neutral_mass_present(
        self, iin_groups, ms1_with_iin, contrast_consistent, merged_annotations_same_key
    ):
        trend = check_trend_consistency(iin_groups, contrast_consistent)
        ann = check_annotation_consistency(iin_groups, merged_annotations_same_key)
        table = build_iin_validation_table(iin_groups, ms1_with_iin, trend, ann)
        assert "neutral_mass" in table.columns
        assert abs(table.loc[1, "neutral_mass"] - 149.1) < 1e-6


# ---------------------------------------------------------------------------
# load_sirius_topk
# ---------------------------------------------------------------------------

class TestLoadSiriusTopk:
    _TSV = (
        "mappingFeatureId\tstructurePerIdRank\tConfidenceScoreApproximate"
        "\tInChIkey2D\tInChI\tname\tsmiles\tmolecularFormula\tadduct\n"
        "1\t1\t0.80\tAAAAAAAAAAAAAAA\tInChI=1/A\tComp A\tCC\tC2H6\t[M+H]+\n"
        "1\t2\t0.50\tBBBBBBBBBBBBBBB\tInChI=1/B\tComp B\tCCC\tC3H8\t[M+H]+\n"
        "1\t3\t0.30\tCCCCCCCCCCCCCCC\tInChI=1/C\tComp C\tO\tH2O\t[M+H]+\n"
        "2\t1\t0.70\tDDDDDDDDDDDDDDD\tInChI=1/D\tComp D\tCC=O\tC2H4O\t[M+Na]+\n"
    )

    @pytest.fixture
    def structures_file(self, tmp_path):
        p = tmp_path / "structure_identifications.tsv"
        p.write_text(self._TSV)
        return p

    def test_top2_keeps_two_rows_for_feature1(self, structures_file):
        from metabproject.annotation.sirius import load_sirius_topk
        df = load_sirius_topk(structures_file, top_k=2)
        assert len(df.loc[[1]]) == 2

    def test_top1_equals_rank1_only(self, structures_file):
        from metabproject.annotation.sirius import load_sirius, load_sirius_topk
        topk = load_sirius_topk(structures_file, top_k=1)
        rank1 = load_sirius(structures_file)
        # Same features present.
        assert set(topk.index.unique()) == set(rank1.index)

    def test_rank_column_retained(self, structures_file):
        from metabproject.annotation.sirius import load_sirius_topk
        df = load_sirius_topk(structures_file, top_k=3)
        assert "structurePerIdRank" in df.columns
        assert set(df.loc[1, "structurePerIdRank"].tolist()) == {1, 2, 3}

    def test_non_unique_index(self, structures_file):
        from metabproject.annotation.sirius import load_sirius_topk
        df = load_sirius_topk(structures_file, top_k=3)
        assert not df.index.is_unique
