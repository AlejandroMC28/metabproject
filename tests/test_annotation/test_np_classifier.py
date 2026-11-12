"""Tests for metabproject.annotation.np_classifier (mocked HTTP)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pandas as pd
import pytest

from metabproject.annotation import np_classifier
from metabproject.annotation.np_classifier import (
    CACHE_COLUMNS,
    classify_features,
    classify_smiles,
    load_cache,
    save_cache,
)


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code=200, payload=None, raise_exc=None):
        self.status_code = status_code
        self._payload = payload
        self._raise_exc = raise_exc

    def raise_for_status(self):
        if self._raise_exc is not None:
            raise self._raise_exc
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "boom", request=None, response=None
            )

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _FakeClient:
    """Replaces httpx.Client. Returns canned responses keyed by SMILES."""

    def __init__(self, responses: dict[str, _FakeResponse], **kwargs):
        self._responses = responses
        self.calls: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def get(self, url, params=None, timeout=None):
        smi = params["smiles"] if params else ""
        self.calls.append(smi)
        if smi not in self._responses:
            return _FakeResponse(
                raise_exc=httpx.RequestError("unexpected smiles in fake")
            )
        return self._responses[smi]


# --------------------------------------------------------------------------
# classify_smiles
# --------------------------------------------------------------------------

def test_classify_smiles_success(monkeypatch):
    payload = {
        "pathway_results": ["Alkaloids"],
        "superclass_results": ["Piperidine alkaloids"],
        "class_results": ["Piperidines"],
    }

    def fake_get(url, params=None, timeout=None):
        return _FakeResponse(payload=payload)

    monkeypatch.setattr(np_classifier.httpx, "get", fake_get)
    result = classify_smiles("CCN1CCCCC1")
    assert result == {
        "npc_pathway": "Alkaloids",
        "npc_superclass": "Piperidine alkaloids",
        "npc_class": "Piperidines",
    }


def test_classify_smiles_empty_lists_returns_nones(monkeypatch):
    payload = {
        "pathway_results": [],
        "superclass_results": [None],
        "class_results": ["", "Real class"],
    }
    monkeypatch.setattr(
        np_classifier.httpx, "get",
        lambda *a, **k: _FakeResponse(payload=payload),
    )
    result = classify_smiles("C")
    assert result["npc_pathway"] is None
    assert result["npc_superclass"] is None
    assert result["npc_class"] == "Real class"


def test_classify_smiles_http_error_returns_none(monkeypatch):
    monkeypatch.setattr(
        np_classifier.httpx, "get",
        lambda *a, **k: _FakeResponse(
            raise_exc=httpx.ConnectError("dead")
        ),
    )
    assert classify_smiles("CCO") is None


def test_classify_smiles_bad_json_returns_none(monkeypatch):
    monkeypatch.setattr(
        np_classifier.httpx, "get",
        lambda *a, **k: _FakeResponse(payload=None),
    )
    assert classify_smiles("CCO") is None


def test_classify_smiles_empty_input():
    assert classify_smiles("") is None
    assert classify_smiles("   ") is None


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def test_load_cache_missing_returns_empty(tmp_path):
    df = load_cache(tmp_path / "no_such.parquet")
    assert df.empty
    assert df.index.name == "InChIkey2D"


def test_round_trip_cache(tmp_path):
    df = pd.DataFrame(
        {
            "smiles": ["CCO"],
            "npc_pathway": ["Other"],
            "npc_superclass": ["Alcohols"],
            "npc_class": ["Ethanols"],
            "queried_at": ["2026-06-10T00:00:00+00:00"],
        },
        index=pd.Index(["KEYAAA"], name="InChIkey2D"),
    )
    path = tmp_path / "cache.parquet"
    save_cache(df, path)
    loaded = load_cache(path)
    assert loaded.index.name == "InChIkey2D"
    assert loaded.loc["KEYAAA", "npc_class"] == "Ethanols"


# --------------------------------------------------------------------------
# classify_features
# --------------------------------------------------------------------------

def test_classify_features_with_cache_hit(monkeypatch, tmp_path):
    cache_path = tmp_path / "cache.parquet"
    existing = pd.DataFrame(
        {
            "smiles": ["CCO"],
            "npc_pathway": ["P"],
            "npc_superclass": ["S"],
            "npc_class": ["C"],
            "queried_at": ["2026-06-10T00:00:00+00:00"],
        },
        index=pd.Index(["KEYAAA"], name="InChIkey2D"),
    )
    save_cache(existing, cache_path)

    keys = pd.DataFrame(
        {"InChIkey2D": ["KEYAAA"], "smiles": ["CCO"]},
        index=pd.Index([1], name="feature_id"),
    )

    def boom(*a, **k):
        raise AssertionError("no HTTP call expected when cache hit")
    monkeypatch.setattr(np_classifier.httpx, "Client", boom)

    out = classify_features(keys, cache_path=cache_path, request_delay=0, verbose=False)
    assert out.loc[1, "sirius_struct_npc_class"] == "C"
    assert out.loc[1, "sirius_struct_npc_pathway"] == "P"


def test_classify_features_cache_miss_calls_api(monkeypatch, tmp_path):
    cache_path = tmp_path / "cache.parquet"
    responses = {
        "CCO": _FakeResponse(payload={
            "pathway_results": ["P"],
            "superclass_results": ["S"],
            "class_results": ["C"],
        }),
    }
    fake = _FakeClient(responses)
    monkeypatch.setattr(np_classifier.httpx, "Client", lambda **kw: fake)

    keys = pd.DataFrame(
        {"InChIkey2D": ["KEYAAA"], "smiles": ["CCO"]},
        index=pd.Index([5], name="feature_id"),
    )
    out = classify_features(keys, cache_path=cache_path, request_delay=0, verbose=False)
    assert fake.calls == ["CCO"]
    assert out.loc[5, "sirius_struct_npc_class"] == "C"
    # Cache was written.
    cached = load_cache(cache_path)
    assert "KEYAAA" in cached.index


def test_classify_features_dedupes_by_inchikey(monkeypatch, tmp_path):
    cache_path = tmp_path / "cache.parquet"
    responses = {
        "CCO": _FakeResponse(payload={
            "pathway_results": ["P"],
            "superclass_results": ["S"],
            "class_results": ["C"],
        }),
    }
    fake = _FakeClient(responses)
    monkeypatch.setattr(np_classifier.httpx, "Client", lambda **kw: fake)

    keys = pd.DataFrame(
        {
            "InChIkey2D": ["KEYAAA", "KEYAAA", "KEYAAA"],
            "smiles": ["CCO", "CCO", "CCO"],
        },
        index=pd.Index([1, 2, 3], name="feature_id"),
    )
    out = classify_features(keys, cache_path=cache_path, request_delay=0, verbose=False)
    # Three feature rows but only one HTTP call.
    assert fake.calls == ["CCO"]
    assert len(out) == 3
    assert (out["sirius_struct_npc_class"] == "C").all()


def test_classify_features_api_failure_caches_nulls(monkeypatch, tmp_path):
    cache_path = tmp_path / "cache.parquet"
    responses = {
        "BAD": _FakeResponse(raise_exc=httpx.ConnectError("dead")),
    }
    fake = _FakeClient(responses)
    monkeypatch.setattr(np_classifier.httpx, "Client", lambda **kw: fake)

    keys = pd.DataFrame(
        {"InChIkey2D": ["KEYBAD"], "smiles": ["BAD"]},
        index=pd.Index([9], name="feature_id"),
    )
    out = classify_features(keys, cache_path=cache_path, request_delay=0, verbose=False)
    assert pd.isna(out.loc[9, "sirius_struct_npc_class"])
    # Failure still cached, so re-runs don't hammer the API.
    cached = load_cache(cache_path)
    assert "KEYBAD" in cached.index


def test_classify_features_skips_null_inchikey(monkeypatch, tmp_path):
    monkeypatch.setattr(
        np_classifier.httpx, "Client",
        lambda **kw: (_ for _ in ()).throw(AssertionError("no http expected")),
    )
    keys = pd.DataFrame(
        {"InChIkey2D": [None], "smiles": [None]},
        index=pd.Index([1], name="feature_id"),
    )
    out = classify_features(keys, cache_path=None, request_delay=0, verbose=False)
    assert out.empty


# --------------------------------------------------------------------------
# Merge integration: sirius_struct_npc fills above CANOPUS, below GNPS
# --------------------------------------------------------------------------

def test_merge_priority_struct_above_canopus():
    from metabproject.annotation.merge import merge_annotations

    sirius_df = pd.DataFrame({
        "sirius_compound_name": ["X"],
        "sirius_InChIkey2D": ["KEYAAA"],
        "sirius_mol_formula": ["C"],
        "sirius_adduct": ["[M+H]+"],
        "sirius_confidence": [0.9],
        "sirius_high_confidence": [True],
        "sirius_InChI": [None],
        "sirius_smiles": ["CCO"],
    }, index=pd.Index([1], name="feature_id"))

    canopus_df = pd.DataFrame({
        "canopus_npc_pathway": ["WrongPath"],
        "canopus_npc_pathway_probability": [0.9],
        "canopus_npc_pathway_passed": [True],
        "canopus_npc_superclass": ["WrongSuper"],
        "canopus_npc_superclass_probability": [0.9],
        "canopus_npc_superclass_passed": [True],
        "canopus_npc_class": ["WrongClass"],
        "canopus_npc_class_probability": [0.9],
        "canopus_npc_class_passed": [True],
        "canopus_classyfire_class": ["CF"],
        "canopus_classyfire_class_probability": [0.9],
        "canopus_classyfire_class_passed": [True],
    }, index=pd.Index([1], name="feature_id"))

    gnps_df = pd.DataFrame(
        {"gnps_compound_name": [], "gnps_InChIkey2D": []},
        index=pd.Index([], name="feature_id"),
    )

    struct_df = pd.DataFrame({
        "sirius_struct_npc_pathway": ["RightPath"],
        "sirius_struct_npc_superclass": ["RightSuper"],
        "sirius_struct_npc_class": ["RightClass"],
    }, index=pd.Index([1], name="feature_id"))

    merged = merge_annotations(
        sirius_df, canopus_df, gnps_df, suspect_df=None,
        sirius_struct_npc_df=struct_df,
    )
    assert merged.loc[1, "npc_pathway"] == "RightPath"
    assert merged.loc[1, "npc_superclass"] == "RightSuper"
    assert merged.loc[1, "npc_class"] == "RightClass"
    # ClassyFire still from CANOPUS (no structure-derived source).
    assert merged.loc[1, "classyfire_class"] == "CF"
    assert "sirius_struct_npc" in merged.loc[1, "annotation_source"]


def test_merge_priority_gnps_still_wins_over_struct():
    from metabproject.annotation.merge import merge_annotations

    sirius_df = pd.DataFrame({
        "sirius_compound_name": ["X"],
        "sirius_InChIkey2D": ["KEYAAA"],
        "sirius_mol_formula": ["C"],
        "sirius_adduct": ["[M+H]+"],
        "sirius_confidence": [0.9],
        "sirius_high_confidence": [True],
        "sirius_InChI": [None],
        "sirius_smiles": ["CCO"],
    }, index=pd.Index([1], name="feature_id"))

    canopus_df = pd.DataFrame(index=pd.Index([], name="feature_id"))

    gnps_df = pd.DataFrame({
        "gnps_compound_name": ["X"],
        "gnps_InChIkey2D": ["KEYAAA"],
        "gnps_MQScore": [0.95],
        "gnps_smiles": ["CCO"],
        "gnps_InChI": [None],
        "gnps_npc_class": ["GNPS_class"],
    }, index=pd.Index([1], name="feature_id"))

    struct_df = pd.DataFrame({
        "sirius_struct_npc_class": ["Struct_class"],
    }, index=pd.Index([1], name="feature_id"))

    merged = merge_annotations(
        sirius_df, canopus_df, gnps_df, suspect_df=None,
        sirius_struct_npc_df=struct_df,
    )
    assert merged.loc[1, "npc_class"] == "GNPS_class"


def test_merge_priority_struct_only_when_sirius_high():
    """Low-conf SIRIUS rank-1 must NOT contribute structure-derived NPC."""
    from metabproject.annotation.merge import merge_annotations

    sirius_df = pd.DataFrame({
        "sirius_compound_name": ["X"],
        "sirius_InChIkey2D": ["KEYAAA"],
        "sirius_mol_formula": ["C"],
        "sirius_adduct": ["[M+H]+"],
        "sirius_confidence": [0.4],
        "sirius_high_confidence": [False],
        "sirius_InChI": [None],
        "sirius_smiles": ["CCO"],
    }, index=pd.Index([1], name="feature_id"))

    canopus_df = pd.DataFrame({
        "canopus_npc_class": ["CanopusClass"],
        "canopus_npc_class_probability": [0.9],
        "canopus_npc_class_passed": [True],
    }, index=pd.Index([1], name="feature_id"))

    gnps_df = pd.DataFrame(
        columns=["gnps_compound_name", "gnps_InChIkey2D"],
        index=pd.Index([], name="feature_id"),
    )

    struct_df = pd.DataFrame({
        "sirius_struct_npc_class": ["StructClass"],
    }, index=pd.Index([1], name="feature_id"))

    merged = merge_annotations(
        sirius_df, canopus_df, gnps_df, suspect_df=None,
        sirius_struct_npc_df=struct_df,
    )
    # CANOPUS wins because sirius is low-confidence.
    assert merged.loc[1, "npc_class"] == "CanopusClass"
