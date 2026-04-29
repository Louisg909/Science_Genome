import json
from pathlib import Path

import pytest

from src.data.corpus_builder import (
    MANIFEST_SCHEMA_VERSION,
    build_manifest,
    persist_manifest,
    validate_manifest,
)


def _sample_sources():
    return [
        {
            "endpoint": "https://api.semanticscholar.org/graph/v1/paper/search",
            "query_terms": ["genomics", "gene regulation"],
            "date_window": {"start": "2025-01-01", "end": "2025-12-31"},
            "retrieved_at": "2026-04-20T10:00:00+00:00",
        },
        {
            "endpoint": "https://export.arxiv.org/api/query",
            "query_terms": ["genomics"],
            "date_window": {"start": "2025-06-01", "end": "2025-12-31"},
            "retrieved_at": "2026-04-21T09:30:00+00:00",
        },
    ]


def _sample_papers():
    return [
        {
            "paper_id": "s2:abc123",
            "doi": "10.1000/example-doi-1",
            "arxiv_id": None,
            "citation_extraction_confidence": 0.93,
            "missingness_flags": {"citations_missing": False, "references_missing": False},
        },
        {
            "paper_id": "arxiv:2501.01234",
            "doi": None,
            "arxiv_id": "2501.01234",
            "citation_extraction_confidence": 0.67,
            "missingness_flags": {"citations_missing": True, "references_missing": False},
        },
    ]


def test_build_manifest_is_deterministic():
    sources = list(reversed(_sample_sources()))
    papers = list(reversed(_sample_papers()))

    manifest_a = build_manifest(
        sources=sources,
        papers=papers,
        retrieval_timestamp="2026-04-26T00:00:00+00:00",
    )
    manifest_b = build_manifest(
        sources=_sample_sources(),
        papers=_sample_papers(),
        retrieval_timestamp="2026-04-26T00:00:00+00:00",
    )

    assert manifest_a == manifest_b
    assert manifest_a["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert [paper["paper_id"] for paper in manifest_a["papers"]] == [
        "arxiv:2501.01234",
        "s2:abc123",
    ]


def test_manifest_schema_validation_rejects_missing_identifiers():
    invalid_manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "retrieval_timestamp": "2026-04-26T00:00:00+00:00",
        "sources": _sample_sources(),
        "papers": [
            {
                "paper_id": "bad-paper",
                "doi": None,
                "arxiv_id": None,
                "citation_extraction_confidence": 0.5,
                "missingness_flags": {"citations_missing": False, "references_missing": True},
            }
        ],
    }

    with pytest.raises(ValueError, match="doi"):
        validate_manifest(invalid_manifest)


def test_persist_manifest_writes_adjacent_json(tmp_path: Path):
    manifest = build_manifest(
        sources=_sample_sources(),
        papers=_sample_papers(),
        retrieval_timestamp="2026-04-26T00:00:00+00:00",
    )

    corpus_path = tmp_path / "assets" / "corpus.json"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text("[]", encoding="utf-8")

    manifest_path = persist_manifest(corpus_path, manifest)

    assert manifest_path == corpus_path.parent / "corpus.manifest.json"
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted == manifest
