"""Corpus manifest builder utilities.

This module centralises metadata recording for dataset/corpus creation so that
retrieval provenance and citation/reference extraction quality are reproducible.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


MANIFEST_SCHEMA_VERSION = "1.0"


def _utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO 8601 string."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalise_source(source: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalise source metadata and enforce deterministic ordering."""

    endpoint = source.get("endpoint")
    query_terms = source.get("query_terms")
    date_window = source.get("date_window")
    retrieved_at = source.get("retrieved_at")

    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("Each source requires a non-empty 'endpoint' string.")
    if not isinstance(query_terms, Sequence) or isinstance(query_terms, (str, bytes)):
        raise ValueError("Each source requires 'query_terms' as a list of strings.")
    if not isinstance(date_window, Mapping):
        raise ValueError("Each source requires 'date_window' with 'start' and 'end'.")
    if not isinstance(retrieved_at, str) or not retrieved_at.strip():
        raise ValueError("Each source requires a non-empty 'retrieved_at' string.")

    start = date_window.get("start")
    end = date_window.get("end")
    if not isinstance(start, str) or not isinstance(end, str):
        raise ValueError("'date_window.start' and 'date_window.end' must be strings.")

    terms = [term.strip() for term in query_terms if isinstance(term, str) and term.strip()]
    if not terms:
        raise ValueError("Each source must include at least one non-empty query term.")

    return {
        "endpoint": endpoint.strip(),
        "query_terms": sorted(terms),
        "date_window": {"start": start, "end": end},
        "retrieved_at": retrieved_at,
    }


def _normalise_paper_identifier(identifier: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalise paper identifiers and citation extraction metadata."""

    paper_id = identifier.get("paper_id")
    doi = identifier.get("doi")
    arxiv_id = identifier.get("arxiv_id")
    confidence = identifier.get("citation_extraction_confidence")
    missingness = identifier.get("missingness_flags")

    if not isinstance(paper_id, str) or not paper_id.strip():
        raise ValueError("Each paper entry requires a non-empty 'paper_id'.")

    doi = doi.strip() if isinstance(doi, str) else None
    arxiv_id = arxiv_id.strip() if isinstance(arxiv_id, str) else None
    if not doi and not arxiv_id:
        raise ValueError("Each paper entry requires at least one of 'doi' or 'arxiv_id'.")

    if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("'citation_extraction_confidence' must be between 0.0 and 1.0.")

    if not isinstance(missingness, Mapping):
        raise ValueError("Each paper entry requires 'missingness_flags'.")

    citations_missing = missingness.get("citations_missing")
    references_missing = missingness.get("references_missing")
    if not isinstance(citations_missing, bool) or not isinstance(references_missing, bool):
        raise ValueError("Missingness flags must include boolean 'citations_missing' and 'references_missing'.")

    return {
        "paper_id": paper_id.strip(),
        "doi": doi,
        "arxiv_id": arxiv_id,
        "citation_extraction_confidence": float(confidence),
        "missingness_flags": {
            "citations_missing": citations_missing,
            "references_missing": references_missing,
        },
    }


def build_manifest(
    *,
    sources: Iterable[Mapping[str, Any]],
    papers: Iterable[Mapping[str, Any]],
    retrieval_timestamp: str | None = None,
) -> Dict[str, Any]:
    """Build a canonical corpus manifest dictionary.

    The output is deterministic for equivalent source/paper data regardless of
    input list ordering.
    """

    source_entries = [_normalise_source(source) for source in sources]
    paper_entries = [_normalise_paper_identifier(paper) for paper in papers]

    source_entries.sort(key=lambda entry: (entry["endpoint"], entry["retrieved_at"], entry["query_terms"]))
    paper_entries.sort(key=lambda entry: entry["paper_id"])

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "retrieval_timestamp": retrieval_timestamp or _utc_now_iso(),
        "sources": source_entries,
        "papers": paper_entries,
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate manifest structure and required fields.

    Raises:
        ValueError: If the manifest is malformed.
    """

    if not isinstance(manifest, Mapping):
        raise ValueError("Manifest must be a mapping.")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be '{MANIFEST_SCHEMA_VERSION}'.")

    retrieval_timestamp = manifest.get("retrieval_timestamp")
    if not isinstance(retrieval_timestamp, str) or not retrieval_timestamp.strip():
        raise ValueError("Manifest requires a non-empty 'retrieval_timestamp' string.")

    sources = manifest.get("sources")
    papers = manifest.get("papers")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Manifest requires non-empty 'sources'.")
    if not isinstance(papers, list) or not papers:
        raise ValueError("Manifest requires non-empty 'papers'.")

    for source in sources:
        _normalise_source(source)
    for paper in papers:
        _normalise_paper_identifier(paper)


def persist_manifest(corpus_path: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Persist a manifest JSON next to a corpus file.

    If ``corpus_path`` is ``assets/corpus.json``, the manifest path becomes
    ``assets/corpus.manifest.json``.
    """

    validate_manifest(manifest)

    corpus = Path(corpus_path)
    corpus.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = corpus.with_name(f"{corpus.stem}.manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def save_corpus_with_manifest(
    *,
    corpus_path: str | Path,
    corpus_payload: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    sources: Iterable[Mapping[str, Any]],
    papers: Iterable[Mapping[str, Any]],
    retrieval_timestamp: str | None = None,
) -> Path:
    """Save corpus JSON and an adjacent manifest JSON."""

    corpus = Path(corpus_path)
    corpus.parent.mkdir(parents=True, exist_ok=True)
    corpus.write_text(json.dumps(corpus_payload, indent=2, sort_keys=True), encoding="utf-8")

    manifest = build_manifest(
        sources=sources,
        papers=papers,
        retrieval_timestamp=retrieval_timestamp,
    )
    return persist_manifest(corpus, manifest)
