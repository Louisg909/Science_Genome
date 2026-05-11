"""Utilities for constructing and validating citation DAG edges."""

from __future__ import annotations

from typing import Iterable

from src.models import Paper


def _normalize_identifier(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    lower = text.lower()
    if "arxiv.org/abs/" in lower:
        text = text[lower.index("arxiv.org/abs/") + len("arxiv.org/abs/") :]
    elif lower.startswith("arxiv:"):
        text = text.split(":", 1)[1]
    return text.strip()


def build_citation_dag(papers: Iterable[Paper]) -> list[tuple[str, str]]:
    """Build directed citation edges (parent_id, child_id) among known papers."""

    paper_list = list(papers)
    known_ids = {_normalize_identifier(p.paper_id): p.paper_id for p in paper_list}

    edges: set[tuple[str, str]] = set()
    for child in paper_list:
        child_id = child.paper_id
        for reference in child.references:
            parent_norm = _normalize_identifier(reference)
            parent_id = known_ids.get(parent_norm)
            if parent_id is not None:
                edges.add((parent_id, child_id))
    return sorted(edges)


def enforce_acyclicity_by_time(
    edges: Iterable[tuple[str, str]], paper_year_map: dict[str, int | None]
) -> list[tuple[str, str]]:
    """Drop edges where a parent appears newer than its child by known years."""

    filtered: list[tuple[str, str]] = []
    for parent_id, child_id in edges:
        parent_year = paper_year_map.get(parent_id)
        child_year = paper_year_map.get(child_id)
        if parent_year is not None and child_year is not None and parent_year > child_year:
            continue
        filtered.append((parent_id, child_id))
    return filtered
