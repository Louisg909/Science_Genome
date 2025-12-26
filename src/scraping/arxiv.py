"""Minimal ArXiv scraper that extracts titles, abstracts, and references."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import List, Optional

import requests

from src.models import Paper


ARXIV_API_URL = "https://export.arxiv.org/api/query"


def _extract_text(element: Optional[ET.Element]) -> str:
    return (element.text or "").strip() if element is not None else ""


def parse_arxiv_feed(feed_text: str) -> List[Paper]:
    """Parse a small ArXiv Atom feed into Paper objects.

    Only keeps the title, summary/abstract, and any child ``reference`` nodes.
    """

    root = ET.fromstring(feed_text)
    papers: List[Paper] = []
    for entry in root.findall(".//{*}entry"):
        title = _extract_text(entry.find("{*}title"))
        abstract = _extract_text(entry.find("{*}summary"))
        refs = [
            _extract_text(ref)
            for ref in entry.findall("{*}reference")
            if _extract_text(ref)
        ]
        papers.append(Paper(title=title, abstract=abstract, references=refs))
    return papers


def fetch_arxiv_papers(query: str, max_results: int = 10, session: Optional[requests.Session] = None) -> List[Paper]:
    """Fetch a small set of papers from ArXiv.

    The call is intentionally lightweight so it can be mocked in tests. The
    returned value is a list of :class:`Paper` instances.
    """

    client = session or requests.Session()
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
    }
    response = client.get(ARXIV_API_URL, params=params, timeout=10)
    response.raise_for_status()
    return parse_arxiv_feed(response.text)
