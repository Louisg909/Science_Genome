"""Lightweight persistence for scraped papers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from src.models import Paper


def save_papers(path: str | Path, papers: Iterable[Paper]) -> None:
    """Write papers to disk as JSON."""

    data = [paper.__dict__ for paper in papers]
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_papers(path: str | Path) -> List[Paper]:
    """Load papers from a JSON file created by :func:`save_papers`."""

    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Paper(**entry) for entry in entries]
