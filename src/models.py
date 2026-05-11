from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Paper:
    """Simple container for scraped paper data."""

    paper_id: str
    title: str
    abstract: str
    year: int | None
    references: List[str]
    field_label: Optional[str] = None
