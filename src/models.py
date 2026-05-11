from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Paper:
    """Simple container for scraped paper data."""

    title: str
    abstract: str
    references: List[str]
    field_label: Optional[str] = None
