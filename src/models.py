from dataclasses import dataclass
from typing import List


@dataclass
class Paper:
    """Simple container for scraped paper data."""

    title: str
    abstract: str
    references: List[str]
