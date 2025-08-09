"""
Scraping Parsers.

Purpose:
    Normalize and validate raw API responses from arXiv/Crossref.

Goals:
    - Clean text (remove LaTeX, whitespace).
    - Deduplicate and validate DOIs.
    - Conform to database schema.

Inputs:
    - Raw JSON/XML from APIs.

Outputs:
    - Sanitized metadata dicts for storage.
"""
