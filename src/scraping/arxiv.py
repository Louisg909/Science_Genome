"""
arXiv Scraper.

Purpose:
    Fetch paper metadata (title, abstract, authors, DOI, categories) via the arXiv API.

Goals:
    - Handle pagination and rate limits.
    - Return clean, structured dicts for each paper.

Inputs:
    - Query strings (keywords, categories, date ranges).
    - Optional max-results parameter.

Outputs:
    - List of metadata dicts ready for database insertion.
"""
