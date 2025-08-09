"""
Crossref Scraper.

Purpose:
    Retrieve metadata and citation pairs from the Crossref API by DOI or search.

Goals:
    - Complete citation links for DAG construction.
    - Supplement missing metadata from other sources.

Inputs:
    - DOI(s) or search terms.

Outputs:
    - Citation records (citing DOI → cited DOI).
    - Supplemental metadata dicts.
"""
