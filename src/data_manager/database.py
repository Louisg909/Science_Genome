"""
Database Core.

Purpose:
    Encapsulate all database interactions (SQLite MVP).

Goals:
    - Define schema (papers, citations, embeddings).
    - Provide safe, reusable CRUD operations.
    - Abstract SQL away from business logic.

Inputs:
    - SQL queries or ORM-style calls.
    - Data dicts for papers, citations, embeddings.

Outputs:
    - Persistent science_papers.db file.
    - Python objects or rows returned to callers.
"""
