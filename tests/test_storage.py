from pathlib import Path

from src.models import Paper
from src.storage import load_papers, save_papers


def test_save_and_load_roundtrip(tmp_path: Path):
    papers = [
        Paper(title="Sample", abstract="An example paper", references=["Ref1"]),
        Paper(title="Another", abstract="Second paper", references=[]),
    ]

    path = tmp_path / "papers.json"
    save_papers(path, papers)

    loaded = load_papers(path)
    assert loaded == papers
