import types

from src.scraping import fetch_arxiv_papers, parse_arxiv_feed


SAMPLE_FEED = """
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>First Paper</title>
    <summary>Abstract one.</summary>
    <reference>Ref A</reference>
    <reference>Ref B</reference>
  </entry>
  <entry>
    <title>Second Paper</title>
    <summary>Abstract two.</summary>
  </entry>
</feed>
"""


def test_parse_arxiv_feed_extracts_fields():
    papers = parse_arxiv_feed(SAMPLE_FEED)
    assert len(papers) == 2
    assert papers[0].title == "First Paper"
    assert papers[0].abstract == "Abstract one."
    assert papers[0].references == ["Ref A", "Ref B"]


def test_fetch_arxiv_papers_uses_session(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        response = types.SimpleNamespace()
        response.text = SAMPLE_FEED
        response.status_code = 200

        def raise_for_status():
            return None

        response.raise_for_status = raise_for_status
        return response

    session = types.SimpleNamespace(get=fake_get)
    papers = fetch_arxiv_papers("quantum", max_results=2, session=session)
    assert [p.title for p in papers] == ["First Paper", "Second Paper"]
