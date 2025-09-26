from __future__ import annotations

from typing import Optional

import feedparser  # type: ignore[import-untyped]
import httpx

from .http_client import create_http_client
from .types import Paper

ARXIV_API_URL = "https://export.arxiv.org/api/query"


def query_arxiv(
    topic: str,
    *,
    max_results: int = 200,
    client: Optional[httpx.Client] = None,
) -> list[Paper]:
    """Query the arXiv API for the given topic and return normalized papers."""

    if not topic.strip():
        raise ValueError("topic must be a non-empty string")

    params: dict[str, str] = {
        "search_query": f"all:{topic}",
        "start": "0",
        "max_results": str(max_results),
    }
    http_client = client or create_http_client()
    close_client = client is None

    try:
        response = http_client.get(ARXIV_API_URL, params=params)
        response.raise_for_status()
    finally:
        if close_client:
            http_client.close()

    feed = feedparser.parse(response.text)
    papers: list[Paper] = []
    for entry in feed.entries:
        entry_id = entry.get("id", "")
        title = entry.get("title")
        summary = entry.get("summary")
        authors = [author.get("name", "") for author in entry.get("authors", [])]
        published = entry.get("published") or entry.get("updated")
        doi = entry.get("arxiv_doi") or entry.get("doi")
        url = entry_id
        for link in entry.get("links", []):
            if link.get("rel") == "alternate" and link.get("href"):
                url = link["href"]
                break

        paper = Paper.from_parts(
            id=entry_id or url,
            title=title,
            abstract=summary,
            authors=authors,
            year=published,
            venue=entry.get("arxiv_journal_ref"),
            doi=doi,
            url=url,
            source="arxiv",
            score=1.0,
        )
        papers.append(paper)
    return papers
