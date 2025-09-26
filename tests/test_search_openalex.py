"""Tests for the OpenAlex search pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
import json
from pathlib import Path

import httpx
import pytest

from la_pkg.search.openalex import (
    SearchConfig,
    SearchMetrics,
    append_audit_log,
    query_openalex,
)


MOCK_WORKS = [
    {
        "id": "https://openalex.org/W1111",
        "display_name": "Genomic screening for cancer",
        "authorships": [
            {"author": {"display_name": "Alice Smith"}},
            {"author": {"display_name": "Carol Doe"}},
        ],
        "abstract_inverted_index": {
            "Genomic": [0],
            "screening": [1],
            "for": [2],
            "cancer": [3],
        },
        "publication_year": 2024,
        "primary_location": {
            "source": {"display_name": "Journal of Screening"},
            "landing_page_url": "https://doi.org/10.123/abc",
        },
        "open_access": {"oa_url": "https://oa.example/abc"},
        "doi": "10.123/abc",
        "relevance_score": 55.5,
    },
    {
        "id": "https://openalex.org/W2222",
        "display_name": "Genomic screening for cancer",
        "authorships": [{"author": {"display_name": "Bob Jones"}}],
        "abstract_inverted_index": {
            "Duplicate": [0],
            "entry": [1],
        },
        "publication_year": 2024,
        "primary_location": {
            "source": {"display_name": "Another Journal"},
            "landing_page_url": "https://example.org/duplicate",
        },
        "open_access": {},
        "doi": "10.123/abc",
        "relevance_score": 50.0,
    },
    {
        "id": "https://openalex.org/W3333",
        "display_name": "Genomic screening for cancer",
        "authorships": [{"author": {"display_name": "Eve Klein"}}],
        "abstract_inverted_index": {
            "Same": [0],
            "title": [1],
        },
        "publication_year": 2023,
        "primary_location": {
            "source": {"display_name": "Mirror Journal"},
            "landing_page_url": "https://example.org/similar",
        },
        "open_access": {},
        "doi": None,
        "relevance_score": 47.0,
    },
    {
        "id": "https://openalex.org/W4444",
        "display_name": "Efficient pipelines for genomic analysis",
        "authorships": [{"author": {"display_name": "Dana Lee"}}],
        "summary": "Overview of pipelines.",
        "publication_year": 2023,
        "primary_location": {
            "source": {"display_name": "Genomics World"},
            "landing_page_url": None,
        },
        "open_access": {"oa_url": "https://oa.example/pipeline"},
        "doi": "10.999/xyz",
        "relevance_score": 48.0,
    },
    {
        "id": "https://openalex.org/W5555",
        "display_name": "Data stewardship in genomic programs",
        "authorships": [],
        "publication_year": 2021,
        "primary_location": {
            "source": {"display_name": "Data Journal"},
            "landing_page_url": "https://data.example/article",
        },
        "open_access": {},
        "doi": None,
        "relevance_score": 42.0,
    },
]


@pytest.fixture()
def mocked_client() -> Iterator[httpx.Client]:
    payload = {
        "results": MOCK_WORKS,
        "meta": {"count": len(MOCK_WORKS), "next_cursor": None},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    try:
        yield client
    finally:
        client.close()


def test_query_openalex_deduplicates_and_matches_golden(
    mocked_client: httpx.Client,
) -> None:
    config = SearchConfig(year_min=2020, languages=["en"], max_results=10)
    result = query_openalex(
        "genomic screening", limit=10, client=mocked_client, config=config
    )

    assert result.metrics.found == len(MOCK_WORKS)
    assert result.metrics.unique == 3
    assert result.metrics.with_doi == 2
    assert result.metrics.fallback_used == "original"
    assert result.metrics.language_used == "en"
    assert result.metrics.query_used == "genomic screening"
    assert result.metrics.queries_tried == ["original:en:genomic screening"]

    golden_path = Path(__file__).parent / "data" / "openalex_golden.json"
    expected = json.loads(golden_path.read_text(encoding="utf-8"))
    actual = [paper.model_dump() for paper in result.papers]
    assert actual == expected


def test_query_openalex_uses_fallback_when_initial_query_empty() -> None:
    responses: dict[str, list[dict[str, Any]]] = {
        "original": [],
        "ascii": [],
        "mapped": MOCK_WORKS[:2],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        query = request.url.params.get("search", "")
        if "cancer" in query:
            dataset = responses["mapped"]
        elif query.isascii():
            dataset = responses["ascii"]
        else:
            dataset = responses["original"]
        return httpx.Response(
            200,
            json={
                "results": dataset,
                "meta": {"count": len(dataset), "next_cursor": None},
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    try:
        config = SearchConfig(year_min=2018, languages=["en", "fi"], max_results=5)
        result = query_openalex(
            "genominen seulonta sy\u00f6v\u00e4ss\u00e4", client=client, config=config
        )
    finally:
        client.close()

    assert result.metrics.fallback_used == "mapped"
    assert result.metrics.language_used in {"en", "any"}
    assert result.metrics.queries_tried[-1].startswith("mapped:")
    assert result.metrics.found == 2


def test_append_audit_log_appends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    metrics = SearchMetrics(
        topic="demo",
        found=5,
        unique=3,
        with_doi=2,
        query_used="demo",
        fallback_used="mapped",
        language_used="en",
        queries_tried=["mapped:en:demo"],
    )
    out_path = Path("artifacts/demo.parquet")

    append_audit_log(metrics, output_path=out_path)
    append_audit_log(metrics, output_path=out_path)

    log_path = tmp_path / "data" / "cache" / "search_log.csv"
    assert log_path.exists()
    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 3
    header = lines[0].split(",")
    assert header == [
        "timestamp",
        "topic",
        "source",
        "found",
        "unique",
        "with_doi",
        "query_used",
        "fallback_used",
        "language_used",
        "queries_tried",
        "output_path",
    ]
    parts = lines[1].split(",")
    assert parts[1] == "demo"
    assert parts[2] == "openalex"
    assert parts[7] == "mapped"
    assert parts[8] == "en"
    assert parts[9] == "mapped:en:demo"
    assert Path(parts[10]) == Path("artifacts/demo.parquet")
