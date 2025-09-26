"""Temporary debug test."""

from __future__ import annotations

import json
from collections.abc import Iterator
import httpx
import pytest

from la_pkg.search.openalex import query_openalex, SearchConfig
from test_search_openalex import MOCK_WORKS


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


def test_query_debug(mocked_client) -> None:
    """Print out actual data for debugging."""
    config = SearchConfig(year_min=2020, languages=["en"], max_results=10)
    result = query_openalex(
        "genomic screening", limit=10, client=mocked_client, config=config
    )
    actual = [paper.model_dump() for paper in result.papers]
    print(json.dumps(actual, indent=2))
