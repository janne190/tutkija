"""Tests for multisource merge and dedupe."""

from __future__ import annotations

import pandas as pd
from typing import Dict, List, TypedDict, cast

from la_pkg.search.merge import merge_and_filter
from la_pkg.search.types import Paper


class Removal(TypedDict, total=False):
    reasons: List[str]


class Stats(TypedDict):
    dup_doi: int
    dup_title: int
    filtered: int
    per_source: Dict[str, int]
    removed: List[Removal]


def _paper(**kwargs) -> Paper:
    return Paper.from_parts(**kwargs)


def test_deduplication_doi_empty_and_normalized():
    # Add test for DOI handling with empty string and case sensitivity
    df, stats = merge_and_filter(
        [Paper(id="1", title="Paper 1", source="openalex", doi="10.1234/ABC")],
        [
            Paper(id="2", title="Paper 2", source="pubmed", doi=""),
            Paper(
                id="3",
                title="Paper 3",
                source="pubmed",
                doi="10.1234/abc",  # Same as first paper but lowercase
            ),
        ],
        None,
    )
    # Should have two papers - one with DOI (keeping either case) and one with empty DOI
    assert len(df) == 2
    dois = set(df["doi"].tolist())
    assert len(dois) == 2
    assert "" in dois  # Empty DOI paper should be kept
    assert any(
        doi.lower() == "10.1234/abc" for doi in dois
    )  # Original or lowercase version kept
    assert stats["dup_doi"] == 1  # One duplicate found by DOI


def test_merge_and_filter_deduplicates_by_doi_and_title() -> None:
    oa = [
        _paper(
            id="oa-1",
            title="Deep Learning for Screening",
            abstract="Details",
            authors=["Alice"],
            year=2024,
            venue="Journal A",
            doi="10.1/abc",
            url="https://example.org/oa1",
            source="openalex",
        ),
        _paper(
            id="oa-dup",
            title="Near Duplicate Title",
            abstract="",
            authors=["Bob"],
            year=2023,
            venue="Journal B",
            doi="",
            url="https://example.org/oa2",
            source="openalex",
        ),
    ]
    pm = [
        _paper(
            id="pm-1",
            title="Deep Learning for Screening",
            abstract="Extended abstract",
            authors=["Alice", "Carol"],
            year=2024,
            venue="Journal A",
            doi="10.1/abc",
            url="https://pubmed.ncbi.nlm.nih.gov/1/",
            source="pubmed",
        )
    ]
    ax = [
        _paper(
            id="ax-1",
            title="Near duplicate title",
            abstract="ArXiv version",
            authors=["Bob"],
            year=2023,
            venue="",
            doi="",
            url="https://arxiv.org/abs/1234",
            source="arxiv",
        ),
        _paper(
            id="ax-unique",
            title="Completely Different Study",
            abstract="Unique",
            authors=["Dana"],
            year=2022,
            venue="Conference X",
            doi="10.9/unique",
            url="https://arxiv.org/abs/9999",
            source="arxiv",
        ),
    ]

    df_raw, stats_raw = merge_and_filter(oa, pm, ax)
    df = cast(pd.DataFrame, df_raw)
    stats: Stats = cast(Stats, stats_raw)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3  # one DOI duplicate and one title duplicate removed
    assert stats["dup_doi"] == 1
    assert stats["dup_title"] == 1
    assert stats["filtered"] == 0
    assert stats["per_source"] == {"openalex": 2, "pubmed": 1, "arxiv": 2}
    assert len(stats["removed"]) == 2
    doi_removals = [
        r
        for r in stats["removed"]
        if "reasons" in r and "doi" in " ".join(r["reasons"])
    ]
    title_removals = [
        r
        for r in stats["removed"]
        if "reasons" in r and "title" in " ".join(r["reasons"])
    ]
    assert doi_removals and title_removals
