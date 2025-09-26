from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Sequence, TypedDict, cast

import pandas as pd
from rapidfuzz import fuzz

from .types import Paper

__all__ = ["merge_and_filter", "MergeStats"]

SIMILARITY_THRESHOLD = 90


class MergeStats(TypedDict):
    per_source: Dict[str, int]
    dup_doi: int
    dup_title: int
    filtered: int
    removed: List[Dict[str, object]]


def merge_and_filter(
    openalex: Sequence[Paper] | None,
    pubmed: Sequence[Paper] | None,
    arxiv: Sequence[Paper] | None,
) -> tuple[pd.DataFrame, MergeStats]:
    """Combine papers from multiple sources and deduplicate."""

    all_papers = list(openalex or []) + list(pubmed or []) + list(arxiv or [])
    if not all_papers:
        empty_stats: MergeStats = {
            "per_source": {},
            "dup_doi": 0,
            "dup_title": 0,
            "filtered": 0,
            "removed": [],
        }
        return pd.DataFrame(), empty_stats
    per_source = Counter(paper.source for paper in all_papers)

    entries: list[dict[str, Any]] = []
    for paper in all_papers:
        record: dict[str, Any] = paper.model_dump()
        record["__drop__"] = False
        record["reasons"] = cast(list[str], [])
        entries.append(record)

    removed: list[dict[str, Any]] = []

    dup_doi = _dedupe_by_doi(entries, removed)
    dup_title = _dedupe_by_title(entries, removed)

    filtered = 0  # Placeholder for future filtering rules.

    remaining = [entry for entry in entries if not entry["__drop__"]]
    for entry in remaining:
        entry.pop("__drop__", None)
    for entry in removed:
        entry.pop("__drop__", None)

    df = pd.DataFrame(remaining)
    if not df.empty and "reasons" in df.columns:
        df["reasons"] = df["reasons"].apply(list)

    stats: MergeStats = {
        "per_source": dict(per_source),
        "dup_doi": dup_doi,
        "dup_title": dup_title,
        "filtered": filtered,
        "removed": removed,
    }
    return df, stats


def _normalize_doi(doi: str) -> str:
    """Normalize DOI for comparison by stripping whitespace, converting to lowercase and removing prefix."""
    doi = doi.strip().lower()
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/") :]
    return doi


def _dedupe_by_doi(entries: list[dict[str, Any]], removed: list[dict[str, Any]]) -> int:
    seen: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for entry in entries:
        if entry["__drop__"]:
            continue
        doi = _normalize_doi(str(entry.get("doi", "")))
        if doi:  # Only process entries with non-empty DOI after normalization
            if doi not in seen:
                seen[doi] = entry
            else:
                keep, drop = _choose_richer(seen[doi], entry)
                cast(list[str], drop["reasons"]).append(
                    f"dup, doi match kept={keep.get('source', '')}"
                )
                drop["__drop__"] = True
                removed.append(drop.copy())
                seen[doi] = keep
                duplicates += 1
    return duplicates


def _dedupe_by_title(
    entries: list[dict[str, Any]], removed: list[dict[str, Any]]
) -> int:
    def _normalize_doi(doi: str) -> str:
        """Normalize DOI for comparison by stripping whitespace, converting to lowercase and removing prefix."""
        doi = doi.strip().lower()
        if doi.startswith("https://doi.org/"):
            doi = doi[len("https://doi.org/") :]
        return doi

    kept_entries = [entry for entry in entries if not entry["__drop__"]]
    duplicates = 0
    for idx, entry in enumerate(kept_entries):
        if entry["__drop__"]:
            continue
        title_a = str(entry.get("title", "")).strip().lower()
        if not title_a:
            continue
        for other in kept_entries[idx + 1 :]:
            if other["__drop__"]:
                continue
            title_b = str(other.get("title", "")).strip().lower()
            if not title_b:
                continue
            score = fuzz.token_sort_ratio(title_a, title_b)
            if score >= SIMILARITY_THRESHOLD:
                keep, drop = _choose_richer(entry, other)
                cast(list[str], drop["reasons"]).append(
                    f"dup, title>={SIMILARITY_THRESHOLD} kept={keep.get('source', '')}"
                )
                drop["__drop__"] = True
                removed.append(drop.copy())
                duplicates += 1
    return duplicates


def _choose_richer(
    first: dict[str, Any],
    second: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _richness_score(second) > _richness_score(first):
        return second, first
    return first, second


def _richness_score(entry: dict[str, Any]) -> int:
    filled = 0
    for key in ("title", "abstract", "authors", "year", "venue", "doi", "url"):
        value = entry.get(key)
        if key == "authors":
            if isinstance(value, list) and any(value):
                filled += 1
        elif isinstance(value, (str, int)) and str(value).strip() not in {"", "0"}:
            filled += 1
    return filled
