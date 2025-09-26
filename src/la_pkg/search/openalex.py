from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Optional
import unicodedata

import httpx
from rapidfuzz import fuzz

from . import Paper, clean_text
from .http_client import apply_contact, create_http_client

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only for Python <3.11
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]


OPENALEX_WORKS_URL = "https://api.openalex.org/works"
DEFAULT_MAX_RESULTS = 200
SIMILARITY_THRESHOLD = 90.0
TERM_MAP = {
    "genominen": "genomic",
    "seulonta": "screening",
    "sy\u00f6p\u00e4": "cancer",
    "sy\u00f6v\u00e4n": "cancer",
    "sy\u00f6v\u00e4ss\u00e4": "cancer",
    "sy\u00f6p\u00e4potilailla": "cancer",
    "biomarkkeri": "biomarker",
    "biomarkkerit": "biomarkers",
    "katsaus": "review",
}


@dataclass
class SearchConfig:
    """Runtime configuration for OpenAlex searches."""

    year_min: Optional[int] = None
    languages: list[str] = field(default_factory=list)
    max_results: int = DEFAULT_MAX_RESULTS


@dataclass
class SearchMetrics:
    """Basic metrics emitted by a search run."""

    topic: str
    found: int
    unique: int
    with_doi: int
    query_used: str = ""
    fallback_used: str = "original"
    language_used: str = "any"
    queries_tried: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "topic": self.topic,
            "found": self.found,
            "unique": self.unique,
            "with_doi": self.with_doi,
            "query_used": self.query_used,
            "fallback_used": self.fallback_used,
            "language_used": self.language_used,
            "queries_tried": ";".join(self.queries_tried),
        }


@dataclass
class OpenAlexSearchResult:
    """Return structure for an OpenAlex query."""

    papers: list[Paper]
    metrics: SearchMetrics


def load_search_config() -> SearchConfig:
    """Load search configuration from config.toml or its example."""

    config_paths = [Path("config.toml"), Path("config.example.toml")]
    for path in config_paths:
        if path.exists():
            with path.open("rb") as handle:
                data: dict[str, Any] = tomllib.load(handle)
            search_cfg = data.get("search", {}) if isinstance(data, dict) else {}
            year_min = (
                search_cfg.get("year_min") if isinstance(search_cfg, dict) else None
            )
            languages: list[str] = []
            if isinstance(search_cfg, dict):
                raw_languages = search_cfg.get("language", [])
                if isinstance(raw_languages, list):
                    languages = [
                        str(lang) for lang in raw_languages if isinstance(lang, str)
                    ]
            max_results = DEFAULT_MAX_RESULTS
            if isinstance(search_cfg, dict) and isinstance(
                search_cfg.get("max_results"), int
            ):
                max_results = int(search_cfg["max_results"])
            if isinstance(year_min, int):
                return SearchConfig(
                    year_min=year_min, languages=languages, max_results=max_results
                )
            return SearchConfig(languages=languages, max_results=max_results)
    return SearchConfig()


def query_openalex(
    topic: str,
    *,
    limit: Optional[int] = None,
    client: Optional[httpx.Client] = None,
    config: Optional[SearchConfig] = None,
    language: str = "auto",
) -> OpenAlexSearchResult:
    """Query OpenAlex Works endpoint and return a normalized result set."""

    if not topic.strip():
        raise ValueError("topic must be a non-empty string")

    cfg = config or load_search_config()
    max_items = limit or cfg.max_results or DEFAULT_MAX_RESULTS
    http_client = client or create_http_client()
    close_client = client is None

    queries = list(_expand_queries(topic))
    languages = _resolve_languages(language, cfg.languages)
    queries_tried: list[str] = []
    fallback_used = "original"
    language_used = "any"
    query_used = queries[0][1]
    works: list[dict[str, Any]] = []

    try:
        for fallback_label, query_text in queries:
            query_used = query_text
            lang_sequence = languages or [None]
            for lang in lang_sequence:
                works = _collect_works(
                    http_client,
                    query=query_text,
                    cfg=cfg,
                    max_items=max_items,
                    language_filter=lang,
                )
                queries_tried.append(_format_attempt(fallback_label, query_text, lang))
                if works:
                    fallback_used = fallback_label
                    language_used = lang or "any"
                    break
            if works:
                break
            if languages:
                works = _collect_works(
                    http_client,
                    query=query_text,
                    cfg=cfg,
                    max_items=max_items,
                    language_filter=None,
                )
                queries_tried.append(_format_attempt(fallback_label, query_text, None))
                if works:
                    fallback_used = fallback_label
                    language_used = "any"
                    break
        else:
            works = []
    finally:
        if close_client:
            http_client.close()

    papers = [_map_work(work) for work in works]
    unique_papers = _deduplicate(papers)
    metrics = SearchMetrics(
        topic=topic,
        found=len(papers),
        unique=len(unique_papers),
        with_doi=sum(1 for paper in unique_papers if paper.doi),
        query_used=query_used,
        fallback_used=fallback_used,
        language_used=language_used,
        queries_tried=queries_tried,
    )
    return OpenAlexSearchResult(papers=unique_papers, metrics=metrics)


def _collect_works(
    http_client: httpx.Client,
    *,
    query: str,
    cfg: SearchConfig,
    max_items: int,
    language_filter: Optional[str],
) -> list[dict[str, Any]]:
    per_page = min(200, max_items)
    params: dict[str, str] = {
        "search": query,
        "per-page": str(per_page),
        "cursor": "*",
        "sort": "relevance_score:desc",
    }
    params = apply_contact(params)
    filters: list[str] = []
    if cfg.year_min:
        filters.append(f"from_publication_date:{int(cfg.year_min)}-01-01")
    if language_filter:
        filters.append(f"language:{language_filter}")
    if filters:
        params["filter"] = ",".join(filters)

    results: list[dict[str, Any]] = []
    cursor: Optional[str] = "*"
    while len(results) < max_items and cursor:
        params["cursor"] = cursor
        response = http_client.get(OPENALEX_WORKS_URL, params=params)
        response.raise_for_status()
        payload = response.json()
        works = payload.get("results", [])
        if not isinstance(works, list):
            break
        results.extend(work for work in works if isinstance(work, dict))
        if len(results) >= max_items:
            break
        meta = payload.get("meta")
        if isinstance(meta, Mapping):
            next_cursor = meta.get("next_cursor")
            cursor = str(next_cursor) if isinstance(next_cursor, str) else None
        else:
            cursor = None
    return results[:max_items]


def _map_work(work: Mapping[str, Any]) -> Paper:
    authors = [
        entry["author"].get("display_name")
        for entry in work.get("authorships", [])
        if isinstance(entry, Mapping) and isinstance(entry.get("author"), Mapping)
    ]
    abstract = _extract_abstract(work)
    venue = _safe_get(work, "primary_location", "source", "display_name")
    url = (
        _safe_get(work, "primary_location", "landing_page_url")
        or _safe_get(work, "open_access", "oa_url")
        or work.get("id")
    )
    doi = work.get("doi")
    score = None
    relevance = work.get("relevance_score")
    if isinstance(relevance, (int, float)):
        score = float(relevance)
    return Paper.from_parts(
        id=str(work.get("id", "")),
        title=work.get("display_name"),
        abstract=abstract,
        authors=authors,
        year=work.get("publication_year"),
        venue=venue,
        doi=doi,
        url=url,
        source="openalex",
        score=score,
    )


def _extract_abstract(work: Mapping[str, Any]) -> str:
    inverted = work.get("abstract_inverted_index")
    if isinstance(inverted, Mapping):
        positions: list[tuple[int, str]] = []
        for word, indexes in inverted.items():
            if not isinstance(word, str) or not isinstance(indexes, list):
                continue
            for index in indexes:
                if isinstance(index, int):
                    positions.append((index, word))
        ordered = [word for _, word in sorted(positions, key=lambda item: item[0])]
        return clean_text(" ".join(ordered))
    summary = work.get("summary")
    if isinstance(summary, str):
        return clean_text(summary)
    if summary:
        return clean_text(json.dumps(summary))
    return ""


def _deduplicate(papers: Iterable[Paper]) -> list[Paper]:
    seen: dict[str, Paper] = {}
    unique: list[Paper] = []
    for paper in papers:
        doi = paper.doi.lower() if paper.doi else None
        if doi:
            if doi in seen:
                continue
            seen[doi] = paper
            unique.append(paper)
            continue
        if paper.title:
            duplicate = False
            for existing in unique:
                if not existing.title:
                    continue
                if (
                    fuzz.token_sort_ratio(existing.title, paper.title)
                    >= SIMILARITY_THRESHOLD
                ):
                    duplicate = True
                    break
            if duplicate:
                continue
        unique.append(paper)
    return unique


def _safe_get(obj: Mapping[str, Any], *path: str) -> Optional[str]:
    current: object = obj
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current if isinstance(current, str) else None


def _expand_queries(topic: str) -> Iterable[tuple[str, str]]:
    cleaned = topic.strip()
    ascii_version = (
        unicodedata.normalize("NFKD", cleaned).encode("ascii", "ignore").decode()
    )
    tokens = cleaned.lower().split()
    mapped_tokens = [TERM_MAP.get(token, token) for token in tokens]
    mapped = " ".join(mapped_tokens)

    yielded: set[str] = set()
    for label, query in (
        ("original", cleaned),
        ("ascii", ascii_version),
        ("mapped", mapped),
    ):
        if query and query not in yielded:
            yielded.add(query)
            yield label, query


def _resolve_languages(option: str, configured: list[str]) -> list[Optional[str]]:
    option_lower = option.lower()
    if option_lower in {"en", "fi"}:
        return [option_lower]
    if option_lower == "auto":
        if configured:
            return [lang.lower() for lang in configured]
        return ["en", "fi"]
    if option_lower == "none":
        return []
    return [option_lower]


def _format_attempt(label: str, query: str, language: Optional[str]) -> str:
    lang = language or "any"
    return f"{label}:{lang}:{query}"


def append_audit_log(
    metrics: SearchMetrics, *, output_path: Path, source: str = "openalex"
) -> None:
    """Persist metrics into a CSV audit log under data/cache."""

    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_path = cache_dir / "search_log.csv"
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    row = {
        "timestamp": timestamp,
        "topic": metrics.topic,
        "source": source,
        "found": metrics.found,
        "unique": metrics.unique,
        "with_doi": metrics.with_doi,
        "query_used": metrics.query_used,
        "fallback_used": metrics.fallback_used,
        "language_used": metrics.language_used,
        "queries_tried": ";".join(metrics.queries_tried),
        "output_path": str(output_path),
    }
    header = [
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
    line = ",".join(str(row[key]) for key in header)
    if not log_path.exists():
        log_path.write_text(",".join(header) + "\n" + line + "\n", encoding="utf-8")
    else:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


__all__ = [
    "SearchConfig",
    "SearchMetrics",
    "OpenAlexSearchResult",
    "append_audit_log",
    "query_openalex",
]
