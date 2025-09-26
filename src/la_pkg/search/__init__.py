from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from pydantic import BaseModel, Field

__all__ = [
    "Paper",
    "clean_text",
    "parse_year",
]


class Paper(BaseModel):
    """Normalized representation of a scientific paper across sources."""

    id: str
    title: str = ""
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int = 0
    venue: str = ""
    doi: str = ""
    url: str = ""
    source: str
    score: float | None = None

    model_config = {"extra": "ignore"}

    @classmethod
    def from_parts(
        cls,
        *,
        id: str,
        title: str | None = None,
        abstract: str | None = None,
        authors: Iterable[str] | None = None,
        year: int | str | None = None,
        venue: str | None = None,
        doi: str | None = None,
        url: str | None = None,
        source: str,
        score: float | None = None,
    ) -> "Paper":
        return cls(
            id=id,
            title=clean_text(title),
            abstract=clean_text(abstract),
            authors=[clean_text(name) for name in (authors or []) if clean_text(name)],
            year=parse_year(year),
            venue=clean_text(venue),
            doi=clean_text(doi),
            url=clean_text(url),
            source=source,
            score=score,
        )


def clean_text(value: str | None) -> str:
    """Normalize whitespace and Unicode for textual metadata."""

    if not value:
        return ""
    normalized = unicodedata.normalize("NFC", value)
    collapsed = " ".join(normalized.split())
    return collapsed.strip()


YEAR_PATTERN = re.compile(r"(19|20|21)\d{2}")


def parse_year(value: int | str | None) -> int:
    """Parse a publication year into an integer, returning 0 when unknown."""

    if isinstance(value, int):
        return value if value > 0 else 0
    if not value:
        return 0
    match = YEAR_PATTERN.search(str(value))
    if match:
        try:
            year_int = int(match.group())
            if year_int > 0:
                return year_int
        except ValueError:
            return 0
    return 0
