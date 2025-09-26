from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, Field


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


class Paper(BaseModel):
    """Normalized representation of a scientific paper across sources."""

    id: str
    title: str = ""
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    url: str | None = None
    source: str
    score: float | None = None
    # Hidden fields that don't appear in serialization
    language: str | None = Field(None, exclude=True)
    type: str | None = Field(None, exclude=True)

    model_config = {
        "validate_assignment": True,
        "exclude_none": True,
        "json_schema_extra": {"additionalProperties": False},
    }

    def model_dump(self, **kwargs):
        """Override model_dump to handle None values and exclusions."""
        # Combine excludes from kwargs with our default excludes
        exclude = kwargs.pop("exclude", set())
        if isinstance(exclude, dict):
            exclude = set(exclude.keys())
        exclude = exclude | {"language", "type"}

        data = super().model_dump(exclude=exclude, **kwargs)
        # Convert None values to empty strings for string fields
        for key in ["doi", "url", "venue", "abstract"]:
            if key in data and data[key] is None:
                data[key] = ""
        return data

    @classmethod
    def from_parts(
        cls,
        *,
        id: str,
        title: str | None = None,
        abstract: str | None = None,
        authors: list[str] | None = None,
        year: int | str | None = None,
        venue: str | None = None,
        doi: str | None = None,
        url: str | None = None,
        source: str,
        score: float | None = None,
        language: str | None = None,
        type: str | None = None,
    ) -> "Paper":
        """Create a Paper instance from individual fields."""
        # Clean and validate required fields
        if not id or not source:
            raise ValueError("id and source are required")

        # Initialize with required fields
        paper = cls(id=id, title=clean_text(title or ""), source=source)

        # Set optional fields only if they have actual values
        if abstract:
            paper.abstract = clean_text(abstract)
        if authors:
            paper.authors = [clean_text(name) for name in authors if clean_text(name)]
        if year is not None:
            paper.year = parse_year(year)
        if venue:
            paper.venue = clean_text(venue)
        if doi is not None:  # Always set DOI, even if empty string
            paper.doi = clean_text(doi)
        if url:
            paper.url = clean_text(url)
        if score is not None:
            paper.score = score

        # Handle optional metadata fields
        if language and language.strip():
            paper.language = language
        if type and type.strip():
            paper.type = type

        return paper
