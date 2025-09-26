"""Rule-based pre-screening helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd

_LANGUAGE_REASON = "language"
_YEAR_REASON = "year"
_TYPE_REASON = "type"
_NON_RESEARCH_TYPES = {"editorial", "letter", "news item", "Editorial"}
_TYPE_COLUMNS = ("type", "document_type", "publication_type", "pub_type")


def _normalize_reasons(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [] if value.strip() == "" else [value]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            value = value.tolist()
    except ImportError:
        pass
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            value = value[0]
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _normalize_iterable(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield str(value).strip().lower()
    elif isinstance(value, Sequence):  # type: ignore[unreachable]
        for item in value:
            if item is None:
                continue
            yield str(item).strip().lower()
    elif value not in (None, ""):
        yield str(value).strip().lower()


def apply_rules(
    df: pd.DataFrame,
    *,
    year_min: int | None = None,
    allowed_lang: Sequence[str] | None = ("en", "fi"),
    drop_non_research: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply conservative heuristics and return annotated DataFrame and rule counts.

    Rules are applied in order: language -> year -> type.
    Records that already have a 'not-research' reason will skip the type check.
    A record can be flagged by multiple rules, but only counted once per rule type.
    """
    result = df.copy()
    # Ensure reasons column exists and is normalized
    if "reasons" not in result.columns:
        result["reasons"] = [[] for _ in range(len(result))]
    else:
        result["reasons"] = result["reasons"].apply(_normalize_reasons)

    # Track rule applications for detailed breakdown in counts
    counts: dict[str, int] = {
        _LANGUAGE_REASON: 0,
        _YEAR_REASON: 0,
        _TYPE_REASON: 0,
    }

    # First pass: Skip type checks for records that already have 'not-research' reason
    # or other type-related exclusion reason
    skip_type_check = set(
        idx
        for idx, row in result.iterrows()
        if any(
            r.endswith("type filter") for r in row["reasons"]
        )  # Skip type check if record already has a type-related reason
    )

    # Rule 1: Language filter
    if allowed_lang and "language" in result.columns:
        allowed = {lang.lower() for lang in allowed_lang}
        for idx, row in result.iterrows():
            value = row.get("language")
            if value in (None, "") or pd.isna(value):
                continue
            normalized = str(value).strip().lower()
            if not normalized:
                continue
            if normalized not in allowed:
                reasons = result.at[idx, "reasons"]
                lang_filter = f"{_LANGUAGE_REASON} filter"
                if lang_filter not in reasons:
                    reasons.append(lang_filter)
                    counts[_LANGUAGE_REASON] += 1

    # Rule 2: Year filter
    if year_min is not None and "year" in result.columns:
        for idx, row in result.iterrows():
            value = row.get("year")
            parsed_year = _to_int(value)
            if parsed_year is None:
                continue
            if parsed_year < year_min:
                reasons = result.at[idx, "reasons"]
                year_filter = "year filter"
                if year_filter not in reasons:
                    reasons.append(year_filter)
                    counts[_YEAR_REASON] += 1

    # Rule 3: Type filter for non-research articles
    type_col = next((col for col in _TYPE_COLUMNS if col in result.columns), None)
    if drop_non_research and type_col:
        for idx, row in result.iterrows():
            if (
                idx in skip_type_check
            ):  # Skip if record already has a type-related reason
                continue

            value = row.get(type_col)
            normalized_types = list(_normalize_iterable(value))
            flagged = False
            for entry in normalized_types:
                if entry.lower() in _NON_RESEARCH_TYPES:
                    flagged = True
                    break
            if flagged:
                reasons = result.at[idx, "reasons"]
                type_filter = "type filter"
                if type_filter not in reasons:
                    reasons.append(type_filter)
                    counts[_TYPE_REASON] += 1

    return result, counts
