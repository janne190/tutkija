"""Rule-based pre-screening helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd

_LANGUAGE_REASON = "language filter"
_YEAR_REASON = "year filter"
_TYPE_REASON = "type filter"
_NON_RESEARCH_TYPES = {"editorial", "letter", "news"}
_TYPE_COLUMNS = ("type", "document_type", "publication_type", "pub_type")


def _normalize_reasons(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    if value in (None, ""):
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    return [str(value)]


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
        yield value
    elif isinstance(value, Sequence):  # type: ignore[unreachable]
        for item in value:
            if item is None:
                continue
            yield str(item)
    elif value not in (None, ""):
        yield str(value)


def apply_rules(
    df: pd.DataFrame,
    *,
    year_min: int | None = None,
    allowed_lang: Sequence[str] | None = ("en", "fi"),
    drop_non_research: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply conservative heuristics and return annotated DataFrame and rule counts."""

    result = df.copy()
    if "reasons" not in result.columns:
        result["reasons"] = [[] for _ in range(len(result))]
    else:
        # Ensure reasons are lists, not something else
        result["reasons"] = result["reasons"].apply(
            lambda x: _normalize_reasons(x) if not isinstance(x, list) else x
        )

    counts: dict[str, int] = {
        _LANGUAGE_REASON: 0,
        _YEAR_REASON: 0,
        _TYPE_REASON: 0,
    }

    # Rule: Language filter
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
                if _LANGUAGE_REASON not in reasons:
                    reasons.append(_LANGUAGE_REASON)
                    counts[_LANGUAGE_REASON] += 1

    # Rule: Year filter
    if year_min is not None and "year" in result.columns:
        for idx, row in result.iterrows():
            value = row.get("year")
            parsed_year = _to_int(value)
            if parsed_year is None:
                continue
            if parsed_year < year_min:
                reasons = result.at[idx, "reasons"]
                if _YEAR_REASON not in reasons:
                    reasons.append(_YEAR_REASON)
                    counts[_YEAR_REASON] += 1

    # Rule: Type filter for non-research articles
    type_col = next((col for col in _TYPE_COLUMNS if col in result.columns), None)
    if drop_non_research and type_col:
        for idx, row in result.iterrows():
            value = row.get(type_col)
            flagged = False
            for entry in _normalize_iterable(value):
                if entry.lower() in _NON_RESEARCH_TYPES:
                    flagged = True
                    break
            if flagged:
                reasons = result.at[idx, "reasons"]
                if _TYPE_REASON not in reasons:
                    reasons.append(_TYPE_REASON)
                    counts[_TYPE_REASON] += 1

    return result, counts
