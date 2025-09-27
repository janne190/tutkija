"""Rule-based pre-screening helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any
import numpy as np
import pandas as pd

_LANGUAGE_REASON = "language filter"
_YEAR_REASON = "year filter"
_TYPE_REASON = "type filter"
_MANUAL_REASON = "manual check"
_ALLOWED_REASONS = {_LANGUAGE_REASON, _YEAR_REASON, _TYPE_REASON, _MANUAL_REASON}
_NON_RESEARCH_TYPES = {"editorial", "letter", "news"}
_TYPE_COLUMNS = ("type", "document_type", "publication_type", "pub_type")


def is_empty(x: Any) -> bool:
    """Check if a value is empty, handling None, pandas/numpy objects, and sequences."""
    if x is None:
        return True
    if hasattr(x, "empty"):  # DataFrame/Series
        return x.empty
    if hasattr(x, "size"):  # ndarray
        return x.size == 0
    try:
        return len(x) == 0  # list, tuple, etc.
    except TypeError:
        return False


def has_reasons(v: Any) -> bool:
    """Check if a value contains any reasons, handling numpy arrays and other types."""
    if isinstance(v, np.ndarray):
        return v.size > 0
    try:
        return len(v) > 0
    except TypeError:
        return bool(v)


def _reason_list(val: Any) -> list[str]:
    if isinstance(val, list):
        return [str(x) for x in val if str(x).strip()]
    return []


def _add_reason_if_missing(df: pd.DataFrame, idx: Any, reason: str) -> bool:
    lst = _reason_list(df.loc[idx, "reasons"])
    if reason not in lst:
        lst.append(str(reason))
        # Aseta lista aina yhteen soluun
        df.at[idx, "reasons"] = lst
        return True
    return False


def _normalize_reasons(value: Any) -> list[str]:
    raw: list[str]
    if isinstance(value, list):
        raw = [str(item) for item in value if not is_empty(str(item))]
    elif isinstance(value, tuple):
        raw = [str(item) for item in value if not is_empty(str(item))]
    elif is_empty(value):
        return []
    elif isinstance(value, float) and pd.isna(value):
        return []
    else:
        raw = [str(value)]

    normalised: list[str] = []
    for item in raw:
        text = item.strip()
        if is_empty(text):
            continue
        reason = text if text in _ALLOWED_REASONS else _MANUAL_REASON
        if reason not in normalised:
            normalised.append(reason)
    return normalised


def _to_int(value: Any) -> int | None:
    if is_empty(value):
        return None
    try:
        if isinstance(value, (list, tuple)):
            if is_empty(value):
                return None
            value = value[0]
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _normalize_iterable(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value.strip()
        return
    if isinstance(value, Sequence):
        for item in value:
            if item is None:
                continue
            yield str(item).strip()
        return
    if value not in (None, ""):
        yield str(value).strip()


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
        result["reasons"] = result["reasons"].apply(_normalize_reasons)

    counts: dict[str, int] = {"language": 0, "year": 0, "type": 0}

    # Käytä eksplisiittistä None-tarkistusta, jotta mypy kaventaa tyypin
    if (
        allowed_lang is not None
        and len(allowed_lang) > 0
        and "language" in result.columns
    ):
        allowed = {str(lang).lower() for lang in allowed_lang}
        for idx, value in result["language"].items():
            normalized = str(value).strip().lower() if value is not None else ""
            if is_empty(normalized):
                continue
            if normalized not in allowed:
                if _add_reason_if_missing(result, idx, _LANGUAGE_REASON):
                    counts["language"] += 1

    if year_min is not None and "year" in result.columns:
        for idx, value in result["year"].items():
            parsed_year = _to_int(value)
            if parsed_year is None:
                continue
            if parsed_year < year_min:
                if _add_reason_if_missing(result, idx, _YEAR_REASON):
                    counts["year"] += 1

    type_col = next((col for col in _TYPE_COLUMNS if col in result.columns), None)
    if drop_non_research and type_col:
        for idx, value in result[type_col].items():
            flagged = False
            for entry in _normalize_iterable(value):
                normalized = str(entry).strip().lower()
                if is_empty(normalized):
                    continue
                words = normalized.split()
                if any(word in _NON_RESEARCH_TYPES for word in words):
                    flagged = True
                    break
                if normalized in _NON_RESEARCH_TYPES:
                    flagged = True
                    break
            if flagged:
                if _add_reason_if_missing(result, idx, _TYPE_REASON):
                    counts["type"] += 1

    return result, counts
