"""Unit tests for rule-based screening."""

from __future__ import annotations

import pandas as pd

from la_pkg.screening import apply_rules


def test_apply_rules_flags_language_year_and_type() -> None:
    df = pd.DataFrame(
        [
            {
                "id": "A1",
                "title": "Baseline research",
                "language": "en",
                "year": 2022,
                "type": "article",
                "reasons": [],
            },
            {
                "id": "B2",
                "title": "Spanish summary",
                "language": "es",
                "year": 2023,
                "type": "article",
                "reasons": [],
            },
            {
                "id": "C3",
                "title": "Older Finnish letter",
                "language": "fi",
                "year": 2016,
                "type": "letter",
                "reasons": ["manual check"],
            },
            {
                "id": "D4",
                "title": "Editorial comment",
                "language": "en",
                "year": 2024,
                "type": ["Editorial"],
                "reasons": [],
            },
            {
                "id": "E5",
                "title": "News item",
                "language": "de",
                "year": 2015,
                "type": "news item",
                "reasons": [],
            },
        ]
    )

    updated, counts = apply_rules(
        df,
        year_min=2018,
        allowed_lang=("en", "fi"),
        drop_non_research=True,
    )

    assert counts == {"language": 2, "year": 2, "type": 3}

    reasons_b2 = updated.loc[updated["id"] == "B2", "reasons"].iloc[0]
    assert "language filter" in reasons_b2

    reasons_c3 = updated.loc[updated["id"] == "C3", "reasons"].iloc[0]
    assert "year filter" in reasons_c3
    assert "manual check" in reasons_c3

    reasons_d4 = updated.loc[updated["id"] == "D4", "reasons"].iloc[0]
    assert reasons_d4 == ["type filter"]

    reasons_e5 = updated.loc[updated["id"] == "E5", "reasons"].iloc[0]
    assert set(reasons_e5) == {"language filter", "year filter", "type filter"}

    reasons_a1 = updated.loc[updated["id"] == "A1", "reasons"].iloc[0]
    assert reasons_a1 == []
