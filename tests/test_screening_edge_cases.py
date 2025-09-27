"""Test edge cases in screening."""

from __future__ import annotations

import pandas as pd

from la_pkg.screening import EXCLUDED, score_and_label


def test_no_gold_no_seeds() -> None:
    """Test that model defaults to 0.5 probs with no training data."""
    df = pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "title": ["A", "B", "C"],
            "abstract": ["test one", "test two", "test three"],
            "reasons": [[], [], []],
        }
    )

    result = score_and_label(df, target_recall=0.9)
    df_result = result.frame

    # Check default probabilities
    assert "probability" in df_result.columns
    assert all(df_result["probability"] == 0.5)

    # Check labels are based on threshold
    assert set(df_result["label"].unique()) == {
        EXCLUDED
    }  # All excluded with 0.9 recall target


def test_single_class_gold() -> None:
    """Test that model handles gold set with only one class."""
    df = pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "title": ["A", "B", "C"],
            "abstract": ["test one", "test two", "test three"],
            "gold_label": ["excluded", "excluded", "excluded"],
            "reasons": [[], [], []],
        }
    )

    result = score_and_label(df, target_recall=0.9)
    df_result = result.frame

    # Should not fail and return reasonable probabilities
    assert "probability" in df_result.columns
    assert all(df_result["probability"] == 0.5)  # Default when untrained
    assert result.engine == "scikit"
    # With target_recall > 0.8, we use threshold 0.6 for more exclusions
    assert result.threshold == 0.5


def test_seeds_increase_nearby_inclusion() -> None:
    """Test that seed papers influence nearby papers positively."""
    papers = [
        (
            "Cancer genomics advances",
            "Important genomic screening results in cancer",
            None,
        ),
        ("More cancer research", "Further genomic studies in cancer treatment", None),
        ("Unrelated topic", "Agriculture and soil studies", None),
        ("Weather patterns", "Climate analysis methods", None),
    ]

    df = pd.DataFrame(
        [
            {
                "id": str(i),
                "title": title,
                "abstract": abstract,
                "gold_label": label,
                "reasons": [],
            }
            for i, (title, abstract, label) in enumerate(papers)
        ]
    )

    # First run without seeds
    result1 = score_and_label(df, target_recall=0.9)
    df_no_seeds = result1.frame
    no_seeds_probs = df_no_seeds["probability"].tolist()

    # Then run with first paper as seed
    result2 = score_and_label(df, target_recall=0.9, seeds=["0"])
    df_with_seeds = result2.frame
    with_seeds_probs = df_with_seeds["probability"].tolist()

    # The second paper (similar to seed) should get higher probability with seeds
    assert with_seeds_probs[1] > no_seeds_probs[1]
    # The unrelated papers should get lower probabilities
    assert with_seeds_probs[2] < with_seeds_probs[1]
    assert with_seeds_probs[3] < with_seeds_probs[1]


def test_rules_override_reasons() -> None:
    """Test that rules-based reasons are correctly assigned."""
    df = pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "title": ["A", "B", "C", "D"],
            "abstract": ["test one", "test two", "test three", "test four"],
            "reasons": [["excluded by rule 1"], [], ["excluded by rule 2"], []],
        }
    )

    result = score_and_label(df, target_recall=0.9)
    df_result = result.frame
    stats = result.metadata

    # Check that records with reasons are always excluded
    assert df_result.loc[df_result["reasons"].str.len() > 0, "label"].eq(EXCLUDED).all()

    # Check stats count the rules correctly
    assert stats["excluded_rules"] == 2
    assert stats["identified"] == 4
    assert stats["screened"] == 4
