"""Smoke tests for data schemas and output formats."""

from __future__ import annotations

import pandas as pd
from pathlib import Path


def test_schema_smoke() -> None:
    """Test that Parquet files have correct schemas and dtypes."""
    merged_path = Path("data/cache/merged.parquet")
    screened_path = Path("data/cache/screened.parquet")

    # Skip if files don't exist (fresh checkout)
    if not merged_path.exists() or not screened_path.exists():
        return

    # Check merged file schema
    merged_df = pd.read_parquet(merged_path)
    assert "id" in merged_df.columns
    assert "title" in merged_df.columns
    assert "abstract" in merged_df.columns
    assert "authors" in merged_df.columns
    assert "year" in merged_df.columns
    assert "venue" in merged_df.columns
    assert "doi" in merged_df.columns
    assert "url" in merged_df.columns
    assert "source" in merged_df.columns
    assert "score" in merged_df.columns
    assert "reasons" in merged_df.columns

    # Check dtypes
    assert merged_df["id"].dtype == "object"  # string
    assert merged_df["year"].dtype == "int64"
    assert merged_df["score"].dtype == "float64"
    reasons = merged_df["reasons"]
    non_null_reasons = reasons[~pd.isna(reasons)]
    assert all(isinstance(r, list) for r in non_null_reasons)

    # Check screened file schema
    screened_df = pd.read_parquet(screened_path)
    assert "probability" in screened_df.columns
    assert "label" in screened_df.columns
    assert screened_df["probability"].dtype == "float64"
    assert screened_df["label"].dtype == "object"  # string
    reasons = screened_df["reasons"]
    non_null_reasons = reasons[~pd.isna(reasons)]
    assert all(isinstance(r, list) for r in non_null_reasons)
