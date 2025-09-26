"""Schema definitions for data storage."""

from __future__ import annotations

import pyarrow as pa

# Schema for screened data
SCREENED_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("title", pa.string()),
        ("abstract", pa.string()),
        ("authors", pa.list_(pa.string())),
        ("year", pa.int64()),
        ("venue", pa.string()),
        ("doi", pa.string()),
        ("url", pa.string()),
        ("source", pa.string()),
        ("score", pa.float64()),
        ("probability", pa.float64()),
        ("label", pa.string()),
        ("reasons", pa.list_(pa.string())),
    ]
)

# Schema for search results
MERGED_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("title", pa.string()),
        ("abstract", pa.string()),
        ("authors", pa.list_(pa.string())),
        ("year", pa.int64()),
        ("venue", pa.string()),
        ("doi", pa.string()),
        ("url", pa.string()),
        ("source", pa.string()),
        ("score", pa.float64()),
        ("reasons", pa.list_(pa.string())),
    ]
)
