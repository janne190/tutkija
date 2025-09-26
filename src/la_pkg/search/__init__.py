"""Search utilities for Tutkija."""

from .openalex import OpenAlexSearchResult, Paper, append_audit_log, query_openalex

__all__ = [
    "Paper",
    "OpenAlexSearchResult",
    "append_audit_log",
    "query_openalex",
]
