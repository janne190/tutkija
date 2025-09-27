"""Search package public exports."""

from .types import Paper, clean_text, parse_year
from .merge import MergeStats

__all__ = ["Paper", "clean_text", "parse_year", "MergeStats"]
