# src/la_pkg/write/linkcheck.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import bibtexparser
import httpx
import pandas as pd

logger = logging.getLogger(__name__)


def check_url(url: str, client: httpx.Client) -> tuple[str, int, str]:
    """Check a single URL and return its status."""
    try:
        response = client.head(url, timeout=10, follow_redirects=True)
        return str(response.url), response.status_code, "OK"
    except httpx.HTTPStatusError as e:
        return str(e.request.url), e.response.status_code, f"HTTP Error: {e}"
    except httpx.RequestError as e:
        return str(e.request.url), 0, f"Request Error: {e}"


def extract_urls_from_bib(bib_path: Path) -> Iterable[str]:
    """Extract all DOI and URL fields from a BibTeX file."""
    if not bib_path.exists():
        raise FileNotFoundError(f"BibTeX file not found: {bib_path}")

    with open(bib_path, encoding="utf-8") as bibfile:
        db = bibtexparser.load(bibfile)

    for entry in db.entries:
        if "doi" in entry:
            yield f"https://doi.org/{entry['doi']}"
        if "url" in entry:
            yield entry["url"]


def run_linkcheck(
    bib_path: Path,
    output_log_path: Path,
) -> None:
    """
    Checks all URLs in a BibTeX file and logs the results.
    """
    urls = set(extract_urls_from_bib(bib_path))
    results = []

    with httpx.Client() as client:
        for url in urls:
            final_url, status, reason = check_url(url, client)
            results.append(
                {
                    "original_url": url,
                    "final_url": final_url,
                    "status_code": status,
                    "reason": reason,
                }
            )

    output_log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_log_path, index=False)
