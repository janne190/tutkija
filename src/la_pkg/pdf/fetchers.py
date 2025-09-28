"""Helpers for locating PDF URLs from known providers."""

from __future__ import annotations

import re
from typing import Mapping, Optional

import httpx

ARXIV_RE = re.compile(r"(arxiv\.org/abs/|^arXiv:)(?P<id>[\w.\-]+)", re.I)
PMC_RE = re.compile(r"/pmc/articles/(PMC\d+)/", re.I)


def _get_field(row: Mapping[str, object], key: str) -> str:
    value = row.get(key) if isinstance(row, Mapping) else getattr(row, key, "")
    if value is None:
        return ""
    return str(value)


def arxiv_pdf_url(row: Mapping[str, object]) -> Optional[str]:
    """Return the canonical PDF URL for an arXiv identifier."""

    text = f"{_get_field(row, 'url')} {_get_field(row, 'source')} {_get_field(row, 'title')}"
    match = ARXIV_RE.search(text)
    if not match:
        return None
    return f"https://arxiv.org/pdf/{match.group('id')}.pdf"


def pmc_pdf_url(row: Mapping[str, object]) -> Optional[str]:
    """Return the PDF endpoint for a PubMed Central record."""

    url = _get_field(row, "url")
    match = PMC_RE.search(url)
    if not match:
        return None
    pmcid = match.group(1)
    return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf"


def unpaywall_pdf_url(
    doi: str,
    email: str,
    client: httpx.Client,
) -> tuple[Optional[str], Optional[str]]:
    """Resolve a PDF via the Unpaywall API."""

    if not doi or not email:
        return (None, None)
    try:
        response = client.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": email},
            timeout=15,
        )
    except httpx.HTTPError:
        return (None, None)
    if response.status_code != 200:
        return (None, None)
    try:
        data = response.json()
    except ValueError:
        return (None, None)
    location = data.get("best_oa_location") or {}
    url_for_pdf = location.get("url_for_pdf")
    license_info = location.get("license")
    return (url_for_pdf, license_info)
