"""PDF fetching utilities for Tutkija."""

from .download import download_all
from .fetchers import arxiv_pdf_url, pmc_pdf_url, unpaywall_pdf_url

__all__ = [
    "download_all",
    "arxiv_pdf_url",
    "pmc_pdf_url",
    "unpaywall_pdf_url",
]
