"""Thin client for interacting with a GROBID service."""

from __future__ import annotations

from pathlib import Path

import httpx


class GrobidClient:
    """Wrap the HTTP interaction with a running GROBID container."""

    def __init__(self, base: str = "http://localhost:8070", timeout: int = 60) -> None:
        self.base = base.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def process_fulltext(self, pdf_path: Path) -> str:
        """Send a PDF to GROBID and return the TEI XML response."""

        files = {
            "input": (pdf_path.name, pdf_path.read_bytes(), "application/pdf"),
        }
        params = {"consolidateHeader": "1", "teiCoordinates": "1"}
        response = self.client.post(
            f"{self.base}/api/processFulltextDocument",
            files=files,
            params=params,
        )
        response.raise_for_status()
        return response.text

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "GrobidClient":  # pragma: no cover - context manager sugar
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - context manager sugar
        self.close()
