"""Tests for the bulk PDF downloader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from la_pkg.pdf import download_all


@dataclass
class FakeResponse:
    status_code: int
    headers: dict[str, str]
    content: bytes = b""
    json_data: dict[str, Any] | None = None

    def json(self) -> dict[str, Any]:
        if self.json_data is None:
            raise ValueError("no json payload")
        return self.json_data


class FakeClient:
    def __init__(self, responses: list[FakeResponse]):
        self._responses = responses
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def __enter__(self) -> "FakeClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - nothing to close
        return None

    def get(self, url: str, params=None, timeout=None):  # type: ignore[override]
        self.calls.append((url, params))
        if not self._responses:
            raise AssertionError("no fake responses left")
        return self._responses.pop(0)


def test_download_all_downloads_and_logs(tmp_path: Path, monkeypatch) -> None:
    pdf_dir = tmp_path / "pdfs"
    audit_csv = tmp_path / "audit.csv"
    responses = [
        FakeResponse(200, {"content-type": "application/pdf"}, content=b"%PDF-1"),
        FakeResponse(
            200,
            {},
            json_data={
                "best_oa_location": {
                    "url_for_pdf": "https://files.example.org/demo.pdf",
                    "license": "cc-by",
                }
            },
        ),
        FakeResponse(200, {"content-type": "application/pdf"}, content=b"%PDF-2"),
    ]

    fake_client = FakeClient(responses)

    import la_pkg.pdf.download as download_mod

    monkeypatch.setattr(download_mod.httpx, "Client", lambda *a, **k: fake_client)
    monkeypatch.setattr(download_mod.time, "sleep", lambda _seconds: None)

    frame = pd.DataFrame(
        [
            {
                "id": "ax-1",
                "url": "https://arxiv.org/abs/2101.12345",
                "source": "arxiv",
                "doi": "",
            },
            {
                "id": "doi-1",
                "url": "https://example.org/record",
                "doi": "10.1000/demo",
                "source": "crossref",
            },
            {
                "id": "missing",
                "source": "unknown",
            },
        ]
    )

    result = download_all(
        frame,
        pdf_dir,
        audit_csv,
        timeout_s=10,
        retries=1,
        throttle_ms=0,
        unpaywall_email="user@example.com",
    )

    assert len(result) == 3
    assert bool(result.loc[0, "has_fulltext"])
    assert bool(result.loc[1, "has_fulltext"])
    assert not bool(result.loc[2, "has_fulltext"])

    first_pdf = Path(result.loc[0, "pdf_path"])
    second_pdf = Path(result.loc[1, "pdf_path"])
    assert first_pdf.exists()
    assert second_pdf.exists()
    assert first_pdf.read_bytes().startswith(b"%PDF")
    assert second_pdf.read_bytes().startswith(b"%PDF")

    audit = pd.read_csv(audit_csv)
    assert list(audit["status"]) == ["downloaded", "downloaded", "skipped"]
    assert audit.loc[1, "license"] == "cc-by"

    assert not responses  # all fake responses were consumed
