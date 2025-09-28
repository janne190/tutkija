"""Tests for the GrobidClient wrapper."""

from __future__ import annotations

from pathlib import Path

from la_pkg.parse.grobid_client import GrobidClient


class DummyResponse:
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code != 200:
            raise RuntimeError("bad status")


class DummyHttpxClient:
    def __init__(self, *args, **kwargs):  # type: ignore[override]
        self.calls: list[tuple[str, dict[str, object], dict[str, object]]] = []

    def post(self, url: str, *, files=None, params=None):  # type: ignore[override]
        self.calls.append((url, files or {}, params or {}))
        return DummyResponse(200, "<TEI>ok</TEI>")

    def close(self) -> None:  # pragma: no cover - nothing to close
        return None


def test_process_fulltext_posts_pdf(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")

    import la_pkg.parse.grobid_client as gc_mod

    dummy_client = DummyHttpxClient()
    monkeypatch.setattr(gc_mod.httpx, "Client", lambda *a, **k: dummy_client)

    client = GrobidClient(base="http://localhost:8070", timeout=5)
    try:
        tei = client.process_fulltext(pdf_path)
    finally:
        client.close()

    assert tei == "<TEI>ok</TEI>"
    assert dummy_client.calls
    url, files, params = dummy_client.calls[0]
    assert url.endswith("/api/processFulltextDocument")
    assert "input" in files
    assert params.get("consolidateHeader") == "1"
