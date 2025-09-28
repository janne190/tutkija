"""Unit tests for PDF fetcher helpers."""

from __future__ import annotations

from dataclasses import dataclass

from la_pkg.pdf.fetchers import arxiv_pdf_url, pmc_pdf_url, unpaywall_pdf_url


@dataclass
class DummyResponse:
    status_code: int
    payload: dict[str, object]

    def json(self) -> dict[str, object]:
        return self.payload


class DummyClient:
    def __init__(self, responses: list[DummyResponse]):
        self._responses = responses
        self.calls: list[tuple[str, dict[str, object] | None]] = []

    def get(self, url: str, *, params=None, timeout=None):  # type: ignore[override]
        self.calls.append((url, params))
        if not self._responses:
            raise AssertionError("no more responses queued")
        return self._responses.pop(0)


def test_arxiv_pdf_url_detects_abs_link() -> None:
    row = {"url": "https://arxiv.org/abs/1234.56789", "source": "arxiv"}
    assert (
        arxiv_pdf_url(row)
        == "https://arxiv.org/pdf/1234.56789.pdf"
    )


def test_pmc_pdf_url_handles_case_insensitive_ids() -> None:
    row = {"url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1234567/"}
    assert (
        pmc_pdf_url(row)
        == "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1234567/pdf"
    )


def test_unpaywall_pdf_url_returns_tuple_when_available() -> None:
    responses = [
        DummyResponse(
            200,
            {
                "best_oa_location": {
                    "url_for_pdf": "https://example.org/demo.pdf",
                    "license": "cc-by",
                }
            },
        )
    ]
    client = DummyClient(responses)
    url, license_info = unpaywall_pdf_url("10.123/demo", "user@example.com", client)  # type: ignore[arg-type]
    assert url == "https://example.org/demo.pdf"
    assert license_info == "cc-by"
    assert client.calls[0][0].startswith("https://api.unpaywall.org/v2/10.123/demo")


def test_unpaywall_pdf_url_handles_non_200() -> None:
    responses = [DummyResponse(404, {})]
    client = DummyClient(responses)
    url, license_info = unpaywall_pdf_url("10.999/demo", "user@example.com", client)  # type: ignore[arg-type]
    assert url is None
    assert license_info is None
