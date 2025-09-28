"""Tests for the parse pipeline helper."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from la_pkg.parse.run import parse_all

SAMPLE_TEI = """
<TEI xmlns=\"http://www.tei-c.org/ns/1.0\">
  <teiHeader>
    <fileDesc><titleStmt><title>Parsed Title</title></titleStmt></fileDesc>
    <profileDesc><abstract><p>Summary text.</p></abstract></profileDesc>
  </teiHeader>
</TEI>
"""


class DummyClient:
    def __init__(self, *args, **kwargs):  # type: ignore[override]
        self.calls: list[Path] = []

    def __enter__(self) -> "DummyClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - nothing to close
        return None

    def process_fulltext(self, pdf_path: Path) -> str:
        self.calls.append(pdf_path)
        return SAMPLE_TEI


def test_parse_all_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "file.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")

    frame = pd.DataFrame(
        [
            {
                "id": "row-1",
                "doi": "10.1000/abc",
                "pdf_path": str(pdf_path),
            }
        ]
    )

    import la_pkg.parse.run as run_mod

    monkeypatch.setattr(run_mod, "GrobidClient", lambda *a, **k: DummyClient())

    parsed_dir = tmp_path / "parsed"
    err_log = tmp_path / "errors.csv"
    result = parse_all(frame, parsed_dir, "http://localhost:8070", err_log)

    assert bool(result.loc[0, "parsed_ok"])
    tei_path = Path(result.loc[0, "parsed_xml_path"])
    txt_path = Path(result.loc[0, "parsed_txt_path"])
    assert tei_path.exists()
    assert txt_path.exists()
    assert "Parsed Title" in tei_path.read_text(encoding="utf-8")
    assert txt_path.read_text(encoding="utf-8").startswith("# Parsed Title")
    assert not err_log.exists()
