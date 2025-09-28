"""Tests for the TEI extraction helpers."""

from __future__ import annotations

from la_pkg.parse.tei_extract import tei_to_title_abstract_refs


SAMPLE_TEI = """
<TEI xmlns=\"http://www.tei-c.org/ns/1.0\">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>Sample Article</title>
      </titleStmt>
    </fileDesc>
    <profileDesc>
      <abstract>
        <p>This is a test abstract.</p>
      </abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <back>
      <listBibl>
        <biblStruct><analytic><title>Ref 1</title></analytic></biblStruct>
        <biblStruct><analytic><title>Ref 2</title></analytic></biblStruct>
      </listBibl>
    </back>
  </text>
</TEI>
"""


def test_tei_to_title_abstract_refs_extracts_fields() -> None:
    result = tei_to_title_abstract_refs(SAMPLE_TEI)
    assert result["title"] == "Sample Article"
    assert "test abstract" in result["abstract"]
    assert result["n_refs"] == 2
