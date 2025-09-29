# tests/write/test_refs.py
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from bibtexparser.db import BibDatabase

from la_pkg.write.refs import (
    collect_and_write_references,
    create_bib_database,
    doi_to_bibtex,
    pmid_to_bibtex,
)

MOCK_DOI_BIBTEX = """@article{some_doi,
    title = {Title for DOI},
    author = {Doe, John},
    year = {2023},
    journal = {Journal of DOI}
}"""

MOCK_PMID_BIBTEX = """@article{some_pmid,
    title = {Title for PMID},
    author = {Smith, Jane},
    year = {2024},
    journal = {Journal of PMID}
}"""


@pytest.mark.network
def test_doi_to_bibtex_real():
    # Simple integration test with a real DOI
    bibtex = doi_to_bibtex("10.1038/s41586-021-03491-6")
    assert bibtex is not None
    assert "DeepMind" in bibtex


@patch("la_pkg.write.refs.httpx.Client.get")
def test_doi_to_bibtex_mocked(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.text = MOCK_DOI_BIBTEX
    bibtex = doi_to_bibtex("10.1234/mock")
    assert bibtex == MOCK_DOI_BIBTEX


@patch("la_pkg.write.refs.httpx.Client.get")
def test_pmid_to_bibtex_mocked(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.text = MOCK_PMID_BIBTEX
    bibtex = pmid_to_bibtex("12345678")
    assert bibtex == MOCK_PMID_BIBTEX


@patch("la_pkg.write.refs.doi_to_bibtex", return_value=MOCK_DOI_BIBTEX)
@patch("la_pkg.write.refs.pmid_to_bibtex", return_value=MOCK_PMID_BIBTEX)
def test_create_bib_database(mock_pmid, mock_doi):
    db, missing = create_bib_database(
        dois=["10.1234/mock"], pmids=["12345678"]
    )
    assert isinstance(db, BibDatabase)
    assert len(db.entries) == 2
    assert not missing
    assert db.entries[0]["title"] == "Title for DOI"
    assert db.entries[1]["title"] == "Title for PMID"


def test_collect_and_write_references(tmp_path: Path):
    parquet_path = tmp_path / "index.parquet"
    qa_path = tmp_path / "qa.jsonl"
    bib_path = tmp_path / "refs.bib"
    missing_log_path = tmp_path / "missing.csv"

    # Create dummy data
    pd.DataFrame({"doi": ["10.1234/mock1"], "pmid": [None]}).to_parquet(parquet_path)
    with open(qa_path, "w") as f:
        f.write('{"citations": [{"doi": "10.1234/mock2"}]}\\n')
        f.write('{"citations": [{"pmid": "12345"}]}\\n')

    with patch("la_pkg.write.refs.doi_to_bibtex") as mock_doi, patch(
        "la_pkg.write.refs.pmid_to_bibtex"
    ) as mock_pmid:
        mock_doi.side_effect = [MOCK_DOI_BIBTEX, MOCK_DOI_BIBTEX.replace("DOI", "DOI2")]
        mock_pmid.return_value = MOCK_PMID_BIBTEX

        collect_and_write_references(parquet_path, qa_path, bib_path, missing_log_path)

        assert bib_path.exists()
        content = bib_path.read_text()
        assert "Title for DOI" in content
        assert "Title for DOI2" in content
        assert "Title for PMID" in content
        assert not missing_log_path.exists()
