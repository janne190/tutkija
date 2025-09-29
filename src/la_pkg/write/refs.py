# src/la_pkg/write/refs.py
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional

import bibtexparser
import httpx
import pandas as pd
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.db import BibDatabase

logger = logging.getLogger(__name__)


def get_email_for_entrez() -> str:
    return os.environ.get("EMAIL_FOR_ENTREZ", "anonymous@example.com")


def doi_to_bibtex(doi: str) -> Optional[str]:
    """Fetch BibTeX for a DOI from Crossref."""
    url = f"https://api.crossref.org/works/{doi}/transform/application/x-bibtex"
    try:
        with httpx.Client(follow_redirects=True) as client:
            response = client.get(url, timeout=10)
            response.raise_for_status()
            return response.text
    except httpx.HTTPStatusError as e:
        logger.warning(f"Failed to fetch BibTeX for DOI {doi}: {e}")
        return None
    except httpx.RequestError as e:
        logger.error(f"Request failed for DOI {doi}: {e}")
        return None


def pmid_to_bibtex(pmid: str) -> Optional[str]:
    """Fetch BibTeX for a PMID from Entrez."""
    email = get_email_for_entrez()
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
        f"db=pubmed&id={pmid}&rettype=bibtex&retmode=text&email={email}"
    )
    try:
        with httpx.Client() as client:
            response = client.get(url, timeout=10)
            response.raise_for_status()
            return response.text
    except httpx.HTTPStatusError as e:
        logger.warning(f"Failed to fetch BibTeX for PMID {pmid}: {e}")
        return None
    except httpx.RequestError as e:
        logger.error(f"Request failed for PMID {pmid}: {e}")
        return None


def generate_citekey(entry: dict) -> str:
    """Generate a consistent citekey for a BibTeX entry."""
    if "doi" in entry:
        return f"doi:{entry['doi']}"
    if "pmid" in entry:
        return f"pmid:{entry['pmid']}"
    
    author = entry.get("author", "unknown").split(",")[0].lower()
    year = entry.get("year", "nodate")
    title_word = entry.get("title", "notitle").split(" ")[0].lower()
    
    return f"{author}{year}{title_word}"


def create_bib_database(
    dois: Iterable[str], pmids: Iterable[str]
) -> tuple[BibDatabase, list[dict]]:
    """Create a BibDatabase from lists of DOIs and PMIDs."""
    db = BibDatabase()
    missing_log = []

    all_dois = set(filter(None, dois))
    all_pmids = set(filter(None, pmids))

    for doi in all_dois:
        bibtex_str = doi_to_bibtex(doi)
        if bibtex_str:
            try:
                entry_db = bibtexparser.loads(bibtex_str)
                if entry_db.entries:
                    entry = entry_db.entries[0]
                    entry["ID"] = generate_citekey(entry)
                    db.entries.append(entry)
                else:
                    missing_log.append({"id": doi, "reason": "crossref_empty_response"})
            except Exception as e:
                logger.error(f"Failed to parse BibTeX for DOI {doi}: {e}")
                missing_log.append({"id": doi, "reason": "bibtex_parsing_failed"})
        else:
            missing_log.append({"id": doi, "reason": "fetch_failed"})

    for pmid in all_pmids:
        bibtex_str = pmid_to_bibtex(pmid)
        if bibtex_str:
            try:
                entry_db = bibtexparser.loads(bibtex_str)
                if entry_db.entries:
                    entry = entry_db.entries[0]
                    entry["ID"] = generate_citekey(entry)
                    db.entries.append(entry)
                else:
                    missing_log.append({"id": pmid, "reason": "entrez_empty_response"})
            except Exception as e:
                logger.error(f"Failed to parse BibTeX for PMID {pmid}: {e}")
                missing_log.append({"id": pmid, "reason": "bibtex_parsing_failed"})
        else:
            missing_log.append({"id": pmid, "reason": "fetch_failed"})

    return db, missing_log


def collect_and_write_references(
    parquet_path: Path,
    qa_path: Path,
    output_path: Path,
    missing_log_path: Path,
) -> None:
    """
    Collects DOIs/PMIDs, fetches BibTeX, and writes to a .bib file.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Input parquet file not found: {parquet_path}")
    if not qa_path.exists():
        raise FileNotFoundError(f"QA JSONL file not found: {qa_path}")

    df = pd.read_parquet(parquet_path)
    qa_df = pd.read_json(qa_path, lines=True)

    # Extract DOIs and PMIDs
    dois = set(df["doi"].dropna().unique())
    pmids = set(df["pmid"].dropna().unique())

    # Add citations from QA file
    for citations in qa_df["citations"].dropna():
        for citation in citations:
            if "doi" in citation:
                dois.add(citation["doi"])
            if "pmid" in citation:
                pmids.add(citation["pmid"])

    db, missing_log = create_bib_database(dois, pmids)

    # Write .bib file
    writer = BibTexWriter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as bibfile:
        bibfile.write(writer.write(db))

    # Write missing log
    if missing_log:
        missing_log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(missing_log).to_csv(missing_log_path, index=False)
