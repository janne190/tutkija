from __future__ import annotations

from typing import Optional
import xml.etree.ElementTree as ET

import httpx

from . import Paper
from .http_client import create_http_client

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def query_pubmed(
    topic: str,
    *,
    max_results: int = 200,
    client: Optional[httpx.Client] = None,
) -> list[Paper]:
    """Query PubMed for the topic using E-utilities."""

    if not topic.strip():
        raise ValueError("topic must be a non-empty string")

    http_client = client or create_http_client()
    close_client = client is None

    try:
        search_resp = http_client.get(
            ESEARCH_URL,
            params={
                "db": "pubmed",
                "term": topic,
                "retmode": "json",
                "retmax": str(max_results),
            },
        )
        search_resp.raise_for_status()
        search_json = search_resp.json()
        id_list = search_json.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return []

        fetch_resp = http_client.get(
            EFETCH_URL,
            params={
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "xml",
            },
        )
        fetch_resp.raise_for_status()
    finally:
        if close_client:
            http_client.close()

    root = ET.fromstring(fetch_resp.text)
    papers: list[Paper] = []
    for article in root.findall(".//PubmedArticle"):
        medline = article.find("MedlineCitation")
        if medline is None:
            continue
        pmid = medline.findtext("PMID", default="").strip()
        article_info = medline.find("Article")
        if article_info is None:
            continue

        title = article_info.findtext("ArticleTitle", default="")
        abstract_chunks: list[str] = []
        for abstract_el in article_info.findall("Abstract/AbstractText"):
            abstract_chunks.append("".join(abstract_el.itertext()))
        abstract = " ".join(abstract_chunks)

        authors: list[str] = []
        for author in article_info.findall("AuthorList/Author"):
            last = author.findtext("LastName")
            fore = author.findtext("ForeName") or author.findtext("Initials")
            name = " ".join(filter(None, [fore, last]))
            if name:
                authors.append(name)

        journal = article_info.findtext("Journal/Title", default="")
        doi = ""
        for elocation in article_info.findall("ELocationID"):
            if elocation.attrib.get("EIdType", "").lower() == "doi" and elocation.text:
                doi = elocation.text
                break

        year = _extract_year(article_info)
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

        paper = Paper.from_parts(
            id=pmid or url,
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            venue=journal,
            doi=doi,
            url=url,
            source="pubmed",
        )
        papers.append(paper)
    return papers


def _extract_year(article_info: ET.Element) -> str | int | None:
    article_date = article_info.find("ArticleDate")
    if article_date is not None:
        year = article_date.findtext("Year")
        if year:
            return year
    pub_date = article_info.find("Journal/JournalIssue/PubDate")
    if pub_date is not None:
        year = pub_date.findtext("Year")
        if year:
            return year
        medline_date = pub_date.findtext("MedlineDate")
        if medline_date:
            return medline_date
    return None
