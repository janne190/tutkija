"""Tests for the PubMed search adapter."""

from __future__ import annotations

import httpx
import pytest

from la_pkg.search import Paper
from la_pkg.search.pubmed import query_pubmed


@pytest.fixture()
def pubmed_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "eutils.ncbi.nlm.nih.gov" and request.url.path.endswith(
            "esearch.fcgi"
        ):
            payload = {"esearchresult": {"idlist": ["12345678", "23456789"]}}
            return httpx.Response(200, json=payload)
        if request.url.host == "eutils.ncbi.nlm.nih.gov" and request.url.path.endswith(
            "efetch.fcgi"
        ):
            xml_body = """
                <PubmedArticleSet>
                  <PubmedArticle>
                    <MedlineCitation>
                      <PMID>12345678</PMID>
                      <Article>
                        <ArticleTitle>Example PubMed Article</ArticleTitle>
                        <Abstract>
                          <AbstractText>Short abstract.</AbstractText>
                        </Abstract>
                        <Journal>
                          <JournalIssue>
                            <PubDate>
                              <Year>2022</Year>
                            </PubDate>
                          </JournalIssue>
                          <Title>Journal of Examples</Title>
                        </Journal>
                        <AuthorList>
                          <Author>
                            <ForeName>Ada</ForeName>
                            <LastName>Lovelace</LastName>
                          </Author>
                          <Author>
                            <ForeName>Grace</ForeName>
                            <LastName>Hopper</LastName>
                          </Author>
                        </AuthorList>
                        <ELocationID EIdType="doi">10.1000/example</ELocationID>
                      </Article>
                    </MedlineCitation>
                  </PubmedArticle>
                  <PubmedArticle>
                    <MedlineCitation>
                      <PMID>23456789</PMID>
                      <Article>
                        <ArticleTitle>Article Without DOI</ArticleTitle>
                        <Journal>
                          <JournalIssue>
                            <PubDate>
                              <MedlineDate>2020 Winter</MedlineDate>
                            </PubDate>
                          </JournalIssue>
                          <Title>Seasonal Science</Title>
                        </Journal>
                        <AuthorList>
                          <Author>
                            <ForeName>Jane</ForeName>
                            <LastName>Doe</LastName>
                          </Author>
                        </AuthorList>
                      </Article>
                    </MedlineCitation>
                  </PubmedArticle>
                </PubmedArticleSet>
            """
            return httpx.Response(
                200, text="\n".join(line.strip() for line in xml_body.splitlines())
            )
        raise AssertionError(f"Unexpected URL: {request.url}")

    return httpx.MockTransport(handler)


def test_query_pubmed_parses_results(pubmed_transport: httpx.MockTransport) -> None:
    with httpx.Client(transport=pubmed_transport) as client:
        papers = query_pubmed("example topic", client=client)

    assert len(papers) == 2
    first = papers[0]
    assert isinstance(first, Paper)
    assert first.id == "12345678"
    assert first.title == "Example PubMed Article"
    assert first.authors == ["Ada Lovelace", "Grace Hopper"]
    assert first.year == 2022
    assert first.doi == "10.1000/example"
    assert first.source == "pubmed"
    second = papers[1]
    assert second.doi == ""
    assert second.year == 2020
