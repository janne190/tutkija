"""Smoke tests for the la CLI."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from la_pkg import cli as cli_module
from la_pkg.search import Paper
from la_pkg.search.openalex import OpenAlexSearchResult, SearchMetrics


def _resolve_cli() -> str:
    candidates = [
        Path(sys.executable).with_name("la.exe"),
        Path(sys.executable).with_name("la"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    which_path = shutil.which("la")
    if which_path:
        return which_path
    raise FileNotFoundError("la entrypoint puuttuu testiajosta")


def test_la_hello_runs() -> None:
    """CLI should emit Tutkija banner and config template."""
    cli_path = _resolve_cli()
    out = subprocess.check_output([cli_path, "hello"], text=True)
    assert "Tutkija, konfiguraation malli alla" in out


def test_la_search_creates_parquet(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    sample_papers = [
        Paper(
            id="openalex:A1",
            title="Sample paper one",
            abstract="Example abstract",
            authors=["Ada"],
            year=2024,
            venue="Test Venue",
            doi="10.1/a1",
            url="https://example.org/a1",
            source="openalex",
        ),
        Paper(
            id="openalex:A2",
            title="Sample paper two",
            authors=[],
            year=2023,
            venue="",
            doi="",
            url="https://example.org/a2",
            source="openalex",
        ),
    ]
    metrics = SearchMetrics(
        topic="demo",
        found=4,
        unique=2,
        with_doi=1,
        query_used="demo topic",
        fallback_used="mapped",
        language_used="any",
        queries_tried=["mapped:any:demo topic"],
    )
    result = OpenAlexSearchResult(papers=sample_papers, metrics=metrics)

    recorded: dict[str, object] = {}

    def fake_query(topic: str, *, limit=None, language="auto"):  # type: ignore[override]
        recorded["topic"] = topic
        recorded["limit"] = limit
        recorded["language"] = language
        return result

    def fake_append(log_metrics: SearchMetrics, *, output_path: Path):
        recorded["metrics"] = log_metrics
        recorded["output_path"] = output_path

    import la_pkg.search.openalex as openalex_mod

    monkeypatch.setattr(openalex_mod, "query_openalex", fake_query)
    monkeypatch.setattr(openalex_mod, "append_audit_log", fake_append)

    output_path = tmp_path / "search.parquet"
    cli_result = runner.invoke(
        cli_module.app,
        [
            "search",
            "--topic",
            "demo topic",
            "--out",
            str(output_path),
            "--limit",
            "5",
            "--lang",
            "en",
        ],
    )
    assert cli_result.exit_code == 0, cli_result.output
    assert "OpenAlex search OK" in cli_result.output
    assert "queries_tried=1" in cli_result.output
    assert "fallback=mapped" in cli_result.output
    assert recorded["topic"] == "demo topic"
    assert recorded["limit"] == 5
    assert recorded["language"] == "en"

    assert output_path.exists()
    frame = pd.read_parquet(output_path)
    assert len(frame) == 2
    assert set(frame.columns) >= {"id", "title", "abstract", "authors", "year"}

    assert isinstance(recorded.get("metrics"), SearchMetrics)
    assert recorded.get("output_path") == output_path


def test_la_search_all_creates_merge(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()

    oa_metrics = SearchMetrics(
        topic="demo",
        found=1,
        unique=1,
        with_doi=1,
        query_used="demo",
        fallback_used="original",
        language_used="en",
        queries_tried=["original:en:demo"],
    )
    oa_papers = [
        Paper(
            id="oa-1",
            title="Paper OA",
            abstract="",
            authors=["Alice"],
            year=2024,
            venue="Journal",
            doi="10.1/demo",
            url="https://example.org/oa",
            source="openalex",
        ),
    ]
    oa_result = OpenAlexSearchResult(papers=oa_papers, metrics=oa_metrics)

    def fake_query_openalex(topic: str, *, limit=None, language="auto"):
        return oa_result

    pubmed_papers = [
        Paper(
            id="pm-1",
            title="Paper PM",
            authors=["Bob"],
            year=2023,
            doi="",
            url="https://pubmed.ncbi.nlm.nih.gov/pm-1/",
            source="pubmed",
        ),
    ]

    def fake_query_pubmed(topic: str, *, max_results=200):
        return pubmed_papers

    arxiv_papers = [
        Paper(
            id="ax-1",
            title="Paper AX",
            authors=["Carol"],
            year=2022,
            doi="",
            url="https://arxiv.org/abs/ax-1",
            source="arxiv",
        ),
    ]

    def fake_query_arxiv(topic: str, *, max_results=200):
        return arxiv_papers

    def fake_merge_and_filter(oa, pm, ax):
        df = pd.DataFrame([paper.model_dump() for paper in oa + pm + ax])
        df["reasons"] = [[] for _ in range(len(df))]
        df["score"] = [0.5 for _ in range(len(df))]
        stats = {
            "per_source": {"openalex": 1, "pubmed": 1, "arxiv": 1},
            "dup_doi": 0,
            "dup_title": 0,
            "filtered": 0,
            "removed": [],
        }
        return df, stats

    import la_pkg.search.arxiv as arxiv_mod
    import la_pkg.search.merge as merge_mod
    import la_pkg.search.openalex as openalex_mod
    import la_pkg.search.pubmed as pubmed_mod

    monkeypatch.setattr(openalex_mod, "query_openalex", fake_query_openalex)
    monkeypatch.setattr(pubmed_mod, "query_pubmed", fake_query_pubmed)
    monkeypatch.setattr(arxiv_mod, "query_arxiv", fake_query_arxiv)
    monkeypatch.setattr(merge_mod, "merge_and_filter", fake_merge_and_filter)
    monkeypatch.chdir(tmp_path)
    out_path = Path("merged.parquet")
    cli_result = runner.invoke(
        cli_module.app,
        ["search-all", "--topic", "demo", "--out", str(out_path), "--save-single"],
    )
    assert cli_result.exit_code == 0, cli_result.output
    assert "Multisource OK" in cli_result.output
    assert out_path.exists()
    assert Path("data/cache/merge_log.csv").exists()

    import la_pkg.search.arxiv as arxiv_mod
    import la_pkg.search.merge as merge_mod
    import la_pkg.search.openalex as openalex_mod
    import la_pkg.search.pubmed as pubmed_mod

    monkeypatch.setattr(openalex_mod, "query_openalex", fake_query_openalex)
    monkeypatch.setattr(pubmed_mod, "query_pubmed", fake_query_pubmed)
    monkeypatch.setattr(arxiv_mod, "query_arxiv", fake_query_arxiv)
    monkeypatch.setattr(merge_mod, "merge_and_filter", fake_merge_and_filter)
    monkeypatch.chdir(tmp_path)
    out_path = Path("merged.parquet")
    cli_result = runner.invoke(
        cli_module.app,
        ["search-all", "--topic", "demo", "--out", str(out_path), "--save-single"],
    )
    assert cli_result.exit_code == 0, cli_result.output
    assert "Multisource OK" in cli_result.output
    assert out_path.exists()
    assert Path("data/cache/merge_log.csv").exists()
    pm_papers = [
        Paper(
            id="pubmed:P1",
            title="PubMed paper",
            abstract="PubMed abstract",
            authors=["Bob"],
            year=2024,
            venue="PubMed Venue",
            doi="10.1/p1",
            url="https://example.org/pm",
            source="pubmed",
        )
    ]
    ax_papers = [
        Paper(
            id="arxiv:2401.1234",
            title="arXiv paper",
            abstract="arXiv abstract",
            authors=["Carol"],
            year=2024,
            venue="arXiv cs.AI",
            doi="10.1/ax1",
            url="https://example.org/ax",
            source="arxiv",
        )
    ]
    papers = oa_papers + pm_papers + ax_papers
    df = pd.DataFrame([paper.model_dump() for paper in papers])

    def merge_fn(out_path: Path):
        stats = {
            "per_source": {"openalex": 1, "pubmed": 1, "arxiv": 1},
            "dup_doi": 0,
            "dup_title": 0,
            "filtered": 0,
        }
        return df, stats
        return df, stats

    import la_pkg.search.arxiv as arxiv_mod
    import la_pkg.search.merge as merge_mod
    import la_pkg.search.openalex as openalex_mod
    import la_pkg.search.pubmed as pubmed_mod

    monkeypatch.setattr(openalex_mod, "query_openalex", fake_query_openalex)
    monkeypatch.setattr(pubmed_mod, "query_pubmed", fake_query_pubmed)
    monkeypatch.setattr(arxiv_mod, "query_arxiv", fake_query_arxiv)
    monkeypatch.setattr(merge_mod, "merge_and_filter", fake_merge_and_filter)
    monkeypatch.chdir(tmp_path)
    out_path = Path("merged.parquet")
    cli_result = runner.invoke(
        cli_module.app,
        ["search-all", "--topic", "demo", "--out", str(out_path), "--save-single"],
    )
    assert cli_result.exit_code == 0, cli_result.output
    assert "Multisource OK" in cli_result.output
    assert out_path.exists()
    assert Path("data/cache/merge_log.csv").exists()
