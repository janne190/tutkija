"""Smoke tests for the la CLI."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from la_pkg import cli as cli_module
from la_pkg.search.openalex import OpenAlexSearchResult, Paper, SearchMetrics


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
            score=12.5,
        ),
        Paper(
            id="openalex:A2",
            title="Sample paper two",
            abstract=None,
            authors=[],
            year=2023,
            venue=None,
            doi=None,
            url="https://example.org/a2",
            score=None,
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

    monkeypatch.setattr(cli_module, "query_openalex", fake_query)
    monkeypatch.setattr(cli_module, "append_audit_log", fake_append)

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
