"""Tests for the PDF subcommands."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from la_pkg import cli as cli_module


def test_pdf_discover_from_parquet(tmp_path: Path) -> None:
    runner = CliRunner()
    records = [
        {
            "id": "oa-1",
            "title": "ArXiv sample",
            "url": "https://arxiv.org/abs/2101.00001",
            "doi": "10.1/demo1",
            "source": "openalex",
        },
        {
            "id": "pm-1",
            "title": "PubMed Central sample",
            "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7654321/",
            "doi": "",
            "source": "pubmed",
        },
        {
            "id": "ua-1",
            "title": "Unpaywall candidate",
            "url": "https://example.org/article",
            "doi": "10.2/demo2",
            "source": "openalex",
        },
    ]
    in_path = tmp_path / "merged.parquet"
    pd.DataFrame(records).to_parquet(in_path, index=False)
    out_path = tmp_path / "pdf_index.parquet"

    cli_result = runner.invoke(
        cli_module.app,
        [
            "pdf",
            "discover",
            "--in",
            str(in_path),
            "--out",
            str(out_path),
        ],
    )

    assert cli_result.exit_code == 0, cli_result.output
    assert "PDF discovery OK" in cli_result.output
    assert out_path.exists()

    frame = pd.read_parquet(out_path)
    assert frame.loc[0, "pdf_provider"] == "arxiv"
    assert frame.loc[0, "pdf_provider_url"].endswith("2101.00001.pdf")
    assert frame.loc[0, "pdf_needs_unpaywall_email"] is False

    assert frame.loc[1, "pdf_provider"] == "pmc"
    assert frame.loc[1, "pdf_provider_url"].endswith("/pdf")

    assert frame.loc[2, "pdf_provider"] == "unpaywall"
    assert pd.isna(frame.loc[2, "pdf_provider_url"])
    assert frame.loc[2, "pdf_needs_unpaywall_email"] is True
    assert frame["has_fulltext"].tolist() == [False, False, False]
    assert set(frame["pdf_discovery_source"]) == {"metadata"}


def test_pdf_discover_seed_fallback(tmp_path: Path) -> None:
    runner = CliRunner()
    seed_csv = tmp_path / "seed_urls.csv"
    seed_csv.write_text(
        "title,url,doi\nSeed article,https://arxiv.org/abs/9876.5432,10.9/demo\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "pdf_index.parquet"

    cli_result = runner.invoke(
        cli_module.app,
        [
            "pdf",
            "discover",
            "--in",
            str(tmp_path / "missing.parquet"),
            "--seed-csv",
            str(seed_csv),
            "--out",
            str(out_path),
        ],
    )

    assert cli_result.exit_code == 0, cli_result.output
    frame = pd.read_parquet(out_path)
    assert frame.loc[0, "pdf_provider"] == "arxiv"
    assert frame.loc[0, "source"] == "seeds"
    assert frame.loc[0, "pdf_discovery_source"] == "seeds"


def test_pdf_discover_limit_truncates_metadata(tmp_path: Path) -> None:
    runner = CliRunner()
    records = [
        {
            "id": f"oa-{idx}",
            "title": f"Article {idx}",
            "url": f"https://arxiv.org/abs/2101.0000{idx}",
            "doi": f"10.1/demo{idx}",
            "source": "openalex",
        }
        for idx in range(4)
    ]
    in_path = tmp_path / "merged.parquet"
    pd.DataFrame(records).to_parquet(in_path, index=False)
    out_path = tmp_path / "pdf_index.parquet"

    cli_result = runner.invoke(
        cli_module.app,
        [
            "pdf",
            "discover",
            "--in",
            str(in_path),
            "--out",
            str(out_path),
            "--limit",
            "2",
        ],
    )

    assert cli_result.exit_code == 0, cli_result.output
    frame = pd.read_parquet(out_path)
    assert len(frame) == 2
    assert frame["id"].tolist() == ["oa-0", "oa-1"]


def test_pdf_discover_limit_truncates_seed_source(tmp_path: Path) -> None:
    runner = CliRunner()
    seed_csv = tmp_path / "seed_urls.csv"
    rows = [
        "title,url,doi",
        "First seed,https://arxiv.org/abs/1234.5678,10.9/demo1",
        "Second seed,https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7654321/,",
    ]
    seed_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    out_path = tmp_path / "pdf_index.parquet"

    cli_result = runner.invoke(
        cli_module.app,
        [
            "pdf",
            "discover",
            "--in",
            str(tmp_path / "missing.parquet"),
            "--seed-csv",
            str(seed_csv),
            "--out",
            str(out_path),
            "--limit",
            "1",
        ],
    )

    assert cli_result.exit_code == 0, cli_result.output
    frame = pd.read_parquet(out_path)
    assert len(frame) == 1
    assert frame.loc[0, "title"] == "First seed"
    assert frame.loc[0, "pdf_discovery_source"] == "seeds"


def test_pdf_download_updates_index(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    index_path = tmp_path / "pdf_index.parquet"
    frame = pd.DataFrame(
        [
            {
                "id": "oa-1",
                "title": "Example",
                "doi": "10.1/demo",
                "url": "https://example.org/demo",
                "source": "openalex",
            }
        ]
    )
    frame.to_parquet(index_path, index=False)

    recorded: dict[str, object] = {}

    def fake_download(df, out_dir, audit_csv, *, timeout_s, retries, throttle_ms, unpaywall_email):  # type: ignore[override]
        recorded["df_shape"] = df.shape
        recorded["out_dir"] = out_dir
        recorded["audit_csv"] = audit_csv
        recorded["timeout_s"] = timeout_s
        recorded["retries"] = retries
        recorded["throttle_ms"] = throttle_ms
        recorded["unpaywall_email"] = unpaywall_email
        out_dir.mkdir(parents=True, exist_ok=True)
        new_df = df.copy()
        new_df["pdf_path"] = [str(out_dir / "oa-1.pdf")]
        new_df["pdf_license"] = ["cc-by"]
        new_df["has_fulltext"] = [True]
        return new_df

    import la_pkg.pdf.download as download_mod

    monkeypatch.setattr(download_mod, "download_all", fake_download)

    pdf_dir = tmp_path / "pdfs"
    audit_csv = tmp_path / "audit.csv"

    cli_result = runner.invoke(
        cli_module.app,
        [
            "pdf",
            "download",
            "--in",
            str(index_path),
            "--pdf-dir",
            str(pdf_dir),
            "--audit",
            str(audit_csv),
            "--mailto",
            "cli@example.com",
            "--timeout",
            "45",
            "--retries",
            "3",
            "--throttle",
            "50",
        ],
    )

    assert cli_result.exit_code == 0, cli_result.output
    assert "PDF download OK" in cli_result.output
    updated = pd.read_parquet(index_path)
    assert updated.loc[0, "pdf_path"].endswith("oa-1.pdf")
    assert updated.loc[0, "has_fulltext"] is True

    assert recorded["df_shape"] == (1, 5)
    assert recorded["out_dir"] == pdf_dir
    assert recorded["audit_csv"] == audit_csv
    assert recorded["timeout_s"] == 45
    assert recorded["retries"] == 3
    assert recorded["throttle_ms"] == 50
    assert recorded["unpaywall_email"] == "cli@example.com"
