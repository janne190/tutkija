#!/usr/bin/env python3
"""Create deterministic Phase 4 sample artefacts without external deps."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
LOGS_DIR = DATA_DIR / "logs"
PDF_DIR = DATA_DIR / "pdfs"
PARSED_DIR = DATA_DIR / "parsed"
SAMPLES_DIR = DATA_DIR / "samples"


@dataclass
class Paper:
    pid: str
    title: str
    doi: str
    source: str
    provider: str
    provider_url: str
    license: str
    needs_email: bool
    has_pdf: bool


BASE_ROWS = [
    Paper(
        pid="arxiv-2101.00001",
        title="Graph Neural Networks for Sampled Data",
        doi="10.5555/arxiv-2101.00001",
        source="arxiv",
        provider="arxiv",
        provider_url="https://arxiv.org/pdf/2101.00001.pdf",
        license="arxiv",
        needs_email=False,
        has_pdf=True,
    ),
    Paper(
        pid="pmc-7654321",
        title="Clinical Insights into Sample Pipelines",
        doi="10.5555/pmc7654321",
        source="pmc",
        provider="pmc",
        provider_url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7654321/pdf",
        license="pmc",
        needs_email=False,
        has_pdf=True,
    ),
    Paper(
        pid="doi-10-1000-demo-1",
        title="Open Access Discovery via Unpaywall",
        doi="10.1000/demo.1",
        source="unpaywall",
        provider="unpaywall",
        provider_url="",
        license="CC-BY-4.0",
        needs_email=True,
        has_pdf=False,
    ),
]


TOPICS = [
    ("Adaptive Control of Sample Pipelines", "10.1000/demo.2"),
    ("Federated Learning for Literature Workflows", "10.1000/demo.3"),
    ("Semi-supervised Screening in Practice", "10.1000/demo.4"),
    ("Benchmarking PDF Parsing", "10.1000/demo.5"),
    ("Maintaining Audit Trails", "10.1000/demo.6"),
    ("Reliable Seeds for PDF Discovery", "10.1000/demo.7"),
    ("RapidFuzz Threshold Experiments", "10.1000/demo.8"),
    ("Fallback Strategies for Metadata", "10.1000/demo.9"),
    ("Evaluating Parsing Latency", "10.1000/demo.10"),
    ("Mitigating Network Constraints", "10.1000/demo.11"),
    ("Sampling Strategies for Phase 4", "10.1000/demo.12"),
    ("PDF Deduplication Heuristics", "10.1000/demo.13"),
    ("Metadata Quality Metrics", "10.1000/demo.14"),
    ("Seed URL Governance", "10.1000/demo.15"),
    ("Graceful CLI Degradation", "10.1000/demo.16"),
    ("Synthetic Artefacts for Testing", "10.1000/demo.17"),
    ("Orchestrating Phase Pipelines", "10.1000/demo.18"),
]


RANDOM = random.Random(42)


def ensure_dirs() -> None:
    for path in (CACHE_DIR, LOGS_DIR, PDF_DIR, PARSED_DIR, SAMPLES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _pdf_filename(pid: str) -> str:
    return pid.replace("/", "_") + ".pdf"


def _tei_dir(pid: str) -> Path:
    return PARSED_DIR / pid.replace("/", "_")


def build_pdf_index(rows: list[Paper]) -> list[dict[str, object]]:
    base_url = "https://example.org/article"
    dataset: list[dict[str, object]] = []
    for idx, paper in enumerate(rows):
        discovery = {
            "id": paper.pid,
            "title": paper.title,
            "url": f"{base_url}/{idx}",
            "doi": paper.doi,
            "source": paper.source,
            "pdf_provider": paper.provider,
            "pdf_provider_url": paper.provider_url,
            "pdf_needs_unpaywall_email": paper.needs_email,
            "pdf_license": paper.license if paper.has_pdf else "",
            "pdf_discovery_source": "synthetic",
        }
        if paper.has_pdf:
            discovery["pdf_path"] = str(PDF_DIR / _pdf_filename(paper.pid))
            discovery["has_fulltext"] = True
        else:
            discovery["pdf_path"] = ""
            discovery["has_fulltext"] = False
        dataset.append(discovery)
    return dataset


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_pdf_audit(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    now = datetime(2024, 1, 15, 9, 0, 0)
    audit_rows: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        elapsed = round(0.4 + (idx % 3) * 0.17, 2)
        status = "downloaded" if row.get("has_fulltext") else "skipped"
        reason = "" if status == "downloaded" else "no_pdf_url"
        bytes_count = 153600 + idx * 1024 if status == "downloaded" else 0
        entry = {
            "paper_id": row["id"],
            "doi": row["doi"],
            "source": row["source"],
            "provider": row["pdf_provider"],
            "pdf_url": row["pdf_provider_url"],
            "license": row["pdf_license"],
            "status": status,
            "reason": reason,
            "http_status": 200 if status == "downloaded" else "",
            "bytes": bytes_count,
            "elapsed_s": elapsed,
            "timestamp": (now + timedelta(seconds=idx * 11)).isoformat(),
        }
        audit_rows.append(entry)
    return audit_rows


def write_minimal_pdf(path: Path, title: str) -> None:
    content = f"BT /F1 24 Tf 72 720 Td ({title}) Tj ET"
    body = "\n".join(
        [
            "%PDF-1.4",
            "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
            "2 0 obj << /Type /Pages /Count 1 /Kids [3 0 R] >> endobj",
            "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj",
            f"4 0 obj << /Length {len(content)} >> stream",
            content,
            "endstream endobj",
            "5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
        ]
    )
    xref_offset = len(body.encode("utf-8")) + 1
    trailer = "\n".join(
        [
            "xref",
            "0 6",
            "0000000000 65535 f ",
            "0000000010 00000 n ",
            "0000000060 00000 n ",
            "0000000113 00000 n ",
            "0000000230 00000 n ",
            "0000000297 00000 n ",
            "trailer << /Root 1 0 R /Size 6 >>",
            "startxref",
            str(xref_offset),
            "%%EOF",
        ]
    )
    path.write_text(body + "\n" + trailer, encoding="latin-1")


def write_minimal_tei(target_dir: Path, title: str, abstract: str) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    tei = f"""<?xml version='1.0' encoding='UTF-8'?>
<TEI xmlns='http://www.tei-c.org/ns/1.0'>
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>{title}</title>
      </titleStmt>
    </fileDesc>
  </teiHeader>
  <text>
    <body>
      <p>{abstract}</p>
    </body>
  </text>
</TEI>
"""
    (target_dir / "tei.xml").write_text(tei, encoding="utf-8")
    summary = f"# {title}\n\n{abstract}\n"
    (target_dir / "text.txt").write_text(summary, encoding="utf-8")


def build_parsed_index(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    parsed_rows: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        has_pdf = bool(row.get("has_fulltext"))
        parsed_rows.append(
            {
                "id": row["id"],
                "pdf_path": row.get("pdf_path", ""),
                "parsed_xml_path": str(_tei_dir(row["id"]) / "tei.xml") if has_pdf else "",
                "parsed_txt_path": str(_tei_dir(row["id"]) / "text.txt") if has_pdf else "",
                "parsed_ok": has_pdf,
            }
        )
    return parsed_rows


def prepare_rows() -> list[Paper]:
    rows = BASE_ROWS.copy()
    for idx, (title, doi) in enumerate(TOPICS, start=1):
        pid = f"seed-{idx:02d}"
        provider = RANDOM.choice(["arxiv", "pmc", "unpaywall"])
        has_pdf = provider != "unpaywall"
        needs_email = provider == "unpaywall"
        provider_url = ""
        license_tag = ""
        if provider == "arxiv":
            provider_url = f"https://arxiv.org/pdf/2401.{idx:05d}.pdf"
            license_tag = "arxiv"
        elif provider == "pmc":
            provider_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC99{idx:04d}/pdf"
            license_tag = "pmc"
        else:
            license_tag = "CC-BY-SA-4.0"
        rows.append(
            Paper(
                pid=pid,
                title=title,
                doi=doi,
                source="openalex",
                provider=provider,
                provider_url=provider_url,
                license=license_tag,
                needs_email=needs_email,
                has_pdf=has_pdf,
            )
        )
    return rows


def create_sample_archive(pdf_index: list[dict[str, object]], audit_rows: list[dict[str, object]], parsed_rows: list[dict[str, object]]) -> Path:
    archive_path = SAMPLES_DIR / "vaihe4_sample.zip"
    with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
        archive.write(CACHE_DIR / "pdf_index.csv", arcname="cache/pdf_index.csv")
        archive.write(LOGS_DIR / "pdf_audit.csv", arcname="logs/pdf_audit.csv")
        archive.write(CACHE_DIR / "parsed_index.csv", arcname="cache/parsed_index.csv")
        meta = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "rows": len(pdf_index),
        }
        archive.writestr("README.txt", "Synthetic Phase 4 sample dataset\n")
        archive.writestr("metadata.json", json.dumps(meta, indent=2))
        for row in pdf_index:
            pdf_path = row.get("pdf_path")
            if pdf_path:
                archive.write(Path(pdf_path), arcname=f"pdfs/{Path(pdf_path).name}")
        for row in parsed_rows:
            if row.get("parsed_ok"):
                pid = row["id"].replace("/", "_")
                tei_dir = _tei_dir(row["id"])
                archive.write(tei_dir / "tei.xml", arcname=f"parsed/{pid}/tei.xml")
                archive.write(tei_dir / "text.txt", arcname=f"parsed/{pid}/text.txt")
    return archive_path


def main() -> None:
    ensure_dirs()
    papers = prepare_rows()
    pdf_index_rows = build_pdf_index(papers)
    audit_rows = build_pdf_audit(pdf_index_rows)
    parsed_rows = build_parsed_index(pdf_index_rows)

    write_csv(CACHE_DIR / "pdf_index.csv", pdf_index_rows)
    write_csv(LOGS_DIR / "pdf_audit.csv", audit_rows)
    write_csv(CACHE_DIR / "parsed_index.csv", parsed_rows)

    for row in pdf_index_rows:
        if row.get("has_fulltext"):
            pdf_path = Path(row["pdf_path"])
            write_minimal_pdf(pdf_path, row["title"][:40])
            abstract = "Synthetic abstract for sample parsing output."
            write_minimal_tei(_tei_dir(row["id"]), row["title"], abstract)

    archive = create_sample_archive(pdf_index_rows, audit_rows, parsed_rows)
    print(f"Sample dataset created at {archive}")


if __name__ == "__main__":
    main()
