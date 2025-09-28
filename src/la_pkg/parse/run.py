"""Pipeline helpers for running GROBID parsing on a dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .grobid_client import GrobidClient
from .tei_extract import tei_to_title_abstract_refs


def _target_rows(df: pd.DataFrame, sample: int | None) -> Iterable[int]:
    indices = list(df.index)
    if sample is not None:
        return indices[: max(sample, 0)]
    return indices


def parse_all(
    df: pd.DataFrame,
    parsed_dir: Path,
    grobid_url: str,
    err_log: Path,
    *,
    sample: int | None = None,
) -> pd.DataFrame:
    """Parse PDFs via GROBID and write TEI/text artefacts."""

    parsed_dir.mkdir(parents=True, exist_ok=True)
    err_log.parent.mkdir(parents=True, exist_ok=True)

    result = df.copy()
    if "parsed_xml_path" not in result.columns:
        result["parsed_xml_path"] = None
    if "parsed_txt_path" not in result.columns:
        result["parsed_txt_path"] = None
    if "parsed_ok" not in result.columns:
        result["parsed_ok"] = False
    else:
        result["parsed_ok"] = result["parsed_ok"].fillna(False).astype(bool)

    errors: list[tuple[object, object, object]] = []
    indices = list(_target_rows(result, sample))

    with GrobidClient(grobid_url) as client:
        for idx in indices:
            row = result.loc[idx]
            pdf_value = row.get("pdf_path")
            pdf_path = Path(str(pdf_value)) if pdf_value else None
            if not pdf_path or not pdf_path.exists():
                errors.append((row.get("id"), row.get("doi"), "no_pdf"))
                result.at[idx, "parsed_ok"] = False
                continue
            try:
                tei_xml = client.process_fulltext(pdf_path)
                pid_source = row.get("doi") or row.get("id") or str(idx)
                pid = str(pid_source).replace("/", "_")
                out_dir = parsed_dir / pid
                out_dir.mkdir(parents=True, exist_ok=True)
                tei_path = out_dir / "tei.xml"
                txt_path = out_dir / "text.txt"
                tei_path.write_text(tei_xml, encoding="utf-8")
                info = tei_to_title_abstract_refs(tei_xml)
                summary = f"# {info['title']}\n\n{info['abstract']}\n"
                txt_path.write_text(summary, encoding="utf-8")
                result.at[idx, "parsed_xml_path"] = str(tei_path)
                result.at[idx, "parsed_txt_path"] = str(txt_path)
                result.at[idx, "parsed_ok"] = True
            except Exception as exc:  # pragma: no cover - defensive
                errors.append((row.get("id"), row.get("doi"), str(exc)[:200]))
                result.at[idx, "parsed_ok"] = False

    if errors:
        pd.DataFrame(errors, columns=["paper_id", "doi", "error"]).to_csv(
            err_log, index=False
        )
    return result
