"""Bulk PDF downloader with audit logging."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Mapping, Optional

import httpx
import pandas as pd

from .fetchers import arxiv_pdf_url, pmc_pdf_url, unpaywall_pdf_url


def _get_field(row: Mapping[str, object], key: str) -> str:
    value = row.get(key) if isinstance(row, Mapping) else getattr(row, key, "")
    if value is None:
        return ""
    return str(value)


def safe_id(row: Mapping[str, object]) -> str:
    """Return a filesystem friendly identifier for the row."""

    doi = _get_field(row, "doi").lower()
    if doi:
        return doi.replace("/", "_")
    fallback_source = _get_field(row, "id") or _get_field(row, "title")
    if fallback_source:
        digest = hashlib.sha1(fallback_source.encode("utf-8")).hexdigest()
        return digest[:16]
    normalized = repr(tuple(sorted((str(key), str(value)) for key, value in row.items())))
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]  # type: ignore[arg-type]


def pick_pdf(
    row: Mapping[str, object],
    client: httpx.Client,
    unpaywall_email: str,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Decide which provider to use and return provider/url/license."""

    for provider, func in (("arxiv", arxiv_pdf_url), ("pmc", pmc_pdf_url)):
        url = func(row)
        if url:
            return provider, url, None
    doi = _get_field(row, "doi")
    url, license_info = unpaywall_pdf_url(doi, unpaywall_email, client)
    if url:
        return "unpaywall", url, license_info
    return None, None, None


def download_all(
    df: pd.DataFrame,
    out_dir: Path,
    audit_csv: Path,
    *,
    timeout_s: int = 30,
    retries: int = 2,
    throttle_ms: int = 200,
    unpaywall_email: str = "",
) -> pd.DataFrame:
    """Download PDFs for each row in the DataFrame."""

    out_dir.mkdir(parents=True, exist_ok=True)
    audit_csv.parent.mkdir(parents=True, exist_ok=True)

    result = df.copy()
    if "pdf_path" not in result.columns:
        result["pdf_path"] = None
    if "pdf_license" not in result.columns:
        result["pdf_license"] = None
    if "has_fulltext" not in result.columns:
        result["has_fulltext"] = False
    else:
        result["has_fulltext"] = result["has_fulltext"].fillna(False).astype(bool)

    audit_rows: list[tuple[object, ...]] = []

    with httpx.Client(follow_redirects=True, timeout=timeout_s) as client:
        for idx, row in result.iterrows():
            pid = safe_id(row)
            provider, url, license_info = pick_pdf(row, client, unpaywall_email)
            doi_value = row.get("doi")
            source_value = row.get("source")

            if not url:
                result.at[idx, "pdf_path"] = None
                result.at[idx, "pdf_license"] = None
                result.at[idx, "has_fulltext"] = False
                audit_rows.append(
                    (
                        pid,
                        doi_value,
                        source_value,
                        provider,
                        url,
                        license_info,
                        "skipped",
                        "no_pdf_url",
                        None,
                        0,
                        0.0,
                    )
                )
                continue

            dest = out_dir / f"{pid}.pdf"
            if dest.exists():
                existing_license = row.get("pdf_license")
                if isinstance(existing_license, str) and existing_license.strip():
                    license_value = existing_license
                else:
                    license_value = license_info
                result.at[idx, "pdf_path"] = str(dest)
                result.at[idx, "pdf_license"] = license_value
                result.at[idx, "has_fulltext"] = True
                audit_rows.append(
                    (
                        pid,
                        doi_value,
                        source_value,
                        provider,
                        url,
                        license_value,
                        "cached",
                        "",
                        200,
                        dest.stat().st_size,
                        0.0,
                    )
                )
                continue

            ok = False
            http_status: Optional[int] = None
            nbytes = 0
            elapsed = 0.0
            reason = "http_error"
            for attempt in range(retries + 1):
                start = time.perf_counter()
                try:
                    response = client.get(url)
                    elapsed = time.perf_counter() - start
                    http_status = response.status_code
                    if response.status_code == 200:
                        if response.headers.get("content-type", "").lower().startswith("application/pdf"):
                            dest.write_bytes(response.content)
                            nbytes = dest.stat().st_size
                            ok = True
                            reason = ""
                            break
                        reason = "not_pdf"
                    elif response.status_code >= 400:
                        reason = f"status_{response.status_code}"
                except httpx.HTTPError:
                    elapsed = time.perf_counter() - start
                    http_status = None
                    reason = "http_error"
                if attempt < retries and not ok:
                    time.sleep(throttle_ms / 1000.0)

            if not ok and dest.exists():
                dest.unlink(missing_ok=True)

            result.at[idx, "pdf_path"] = str(dest) if ok else None
            result.at[idx, "pdf_license"] = license_info if ok else None
            result.at[idx, "has_fulltext"] = bool(ok)

            audit_rows.append(
                (
                    pid,
                    doi_value,
                    source_value,
                    provider,
                    url,
                    license_info,
                    "downloaded" if ok else "failed",
                    reason,
                    http_status,
                    nbytes,
                    elapsed,
                )
            )

    columns = [
        "paper_id",
        "doi",
        "source",
        "provider",
        "pdf_url",
        "license",
        "status",
        "reason",
        "http_status",
        "bytes",
        "elapsed_s",
    ]
    pd.DataFrame(audit_rows, columns=columns).to_csv(audit_csv, index=False)
    return result
