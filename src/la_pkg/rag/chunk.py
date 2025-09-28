"""TEI XML chunking logic for RAG."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import tiktoken
from lxml import etree
from pydantic import BaseModel


class Chunk(BaseModel):
    """Yksi tekstipala, joka voidaan indeksoida."""

    paper_id: str
    section_id: str
    section_title: str | None
    chunk_id: str  # Changed to string
    text: str
    n_tokens: int
    page_start: int | None = None
    page_end: int | None = None
    tei_path: str
    source: str = "tei"


def chunk_document(
    row: dict[str, Any], max_tokens: int, overlap: int, min_tokens: int = 50
) -> list[Chunk]:
    """Jäsennä yksi TEI XML -dokumentti ja palasta se osiin."""
    tei_path = Path(row["parsed_xml_path"])  # Corrected column name
    if not tei_path.exists():
        return []

    paper_id = row.get("id", tei_path.stem)
    chunks: list[Chunk] = []

    try:
        tree = etree.parse(str(tei_path))
        ns = {"tei": "http://www.tei-c.org/ns/1.0"}

        # 1. Käsittele 'front' (otsikko ja abstrakti)
        front_element = tree.find(".//tei:front", namespaces=ns)
        if front_element is not None:
            title = " ".join(front_element.xpath(".//tei:title/text()", namespaces=ns))
            abstract = " ".join(
                front_element.xpath(".//tei:abstract//text()", namespaces=ns)
            )
            front_text = f"{title}\n\n{abstract}".strip()
            if front_text:
                chunks.extend(
                    _create_chunks_for_section(
                        text=front_text,
                        paper_id=paper_id,
                        section_id="front",
                        section_title="Front Matter",
                        tei_path=str(tei_path),
                        max_tokens=max_tokens,
                        overlap=overlap,
                        min_tokens=min_tokens,
                    )
                )

        # 2. Käsittele 'body/div' (luvut ja alaluvut)
        for i, div in enumerate(tree.xpath(".//tei:body/tei:div", namespaces=ns)):
            heading_element = div.find("tei:head", namespaces=ns)
            heading = (
                heading_element.text
                if heading_element is not None
                else f"Section {i+1}"
            )

            # Poista viittaukset ja taulukot ennen tekstin yhdistämistä
            etree.strip_elements(
                div, "{http://www.tei-c.org/ns/1.0}ref", with_tail=False
            )
            etree.strip_elements(
                div, "{http://www.tei-c.org/ns/1.0}figure", with_tail=False
            )

            section_text = " ".join(div.xpath(".//text()")).strip()
            if section_text:
                chunks.extend(
                    _create_chunks_for_section(
                        text=section_text,
                        paper_id=paper_id,
                        section_id=f"body_div_{i}",
                        section_title=heading,
                        tei_path=str(tei_path),
                        max_tokens=max_tokens,
                        overlap=overlap,
                        min_tokens=min_tokens,
                    )
                )

        # 3. Jätä 'back' (viitteet) pois tarkoituksella

    except (etree.XMLSyntaxError, IOError):
        # Jos XML-tiedosto on rikki tai ei luettavissa, ohita se
        return []

    return chunks


def _create_chunks_for_section(
    text: str,
    paper_id: str,
    section_id: str,
    section_title: str,
    tei_path: str,
    max_tokens: int,
    overlap: int,
    min_tokens: int,
) -> list[Chunk]:
    """Paloittele yhden osion teksti pienemmiksi chunkeiksi."""
    if not text:
        return []

    # Käytä tiktokenia tokenointiin
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    chunks: list[Chunk] = []
    chunk_id_counter = 0

    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]

        if len(chunk_tokens) < min_tokens:  # Apply min_tokens filter
            start += max_tokens - overlap
            continue

        chunk_text = enc.decode(chunk_tokens)

        # Create a unique ID for the chunk
        unique_chunk_id = f"{paper_id}-{section_id}-{chunk_id_counter}"

        chunks.append(
            Chunk(
                paper_id=paper_id,
                section_id=section_id,
                section_title=section_title,
                chunk_id=unique_chunk_id,  # Use unique_chunk_id
                text=chunk_text,
                n_tokens=len(chunk_tokens),
                tei_path=tei_path,
            )
        )
        chunk_id_counter += 1

        if end >= len(tokens):
            break
        start += max_tokens - overlap

    return chunks


def run_chunking(
    parsed_index_path: Path,
    out_path: Path,
    max_tokens: int,
    overlap: int,
    min_tokens: int = 50,
) -> None:
    """Aja chunkkaus koko aineistolle."""
    if not parsed_index_path.exists():
        raise FileNotFoundError(f"Input file not found: {parsed_index_path}")

    df = pd.read_parquet(parsed_index_path)
    all_chunks: list[Chunk] = []

    for _, row in df.iterrows():
        if not row.get("parsed_ok") or not row.get(
            "parsed_xml_path"
        ):  # Corrected column name
            continue
        chunks = chunk_document(
            cast(dict[str, Any], row.to_dict()), max_tokens, overlap, min_tokens
        )
        all_chunks.extend(chunks)

    chunk_df = pd.DataFrame([c.model_dump() for c in all_chunks])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_df.to_parquet(out_path, index=False)

    # Laske ja tallenna statistiikat
    stats: dict[str, Any] = {}
    if not chunk_df.empty:
        stats["n_chunks"] = len(chunk_df)
        stats["avg_tokens_per_chunk"] = chunk_df["n_tokens"].mean()
        stats["percent_missing_pages"] = (
            chunk_df["page_start"].isnull().sum() / len(chunk_df) * 100
        )
    else:
        stats["n_chunks"] = 0
        stats["avg_tokens_per_chunk"] = 0
        stats["percent_missing_pages"] = 100

    stats_path = out_path.parent / "chunk_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Chunking complete. Stats: {stats}")
