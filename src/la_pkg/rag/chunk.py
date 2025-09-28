"""TEI XML chunking logic for RAG."""

from __future__ import annotations

import json
from pathlib import Path
from collections import namedtuple
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
    chunk_id: str
    text: str
    n_tokens: int
    page_start: int | None = None
    page_end: int | None = None
    file_path: str
    source: str = "tei"


# Helper types for clarity
SectionSpan = namedtuple(
    "SectionSpan", ["section_id", "abs_start", "abs_end", "text", "label"]
)
PageBreak = namedtuple("PageBreak", ["abs_offset", "page_no"])


def _normalize_whitespace(text: str) -> str:
    """Normalizes whitespace: replaces multiple spaces/newlines with a single space."""
    return " ".join(text.split()).strip()


def _flatten_tei(
    tree: etree._ElementTree, include_front: bool
) -> tuple[str, list[SectionSpan], list[PageBreak]]:
    """
    Flattens a TEI XML tree into a single string, extracts sections, and page breaks
    with global offsets.
    """
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    full_text_parts: list[str] = []
    sections: list[SectionSpan] = []
    page_breaks: list[PageBreak] = []
    abs_cursor = 0

    # Process front matter
    if include_front:
        front_element = tree.find(".//tei:front", namespaces=ns)
        if front_element is not None:
            title = _normalize_whitespace(
                " ".join(front_element.xpath(".//tei:title/text()", namespaces=ns))
            )
            abstract = _normalize_whitespace(
                " ".join(front_element.xpath(".//tei:abstract//text()", namespaces=ns))
            )
            front_text = f"{title}\n\n{abstract}".strip()
            if front_text:
                normalized_front_text = _normalize_whitespace(front_text)
                sections.append(
                    SectionSpan(
                        "front",
                        abs_cursor,
                        abs_cursor + len(normalized_front_text),
                        normalized_front_text,
                        "Front Matter",
                    )
                )
                full_text_parts.append(normalized_front_text)
                abs_cursor += len(normalized_front_text)
                if abs_cursor > 0:
                    full_text_parts.append(" ")  # Add a space between sections
                    abs_cursor += 1

    # Process body divisions
    body_divs = tree.xpath(".//tei:body/tei:div", namespaces=ns)
    for i, div in enumerate(body_divs):
        # Extract page breaks within this div
        for pb_element in div.xpath(".//tei:pb", namespaces=ns):
            page_num = pb_element.get("n")
            if page_num and page_num.isdigit():
                # Calculate offset relative to the current full_text_parts length
                pb_offset_in_full_text = abs_cursor + len(
                    _normalize_whitespace(
                        " ".join(
                            div.xpath(
                                f".//text()[preceding::tei:pb[@n='{page_num}']]",
                                namespaces=ns,
                            )
                        )
                    )
                )
                page_breaks.append(PageBreak(pb_offset_in_full_text, int(page_num)))

        heading_element = div.find("tei:head", namespaces=ns)
        heading = (
            heading_element.text if heading_element is not None else f"Section {i+1}"
        )

        # Strip elements that should not be part of the chunk text
        etree.strip_elements(div, "{http://www.tei-c.org/ns/1.0}ref", with_tail=False)
        etree.strip_elements(
            div, "{http://www.tei-c.org/ns/1.0}figure", with_tail=False
        )

        section_raw_text = " ".join(div.xpath(".//text()"))
        normalized_section_text = _normalize_whitespace(section_raw_text)

        if normalized_section_text:
            sections.append(
                SectionSpan(
                    f"body_div_{i}",
                    abs_cursor,
                    abs_cursor + len(normalized_section_text),
                    normalized_section_text,
                    heading,
                )
            )
            full_text_parts.append(normalized_section_text)
            abs_cursor += len(normalized_section_text)
            if i < len(body_divs) - 1:
                full_text_parts.append(" ")  # Add a space between sections
                abs_cursor += 1

    # Re-collect page breaks from the entire tree after stripping elements
    # This ensures page breaks are correctly positioned relative to the final full_text
    page_breaks = []
    current_offset_for_pb = 0
    for element in tree.iter():
        if element.tag == "{http://www.tei-c.org/ns/1.0}pb":
            page_num = element.get("n")
            if page_num and page_num.isdigit():
                page_breaks.append(PageBreak(current_offset_for_pb, int(page_num)))
        if element.text:
            current_offset_for_pb += len(_normalize_whitespace(element.text))
        if element.tail:
            current_offset_for_pb += len(_normalize_whitespace(element.tail))

    full_text = "".join(full_text_parts).strip()
    return full_text, sections, sorted(page_breaks, key=lambda x: x.abs_offset)


def _get_page_range(
    start_offset: int, end_offset: int, page_breaks: list[PageBreak]
) -> tuple[int | None, int | None]:
    """Determines the page range for a given text offset range using global page breaks."""
    start_page: int | None = None
    end_page: int | None = None

    if not page_breaks:
        return None, None

    # Find start page
    for i in range(len(page_breaks)):
        pb_offset, page_num = page_breaks[i]
        if start_offset >= pb_offset:
            start_page = page_num
        else:
            break
    # If chunk starts before the first page break, assume it's on the first page
    if start_page is None and page_breaks:
        start_page = page_breaks[0].page_no if page_breaks[0].page_no == 1 else None

    # Find end page
    for i in range(len(page_breaks)):
        pb_offset, page_num = page_breaks[i]
        if end_offset > pb_offset:  # Use > for end_offset to include content on the page
            end_page = page_num
        else:
            break
    # If chunk ends before the first page break, assume it's on the first page
    if end_page is None and page_breaks:
        end_page = page_breaks[0].page_no if page_breaks[0].page_no == 1 else None

    return start_page, end_page


def _chunk_from_text_file(
    row: dict[str, Any],
    max_tokens: int,
    overlap: int,
    min_tokens: int,
    min_chars: int,
    page_breaks: list[PageBreak] | None = None,
) -> list[Chunk]:
    """Luo chunkit suoraan tekstitiedostosta fallbackina."""
    txt_path_str = row.get("parsed_txt_path")
    if not txt_path_str or not isinstance(txt_path_str, str):
        return []

    txt_path = Path(txt_path_str)
    if not txt_path.exists():
        return []

    paper_id = row.get("id", txt_path.stem)
    try:
        text = txt_path.read_text(encoding="utf-8")
        if not text.strip():
            return []

        return _create_chunks_for_section(
            text=text,
            paper_id=paper_id,
            section_id="full_text",
            section_title="Full Text",
            file_path=str(txt_path),
            max_tokens=max_tokens,
            overlap=overlap,
            min_tokens=min_tokens,
            min_chars=min_chars,
            source="text",
            abs_base=0,  # For text files, the base offset is 0
            page_breaks=page_breaks,
        )
    except IOError:
        return []


def chunk_document(
    row: dict[str, Any],
    max_tokens: int,
    overlap: int,
    min_tokens: int,
    include_front: bool,
    use_text_txt: bool,
    min_chars: int,
) -> list[Chunk]:
    """Jäsennä yksi TEI XML -dokumentti ja palasta se osiin."""
    xml_path_str = row.get("parsed_xml_path")
    if not xml_path_str or not isinstance(xml_path_str, str):
        if use_text_txt:
            return _chunk_from_text_file(
                row, max_tokens, overlap, min_tokens, min_chars
            )
        return []

    tei_path = Path(xml_path_str)
    if not tei_path.exists():
        if use_text_txt:
            return _chunk_from_text_file(
                row, max_tokens, overlap, min_tokens, min_chars
            )
        return []

    paper_id = row.get("id", tei_path.stem)
    chunks: list[Chunk] = []

    try:
        tree = etree.parse(str(tei_path))
        full_text, sections, page_breaks = _flatten_tei(tree, include_front)

        for section in sections:
            effective_min_tokens = (
                0 if section.section_id == "front" and include_front else min_tokens
            )
            chunks.extend(
                _create_chunks_for_section(
                    text=section.text,
                    paper_id=paper_id,
                    section_id=section.section_id,
                    section_title=section.label,
                    file_path=str(tei_path),
                    max_tokens=max_tokens,
                    overlap=overlap,
                    min_tokens=effective_min_tokens,
                    min_chars=min_chars,
                    abs_base=section.abs_start,
                    page_breaks=page_breaks,
                )
            )

        if not chunks and use_text_txt:
            return _chunk_from_text_file(
                row, max_tokens, overlap, min_tokens, min_chars, page_breaks
            )

    except (etree.XMLSyntaxError, IOError):
        if use_text_txt:
            return _chunk_from_text_file(
                row, max_tokens, overlap, min_tokens, min_chars
            )
        return []

    return chunks


def _create_chunks_for_section(
    text: str,
    paper_id: str,
    section_id: str,
    section_title: str,
    file_path: str,
    max_tokens: int,
    overlap: int,
    min_tokens: int,
    min_chars: int,
    source: str = "tei",
    abs_base: int = 0,  # New parameter for global base offset
    page_breaks: list[PageBreak] | None = None,
) -> list[Chunk]:
    """Paloittele yhden osion teksti pienemmiksi chunkeiksi."""
    if not text or len(text) < min_chars:
        return []

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    if len(tokens) < min_tokens:
        return []

    chunks: list[Chunk] = []
    chunk_id_counter = 0
    start_token_idx = 0

    while start_token_idx < len(tokens):
        end_token_idx = start_token_idx + max_tokens
        chunk_tokens = tokens[start_token_idx:end_token_idx]

        is_last_chunk = end_token_idx >= len(tokens)
        if not chunks and is_last_chunk and len(chunk_tokens) < min_tokens:
            pass
        elif len(chunk_tokens) < min_tokens:
            break

        chunk_text = enc.decode(chunk_tokens)
        unique_chunk_id = f"{paper_id}-{section_id}-{chunk_id_counter}"

        # Calculate global offsets for the chunk
        # We need to find the start and end character index of chunk_text within the original 'text'
        # This is more robust than text.find() if the text has been normalized
        local_chunk_start = text.find(chunk_text)
        if local_chunk_start == -1: # Fallback if find fails due to normalization differences
            local_chunk_start = 0
            # This might lead to less accurate page numbers, but prevents crashes
            # A more robust solution would involve token-to-char mapping
        local_chunk_end = local_chunk_start + len(chunk_text)

        global_chunk_start_offset = abs_base + local_chunk_start
        global_chunk_end_offset = abs_base + local_chunk_end

        page_start, page_end = _get_page_range(
            global_chunk_start_offset, global_chunk_end_offset, page_breaks or []
        )

        chunks.append(
            Chunk(
                paper_id=paper_id,
                section_id=section_id,
                section_title=section_title,
                chunk_id=unique_chunk_id,
                text=chunk_text,
                n_tokens=len(chunk_tokens),
                page_start=page_start,
                page_end=page_end,
                file_path=file_path,
                source=source,
            )
        )
        chunk_id_counter += 1

        if end_token_idx >= len(tokens):
            break
        start_token_idx += max_tokens - overlap

    return chunks


def run_chunking(
    parsed_index_path: Path,
    out_path: Path,
    max_tokens: int,
    overlap: int,
    min_tokens: int,
    include_front: bool,
    use_text_txt: bool,
    min_chars: int,
) -> None:
    """Aja chunkkaus koko aineistolle."""
    if not parsed_index_path.exists():
        raise FileNotFoundError(f"Input file not found: {parsed_index_path}")

    df = pd.read_parquet(parsed_index_path)
    all_chunks: list[Chunk] = []
    fallback_count = 0

    for _, row in df.iterrows():
        if not row.get("parsed_ok"):
            continue

        row_dict = cast(dict[str, Any], row.to_dict())
        chunks = chunk_document(
            row_dict,
            max_tokens,
            overlap,
            min_tokens,
            include_front,
            use_text_txt,
            min_chars,
        )
        if chunks and chunks[0].source == "text":
            fallback_count += 1
        all_chunks.extend(chunks)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if all_chunks:
        chunk_df = pd.DataFrame([c.model_dump() for c in all_chunks])
    else:
        # Luo tyhjä DataFrame oikealla skeemalla
        schema = Chunk.model_json_schema()
        columns = list(schema["properties"].keys())
        chunk_df = pd.DataFrame(columns=columns)

    chunk_df.to_parquet(out_path, index=False)

    stats: dict[str, Any] = {
        "n_chunks": 0,
        "avg_tokens_per_chunk": 0.0,
        "percent_missing_pages": 100.0,
        "fallback_used_count": fallback_count,
        "total_docs_processed": len(df[df["parsed_ok"]]),
    }
    if not chunk_df.empty:
        stats["n_chunks"] = len(chunk_df)
        stats["avg_tokens_per_chunk"] = chunk_df["n_tokens"].mean()
        stats["percent_missing_pages"] = (
            chunk_df["page_start"].isnull().sum() / len(chunk_df) * 100
        )

    stats_path = out_path.parent / "chunk_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Chunking complete. Stats: {json.dumps(stats)}")
