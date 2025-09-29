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
from lxml import etree as ET # Use ET for etree to avoid conflict with _get_page_map parameter


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
    "SectionSpan", ["section_id", "abs_start", "abs_end", "label"]
)
PageBreak = namedtuple("PageBreak", ["abs_offset", "page_no"])


def _normalize_whitespace(text: str) -> str:
    """Normalizes whitespace: replaces multiple spaces/newlines with a single space."""
    return " ".join(text.split()).strip()


def _get_page_map(tree: ET._ElementTree) -> list[PageBreak]:
    """
    Palauttaa listan (abs_offset, page_no). Säilyttää taaksepäin-yhteensopivuuden.
    """
    # _flatten_tei expects the root element, not the entire tree
    full_text, sections, page_breaks = _flatten_tei(tree.getroot(), include_front=True)
    return page_breaks


def _flatten_tei(
    root: ET._Element, include_front: bool
) -> tuple[str, list[SectionSpan], list[PageBreak]]:
    """
    Flattens a TEI XML tree into a single string, extracts sections, and page breaks
    with global offsets in a single pass.
    """
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    full_text_parts: list[str] = []
    sections: list[SectionSpan] = []
    page_breaks: list[PageBreak] = []
    abs_cursor = 0

    def emit(s: str):
        nonlocal abs_cursor
        s = _normalize_whitespace(s)
        if not s:
            return

        if full_text_parts and not full_text_parts[-1].endswith(" "):
            full_text_parts.append(" ")
            abs_cursor += 1
        
        full_text_parts.append(s)
        abs_cursor += len(s)

    # Process front matter
    if include_front:
        front = root.find("tei:front", ns)
        if front is not None:
            section_start = abs_cursor
            
            # Capture text by tracking the state of full_text_parts
            start_parts_len = len(full_text_parts)
            
            for node in front.iter():
                if node.tag == f"{{{ns['tei']}}}pb":
                    n = node.get("n")
                    if n and n.isdigit():
                        page_breaks.append(PageBreak(abs_cursor, int(n)))
                if node.text:
                    emit(node.text)
                if node.tail:
                    emit(node.tail)

            section_end = abs_cursor
            
            if section_end > section_start: # Check if any text was emitted for this section
                sections.append(
                    SectionSpan(
                        "front",
                        section_start,
                        section_end,
                        "Front Matter",
                    )
                )

    # Process body divisions
    body = root.find("tei:body", ns)
    if body is not None:
        # Find only direct children divs to avoid processing nested divs twice
        for i, div in enumerate(body.findall("tei:div", ns)):
            section_start = abs_cursor
            
            heading_element = div.find("tei:head", ns)
            heading = (
                _normalize_whitespace(heading_element.text)
                if heading_element is not None and heading_element.text
                else f"Section {i+1}"
            )

            start_parts_len = len(full_text_parts)

            # Iterate within the div
            for node in div.iter():
                if node.tag == f"{{{ns['tei']}}}pb":
                    n = node.get("n")
                    if n and n.isdigit():
                        page_breaks.append(PageBreak(abs_cursor, int(n)))
                if node.text:
                    emit(node.text)
                if node.tail:
                    emit(node.tail)

            section_end = abs_cursor
            
            if section_end > section_start: # Check if any text was emitted for this section
                sections.append(
                    SectionSpan(
                        f"body_div_{i}",
                        section_start,
                        section_end,
                        heading,
                    )
                )

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
            full_document_text=text,
            section_abs_start=0,
            section_abs_end=len(text),
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
        full_text, sections, page_breaks = _flatten_tei(tree.getroot(), include_front)

        for section in sections:
            effective_min_tokens = (
                0 if section.section_id == "front" and include_front else min_tokens
            )
            chunks.extend(
                _create_chunks_for_section(
                    full_document_text=full_text,
                    section_abs_start=section.abs_start,
                    section_abs_end=section.abs_end,
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
    full_document_text: str,
    section_abs_start: int,
    section_abs_end: int,
    paper_id: str,
    section_id: str,
    section_title: str,
    file_path: str,
    max_tokens: int,
    overlap: int,
    min_tokens: int,
    min_chars: int,
    source: str = "tei",
    abs_base: int = 0,  # This is now redundant, as section_abs_start is the base
    page_breaks: list[PageBreak] | None = None,
) -> list[Chunk]:
    """Paloittele yhden osion teksti pienemmiksi chunkeiksi."""
    # Extract the section's text from the full document text
    text = full_document_text[section_abs_start:section_abs_end]

    if not text or len(text) < min_chars:
        return []

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    if len(tokens) < min_tokens:
        return []

    # Create a token-to-character map to replace str.find()
    decoded_tokens = [enc.decode([t]) for t in tokens]
    char_starts: list[int] = []
    current_char_offset = 0
    # Reconstruct the text with a single space between tokens to map offsets
    for i, token_str in enumerate(decoded_tokens):
        char_starts.append(current_char_offset)
        current_char_offset += len(token_str)
        if i < len(decoded_tokens) - 1:
            current_char_offset += 1  # Account for the single space separator

    chunks: list[Chunk] = []
    chunk_id_counter = 0
    start_token_idx = 0

    while start_token_idx < len(tokens):
        end_token_idx = min(start_token_idx + max_tokens, len(tokens))
        chunk_tokens = tokens[start_token_idx:end_token_idx]

        if len(chunk_tokens) < min_tokens:
            # If it's the first and only chunk, keep it if it meets min_chars.
            # Otherwise, if it's a small trailing chunk, discard it.
            if chunks or len(text) < min_chars:
                break

        chunk_text = enc.decode(chunk_tokens)
        unique_chunk_id = f"{paper_id}-{section_id}-{chunk_id_counter}"

        # Calculate offsets using the token->char map
        local_start = char_starts[start_token_idx]
        # The end offset is the start of the last token plus its length
        last_token_idx = end_token_idx - 1
        local_end = char_starts[last_token_idx] + len(decoded_tokens[last_token_idx])

        global_chunk_start_offset = section_abs_start + local_start
        global_chunk_end_offset = section_abs_start + local_end

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
