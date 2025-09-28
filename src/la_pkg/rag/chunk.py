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
    "SectionSpan", ["section_id", "abs_start", "abs_end", "text", "label"]
)
PageBreak = namedtuple("PageBreak", ["abs_offset", "page_no"])


def _normalize_whitespace(text: str) -> str:
    """Normalizes whitespace: replaces multiple spaces/newlines with a single space."""
    return " ".join(text.split()).strip()


def _get_page_map(tree: ET._ElementTree) -> list[tuple[int, int]]:
    """
    Palauttaa listan (abs_offset, page_no). Säilyttää taaksepäin-yhteensopivuuden.
    """
    # _flatten_tei expects the root element, not the entire tree
    full_text, sections, page_breaks = _flatten_tei(tree.getroot(), include_front=True) # Assuming include_front=True for page map
    return [(pb.abs_offset, pb.page_no) for pb in page_breaks]


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

    def emit(text: str):
        nonlocal abs_cursor
        normalized_text = _normalize_whitespace(text)
        if normalized_text:
            full_text_parts.append(normalized_text)
            abs_cursor += len(normalized_text)

    # Process front matter
    if include_front:
        front_element = root.find(".//tei:front", namespaces=ns)
        if front_element is not None:
            title = " ".join(front_element.xpath(".//tei:title/text()", namespaces=ns))
            abstract = " ".join(
                front_element.xpath(".//tei:abstract//text()", namespaces=ns)
            )
            front_text = f"{title}\n\n{abstract}".strip()
            if front_text:
                start_offset = abs_cursor
                emit(front_text)
                sections.append(
                    SectionSpan(
                        "front",
                        start_offset,
                        abs_cursor,
                        _normalize_whitespace(front_text),
                        "Front Matter",
                    )
                )
                if abs_cursor > 0:
                    emit(" ") # Add a space between sections

    # Process body divisions
    body_divs = root.xpath(".//tei:body/tei:div", namespaces=ns)
    for i, div in enumerate(body_divs):
        # Record page breaks encountered during traversal
        for element in div.iter():
            if element.tag == "{http://www.tei-c.org/ns/1.0}pb":
                page_num = element.get("n")
                if page_num and page_num.isdigit():
                    page_breaks.append(PageBreak(abs_cursor, int(page_num)))
            if element.text:
                emit(element.text)
            if element.tail:
                emit(element.tail)

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
            start_offset = abs_cursor
            emit(normalized_section_text)
            sections.append(
                SectionSpan(
                    f"body_div_{i}",
                    start_offset,
                    abs_cursor,
                    normalized_section_text,
                    heading,
                )
            )
            if i < len(body_divs) - 1:
                emit(" ") # Add a space between sections

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

    # Pre-calculate character start positions for each word in the normalized text
    words = _normalize_whitespace(text).split()
    char_starts = []
    current_char_offset = 0
    for i, word in enumerate(words):
        if i > 0:
            current_char_offset += 1  # Account for the single space between words
        char_starts.append(current_char_offset)
        current_char_offset += len(word)
    joined_text = " ".join(words) # This is the normalized text we'll use for char offsets

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

        # Find the character offsets for the chunk_text within the normalized section text
        # This is more robust than text.find() for sub-chunks and normalization differences
        # We need to find the token indices of the chunk_text within the original tokens
        # Then map those token indices to character offsets using the pre-calculated char_starts
        
        # A more robust way to get local_chunk_start/end without text.find()
        # is to re-tokenize the chunk_text and find its position in the original tokens.
        # However, given the current tokenization approach, a simpler method is to
        # find the character span of the decoded chunk_text within the *normalized* section text.
        # This assumes that `enc.decode(chunk_tokens)` produces text that is a substring
        # of `_normalize_whitespace(text)`.

        # For now, we'll use a simplified approach that relies on the decoded chunk_text
        # being a direct substring of the normalized section text.
        # A more advanced solution would involve token-to-character mapping during tokenization.
        
        # Fallback to 0,0 if chunk_text is not found in joined_text (should not happen with proper normalization)
        local_chunk_start = joined_text.find(chunk_text)
        if local_chunk_start == -1:
            local_chunk_start = 0
            local_chunk_end = len(chunk_text)
        else:
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
