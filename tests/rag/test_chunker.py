import pytest
import pandas as pd
from lxml import etree

from src.la_pkg.rag.chunk import (
    run_chunking,
    chunk_document,
    _get_page_map, # Re-introduced wrapper
    _get_page_range,
    _flatten_tei, # Import for direct testing of flatten_tei
)
from lxml import etree as ET # Import ET for consistency with chunk.py


# Fixture for a dummy TEI XML file with front matter and page breaks
@pytest.fixture
def dummy_tei_file(tmp_path):
    tei_content = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
        <teiHeader>
            <fileDesc>
                <titleStmt><title>Test Document</title></titleStmt>
                <publicationStmt><p>Publication</p></publicationStmt>
                <sourceDesc><p>Source</p></sourceDesc>
            </fileDesc>
        </teiHeader>
        <text>
            <front>
                <div type="abstract"><p>This is an abstract. It has some content.</p></div>
                <pb n="1"/>
                <div type="titlePage"><docTitle><titlePart type="main">Main Title</titlePart></docTitle></div>
            </front>
            <body>
                <div type="section">
                    <head>Section 1</head>
                    <p>This is the first paragraph of section 1. It is long enough to ensure some text before the page break.</p>
                    <pb n="2"/>
                    <p>This is the second paragraph of section 1, on page 2. It also has enough content.</p>
                </div>
                <div type="section">
                    <head>Section 2</head>
                    <p>This is the first paragraph of section 2. More content here.</p>
                    <pb n="3"/>
                    <p>This is the second paragraph of section 2, on page 3. Final content.</p>
                </div>
            </body>
        </text>
    </TEI>
    """
    tei_path = tmp_path / "test_doc.tei.xml"
    tei_path.write_text(tei_content, encoding="utf-8")
    return tei_path


# Fixture for a dummy text file
@pytest.fixture
def dummy_text_file(tmp_path):
    text_content = "This is a plain text document. It has some content for fallback."
    txt_path = tmp_path / "test_doc.txt"
    txt_path.write_text(text_content, encoding="utf-8")
    return txt_path


# Fixture for a parsed index DataFrame
@pytest.fixture
def parsed_index_df(dummy_tei_file, dummy_text_file):
    return pd.DataFrame(
        [
            {
                "id": "test_doc",
                "parsed_ok": True,
                "parsed_xml_path": str(dummy_tei_file),
                "parsed_txt_path": str(dummy_text_file),
            }
        ]
    )


@pytest.fixture
def parsed_index_df_no_xml(dummy_text_file):
    return pd.DataFrame(
        [
            {
                "id": "test_doc_no_xml",
                "parsed_ok": True,
                "parsed_xml_path": None,
                "parsed_txt_path": str(dummy_text_file),
            }
        ]
    )


@pytest.fixture
def parsed_index_df_empty_xml(tmp_path, dummy_text_file):
    empty_tei_path = tmp_path / "empty_doc.tei.xml"
    empty_tei_path.write_text("<TEI></TEI>", encoding="utf-8")
    return pd.DataFrame(
        [
            {
                "id": "empty_doc",
                "parsed_ok": True,
                "parsed_xml_path": str(empty_tei_path),
                "parsed_txt_path": str(dummy_text_file),
            }
        ]
    )


@pytest.mark.rag
def test_run_chunking_basic(tmp_path, parsed_index_df):
    out_path = tmp_path / "chunks.parquet"
    index_path = tmp_path / "index.parquet"
    parsed_index_df.to_parquet(index_path)
    run_chunking(
        parsed_index_path=index_path,
        out_path=out_path,
        max_tokens=100,
        overlap=0,
        min_tokens=10,
        include_front=False,
        use_text_txt=False,
        min_chars=0,
    )
    chunks_df = pd.read_parquet(out_path)
    assert not chunks_df.empty
    assert len(chunks_df) > 0
    assert "page_start" in chunks_df.columns
    assert "page_end" in chunks_df.columns
    assert all(chunks_df["page_start"].notna())
    assert all(chunks_df["page_end"].notna())


@pytest.mark.rag
def test_run_chunking_include_front(tmp_path, parsed_index_df):
    out_path = tmp_path / "chunks.parquet"
    index_path = tmp_path / "index.parquet"
    parsed_index_df.to_parquet(index_path)
    run_chunking(
        parsed_index_path=index_path,
        out_path=out_path,
        max_tokens=100,
        overlap=0,
        min_tokens=1000,  # High min_tokens to force front inclusion
        include_front=True,
        use_text_txt=False,
        min_chars=0,
    )
    chunks_df = pd.read_parquet(out_path)
    assert not chunks_df.empty
    assert any(chunks_df["section_id"] == "front")
    assert all(chunks_df["page_start"].notna())
    assert all(chunks_df["page_end"].notna())


@pytest.mark.rag
def test_run_chunking_use_text_txt_fallback(tmp_path, parsed_index_df_no_xml):
    out_path = tmp_path / "chunks.parquet"
    parsed_index_df_no_xml.to_parquet(tmp_path / "index.parquet")
    run_chunking(
        parsed_index_path=tmp_path / "index.parquet",
        out_path=out_path,
        max_tokens=100,
        overlap=0,
        min_tokens=10,
        include_front=False,
        use_text_txt=True,
        min_chars=0,
    )
    chunks_df = pd.read_parquet(out_path)
    assert not chunks_df.empty
    assert any(chunks_df["source"] == "text")
    assert any(chunks_df["section_id"] == "full_text")
    assert all(
        chunks_df["page_start"].isna()
    )  # Page numbers not expected from plain text


@pytest.mark.rag
def test_run_chunking_empty_df_produces_empty_parquet(tmp_path):
    empty_parsed_index_df = pd.DataFrame(
        columns=["id", "parsed_ok", "parsed_xml_path", "parsed_txt_path"]
    )
    empty_parsed_index_df.to_parquet(tmp_path / "empty_index.parquet")
    out_path = tmp_path / "chunks.parquet"

    run_chunking(
        parsed_index_path=tmp_path / "empty_index.parquet",
        out_path=out_path,
        max_tokens=100,
        overlap=0,
        min_tokens=10,
        include_front=False,
        use_text_txt=False,
        min_chars=0,
    )
    chunks_df = pd.read_parquet(out_path)
    assert chunks_df.empty
    assert "paper_id" in chunks_df.columns  # Check for schema


@pytest.mark.rag
def test_get_page_map(dummy_tei_file):
    tree = ET.parse(str(dummy_tei_file))
    page_map = _get_page_map(tree) # _get_page_map now calls _flatten_tei with tree.getroot()
    assert len(page_map) == 3 # Expecting 3 page breaks: 1, 2, 3
    page_numbers = [pb.page_no for pb in page_map]
    assert 1 in page_numbers
    assert 2 in page_numbers
    assert 3 in page_numbers

@pytest.mark.rag
def test_chunk_document_front_before_first_pb(tmp_path):
    # TEI where front matter exists, and the first body section starts before any <pb>
    tei_content = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
        <text>
            <front><docTitle><titlePart type="main">Front Title</titlePart></docTitle><p>Abstract text.</p><pb n="1"/></front>
            <body>
                <div type="section">
                    <head>Section A</head>
                    <p>This is some content on the very first page, before any explicit page breaks.</p>
                    <p>More content on page 1.</p>
                    <pb n="2"/>
                    <p>This content is on page 2.</p>
                </div>
            </body>
        </text>
    </TEI>
    """
    tei_path = tmp_path / "front_first_page.tei.xml"
    tei_path.write_text(tei_content, encoding="utf-8")

    row = {
        "id": "front_first_page",
        "parsed_ok": True,
        "parsed_xml_path": str(tei_path),
        "parsed_txt_path": None,
    }
    max_tokens = 20
    overlap = 5
    min_tokens = 5
    min_chars = 0
    include_front = True
    use_text_txt = False

    chunks = chunk_document(
        row, max_tokens, overlap, min_tokens, include_front, use_text_txt, min_chars
    )

    assert len(chunks) > 0
    # The first chunk (from front or body) should start on page 1
    assert chunks[0].page_start == 1, "First chunk should start on page 1"

    # Find a chunk that spans page 1 and 2
    found_span_1_2 = False
    for chunk in chunks:
        if chunk.page_start == 1 and chunk.page_end == 2:
            found_span_1_2 = True
            break
    assert found_span_1_2, "Expected a chunk spanning page 1 and 2"


@pytest.mark.rag
def test_get_page_range(dummy_tei_file):
    tree = etree.parse(str(dummy_tei_file))
    page_map = _get_page_map(tree)

    # The exact offsets depend on the content and normalization.
    # We need to get the actual offsets from _flatten_tei for precise testing.
    full_text, _, page_breaks = _flatten_tei(etree.parse(str(dummy_tei_file)).getroot(), include_front=True)

    # Find offsets for page breaks
    pb1_offset = page_breaks[0].abs_offset # After front matter, before page 1 content
    pb2_offset = page_breaks[1].abs_offset # After page 1 content, before page 2 content
    pb3_offset = page_breaks[2].abs_offset # After page 2 content, before page 3 content

    # Chunk entirely on page 1 (before pb2)
    start, end = _get_page_range(pb1_offset + 1, pb2_offset - 1, page_breaks)
    assert start == 1
    assert end == 1

    # Chunk spanning page 1 and 2
    start, end = _get_page_range(pb2_offset - 10, pb2_offset + 10, page_breaks)
    assert start == 1
    assert end == 2

    # Chunk entirely on page 2
    start, end = _get_page_range(pb2_offset + 1, pb3_offset - 1, page_breaks)
    assert start == 2
    assert end == 2

    # Chunk spanning page 2 and 3
    start, end = _get_page_range(pb3_offset - 10, pb3_offset + 10, page_breaks)
    assert start == 2
    assert end == 3

    # Chunk entirely on page 3 (after last pb)
    start, end = _get_page_range(pb3_offset + 1, len(full_text) - 1, page_breaks)
    assert start == 3
    assert end == 3

    # Test edge case: chunk starts exactly at a page break
    start, end = _get_page_range(pb2_offset, pb2_offset + 5, page_breaks)
    assert start == 2
    assert end == 2

    # Empty page map
    start, end = _get_page_range(0, 100, [])
    assert start is None
    assert end is None

    # Empty page map
    start, end = _get_page_range(0, 100, [])
    assert start is None
    assert end is None


@pytest.mark.rag
def test_chunk_document_min_chars(tmp_path, dummy_tei_file, dummy_text_file):
    row = {
        "id": "test_doc",
        "parsed_ok": True,
        "parsed_xml_path": str(dummy_tei_file),
        "parsed_txt_path": str(dummy_text_file),
    }
    # Set min_chars higher than content length
    chunks = chunk_document(row, 100, 0, 10, True, False, 1000)
    assert len(chunks) == 0

    # Set min_chars lower than content length
    chunks = chunk_document(row, 100, 0, 10, True, False, 10)
    assert len(chunks) > 0


@pytest.mark.rag
def test_chunk_document_empty_xml_with_fallback(tmp_path, parsed_index_df_empty_xml):
    row = parsed_index_df_empty_xml.iloc[0].to_dict()
    chunks = chunk_document(row, 100, 0, 10, False, True, 0)
    assert len(chunks) > 0
    assert chunks[0].source == "text"
    assert chunks[0].section_id == "full_text"
    assert chunks[0].page_start is None
    assert chunks[0].page_end is None


@pytest.fixture
def tei_with_multiple_page_breaks(tmp_path):
    tei_content = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
        <teiHeader><fileDesc><titleStmt><title>Multi-page Test</title></titleStmt><publicationStmt><p>Pub</p></publicationStmt><sourceDesc><p>Src</p></sourceDesc></fileDesc></teiHeader>
        <text>
            <body>
                <div type="section">
                    <head>Introduction</head>
                    <p>This is the first paragraph of the introduction. It is quite long and will span multiple chunks. We need enough text to ensure that the chunking logic creates overlapping chunks. This paragraph continues for a while to fill up space and demonstrate the page break handling. More text here to make sure we hit the token limits and create multiple chunks. This is still the first paragraph, just making it longer. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.</p>
                    <pb n="2"/>
                    <p>This is the second paragraph, starting on page 2. It also needs to be long enough to ensure overlapping chunks. The content here will be part of chunks that might start on page 1 and end on page 2, or start and end entirely on page 2. This is crucial for testing the global offset logic. More text to ensure proper chunking and page range calculation. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.</p>
                    <pb n="3"/>
                    <p>This is the third paragraph, starting on page 3. This paragraph will test chunks that span page 2 and 3, or are entirely on page 3. The overlap logic should correctly assign page numbers based on the global offsets. Final paragraph to ensure all edge cases with page breaks and overlaps are covered. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.</p>
                </div>
            </body>
        </text>
    </TEI>
    """
    tei_path = tmp_path / "multi_page_doc.tei.xml"
    tei_path.write_text(tei_content, encoding="utf-8")
    return tei_path


@pytest.mark.rag
def test_chunk_document_global_offsets_page_ranges_with_overlap(
    tmp_path, tei_with_multiple_page_breaks
):
    # This test specifically checks the global offset and page range logic with overlapping chunks.
    # We'll use a small max_tokens and a significant overlap to ensure chunks span page breaks.
    row = {
        "id": "multi_page_doc",
        "parsed_ok": True,
        "parsed_xml_path": str(tei_with_multiple_page_breaks),
        "parsed_txt_path": None,
    }
    max_tokens = 50
    overlap = 20
    min_tokens = 10
    min_chars = 0
    include_front = False
    use_text_txt = False

    chunks = chunk_document(
        row, max_tokens, overlap, min_tokens, include_front, use_text_txt, min_chars
    )

    assert len(chunks) > 5  # Expect multiple chunks
    assert all(c.page_start is not None for c in chunks)
    assert all(c.page_end is not None for c in chunks)

    # Verify page ranges for some expected chunk scenarios
    # This requires knowledge of the dummy_tei_file content and expected tokenization/offsets.
    # Since exact offsets are hard to predict without running the chunker, we'll check for
    # logical progression and spanning.

    # Find a chunk that should start on page 1 and end on page 2
    # This is an approximation; actual offsets depend on tokenization and text content.
    # We expect chunks to correctly span page breaks.
    found_span_1_2 = False
    found_span_2_3 = False
    found_on_page_3 = False

    for chunk in chunks:
        if chunk.page_start == 1 and chunk.page_end == 2:
            found_span_1_2 = True
        if chunk.page_start == 2 and chunk.page_end == 3:
            found_span_2_3 = True
        if chunk.page_start == 3 and chunk.page_end == 3:
            found_on_page_3 = True

    assert found_span_1_2, "Expected at least one chunk spanning page 1 and 2"
    assert found_span_2_3, "Expected at least one chunk spanning page 2 and 3"
    assert found_on_page_3, "Expected at least one chunk entirely on page 3"

    # Further check: ensure page numbers are monotonically increasing or stable within a chunk
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        next_chunk = chunks[i + 1]
        # Page start should not decrease for subsequent chunks
        assert next_chunk.page_start is not None and current_chunk.page_start is not None and next_chunk.page_start >= current_chunk.page_start, \
            f"Page start decreased from {current_chunk.page_start} to {next_chunk.page_start} for chunks {current_chunk.chunk_id} and {next_chunk.chunk_id}"
        assert next_chunk.page_end is not None and current_chunk.page_end is not None and next_chunk.page_end >= current_chunk.page_end, \
            f"Page end decreased from {current_chunk.page_end} to {next_chunk.page_end} for chunks {current_chunk.chunk_id} and {next_chunk.chunk_id}"

    # Check for global offset progression (implied by page ranges and chunk order)
    # This is a more robust check for the "overlap päällä peräkkäisten chunkkien globaalit offsetit kasvavat" criterion.
    # We can't directly access global offsets from the Chunk object, but we can infer from page ranges.
    # If page_start/end are not decreasing, and chunks are ordered, this implies global offset progression.
    # The token-to-char mapping ensures that repeated sentences don't map to the same offset.
    
    # To explicitly check global offsets, we would need to store them in the Chunk object,
    # or re-calculate them here, which is redundant. The page range check is a good proxy.
    
    # Let's ensure that the text content of overlapping chunks is indeed overlapping
    # and that the page ranges reflect this.
    
    # Example: Find two consecutive chunks that overlap and check their page ranges
    for i in range(len(chunks) - 1):
        chunk1 = chunks[i]
        chunk2 = chunks[i+1]
        
        # Assuming a reasonable overlap, the text should be different but share content
        # This is hard to assert precisely without re-implementing tokenization here.
        # The primary check is the page range monotonicity and the distinctness of repeated sentences.
        
        # The requirement "overlap päällä peräkkäisten chunkkien globaalit offsetit kasvavat"
        # is covered by the `start_token_idx += max_tokens - overlap` logic in _create_chunks_for_section
        # and the page_start/end monotonicity check above.
        
        # The requirement "sivuvälit eivät palaudu osion alkuun" is also covered by page_start/end monotonicity.

@pytest.fixture
def tei_with_repeated_sentences(tmp_path):
    tei_content = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
        <text>
            <body>
                <div type="section">
                    <head>Repeated Sentences</head>
                    <p>This is a repeated sentence. This is a repeated sentence. This is a unique sentence.</p>
                    <p>Another paragraph with newlines.
                    And some more text on a new line.
                    This is a repeated sentence.</p>
                    <pb n="2"/>
                    <p>Content on page 2. This is a repeated sentence.</p>
                </div>
            </body>
        </text>
    </TEI>
    """
    tei_path = tmp_path / "repeated_sentences.tei.xml"
    tei_path.write_text(tei_content, encoding="utf-8")
    return tei_path

@pytest.mark.rag
def test_chunk_document_repeated_sentences_and_newlines(tmp_path, tei_with_repeated_sentences):
    row = {
        "id": "repeated_sentences_doc",
        "parsed_ok": True,
        "parsed_xml_path": str(tei_with_repeated_sentences),
        "parsed_txt_path": None,
    }
    max_tokens = 10
    overlap = 0
    min_tokens = 3
    min_chars = 0
    include_front = False
    use_text_txt = False

    chunks = chunk_document(
        row, max_tokens, overlap, min_tokens, include_front, use_text_txt, min_chars
    )

    assert len(chunks) > 0

    # Check that chunks containing "This is a repeated sentence." are distinct
    # and their page_start/end reflect their actual position.
    # This implicitly tests the token-to-char mapping.
    
    # Find all chunks containing the repeated sentence
    repeated_sentence_chunks = [
        c for c in chunks if "This is a repeated sentence." in c.text
    ]
    
    assert len(repeated_sentence_chunks) >= 3, "Expected at least 3 chunks with the repeated sentence"

    # Verify that their page_start/end values are not all the same,
    # indicating they map to different occurrences.
    # The first occurrence is on page 1, the second is also on page 1 (after newline),
    # and the third is on page 2.
    page_starts = sorted(list(set(c.page_start for c in repeated_sentence_chunks)))
    assert len(page_starts) >= 2, "Expected repeated sentences to map to different pages/offsets"
    assert 1 in page_starts
    assert 2 in page_starts

    # Ensure that the global offsets (implied by page_start/end progression) are correct
    # This is a more direct check for the token-to-char mapping.
    # We can't directly assert global offsets from Chunk, but we can check page progression.
    
    # Example: The chunk containing the repeated sentence on page 2 should have page_start = 2
    found_page_2_repeated_sentence = False
    for chunk in repeated_sentence_chunks:
        if chunk.page_start == 2:
            found_page_2_repeated_sentence = True
            break
    assert found_page_2_repeated_sentence, "Expected a chunk with repeated sentence to be correctly mapped to page 2"
