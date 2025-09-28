import pytest
import pandas as pd
from lxml import etree

from src.la_pkg.rag.chunk import (
    run_chunking,
    chunk_document,
    _get_page_map,
    _get_page_range,
)


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
                <div type="titlePage"><docTitle><titlePart type="main">Main Title</titlePart></docTitle></div>
            </front>
            <body>
                <div type="section">
                    <head>Section 1</head>
                    <p>This is the first paragraph of section 1.</p>
                    <pb n="2"/>
                    <p>This is the second paragraph of section 1, on page 2.</p>
                </div>
                <div type="section">
                    <head>Section 2</head>
                    <p>This is the first paragraph of section 2.</p>
                    <pb n="3"/>
                    <p>This is the second paragraph of section 2, on page 3.</p>
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
    tree = etree.parse(str(dummy_tei_file))
    page_map = _get_page_map(tree)
    assert len(page_map) == 2
    assert (100, 2) in page_map  # Approximate offset for page 2
    assert (200, 3) in page_map  # Approximate offset for page 3


@pytest.mark.rag
def test_get_page_range(dummy_tei_file):
    tree = etree.parse(str(dummy_tei_file))
    page_map = _get_page_map(tree)

    # Chunk entirely on page 1 (before first pb)
    start, end = _get_page_range(0, 50, page_map)
    assert start == 1
    assert end == 1

    # Chunk spanning page 1 and 2
    start, end = _get_page_range(50, 150, page_map)
    assert start == 1
    assert end == 2

    # Chunk entirely on page 2
    start, end = _get_page_range(110, 180, page_map)
    assert start == 2
    assert end == 2

    # Chunk spanning page 2 and 3
    start, end = _get_page_range(150, 250, page_map)
    assert start == 2
    assert end == 3

    # Chunk entirely on page 3 (after last pb)
    start, end = _get_page_range(210, 280, page_map)
    assert start == 3
    assert end == 3

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
        assert next_chunk.page_start >= current_chunk.page_start

        # If chunks overlap significantly, their page ranges might be identical or shift slightly
        # The main goal is that the page numbers are correctly assigned based on global offsets.
        # This test is more about ensuring the _get_page_range logic works with global offsets
        # and the _flatten_tei provides correct page_breaks.
