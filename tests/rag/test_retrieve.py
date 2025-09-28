import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.la_pkg.rag.index import build_index
from src.la_pkg.rag.qa import retrieve
from chromadb.utils import embedding_functions


# Fixture for a temporary ChromaDB directory
@pytest.fixture
def temp_chroma_dir(tmp_path):
    return tmp_path / "chroma_test_db"


# Fixture for dummy chunks DataFrame
@pytest.fixture
def dummy_chunks_df():
    data = {
        "paper_id": ["paper1", "paper1", "paper2", "paper2", "paper3"],
        "section_id": ["s1", "s2", "s1", "s2", "s1"],
        "section_title": ["Intro", "Methods", "Intro", "Results", "Discussion"],
        "chunk_id": ["p1-s1-0", "p1-s2-0", "p2-s1-0", "p2-s2-0", "p3-s1-0"],
        "text": [
            "This is the introduction of paper 1. It talks about genomic screening.",
            "Paper 1 methods section. We used a novel approach for data analysis.",
            "Introduction to paper 2. Discusses cancer research and new therapies.",
            "Results of paper 2. Significant findings in cancer treatment.",
            "Discussion in paper 3. Concludes on the impact of genomic screening.",
        ],
        "n_tokens": [15, 18, 16, 14, 17],
        "page_start": [1, 2, 1, 3, 4],
        "page_end": [1, 2, 1, 3, 4],
        "file_path": [
            "path/to/p1.tei",
            "path/to/p1.tei",
            "path/to/p2.tei",
            "path/to/p2.tei",
            "path/to/p3.tei",
        ],
        "source": ["tei", "tei", "tei", "tei", "tei"],
    }
    return pd.DataFrame(data)


# Mock embedding function for deterministic tests
class MockEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __call__(self, texts: list[str]) -> list[list[float]]:
        # Return a fixed vector based on the index of the text
        # This makes vector search deterministic
        embeddings = []
        for i, text in enumerate(texts):
            # Create a simple, distinct embedding for each text
            embedding = [float((i + j) % 7) for j in range(128)]  # Example dim 128
            embeddings.append(embedding)
        return embeddings


@pytest.fixture(autouse=True)
def mock_embedding_function():
    with patch(
        "chromadb.utils.embedding_functions.GoogleGenerativeAiEmbeddingFunction",
        new=MockEmbeddingFunction,
    ) as mock_embed_func:
        yield mock_embed_func


# Fixture for building a test index
@pytest.fixture
def test_index(temp_chroma_dir, dummy_chunks_df):
    chunks_path = temp_chroma_dir / "chunks.parquet"
    dummy_chunks_df.to_parquet(chunks_path, index=False)

    # Mock os.getenv for API key
    with patch(
        "os.getenv",
        side_effect=lambda x: "dummy_api_key"
        if x in ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
        else None,
    ):
        build_index(
            chunks_path=chunks_path,
            index_dir=temp_chroma_dir,
            embed_provider="google",
            embed_model="text-embedding-004",
            batch_size=2,
        )
    return temp_chroma_dir, dummy_chunks_df


@pytest.mark.rag
def test_retrieve_hybrid_search(test_index):
    index_dir, chunks_df = test_index
    question = "genomic screening methods"
    k = 2
    embed_model = "text-embedding-004"
    embed_provider = "google"

    retrieved = retrieve(question, index_dir, k, chunks_df, embed_model, embed_provider)

    assert len(retrieved) <= k
    assert len(retrieved) > 0

    # Check if retrieved chunks contain expected metadata
    for chunk in retrieved:
        assert "paper_id" in chunk
        assert "chunk_id" in chunk
        assert "text" in chunk
        assert "page_start" in chunk
        assert "page_end" in chunk

    # Check if relevant chunks are retrieved (based on dummy data)
    retrieved_texts = [c["text"] for c in retrieved]
    assert any("genomic screening" in text for text in retrieved_texts)
    assert any("methods" in text for text in retrieved_texts)


@pytest.mark.rag
def test_retrieve_empty_chunks_df(temp_chroma_dir):
    empty_df = pd.DataFrame(
        columns=[
            "paper_id",
            "section_id",
            "section_title",
            "chunk_id",
            "text",
            "n_tokens",
            "page_start",
            "page_end",
            "file_path",
            "source",
        ]
    )
    chunks_path = temp_chroma_dir / "empty_chunks.parquet"
    empty_df.to_parquet(chunks_path, index=False)

    # Build an empty index
    with patch(
        "os.getenv",
        side_effect=lambda x: "dummy_api_key"
        if x in ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
        else None,
    ):
        build_index(
            chunks_path=chunks_path,
            index_dir=temp_chroma_dir,
            embed_provider="google",
            embed_model="text-embedding-004",
            batch_size=2,
        )

    question = "any question"
    k = 2
    embed_model = "text-embedding-004"
    embed_provider = "google"

    retrieved = retrieve(
        question, temp_chroma_dir, k, empty_df, embed_model, embed_provider
    )
    assert len(retrieved) == 0


@pytest.mark.rag
def test_retrieve_no_api_key_raises_error(test_index):
    index_dir, chunks_df = test_index
    question = "test"
    k = 1
    embed_model = "text-embedding-004"
    embed_provider = "google"

    with patch("os.getenv", return_value=None):  # Simulate missing API key
        with pytest.raises(RuntimeError, match="Missing GEMINI_API_KEY/GOOGLE_API_KEY"):
            retrieve(question, index_dir, k, chunks_df, embed_model, embed_provider)


@pytest.mark.rag
def test_retrieve_bm25_only_hits_when_vector_fails(temp_chroma_dir, dummy_chunks_df):
    index_dir, chunks_df = temp_chroma_dir, dummy_chunks_df
    question = "genomic screening"  # This should hit paper1 and paper3 via BM25
    k = 2
    embed_model = "text-embedding-004"
    embed_provider = "google"

    # Build index with mock embedding function
    chunks_path = index_dir / "chunks.parquet"
    dummy_chunks_df.to_parquet(chunks_path, index=False)
    with patch(
        "os.getenv",
        side_effect=lambda x: "dummy_api_key"
        if x in ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
        else None,
    ):
        build_index(
            chunks_path=chunks_path,
            index_dir=index_dir,
            embed_provider="google",
            embed_model=embed_model,
            batch_size=2,
        )

    # Mock the ChromaDB collection.query to return no vector results
    with patch("chromadb.api.models.Collection.Collection.query") as mock_query:
        mock_query.return_value = {
            "ids": [[]],
            "embeddings": None,
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        retrieved = retrieve(
            question, index_dir, k, chunks_df, embed_model, embed_provider
        )

        assert len(retrieved) > 0, "Expected BM25 hits even if vector search fails"
        assert len(retrieved) <= k

        retrieved_paper_ids = {chunk["paper_id"] for chunk in retrieved}
        # "genomic screening" is in paper1 and paper3
        assert "paper1" in retrieved_paper_ids
        assert "paper3" in retrieved_paper_ids
        assert "paper2" not in retrieved_paper_ids
