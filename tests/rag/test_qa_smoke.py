import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
import json

from src.la_pkg.rag.qa import run_qa, QAResult
from src.la_pkg.rag.index import build_index


# Fixture for a temporary ChromaDB directory
@pytest.fixture
def temp_chroma_dir(tmp_path):
    return tmp_path / "chroma_test_db"


# Fixture for dummy chunks DataFrame with multiple paper_ids
@pytest.fixture
def dummy_chunks_df_qa():
    data = {
        "paper_id": ["paperA", "paperA", "paperB", "paperB", "paperC"],
        "section_id": ["s1", "s2", "s1", "s2", "s1"],
        "section_title": [
            "Intro A",
            "Methods A",
            "Intro B",
            "Results B",
            "Discussion C",
        ],
        "chunk_id": ["pA-s1-0", "pA-s2-0", "pB-s1-0", "pB-s2-0", "pC-s1-0"],
        "text": [
            "This is chunk from paper A. It discusses methods.",
            "Another chunk from paper A. More details on methods.",
            "Chunk from paper B. Focuses on results.",
            "Another chunk from paper B. Key findings.",
            "Chunk from paper C. Concluding remarks.",
        ],
        "n_tokens": [10, 12, 11, 13, 10],
        "page_start": [1, 2, 1, 3, 1],
        "page_end": [1, 2, 1, 3, 1],
        "file_path": [
            "path/to/pA.tei",
            "path/to/pA.tei",
            "path/to/pB.tei",
            "path/to/pB.tei",
            "path/to/pC.tei",
        ],
        "source": ["tei", "tei", "tei", "tei", "tei"],
    }
    return pd.DataFrame(data)


# Fixture for building a test index for QA
@pytest.fixture
def test_qa_index(temp_chroma_dir, dummy_chunks_df_qa):
    chunks_path = temp_chroma_dir / "chunks_qa.parquet"
    dummy_chunks_df_qa.to_parquet(chunks_path, index=False)

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
    return temp_chroma_dir, dummy_chunks_df_qa


@pytest.mark.rag
def test_qa_smoke_structured_output_and_guardrail(test_qa_index, tmp_path):
    index_dir, chunks_df = test_qa_index
    out_path = tmp_path / "qa_output.jsonl"
    audit_path = tmp_path / "qa_audit.csv"
    chunks_path = tmp_path / "chunks_qa.parquet" # Explicitly define chunks_path for run_qa

    question = "What are the key findings and methods?"
    k = 2  # Initial k, expecting guardrail to trigger

    # Mock LLM response to return structured JSON with two different paper_ids
    mock_llm_response_initial = MagicMock()
    mock_llm_response_initial.text = json.dumps(
        {
            "answer": "The key findings indicate X [paperB:1-1] and the methods involved Y [paperA:1-2].",
            "claims": [
                {
                    "text": "The key findings indicate X.",
                    "citations": [
                        {
                            "paper_id": "paperB",
                            "page_start": 1,
                            "page_end": 1,
                            "quote": "findings indicate X",
                        }
                    ],
                },
                {
                    "text": "The methods involved Y.",
                    "citations": [
                        {
                            "paper_id": "paperA",
                            "page_start": 1,
                            "page_end": 2,
                            "quote": "methods involved Y",
                        }
                    ],
                },
            ],
        }
    )

    # Mock LLM response for retry (if guardrail triggers and needs more sources)
    mock_llm_response_retry = MagicMock()
    mock_llm_response_retry.text = json.dumps(
        {
            "answer": "The key findings indicate X [paperB:1-1], methods involved Y [paperA:1-2], and further discussion Z [paperC:1-1].",
            "claims": [
                {
                    "text": "The key findings indicate X.",
                    "citations": [
                        {
                            "paper_id": "paperB",
                            "page_start": 1,
                            "page_end": 1,
                            "quote": "findings indicate X",
                        }
                    ],
                },
                {
                    "text": "The methods involved Y.",
                    "citations": [
                        {
                            "paper_id": "paperA",
                            "page_start": 1,
                            "page_end": 2,
                            "quote": "methods involved Y",
                        }
                    ],
                },
                {
                    "text": "Further discussion Z.",
                    "citations": [
                        {
                            "paper_id": "paperC",
                            "page_start": 1,
                            "page_end": 1,
                            "quote": "discussion Z",
                        }
                    ],
                },
            ],
        }
    )

    with patch(
        "os.getenv",
        side_effect=lambda x: "dummy_api_key"
        if x in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"]
        else None,
    ), patch("google.generativeai.GenerativeModel") as mock_generative_model, patch(
        "src.la_pkg.rag.qa.retrieve"
    ) as mock_retrieve, patch(
        "src.la_pkg.rag.qa.MockOpenAIModel"
    ) as mock_openai_model: # Patch the MockOpenAIModel
        # Configure mock_generative_model to return different responses for initial and retry calls
        mock_generative_model.return_value.generate_content.side_effect = [
            mock_llm_response_initial,
            mock_llm_response_retry,
        ]
        # Configure mock_openai_model to return different responses for initial and retry calls
        mock_openai_model.return_value.generate_content.side_effect = [
            mock_llm_response_initial,
            mock_llm_response_retry,
        ]

        # Mock retrieve to return chunks that allow for the desired citations
        mock_retrieve.side_effect = [
            [  # Initial retrieve call (k=2)
                {
                    "paper_id": "paperA",
                    "section_title": "Methods A",
                    "chunk_id": "pA-s1-0",
                    "text": "methods involved Y",
                    "page_start": 1,
                    "page_end": 2,
                },
                {
                    "paper_id": "paperB",
                    "section_title": "Intro B",
                    "chunk_id": "pB-s1-0",
                    "text": "findings indicate X",
                    "page_start": 1,
                    "page_end": 1,
                },
            ],
            [  # Retry retrieve call (k=6, or k+4)
                {
                    "paper_id": "paperA",
                    "section_title": "Methods A",
                    "chunk_id": "pA-s1-0",
                    "text": "methods involved Y",
                    "page_start": 1,
                    "page_end": 2,
                },
                {
                    "paper_id": "paperB",
                    "section_title": "Intro B",
                    "chunk_id": "pB-s1-0",
                    "text": "findings indicate X",
                    "page_start": 1,
                    "page_end": 1,
                },
                {
                    "paper_id": "paperC",
                    "section_title": "Discussion C",
                    "chunk_id": "pC-s1-0",
                    "text": "discussion Z",
                    "page_start": 1,
                    "page_end": 1,
                },
                {
                    "paper_id": "paperD",
                    "section_title": "Extra D",
                    "chunk_id": "pD-s1-0",
                    "text": "extra content",
                    "page_start": 1,
                    "page_end": 1,
                },
                {
                    "paper_id": "paperE",
                    "section_title": "Extra E",
                    "chunk_id": "pE-s1-0",
                    "text": "more extra content",
                    "page_start": 1,
                    "page_end": 1,
                },
                {
                    "paper_id": "paperF",
                    "section_title": "Extra F",
                    "chunk_id": "pF-s1-0",
                    "text": "even more extra content",
                    "page_start": 1,
                    "page_end": 1,
                },
            ],
        ]

        run_qa(
            question=question,
            index_dir=index_dir,
            k=k,
            llm_provider="openai", # Test with openai provider
            llm_model="gpt-4o-mini",
            out_path=out_path,
            chunks_path=chunks_path,
            audit_path=audit_path,
        )

    assert out_path.exists()
    with open(out_path, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f]

    assert len(results) == 1
    qa_result = QAResult.model_validate(results[0])

    # Assertions for the final (retried) result
    assert len(qa_result.claims) == 3
    assert len(qa_result.sources_used) >= 2  # Should be 3 after retry

    for claim in qa_result.claims:
        assert len(claim.citations) >= 1
        for citation in claim.citations:
            assert citation.paper_id in ["paperA", "paperB", "paperC"]
            assert citation.page_start is not None
            assert citation.page_end is not None
            assert citation.quote != ""

    # Check audit log
    assert audit_path.exists()
    audit_df = pd.read_csv(audit_path)
    assert len(audit_df) == 1
    assert audit_df["guardrail_triggered"].iloc[0] == True
    assert audit_df["initial_k"].iloc[0] == k
    assert audit_df["retry_k"].iloc[0] == k + 4
    assert (
        audit_df["final_sources_used"].iloc[0] == 3
    ) # After retry, should have 3 sources
