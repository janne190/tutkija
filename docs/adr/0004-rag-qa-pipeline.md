# 0004-rag-qa-pipeline

## Status

Accepted

## Context

This ADR outlines the design and implementation plan for the RAG (Retrieval Augmented Generation) and QA (Question Answering) pipeline. The goal is to enable the system to answer questions based on a corpus of TEI documents, providing answers with citations and page numbers.

## Decision

We will implement a RAG and QA pipeline with the following key components and decisions:

1.  **Branch:** `feat/rag-paperqa2`
2.  **Dependencies:**
    *   `google-generativeai` for LLM and embeddings.
    *   `chromadb` for vector database (with `faiss-cpu` as an alternative).
    *   `rank-bm25` for BM25 fallback.
    *   `tiktoken` for text tokenization (with `tokenizers` as an alternative).
    *   `pandas`, `pyarrow`, `pydantic` for data handling and validation.
3.  **Chunking TEI:**
    *   **Input:** `data/cache/parsed_index.parquet`
    *   **Output:** `data/cache/chunks.parquet` and `data/cache/chunk_stats.json`
    *   **Logic:** Read `tei.xml`, extract logical sections (`front`, `body/div`), chunk within sections (~800-1200 tokens, 100-150 overlap).
    *   **Fallback:** If `parsed_xml_path` is missing/invalid or produces no chunks, fallback to `parsed_txt_path` if `--use-text-txt` is enabled.
    *   **Front Matter:** `--include-front` forces chunking of `front` even if below `min_tokens`.
    *   **Minimums:** `--min-tokens` and `--min-chars` control minimum chunk size.
    *   **Metadata:** `paper_id`, `section_id`, `section_title`, `chunk_id`, `text`, `n_tokens`, `page_start`, `page_end`, `file_path`, `source` (either "tei" or "text").
    *   **Page Numbers:** Extracted from `tei:pb/@n` with **global offsets** calculated from the flattened document text. If missing, `null`.
    *   **Exclusions:** `back` section (references) will be excluded from indexing.
    *   **Zero Chunks:** If no chunks are generated, an empty Parquet file with the correct schema is written to prevent crashes.
4.  **Vector Database + Hybrid Search:**
    *   **Target:** `data/index/chroma/` (or `data/index/faiss/`), and `data/cache/index_meta.json`.
    *   **Embeddings:** `EMBED_PROVIDER=google`, `EMBED_MODEL=text-embedding-004`.
    *   **API Key:** Reads `GEMINI_API_KEY` or `GOOGLE_API_KEY` from `.env`.
    *   **Index Build:** CLI command `la rag index build`.
    *   **Hybrid Retrieval:** Combines BM25 (top-N candidates) and vector search (Chroma top-K candidates). The union of `chunk_id`s from both methods is retrieved from Chroma. A blended scoring mechanism is applied: `score = w_vec*(-rank_vec) + w_bm25*(-rank_bm25)`, where `rank_vec` and `rank_bm25` are 0-based ranks from vector and BM25 searches respectively, and `w_vec` (e.g., 1.0) and `w_bm25` (e.g., 0.5) are configurable weights. The final top-K chunks are selected based on this blended score.
    *   **Chroma `ids`:** Set to `chunk_id` (string).
    *   **Metadata in Chroma:** `chunk_id`, `paper_id`, `section`, `page_start`, `page_end` are preserved.
5.  **PaperQA2-like QA:**
    *   **Goal:** Produce answers with claims, 2+ distinct sources, and page numbers.
    *   **Approach:**
        1.  `retrieve(question, k=6)` to get relevant chunks using hybrid search.
        2.  Prompt Gemini with a system/prompt-template to generate an answer, forcing structured JSON output with claims and citations in the format `{"answer": "...", "claims": [{"text":"...", "citations":[{"paper_id":"...", "page_start":1, "page_end":1, "quote":"..."}]}]}`.
        3.  Parse the LLM response into a Pydantic `QAResult` model, with a fallback regex parser for malformed JSON.
    *   **Output Model:** `Citation`, `Claim`, `QAResult` Pydantic models as specified in the task.
    *   **Output Format:** `qa.jsonl` (JSONL, one row per question).
    *   **Logging:** `data/logs/qa_audit.csv` with timestamp, question, initial_k, sources_used_initial, guardrail_triggered, retry_k, final_sources_used, final_answer_len.
6.  **Guardrails:**
    *   **Source Count:** If `sources_used < 2`, retry LLM once with a wider `k` (e.g., `k+4`). Log this event to `qa_audit.csv`.
    *   **Missing Page Numbers:** `page_start/end = null` is handled in chunking and reflected in citations.
    *   **Context Packing:** Prioritize chunks by combined BM25 and vector rank, merge and deduplicate paper-internal overlaps, limit context to ~8-12k tokens for LLM.
7.  **Tests:**
    *   **Unit Tests:** `test_chunker.py` (section boundaries, token count, overlap, page order, front/text.txt fallback, min_chars), `test_retriever.py` (BM25+vector returns correct paper, empty chunks, API key handling).
    *   **Integration Tests:** `tests/rag/test_qa_smoke.py` (mock LLM for structured JSON, 2+ sources, citations with page numbers, guardrail trigger).
    *   **Golden Snapshot:** `qa.jsonl` top-level fields check.
    *   **CI:** Nightly build of small demo index, run 3 questions, artifact `qa.jsonl`, schema check, `sources_used >= 2`.
8.  **Documentation:**
    *   **README:** New "RAG & QA" section with 3 commands (chunk, index, qa), instructions for changing providers, `GEMINI_API_KEY` environment variable, and `qa.jsonl` interpretation.
    *   **ADR:** This document (`docs/adr/0004-rag-qa-pipeline.md`) detailing decisions, risks, and metrics.

## Risks

*   **Missing Page Numbers:** Some TEI files may lack page number information, requiring robust handling and auditing.
*   **LLM Hallucinations:** LLM might generate citations or page numbers that are not present in the provided context. Guardrails and prompt engineering will mitigate this.
*   **Performance:** Indexing and retrieval performance with large corpora needs to be monitored and optimized.

## Metrics and Definition of Done (DoD)

*   **DoD:**
    *   `la rag chunk` produces `chunks.parquet` and logs chunk count, `%fallback_used`, and `%missing_pages`.
    *   `la rag index build` creates index and `index_meta.json`; `la rag index stats` functions.
    *   `la qa` produces `qa.jsonl` with claims, sources, and pages, and logs guardrail events to `qa_audit.csv`.
    *   All unit and integration tests (rag-smoke) pass.
    *   `README.md` and `docs/adr/0004-rag-qa-pipeline.md` are updated.
*   **Exit (Phase 5):**
    *   For 3 test questions, each answer has 2+ distinct sources and page numbers in at least some citations.
    *   `data/index/*` and `data/output/qa.jsonl` are generated.
    *   Verification script confirms each claim has >=1 citation.
