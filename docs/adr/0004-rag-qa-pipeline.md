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
    *   **Metadata:** `paper_id`, `section_id`, `section_title`, `chunk_id`, `text`, `n_tokens`, `page_start`, `page_end`, `tei_path`, `source="tei"`.
    *   **Page Numbers:** Attempt to extract from `tei:pb/@n`. If missing, `null`.
    *   **Exclusions:** `back` section (references) will be excluded from indexing.
4.  **Vector Database + Hybrid Search:**
    *   **Target:** `data/index/chroma/` (or `data/index/faiss/`), and `data/cache/index_meta.json`.
    *   **Embeddings:** `EMBED_PROVIDER=google`, `EMBED_MODEL=text-embedding-004`.
    *   **Index Build:** CLI command `la rag index build`.
    *   **Hybrid Retrieval:** BM25 top-50, then vector top-k (e.g., 20) rerank. Fallback to pure BM25 if embeddings fail.
5.  **PaperQA2-like QA:**
    *   **Goal:** Produce answers with claims, 2+ distinct sources, and page numbers.
    *   **Approach:**
        1.  `retrieve(question, k=6)` to get relevant chunks.
        2.  Prompt Gemini with a system/prompt-template to generate an answer, forcing citations in the format `[paper_id:page_start–page_end]`.
        3.  Parse the LLM response into a Pydantic `QAResult` model.
    *   **Output Model:** `Citation`, `Claim`, `QAResult` Pydantic models as specified in the task.
    *   **Output Format:** `qa.jsonl` (JSONL, one row per question).
    *   **Logging:** `data/logs/qa_audit.csv` with timestamp, question, retrieved_k, tokens, duration, sources_used_count.
6.  **Guardrails:**
    *   **Source Count:** If `sources_used < 2`, retry LLM once with a wider `k` (e.g., 10).
    *   **Missing Page Numbers:** Mark `page_start/end = null` and flag `needs_page_review=true` in audit log.
    *   **Context Packing:** Prioritize chunks by combined BM25 and vector rank, merge and deduplicate paper-internal overlaps, limit context to ~8-12k tokens for LLM.
7.  **Tests:**
    *   **Unit Tests:** `test_chunker.py` (section boundaries, token count, overlap, page order), `test_retriever.py` (BM25+vector returns correct paper).
    *   **Integration Tests:** `tests/rag/test_qa_smoke.py` (3 questions, `sources_used >= 2`, at least one claim with `page_start != None`).
    *   **Golden Snapshot:** `qa.jsonl` top-level fields check.
    *   **CI:** Nightly build of small demo index, run 3 questions, artifact `qa.jsonl`, schema check, `sources_used >= 2`.
8.  **Documentation:**
    *   **README:** New "RAG & QA" section with 3 commands (chunk, index, qa), instructions for changing providers, and `qa.jsonl` interpretation.
    *   **ADR:** This document (`docs/adr/0004-rag-qa-pipeline.md`) detailing decisions, risks, and metrics.

## Risks

*   **Missing Page Numbers:** Some TEI files may lack page number information, requiring robust handling and auditing.
*   **LLM Hallucinations:** LLM might generate citations or page numbers that are not present in the provided context. Guardrails and prompt engineering will mitigate this.
*   **Performance:** Indexing and retrieval performance with large corpora needs to be monitored and optimized.

## Metrics and Definition of Done (DoD)

*   **DoD:**
    *   `la rag chunk` produces `chunks.parquet` and logs chunk count and % of chunks with page numbers.
    *   `la rag index build` creates index and `index_meta.json`; `la rag index stats` functions.
    *   `la qa` produces `qa.jsonl` with claims, sources, and pages.
    *   All unit and integration tests (rag-smoke) pass.
    *   `README.md` and `docs/adr/0004-rag-qa-pipeline.md` are updated.
*   **Exit (Phase 5):**
    *   For 3 test questions, each answer has 2+ distinct sources and page numbers in at least some citations.
    *   `data/index/*` and `data/output/qa.jsonl` are generated.
    *   Verification script confirms each claim has >=1 citation.
