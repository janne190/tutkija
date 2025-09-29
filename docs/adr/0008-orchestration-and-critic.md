# ADR 0008 – Orchestration & Critic

## 0. Status

Proposed

## 1. Context

This ADR outlines the design and implementation for Phase 8, focusing on multi-agent orchestration and the introduction of a critic agent. The goal is to automate the entire research pipeline, ensuring quality control and iterative refinement.

## 2. Decision

### 2.1. Chosen Orchestration Method: LangGraph

**LangGraph** will be used for orchestrating the multi-agent pipeline. This choice is based on its robust capabilities for building stateful, multi-actor applications with cycles, which is essential for the iterative nature of the critic feedback loop.

**Fallback/Alternative**: For CI environments or scenarios where `langgraph` might introduce undue dependencies, a lightweight internal DAG-runner will be implemented as a fallback. This DAG-runner will support basic sequential execution and conditional branching, but without the full state management and cyclical capabilities of LangGraph.

### 2.2. Node List and Responsibilities

The pipeline will consist of the following nodes, each responsible for a specific stage of the research process:

*   **plan**: Generates an execution plan (question list, budget, paths, seeds).
*   **search**: Executes `la search` and deduplicates results.
*   **screen**: Executes `la screen` (or skips if `screened.parquet` is fresh).
*   **ingest**: Handles PDF discovery, download, and parsing (`la pdf discover`, `la pdf download`, `la parse run`).
*   **index**: Chunks and indexes parsed documents (`la rag chunk`, `la rag index build`).
*   **qa**: Answers questions using RAG (`la qa`). Includes a guardrail for source requirements.
*   **write**: Drafts the report (`la write draft`).
*   **prisma**: Generates PRISMA diagrams (`la prisma all`).
*   **critic**: Evaluates claims/citations, generates correction suggestions (diffs), and triggers re-runs if critical issues are found.
*   **finalize**: Compiles a final run report and updates symlinks.

### 2.3. Artifact Paths

All artifacts for a single run will be stored under `data/runs/<timestamp>/...` for auditability and reproducibility. "Golden artifacts" (e.g., `data/cache/`, `data/output/`) will be linked or moved at the end of a successful run.

### 2.4. Budget Rules

A `budget.py` module will track LLM costs (prompt/response tokens), timestamps, and duration per node. Configurable budget limits (`budget_usd`, `max_iterations`) will be enforced.

### 2.5. Error Handling

Errors will be logged to `node_errors.csv` and raised as clear `Typer` errors with actionable instructions. Nodes will check for existing, fresh artifacts to enable skipping and partial success.

### 2.6. Critic Evaluation and Correction Suggestions

The critic agent will evaluate the generated report and QA outputs based on the following rules:

*   **Rule 1**: Each answer must have ≥ `require_sources` unique `paper_id`s.
*   **Rule 2**: Each `Claim` object must have ≥1 `Citation` with valid `page_start`/`page_end` that corresponds to `parsed_index` entries.
*   **Rule 3**: Citations must refer to an indexed chunk (`chunk_id` exists) and the page range must intersect the chunk's page interval.
*   **Rule 4**: In-text citations in the report must be in `[doi:..., pmid:...]` format.

The critic will produce:

*   `data/runs/<ts>/critic_report.md`: Detailed findings and correction instructions.
*   `data/runs/<ts>/report.patch`: A Unix unified diff to `report.qmd`, suggesting corrections like adding missing citations, fixing page numbers, or rephrasing claims if not supported by sources.
*   `qa_audit.csv`: A line-by-line pass/fail audit for QA responses with reasons.

If critical failures are detected (e.g., insufficient sources, missing page numbers), the critic will trigger a re-run by sending the graph back to `qa` (potentially via `index`) for a maximum of `max_iterations`.

## 3. Consequences

*   **Increased Automation**: The entire research pipeline can be executed with a single command, reducing manual effort.
*   **Improved Quality Control**: The critic agent ensures adherence to quality standards, leading to more reliable outputs.
*   **Iterative Refinement**: The feedback loop from the critic allows for automatic correction and improvement of generated reports.
*   **Enhanced Auditability**: Detailed run artifacts and metrics provide a comprehensive audit trail for each execution.
*   **Complexity**: Introducing LangGraph and a multi-agent system adds complexity to the codebase, requiring careful design and testing.
*   **Dependencies**: `langgraph` and `pydantic` will become core dependencies. The fallback DAG-runner mitigates this for minimal environments.
