# SPDX-FileCopyrightText: 2024-present tutkija <tutkija@tutkija.fi>
#
# SPDX-License-Identifier: MIT

import json
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from src.la_pkg.runner.schemas import CriticReport, RunConfig, RunState


def plan_node(state: RunState, config: RunConfig) -> RunState:
    """
    Generates an execution plan for the research run.

    Args:
        state: The current state of the research run.
        config: The run configuration.

    Returns:
        The updated run state with the execution plan.
    """
    state.current_node = "plan"
    run_id = str(uuid.uuid4())
    run_dir = config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    questions: List[str] = []
    if config.questions:
        if isinstance(config.questions, Path):
            with open(config.questions, "r") as f:
                questions = yaml.safe_load(f)
        else:
            questions = config.questions

    state.run_id = run_id
    state.config = config
    state.plan = {
        "questions": questions,
        "budget_usd": config.budget_usd,
        "max_iterations": config.max_iterations,
        "output_dir": str(run_dir),
        "search_results_path": str(run_dir / "search.parquet"),
        "screened_results_path": str(run_dir / "screened.parquet"),
        "parsed_index_path": str(run_dir / "parsed_index.parquet"),
        "index_meta_path": str(run_dir / "index_meta.json"),
        "qa_results_path": str(run_dir / "qa.jsonl"),
        "report_draft_path": str(run_dir / "report.qmd"),
        "prisma_path": str(run_dir / "prisma.svg"),
        "critic_report_path": str(run_dir / "critic_report.md"),
        "report_patch_path": str(run_dir / "report.patch"),
        "qa_audit_path": str(run_dir / "qa_audit.csv"),
    }
    state.search_results_path = Path(state.plan["search_results_path"])
    state.screened_results_path = Path(state.plan["screened_results_path"])
    state.parsed_index_path = Path(state.plan["parsed_index_path"])
    state.index_meta_path = Path(state.plan["index_meta_path"])
    state.qa_results_path = Path(state.plan["qa_results_path"])
    state.report_draft_path = Path(state.plan["report_draft_path"])
    state.prisma_path = Path(state.plan["prisma_path"])

    return state


def search_node(state: RunState) -> RunState:
    """
    Executes the search command and merges/deduplicates results.

    Args:
        state: The current state of the research run.

    Returns:
        The updated run state with search results paths.
    """
    state.current_node = "search"
    output_dir = state.config.output_dir / state.run_id

    # Assuming 'la search' command exists and outputs to a specified path
    # and 'la search merge' command exists for deduplication
    search_command = [
        "la",
        "search",
        "--topic",
        state.config.topic,
        "--output",
        str(output_dir / "raw_search_results.parquet"),
    ]
    merge_command = [
        "la",
        "search",
        "merge",
        "--input",
        str(output_dir / "raw_search_results.parquet"),
        "--output",
        str(state.search_results_path),
    ]

    try:
        subprocess.run(search_command, check=True, capture_output=True, text=True)
        subprocess.run(merge_command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        state.errors.append(f"Search node failed: {e.stderr}")
        raise

    return state


def screen_node(state: RunState) -> RunState:
    """
    Executes the screening command.

    Args:
        state: The current state of the research run.

    Returns:
        The updated run state with screened results path.
    """
    state.current_node = "screen"

    if state.screened_results_path.exists():
        # Check if the screened.parquet is fresh, for now, just check existence
        # TODO: Implement freshness check based on mtime or hash
        state.warnings.append(f"Skipping screening, {state.screened_results_path} already exists.")
        return state

    screen_command = [
        "la",
        "screen",
        "--input",
        str(state.search_results_path),
        "--output",
        str(state.screened_results_path),
    ]

    try:
        subprocess.run(screen_command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        state.errors.append(f"Screen node failed: {e.stderr}")
        raise

    return state


def ingest_node(state: RunState) -> RunState:
    """
    Handles PDF discovery, download, and parsing.

    Args:
        state: The current state of the research run.

    Returns:
        The updated run state with parsed index path.
    """
    state.current_node = "ingest"
    output_dir = state.config.output_dir / state.run_id
    pdf_dir = output_dir / "pdfs"
    parsed_dir = output_dir / "parsed"

    # la pdf discover
    discover_command = [
        "la",
        "pdf",
        "discover",
        "--input",
        str(state.screened_results_path),
        "--output",
        str(pdf_dir / "discovered_pdfs.parquet"),
    ]
    # la pdf download
    download_command = [
        "la",
        "pdf",
        "download",
        "--input",
        str(pdf_dir / "discovered_pdfs.parquet"),
        "--output-dir",
        str(pdf_dir),
    ]
    # la parse run
    parse_command = [
        "la",
        "parse",
        "run",
        "--input-dir",
        str(pdf_dir),
        "--output-dir",
        str(parsed_dir),
        "--output-index",
        str(state.parsed_index_path),
    ]

    try:
        subprocess.run(discover_command, check=True, capture_output=True, text=True)
        subprocess.run(download_command, check=True, capture_output=True, text=True)
        subprocess.run(parse_command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        state.errors.append(f"Ingest node failed: {e.stderr}")
        raise

    return state


def index_node(state: RunState) -> RunState:
    """
    Chunks and indexes parsed documents.

    Args:
        state: The current state of the research run.

    Returns:
        The updated run state with index metadata path.
    """
    state.current_node = "index"
    output_dir = state.config.output_dir / state.run_id
    chroma_dir = output_dir / "chroma"

    # la rag chunk
    chunk_command = [
        "la",
        "rag",
        "chunk",
        "--input",
        str(state.parsed_index_path),
        "--output",
        str(output_dir / "chunks.parquet"),
    ]
    # la rag index build
    index_command = [
        "la",
        "rag",
        "index",
        "build",
        "--input",
        str(output_dir / "chunks.parquet"),
        "--output-dir",
        str(chroma_dir),
        "--output-meta",
        str(state.index_meta_path),
    ]

    try:
        subprocess.run(chunk_command, check=True, capture_output=True, text=True)
        subprocess.run(index_command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        state.errors.append(f"Index node failed: {e.stderr}")
        raise

    return state


def qa_node(state: RunState) -> RunState:
    """
    Answers questions using RAG, with a guardrail for source requirements.

    Args:
        state: The current state of the research run.

    Returns:
        The updated run state with QA results path.
    """
    state.current_node = "qa"
    questions = state.plan["questions"]
    output_dir = state.config.output_dir / state.run_id
    chroma_dir = output_dir / "chroma"

    for i, question in enumerate(questions):
        qa_command = [
            "la",
            "qa",
            "--question",
            question,
            "--index-dir",
            str(chroma_dir),
            "--output",
            str(state.qa_results_path),
            "--append",  # Append to the same file for multiple questions
            "--top-k",
            str(state.config.top_k),
            "--bm25-k",
            str(state.config.bm25_k),
        ]

        try:
            subprocess.run(qa_command, check=True, capture_output=True, text=True)
            # TODO: Implement guardrail for require_sources and max_iterations
            # This would involve reading the qa_results_path, parsing the output
            # to check source count, and potentially re-running with increased k
            # or raising an error if max_iterations is reached.
        except subprocess.CalledProcessError as e:
            state.errors.append(f"QA node failed for question '{question}': {e.stderr}")
            raise

    return state


def write_node(state: RunState) -> RunState:
    """
    Drafts the report.

    Args:
        state: The current state of the research run.

    Returns:
        The updated run state with report draft path.
    """
    state.current_node = "write"
    output_dir = state.config.output_dir / state.run_id

    write_command = [
        "la",
        "write",
        "draft",
        "--qa-input",
        str(state.qa_results_path),
        "--output",
        str(state.report_draft_path),
    ]

    try:
        subprocess.run(write_command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        state.errors.append(f"Write node failed: {e.stderr}")
        raise

    return state


def prisma_node(state: RunState) -> RunState:
    """
    Generates PRISMA diagrams.

    Args:
        state: The current state of the research run.

    Returns:
        The updated run state with prisma path.
    """
    state.current_node = "prisma"
    output_dir = state.config.output_dir / state.run_id

    prisma_command = [
        "la",
        "prisma",
        "all",
        "--input",
        str(state.screened_results_path),  # Assuming screened results are input for PRISMA
        "--output",
        str(state.prisma_path),
    ]

    try:
        subprocess.run(prisma_command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        state.errors.append(f"Prisma node failed: {e.stderr}")
        raise

    return state


def critic_node(state: RunState) -> RunState:
    """
    Evaluates claims/citations, generates correction suggestions (diffs),
    and triggers re-runs if critical issues are found.

    Args:
        state: The current state of the research run.

    Returns:
        The updated run state with critic report.
    """
    state.current_node = "critic"
    # Assuming CriticAgent is imported and initialized correctly
    from src.la_pkg.runner.critic import CriticAgent

    critic_agent = CriticAgent(state.config)

    try:
        critic_report = critic_agent.run_criticism(
            qa_results_path=state.qa_results_path,
            report_qmd_path=state.report_draft_path,
            parsed_index_path=state.parsed_index_path,
            index_meta_path=state.index_meta_path,
        )
        state.critic_report = critic_report

        # Save critic report
        critic_report_path = Path(state.plan["critic_report_path"])
        critic_report_path.write_text(critic_report.json(indent=2))

        # Save patch if generated
        if critic_report.patch_file and critic_report.patch_file.exists():
            patch_content = critic_report.patch_file.read_text()
            Path(state.plan["report_patch_path"]).write_text(patch_content)

        # TODO: Implement re-run logic based on critic_report.overall_status
        # If "fail" and iterations < max_iterations, set next node to "qa" or "index"
        # For now, just log findings as errors/warnings
        if critic_report.overall_status == "fail":
            state.errors.extend(critic_report.findings)
        elif critic_report.overall_status == "pass_with_warnings":
            state.warnings.extend(critic_report.findings)

    except Exception as e:
        state.errors.append(f"Critic node failed: {e}")
        raise

    return state


def finalize_node(state: RunState) -> RunState:
    """
    Compiles a final run report and updates symlinks.

    Args:
        state: The current state of the research run.

    Returns:
        The updated run state.
    """
    state.current_node = "finalize"
    output_dir = state.config.output_dir / state.run_id

    # Compile run_report.json
    run_report = {
        "run_id": state.run_id,
        "config": state.config.dict(),
        "metrics": [m.dict() for m in state.metrics],
        "total_llm_cost_usd": state.total_llm_cost_usd,
        "total_duration_s": state.total_duration_s,
        "critic_report": state.critic_report.dict() if state.critic_report else None,
        "errors": state.errors,
        "warnings": state.warnings,
        "final_status": "success" if not state.errors else "failed",
    }
    run_report_path = output_dir / "run_report.json"
    run_report_path.write_text(json.dumps(run_report, indent=2))

    # Update "latest successful" symlink
    latest_symlink_dir = state.config.output_dir / "latest"
    if latest_symlink_dir.exists():
        latest_symlink_dir.unlink()  # Remove existing symlink
    latest_symlink_dir.symlink_to(output_dir, target_is_directory=True)

    return state
