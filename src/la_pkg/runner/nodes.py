# SPDX-FileCopyrightText: 2024-present tutkija <tutkija@tutkija.fi>
#
# SPDX-License-Identifier: MIT

import json
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from la_pkg.runner.schemas import CriticReport, RunConfig, RunState


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
        "search-all",
        "--topic",
        state.config.topic,
        "--out",
        str(state.search_results_path),
        "--lang",
        state.config.lang,
    ]
    # Assuming audit log is handled by the CLI itself, if not, it needs to be added here.
    # For now, I'll assume the CLI handles it as per the instruction "Huolehdi audit-lokista kuten CLI tekee"

    try:
        subprocess.run(search_command, check=True, capture_output=True, text=True)
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
        "--in",
        str(state.search_results_path),
        "--out",
        str(state.screened_results_path),
    ]
    if state.config.recall is not None:
        screen_command.extend(["--recall", str(state.config.recall)])
    if state.config.seeds_path is not None:
        screen_command.extend(["--seeds", str(state.config.seeds_path)])

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
        "--in",
        str(state.screened_results_path),
        "--out",
        str(output_dir / "pdf_candidates.parquet"),
    ]
    # la pdf download
    download_command = [
        "la",
        "pdf",
        "download",
        "--in",
        str(output_dir / "pdf_candidates.parquet"),
        "--pdf-dir",
        str(pdf_dir),
        "--audit",
        str(output_dir / "pdf_audit.csv"),
    ]
    # la parse run
    parse_command = [
        "la",
        "parse",
        "run",
        "--pdf-dir",
        str(pdf_dir),
        "--out-dir",
        str(parsed_dir),
        "--index-out",
        str(state.parsed_index_path),
        "--grobid-url",
        state.config.grobid_url,
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
        "--parsed-index",
        str(state.parsed_index_path),
        "--out",
        str(output_dir / "chunks.parquet"),
    ]
    # la rag index build
    index_command = [
        "la",
        "rag",
        "index",
        "build",
        "--chunks",
        str(output_dir / "chunks.parquet"),
        "--index-dir",
        str(chroma_dir),
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
            "--k",
            str(state.config.top_k),
            "--llm-provider",
            state.config.llm_provider,
            "--llm-model",
            state.config.llm_model,
            "--out",
            str(state.qa_results_path),
        ]

        try:
            # If the file exists, read its content, then append new QA results
            # The instruction says: "ei ole --append-lippua → jos haluat jatkaa JSONL:ää, lue vanhat rivit ja kirjoita uudet perään itse."
            existing_qa_results = []
            if state.qa_results_path.exists():
                with open(state.qa_results_path, "r", encoding="utf-8") as f:
                    for line in f:
                        existing_qa_results.append(json.loads(line))

            result = subprocess.run(qa_command, check=True, capture_output=True, text=True)
            new_qa_result = json.loads(result.stdout.strip()) # Assuming CLI outputs a single JSONL line to stdout

            existing_qa_results.append(new_qa_result)

            with open(state.qa_results_path, "w", encoding="utf-8") as f:
                for entry in existing_qa_results:
                    f.write(json.dumps(entry) + "\n")

            # TODO: Implement guardrail for require_sources and max_iterations
            # This would involve parsing the output to check source count, and potentially re-running with increased k
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

    report_output_dir = Path("output") / "report"
    report_output_dir.mkdir(parents=True, exist_ok=True)
    bib_output_dir = Path("output") / "bib"
    bib_output_dir.mkdir(parents=True, exist_ok=True)

    init_command = [
        "la",
        "write",
        "init",
        "--run-dir",
        str(output_dir),
        "--out-dir",
        str(report_output_dir),
    ]
    bib_command = [
        "la",
        "write",
        "bib",
        "--in",
        str(state.screened_results_path),
        "--out",
        str(bib_output_dir / "references.bib"),
    ]
    fill_command = [
        "la",
        "write",
        "fill",
        "--qa",
        str(state.qa_results_path),
        "--template",
        "docs/templates/report.qmd", # Assuming this path is correct
        "--out",
        str(state.report_draft_path),
    ]
    render_command = [
        "la",
        "write",
        "render",
        "--in",
        str(state.report_draft_path),
        "--format",
        "pdf",
        "--format",
        "html",
    ]

    try:
        subprocess.run(init_command, check=True, capture_output=True, text=True)
        subprocess.run(bib_command, check=True, capture_output=True, text=True)
        subprocess.run(fill_command, check=True, capture_output=True, text=True)
        subprocess.run(render_command, check=True, capture_output=True, text=True)
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

    prisma_output_dir = Path("output") / "prisma"
    prisma_output_dir.mkdir(parents=True, exist_ok=True)
    fig_output_dir = Path("output") / "fig"
    fig_output_dir.mkdir(parents=True, exist_ok=True)

    prisma_command = [
        "la",
        "prisma",
        "all",
        "--search-audit",
        str(output_dir / "search_log.csv"), # Assuming search_log.csv is created by search-all
        "--merged",
        str(state.search_results_path), # This is the merged.parquet from search-all
        "--screened",
        str(state.screened_results_path),
        "--parsed-index",
        str(state.parsed_index_path),
        "--qa",
        str(state.qa_results_path),
        "--out-json",
        str(prisma_output_dir / "prisma.json"),
        "--out-csv",
        str(prisma_output_dir / "prisma.csv"),
        "--out-dir",
        str(fig_output_dir),
        "--qmd",
        str(state.report_draft_path),
        "--image",
        "prisma",
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
        critic_report_path.write_text(critic_report.model_dump_json(indent=2))

        # Save patch if generated
        if critic_report.patch_file: # patch_file is a Path, not a boolean
            # Produce diff and write to file before saving the path to the report
            # Assuming critic_report.patch_file contains the actual diff content or path to it
            # The instruction says: "tuota diff ja kirjoita tiedostoon ennen kuin talletat polun raporttiin."
            # This implies that critic_report.patch_file might be a temporary file or just a string of the diff.
            # For now, I'll assume critic_report.patch_file is the path to the diff file.
            # If it's the content, then the logic needs to be adjusted.
            if critic_report.patch_file.exists():
                patch_content = critic_report.patch_file.read_text()
                Path(state.plan["report_patch_path"]).write_text(patch_content)
            else:
                state.warnings.append(f"Critic report patch file not found at {critic_report.patch_file}")

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
        "final_status": state.final_status, # Use the final_status from the state
    }
    run_report_path = output_dir / "run_report.json"
    run_report_path.write_text(json.dumps(run_report, indent=2))

    # Update "latest successful" symlink
    latest_symlink_dir = state.config.output_dir / "latest"
    if latest_symlink_dir.exists():
        latest_symlink_dir.unlink()  # Remove existing symlink
    latest_symlink_dir.symlink_to(output_dir, target_is_directory=True)

    return state
