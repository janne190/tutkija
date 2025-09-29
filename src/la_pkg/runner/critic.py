# SPDX-FileCopyrightText: 2024-present tutkija <tutkija@tutkija.fi>
#
# SPDX-License-Identifier: MIT

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from la_pkg.runner.schemas import CriticReport, RunState, Patch


class CriticAgent:
    """
    The critic agent evaluates the generated report and QA outputs.
    It checks for citation quality, source requirements, and page number validation.
    """

    def __init__(self, config: Any):
        self.config = config

    def _check_rule_1_sources(self, qa_results: List[Dict[str, Any]]) -> List[str]:
        """
        Rule 1: Each answer must have >= require_sources unique paper_id's.
        """
        findings = []
        for qa_entry in qa_results:
            question = qa_entry.get("question", "N/A")
            citations = qa_entry.get("citations", [])
            unique_paper_ids = set()
            # Use "sources_used" if available, otherwise aggregate from "claims" -> "citations"
            if "sources_used" in qa_entry and qa_entry["sources_used"]:
                unique_paper_ids.update(qa_entry["sources_used"])
            else:
                claims = qa_entry.get("claims", [])
                for claim in claims:
                    citations = claim.get("citations", [])
                    for citation in citations:
                        if citation.get("paper_id"):
                            unique_paper_ids.add(citation.get("paper_id"))

            if len(unique_paper_ids) < self.config.require_sources:
                findings.append(
                    f"FAIL: Question '{question}' has only {len(unique_paper_ids)} unique sources, "
                    f"but requires at least {self.config.require_sources}."
                )
        return findings

    def _check_rule_2_citations_and_pages(
        self, qa_results: List[Dict[str, Any]], parsed_index_df: pd.DataFrame
    ) -> List[str]:
        """
        Rule 2: Each Claim object must have >=1 Citation with valid page_start/page_end
                that corresponds to parsed_index entries.
        """
        findings = []
        for qa_entry in qa_results:
            claims = qa_entry.get("claims", [])
            for claim in claims:
                citations = claim.get("citations", [])
                if not citations:
                    findings.append(f"FAIL: Claim '{claim.get('text', 'N/A')}' has no citations.")
                    continue

                for citation in citations:
                    page_start = citation.get("page_start")
                    page_end = citation.get("page_end")
                    paper_id = citation.get("paper_id")

                    if page_start is None or page_end is None:
                        findings.append(
                            f"FAIL: Citation for paper '{paper_id}' in claim '{claim.get('text', 'N/A')}' "
                            f"is missing page_start or page_end."
                        )
                        continue

                    # Check if page range exists in parsed_index for the given paper_id
                    matching_pages = parsed_index_df[
                        (parsed_index_df["paper_id"] == paper_id)
                        & (parsed_index_df["page_num"] >= page_start)
                        & (parsed_index_df["page_num"] <= page_end)
                    ]
                    if matching_pages.empty:
                        findings.append(
                            f"FAIL: Citation for paper '{paper_id}' (pages {page_start}-{page_end}) "
                            f"in claim '{claim.get('text', 'N/A')}' does not match any pages in parsed_index."
                        )
        return findings

    def _check_rule_3_chunk_references(
        self, qa_results: List[Dict[str, Any]], index_meta: Dict[str, Any]
    ) -> List[str]:
        """
        Rule 3: Citations refer to an indexed chunk (chunk_id found), and page range intersects chunk's page interval.
        """
        findings = []
        chunks_path = index_meta.get("chunks_path")
        if not chunks_path or not Path(chunks_path).exists():
            findings.append("WARNING: Chunks parquet path not found in index metadata or file does not exist. Skipping chunk reference check.")
            return findings

        try:
            chunks_df = pd.read_parquet(chunks_path)
            chunk_metadata = {row["chunk_id"]: row for _, row in chunks_df.iterrows()}
        except Exception as e:
            findings.append(f"ERROR: Failed to load chunks from {chunks_path}: {e}. Skipping chunk reference check.")
            return findings

        for qa_entry in qa_results:
            claims = qa_entry.get("claims", [])
            for claim in claims:
                citations = claim.get("citations", [])
                for citation in citations:
                    chunk_id = citation.get("chunk_id")
                    page_start = citation.get("page_start")
                    page_end = citation.get("page_end")
                    paper_id = citation.get("paper_id")

                    if chunk_id:
                        if chunk_id not in chunk_metadata:
                            findings.append(
                                f"FAIL: Citation for chunk_id '{chunk_id}' in claim '{claim.get('text', 'N/A')}' "
                                f"does not refer to an existing indexed chunk."
                            )
                            continue

                        chunk_row = chunk_metadata[chunk_id]
                        chunk_page_start = chunk_row.get("page_start")
                        chunk_page_end = chunk_row.get("page_end")

                        if chunk_page_start is None or chunk_page_end is None:
                            findings.append(
                                f"WARNING: Chunk '{chunk_id}' is missing page interval metadata."
                            )
                            continue

                        # Check for intersection
                        if not (
                            max(page_start, chunk_page_start) <= min(page_end, chunk_page_end)
                        ):
                            findings.append(
                                f"FAIL: Citation for chunk_id '{chunk_id}' (pages {page_start}-{page_end}) "
                                f"in claim '{claim.get('text', 'N/A')}' does not intersect with chunk's page interval ({chunk_page_start}-{chunk_page_end})."
                            )
                    else:
                        # Fallback: if chunk_id is missing, validate paper_id + page_start/page_end
                        if paper_id and page_start is not None and page_end is not None:
                            # This check is already covered by _check_rule_2_citations_and_pages
                            # We can add a warning here if we want to enforce chunk_id presence
                            findings.append(
                                f"WARNING: Citation for paper '{paper_id}' (pages {page_start}-{page_end}) "
                                f"in claim '{claim.get('text', 'N/A')}' is missing 'chunk_id'. "
                                f"Falling back to paper_id + page range validation."
                            )
                        else:
                            findings.append(
                                f"FAIL: Citation in claim '{claim.get('text', 'N/A')}' is missing 'chunk_id', 'paper_id', or page range."
                            )
        return findings

    def _check_rule_4_link_format(self, report_qmd_content: str) -> List[str]:
        """
        Rule 4: In-text citations in the report `[doi:..., pmid:...]` format — correct/format if missing.
        (This rule is more about formatting/correction, so it might generate suggestions rather than just failures)
        """
        findings = []
        # This is a placeholder. Actual implementation would involve regex to find citations
        # and validate their format. For now, we'll just check for a simple pattern.
        if "[doi:" not in report_qmd_content and "[pmid:" not in report_qmd_content:
            findings.append("WARNING: No DOI/PMID formatted citations found in the report. "
                            "Ensure in-text citations follow `[doi:..., pmid:...]` format.")
        return findings

    def _generate_patch(self, report_qmd_path: Path, findings: List[str]) -> Optional[Patch]:
        """
        Generates a unified diff patch for the report.qmd based on findings.
        This is a highly simplified placeholder. A real implementation would
        require more sophisticated NLP and diff generation.
        """
        if not findings:
            return None

        original_content = report_qmd_path.read_text()
        # In a real scenario, you'd parse the report, apply corrections based on findings,
        # and then generate a diff. For this example, we'll just add a comment.
        correction_comment = "\n<!-- Critic suggested corrections based on findings: -->\n" + "\n".join(
            [f"<!-- - {f} -->" for f in findings]
        )
        new_content = original_content + correction_comment

        # Generate a temporary patch file
        patch_file_path = report_qmd_path.with_suffix(".patch")
        # This is a very basic diff. A proper diff utility would be needed.
        diff_content = f"""--- a/{report_qmd_path.name}
+++ b/{report_qmd_path.name}
@@ -X,Y +A,B @@
{original_content}
{correction_comment}
"""
        patch_file_path.write_text(diff_content)
        return Patch(original_file=report_qmd_path, patch_content=diff_content)

    def run_criticism(
        self,
        qa_results_path: Path,
        report_qmd_path: Path,
        parsed_index_path: Path,
        index_meta_path: Path,
    ) -> CriticReport:
        """
        Runs all critic checks and generates a CriticReport.
        """
        findings: List[str] = []
        correction_suggestions: List[str] = []

        # Load data
        qa_results = []
        if qa_results_path.exists():
            with open(qa_results_path, "r") as f:
                for line in f:
                    qa_results.append(json.loads(line))
        else:
            findings.append(f"ERROR: QA results file not found: {qa_results_path}")

        parsed_index_df = pd.DataFrame()
        if parsed_index_path.exists():
            parsed_index_df = pd.read_parquet(parsed_index_path)
        else:
            findings.append(f"ERROR: Parsed index file not found: {parsed_index_path}")

        index_meta = {}
        if index_meta_path.exists():
            with open(index_meta_path, "r") as f:
                index_meta = json.load(f)
        else:
            findings.append(f"ERROR: Index metadata file not found: {index_meta_path}")

        report_qmd_content = ""
        if report_qmd_path.exists():
            report_qmd_content = report_qmd_path.read_text()
        else:
            findings.append(f"ERROR: Report QMD file not found: {report_qmd_path}")

        # Apply rules
        if qa_results:
            findings.extend(self._check_rule_1_sources(qa_results))
            if not parsed_index_df.empty:
                findings.extend(self._check_rule_2_citations_and_pages(qa_results, parsed_index_df))
            if index_meta:
                findings.extend(self._check_rule_3_chunk_references(qa_results, index_meta))
        if report_qmd_content:
            findings.extend(self._check_rule_4_link_format(report_qmd_content))

        # Determine overall status
        overall_status = "pass"
        if any("FAIL" in f for f in f for f in findings):
            overall_status = "fail"
        elif any("WARNING" in f for f in findings):
            overall_status = "pass_with_warnings"

        # Generate patch (simplified)
        patch_file_path = None
        if overall_status != "pass":
            patch_obj = self._generate_patch(report_qmd_path, findings)
            if patch_obj:
                patch_file_path = report_qmd_path.with_suffix(".patch")
                patch_file_path.write_text(patch_obj.patch_content)
                correction_suggestions.append(f"Apply patch file: {patch_file_path.name}")

        return CriticReport(
            overall_status=overall_status,
            findings=findings,
            correction_suggestions=correction_suggestions,
            patch_file=patch_file_path,
            qa_audit_file=qa_results_path.with_suffix(".csv"),  # Placeholder for actual audit CSV
        )


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
