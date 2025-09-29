# src/la_pkg/write/tables.py
from __future__ import annotations

from pathlib import Path

import pandas as pd


def create_claims_table(qa_path: Path, parsed_index_path: Path) -> str:
    """
    Creates a Markdown table of claims and their citations.
    """
    if not qa_path.exists():
        raise FileNotFoundError(f"QA JSONL file not found: {qa_path}")
    if not parsed_index_path.exists():
        raise FileNotFoundError(f"Parsed index file not found: {parsed_index_path}")

    qa_df = pd.read_json(qa_path, lines=True)
    parsed_df = pd.read_parquet(parsed_index_path)

    # Create a mapping from paper_id to a BibTeX key
    # This is a simplified placeholder. A real implementation would need a robust
    # mapping from the internal ID to the generated BibTeX key.
    def paper_id_to_bibkey(paper_id: str) -> str:
        # Assuming the paper_id might be a DOI or a filename-based ID
        # A proper implementation would look up the real key from the .bib file
        # or use the DOI directly if that's the convention.
        return f"@{paper_id.replace('_', '-').split('.')[0]}"

    table_rows = [
        "| Claim | Citations | Confidence |",
        "|-------|-----------|------------|",
    ]

    for _, row in qa_df.iterrows():
        claim = row.get("question", "N/A").replace("\n", " ")
        confidence = row.get("confidence", "N/A")
        
        citations_str = []
        if "citations" in row and row["citations"]:
            for citation in row["citations"]:
                paper_id = citation.get("paper_id", "unknown")
                bibkey = paper_id_to_bibkey(paper_id)
                pages = citation.get("pages", "")
                page_str = f"p. {pages}" if pages else ""
                citations_str.append(f"[{bibkey} {page_str}]".strip())
        
        citations_md = ", ".join(citations_str) if citations_str else "N/A"
        table_rows.append(f"| {claim} | {citations_md} | {confidence} |")

    return "\n".join(table_rows)


def create_methods_summary_table(logs_dir: Path) -> str:
    """
    Creates a Markdown table summarizing the methods from log files.
    This is a placeholder for a more detailed implementation.
    """
    # In a real implementation, this function would read various log files
    # from `logs_dir` to extract metrics about the search, screening,
    # and parsing steps.
    
    # Placeholder content:
    table = [
        "| Stage | Parameter | Value |",
        "|-------|-----------|-------|",
        "| Search | Sources | OpenAlex, PubMed, arXiv |",
        "| Dedup | Rules | DOI + 90% title similarity |",
        "| Screening | Model | scikit-learn Logistic Regression |",
        "| Screening | Recall Target | 0.90 |",
        "| Parsing | Engine | GROBID v0.8.0 |",
        "| RAG | Chunk Size | 1024 tokens |",
    ]
    return "\n".join(table)
