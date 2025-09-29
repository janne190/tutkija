"""
Computes PRISMA counts from various data sources.
"""
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict

from .schema import PrismaCounts, IdentificationCounts

logger = logging.getLogger(__name__)

def compute_counts(
    search_audit_path: Path,
    merged_path: Path,
    screened_path: Optional[Path] = None,
    parsed_index_path: Optional[Path] = None,
    qa_path: Optional[Path] = None,
) -> PrismaCounts:
    """
    Computes the counts for the PRISMA diagram based on pipeline artifacts.
    Includes fallback logic for missing files.
    """
    # 1. Identified
    search_audit_df = pd.read_csv(search_audit_path)
    identified_total = int(search_audit_df['n_found'].sum())
    by_source = search_audit_df.groupby('source')['n_found'].sum().to_dict()
    identification_counts = IdentificationCounts(total=identified_total, by_source=by_source)

    # 2. Duplicates
    merged_df = pd.read_parquet(merged_path)
    n_unique_after_dedupe = len(merged_df)
    duplicates_removed = identified_total - n_unique_after_dedupe

    # 3. Screened and Excluded
    records_screened = 0
    records_excluded = 0
    if screened_path and screened_path.exists():
        screened_df = pd.read_parquet(screened_path)
        records_screened = len(screened_df[screened_df['label'].notna()])
        records_excluded = len(screened_df[screened_df['label'] == 'excluded'])
        logger.info(f"Using '{screened_path}' for screening counts.")
    else:
        logger.warning(f"Screening file '{screened_path}' not found. Counts will be 0.")

    # 4. Full-text assessed
    full_text_assessed = 0
    if screened_path and screened_path.exists() and parsed_index_path and parsed_index_path.exists():
        screened_df = pd.read_parquet(screened_path)
        parsed_df = pd.read_parquet(parsed_index_path)
        included_ids = set(screened_df[screened_df['label'] == 'included']['paper_id'])
        parsed_ok_ids = set(parsed_df[parsed_df['parsed_ok'] == True]['paper_id'])
        full_text_assessed = len(included_ids.intersection(parsed_ok_ids))
        logger.info(f"Using primary path for 'full_text_assessed' (screened ∩ parsed).")
    elif parsed_index_path and parsed_index_path.exists():
        parsed_df = pd.read_parquet(parsed_index_path)
        full_text_assessed = len(parsed_df[parsed_df['parsed_ok'] == True])
        logger.warning(f"Using fallback for 'full_text_assessed' (parsed_ok only).")
    else:
        logger.warning(f"No data available for 'full_text_assessed'. Count will be 0.")

    # 5. Studies included
    studies_included = 0
    if screened_path and screened_path.exists():
        screened_df = pd.read_parquet(screened_path)
        studies_included = len(screened_df[screened_df['label'] == 'included'])
        logger.info(f"Using primary path for 'studies_included' (from screened.parquet).")
    elif qa_path and qa_path.exists():
        with open(qa_path, 'r') as f:
            paper_ids = set()
            for line in f:
                entry = json.loads(line)
                for claim in entry.get('claims', []):
                    for citation in claim.get('citations', []):
                        paper_ids.add(citation['paper_id'])
        studies_included = len(paper_ids)
        logger.warning(f"Using fallback for 'studies_included' (from qa.jsonl).")
    else:
        logger.warning(f"No data available for 'studies_included'. Count will be 0.")

    return PrismaCounts(
        identified=identification_counts,
        duplicates_removed=duplicates_removed,
        records_screened=records_screened,
        records_excluded=records_excluded,
        full_text_assessed=full_text_assessed,
        studies_included=studies_included,
    )

def validate_counts(counts: PrismaCounts) -> List[Dict[str, str]]:
    """Validates the computed PRISMA counts and returns a list of validation issues."""
    issues = []
    
    if counts.duplicates_removed < 0:
        issues.append({"rule": "duplicates_removed >= 0", "status": "FAIL", "details": f"Value was {counts.duplicates_removed}"})
    
    records_after_dedupe = counts.identified.total - counts.duplicates_removed
    if counts.records_screened > records_after_dedupe:
        issues.append({"rule": "records_screened <= identified - duplicates", "status": "FAIL", "details": f"{counts.records_screened} > {records_after_dedupe}"})

    if counts.records_excluded > counts.records_screened:
        issues.append({"rule": "records_excluded <= records_screened", "status": "FAIL", "details": f"{counts.records_excluded} > {counts.records_screened}"})

    records_after_screening = counts.records_screened - counts.records_excluded
    if counts.full_text_assessed > records_after_screening:
        issues.append({"rule": "full_text_assessed <= records_screened - excluded", "status": "WARN", "details": f"{counts.full_text_assessed} > {records_after_screening} (may be due to fallback logic)"})

    if counts.studies_included > counts.full_text_assessed:
        issues.append({"rule": "studies_included <= full_text_assessed", "status": "WARN", "details": f"{counts.studies_included} > {counts.full_text_assessed} (may be due to fallback logic)"})

    return issues

def write_counts_json_csv(counts: PrismaCounts, out_json: Path, out_csv: Path):
    """Writes the PRISMA counts to JSON and CSV files."""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        f.write(counts.model_dump_json(indent=2))
    logger.info(f"Wrote PRISMA counts to {out_json}")

    # Flatten for CSV
    flat_data = {
        "identified_total": counts.identified.total,
        **{f"identified_{k}": v for k, v in counts.identified.by_source.items()},
        "duplicates_removed": counts.duplicates_removed,
        "records_screened": counts.records_screened,
        "records_excluded": counts.records_excluded,
        "full_text_assessed": counts.full_text_assessed,
        "studies_included": counts.studies_included,
    }
    df = pd.DataFrame([flat_data])
    df.to_csv(out_csv, index=False)
    logger.info(f"Wrote PRISMA counts to {out_csv}")
