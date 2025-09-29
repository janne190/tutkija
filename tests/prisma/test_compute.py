"""
Unit tests for PRISMA count computation.
"""
import pytest
import pandas as pd
import json
from pathlib import Path
from src.la_pkg.prisma.compute import compute_counts, validate_counts, write_counts_json_csv
from src.la_pkg.prisma.schema import PrismaCounts, IdentificationCounts

@pytest.fixture
def mock_data_dir(tmp_path: Path) -> Path:
    """Creates mock data files for testing compute_counts."""
    # search_audit.csv
    search_audit_df = pd.DataFrame({
        'source': ['PubMed', 'OpenAlex', 'arXiv'],
        'query': ['test1', 'test2', 'test3'],
        'ts': ['2023-01-01', '2023-01-01', '2023-01-01'],
        'n_found': [100, 150, 50]
    })
    search_audit_path = tmp_path / "search_audit.csv"
    search_audit_df.to_csv(search_audit_path, index=False)

    # merged.parquet
    merged_df = pd.DataFrame({
        'paper_id': [f'id_{i}' for i in range(250)],
        'title': [f'title_{i}' for i in range(250)],
        'doi': [f'doi_{i}' for i in range(250)],
        'reasons': [[] for _ in range(250)]
    })
    merged_path = tmp_path / "merged.parquet"
    merged_df.to_parquet(merged_path, index=False)

    # screened.parquet
    screened_df = pd.DataFrame({
        'paper_id': [f'id_{i}' for i in range(200)],
        'label': ['included'] * 150 + ['excluded'] * 50,
        'reasons': [[]] * 150 + [['reason_a']] * 50
    })
    screened_path = tmp_path / "screened.parquet"
    screened_df.to_parquet(screened_path, index=False)

    # parsed_index.parquet
    parsed_index_df = pd.DataFrame({
        'paper_id': [f'id_{i}' for i in range(180)],
        'parsed_ok': [True] * 140 + [False] * 40
    })
    parsed_index_path = tmp_path / "parsed_index.parquet"
    parsed_index_df.to_parquet(parsed_index_path, index=False)

    # qa.jsonl (fallback for studies_included)
    qa_data = [
        {"claims": [{"citations": [{"paper_id": "id_0"}, {"paper_id": "id_1"}]}]},
        {"claims": [{"citations": [{"paper_id": "id_2"}]}]},
    ]
    qa_path = tmp_path / "qa.jsonl"
    with open(qa_path, 'w') as f:
        for item in qa_data:
            f.write(json.dumps(item) + '\n')

    return tmp_path

def test_compute_counts_full_path(mock_data_dir: Path):
    """Tests compute_counts with all files present."""
    counts = compute_counts(
        search_audit_path=mock_data_dir / "search_audit.csv",
        merged_path=mock_data_dir / "merged.parquet",
        screened_path=mock_data_dir / "screened.parquet",
        parsed_index_path=mock_data_dir / "parsed_index.parquet",
        qa_path=mock_data_dir / "qa.jsonl",
    )

    assert counts.identified.total == 300
    assert counts.identified.by_source == {'PubMed': 100, 'OpenAlex': 150, 'arXiv': 50}
    assert counts.duplicates_removed == 50  # 300 - 250
    assert counts.records_screened == 200
    assert counts.records_excluded == 50
    assert counts.full_text_assessed == 140 # 150 included & 140 parsed_ok
    assert counts.studies_included == 150

def test_compute_counts_fallback_screened_missing(mock_data_dir: Path):
    """Tests compute_counts when screened.parquet is missing."""
    (mock_data_dir / "screened.parquet").unlink() # Remove screened file

    counts = compute_counts(
        search_audit_path=mock_data_dir / "search_audit.csv",
        merged_path=mock_data_dir / "merged.parquet",
        screened_path=mock_data_dir / "screened.parquet", # Still pass path, but it doesn't exist
        parsed_index_path=mock_data_dir / "parsed_index.parquet",
        qa_path=mock_data_dir / "qa.jsonl",
    )

    assert counts.records_screened == 0
    assert counts.records_excluded == 0
    assert counts.full_text_assessed == 140 # Fallback to parsed_ok only
    assert counts.studies_included == 3 # Fallback to qa.jsonl

def test_compute_counts_fallback_parsed_index_missing(mock_data_dir: Path):
    """Tests compute_counts when parsed_index.parquet is missing."""
    (mock_data_dir / "parsed_index.parquet").unlink() # Remove parsed_index file

    counts = compute_counts(
        search_audit_path=mock_data_dir / "search_audit.csv",
        merged_path=mock_data_dir / "merged.parquet",
        screened_path=mock_data_dir / "screened.parquet",
        parsed_index_path=mock_data_dir / "parsed_index.parquet", # Still pass path
        qa_path=mock_data_dir / "qa.jsonl",
    )

    assert counts.full_text_assessed == 0 # No fallback possible

def test_compute_counts_fallback_qa_missing(mock_data_dir: Path):
    """Tests compute_counts when qa.jsonl is missing."""
    (mock_data_dir / "qa.jsonl").unlink() # Remove qa file

    counts = compute_counts(
        search_audit_path=mock_data_dir / "search_audit.csv",
        merged_path=mock_data_dir / "merged.parquet",
        screened_path=mock_data_dir / "screened.parquet",
        parsed_index_path=mock_data_dir / "parsed_index.parquet",
        qa_path=mock_data_dir / "qa.jsonl", # Still pass path
    )

    assert counts.studies_included == 150 # Primary path from screened.parquet

def test_validate_counts_no_issues():
    """Tests validate_counts with valid counts."""
    counts = PrismaCounts(
        identified=IdentificationCounts(total=300, by_source={'A': 100, 'B': 200}),
        duplicates_removed=50,
        records_screened=200,
        records_excluded=50,
        full_text_assessed=140,
        studies_included=100,
    )
    issues = validate_counts(counts)
    assert not issues

def test_validate_counts_with_issues():
    """Tests validate_counts with invalid counts."""
    counts = PrismaCounts(
        identified=IdentificationCounts(total=100, by_source={'A': 100}),
        duplicates_removed=-10, # FAIL
        records_screened=100,
        records_excluded=110, # FAIL
        full_text_assessed=100,
        studies_included=100,
    )
    issues = validate_counts(counts)
    assert len(issues) == 2
    assert any(issue['rule'] == "duplicates_removed >= 0" and issue['status'] == "FAIL" for issue in issues)
    assert any(issue['rule'] == "records_excluded <= records_screened" and issue['status'] == "FAIL" for issue in issues)

    # Test WARN cases
    counts_warn = PrismaCounts(
        identified=IdentificationCounts(total=300, by_source={'A': 100, 'B': 200}),
        duplicates_removed=50,
        records_screened=200,
        records_excluded=50,
        full_text_assessed=160, # 160 > (200-50=150) -> WARN
        studies_included=160, # 160 > 160 (equal, but if it was 161 it would be WARN)
    )
    issues_warn = validate_counts(counts_warn)
    assert len(issues_warn) == 1
    assert any(issue['rule'] == "full_text_assessed <= records_screened - excluded" and issue['status'] == "WARN" for issue in issues_warn)

def test_write_counts_json_csv(mock_data_dir: Path):
    """Tests writing counts to JSON and CSV."""
    counts = PrismaCounts(
        identified=IdentificationCounts(total=300, by_source={'PubMed': 100, 'OpenAlex': 200}),
        duplicates_removed=50,
        records_screened=200,
        records_excluded=50,
        full_text_assessed=140,
        studies_included=100,
    )
    out_json = mock_data_dir / "output" / "prisma_counts.json"
    out_csv = mock_data_dir / "output" / "prisma_counts.csv"

    write_counts_json_csv(counts, out_json, out_csv)

    assert out_json.exists()
    assert out_csv.exists()

    # Verify JSON content
    with open(out_json, 'r') as f:
        json_content = json.load(f)
    assert json_content['identified']['total'] == 300
    assert json_content['duplicates_removed'] == 50

    # Verify CSV content
    csv_df = pd.read_csv(out_csv)
    assert csv_df['identified_total'][0] == 300
    assert csv_df['duplicates_removed'][0] == 50
    assert csv_df['identified_PubMed'][0] == 100
    assert csv_df['identified_OpenAlex'][0] == 200
