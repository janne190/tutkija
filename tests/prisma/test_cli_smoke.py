"""
Smoke tests for the 'la prisma' CLI commands.
"""
import pytest
import pandas as pd
import json
from pathlib import Path
from typer.testing import CliRunner
from src.la_pkg.cli import app
from src.la_pkg.prisma.schema import PrismaCounts, IdentificationCounts

runner = CliRunner()

@pytest.fixture
def mock_data_for_cli(tmp_path: Path) -> Path:
    """Creates mock data files for CLI smoke tests."""
    # search_audit.csv
    search_audit_df = pd.DataFrame({
        'source': ['PubMed', 'OpenAlex'],
        'query': ['test1', 'test2'],
        'ts': ['2023-01-01', '2023-01-01'],
        'n_found': [100, 150]
    })
    (tmp_path / "data" / "logs").mkdir(parents=True, exist_ok=True)
    search_audit_path = tmp_path / "data" / "logs" / "search_audit.csv"
    search_audit_df.to_csv(search_audit_path, index=False)

    # merged.parquet
    merged_df = pd.DataFrame({
        'paper_id': [f'id_{i}' for i in range(200)],
        'title': [f'title_{i}' for i in range(200)],
        'doi': [f'doi_{i}' for i in range(200)],
        'reasons': [[] for _ in range(200)]
    })
    (tmp_path / "data" / "cache").mkdir(parents=True, exist_ok=True)
    merged_path = tmp_path / "data" / "cache" / "merged.parquet"
    merged_df.to_parquet(merged_path, index=False)

    # screened.parquet
    screened_df = pd.DataFrame({
        'paper_id': [f'id_{i}' for i in range(180)],
        'label': ['included'] * 130 + ['excluded'] * 50,
        'reasons': [[]] * 130 + [['reason_a']] * 50
    })
    screened_path = tmp_path / "data" / "cache" / "screened.parquet"
    screened_df.to_parquet(screened_path, index=False)

    # parsed_index.parquet
    parsed_index_df = pd.DataFrame({
        'paper_id': [f'id_{i}' for i in range(150)],
        'parsed_ok': [True] * 120 + [False] * 30
    })
    parsed_index_path = tmp_path / "data" / "cache" / "parsed_index.parquet"
    parsed_index_df.to_parquet(parsed_index_path, index=False)

    # qa.jsonl
    qa_data = [
        {"claims": [{"citations": [{"paper_id": "id_0"}, {"paper_id": "id_1"}]}]},
        {"claims": [{"citations": [{"paper_id": "id_2"}]}]},
    ]
    qa_path = tmp_path / "data" / "qa" / "qa.jsonl"
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    with open(qa_path, 'w') as f:
        for item in qa_data:
            f.write(json.dumps(item) + '\n')

    # report.qmd
    (tmp_path / "output" / "report").mkdir(parents=True, exist_ok=True)
    qmd_path = tmp_path / "output" / "report" / "report.qmd"
    qmd_path.write_text("---\ntitle: 'Test Report'\n---\n\n# Introduction\n\n# References\n")

    # prisma_template.svg.j2
    (tmp_path / "docs" / "assets").mkdir(parents=True, exist_ok=True)
    template_path = tmp_path / "docs" / "assets" / "prisma_template.svg.j2"
    template_path.write_text("""
    <svg>
        <text>Records identified: {{ identified.total }}</text>
        <text>(Sources: {{ identified.by_source | tojson }})</text>
        <text>Duplicates removed: {{ duplicates_removed }}</text>
        <text>Records screened: {{ records_screened }}</text>
        <text>Records excluded: {{ records_excluded }}</text>
        <text>Full-text assessed: {{ full_text_assessed }}</text>
        <text>Studies included in synthesis: {{ studies_included }}</text>
    </svg>
    """)

    return tmp_path

def test_prisma_compute_cli(mock_data_for_cli: Path):
    """Tests 'la prisma compute' command."""
    result = runner.invoke(app, [
        "prisma", "compute",
        "--search-audit", str(mock_data_for_cli / "data" / "logs" / "search_audit.csv"),
        "--merged", str(mock_data_for_cli / "data" / "cache" / "merged.parquet"),
        "--screened", str(mock_data_for_cli / "data" / "cache" / "screened.parquet"),
        "--parsed-index", str(mock_data_for_cli / "data" / "cache" / "parsed_index.parquet"),
        "--qa", str(mock_data_for_cli / "data" / "qa" / "qa.jsonl"),
        "--out-json", str(mock_data_for_cli / "data" / "cache" / "prisma_counts.json"),
        "--out-csv", str(mock_data_for_cli / "data" / "cache" / "prisma_counts.csv"),
        "--validation-log", str(mock_data_for_cli / "data" / "logs" / "prisma_validation.csv"),
    ])
    assert result.exit_code == 0, result.stdout
    assert "PRISMA compute OK." in result.stdout
    assert (mock_data_for_cli / "data" / "cache" / "prisma_counts.json").exists()
    assert (mock_data_for_cli / "data" / "cache" / "prisma_counts.csv").exists()

def test_prisma_render_cli(mock_data_for_cli: Path):
    """Tests 'la prisma render' command."""
    # First, create a mock prisma_counts.json
    counts_data = PrismaCounts(
        identified=IdentificationCounts(total=250, by_source={'PubMed': 100, 'OpenAlex': 150}),
        duplicates_removed=50,
        records_screened=180,
        records_excluded=50,
        full_text_assessed=120,
        studies_included=130,
    )
    json_path = mock_data_for_cli / "data" / "cache" / "prisma_counts.json"
    with open(json_path, 'w') as f:
        f.write(counts_data.model_dump_json(indent=2))

    result = runner.invoke(app, [
        "prisma", "render",
        "--counts", str(json_path),
        "--out-dir", str(mock_data_for_cli / "output" / "report"),
        "--engine", "python",
        "--formats", "svg,png",
    ])
    assert result.exit_code == 0, result.stdout
    assert "PRISMA render OK." in result.stdout
    assert (mock_data_for_cli / "output" / "report" / "prisma.svg").exists()
    # PNG generation depends on cairosvg, which might not be in test env
    # assert (mock_data_for_cli / "output" / "report" / "prisma.png").exists()

def test_prisma_attach_cli(mock_data_for_cli: Path):
    """Tests 'la prisma attach' command."""
    # Ensure a mock SVG exists for attachment
    (mock_data_for_cli / "output" / "report" / "prisma.svg").write_text("<svg></svg>")
    
    qmd_path = mock_data_for_cli / "output" / "report" / "report.qmd"
    image_path = mock_data_for_cli / "output" / "report" / "prisma.svg"

    result = runner.invoke(app, [
        "prisma", "attach",
        "--qmd", str(qmd_path),
        "--image", str(image_path),
    ])
    assert result.exit_code == 0, result.stdout
    assert "PRISMA attach OK." in result.stdout
    qmd_content = qmd_path.read_text()
    assert '![PRISMA 2020](prisma.svg){fig-cap="PRISMA 2020 flow diagram"}' in qmd_content

def test_prisma_all_cli(mock_data_for_cli: Path):
    """Tests 'la prisma all' command."""
    # Ensure template exists for render step
    (mock_data_for_cli / "docs" / "assets").mkdir(parents=True, exist_ok=True)
    template_path = mock_data_for_cli / "docs" / "assets" / "prisma_template.svg.j2"
    template_path.write_text("""
    <svg>
        <text>Records identified: {{ identified.total }}</text>
        <text>(Sources: {{ identified.by_source | tojson }})</text>
        <text>Duplicates removed: {{ duplicates_removed }}</text>
        <text>Records screened: {{ records_screened }}</text>
        <text>Records excluded: {{ records_excluded }}</text>
        <text>Full-text assessed: {{ full_text_assessed }}</text>
        <text>Studies included in synthesis: {{ studies_included }}</text>
    </svg>
    """)

    result = runner.invoke(app, [
        "prisma", "all",
        "--search-audit", str(mock_data_for_cli / "data" / "logs" / "search_audit.csv"),
        "--merged", str(mock_data_for_cli / "data" / "cache" / "merged.parquet"),
        "--screened", str(mock_data_for_cli / "data" / "cache" / "screened.parquet"),
        "--parsed-index", str(mock_data_for_cli / "data" / "cache" / "parsed_index.parquet"),
        "--qa", str(mock_data_for_cli / "data" / "qa" / "qa.jsonl"),
        "--out-json", str(mock_data_for_cli / "data" / "cache" / "prisma_counts.json"),
        "--out-csv", str(mock_data_for_cli / "data" / "cache" / "prisma_counts.csv"),
        "--validation-log", str(mock_data_for_cli / "data" / "logs" / "prisma_validation.csv"),
        "--out-dir", str(mock_data_for_cli / "output" / "report"),
        "--engine", "python",
        "--formats", "svg", # Only test SVG for simplicity in CI
        "--qmd", str(mock_data_for_cli / "output" / "report" / "report.qmd"),
        "--image", str(mock_data_for_cli / "output" / "report" / "prisma.svg"),
    ])
    assert result.exit_code == 0, result.stdout
    assert "PRISMA all pipeline completed successfully." in result.stdout
    assert (mock_data_for_cli / "data" / "cache" / "prisma_counts.json").exists()
    assert (mock_data_for_cli / "output" / "report" / "prisma.svg").exists()
    qmd_content = (mock_data_for_cli / "output" / "report" / "report.qmd").read_text()
    assert '![PRISMA 2020](prisma.svg){fig-cap="PRISMA 2020 flow diagram"}' in qmd_content
