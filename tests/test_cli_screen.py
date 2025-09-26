"""CLI screening command tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from la_pkg import cli as cli_module
from la_pkg.schema import MERGED_SCHEMA


def test_la_screen_creates_outputs(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)

    # Create data/cache directory for logs
    (tmp_path / "data" / "cache").mkdir(parents=True)

    input_path = Path("merged.parquet")
    output_path = Path("screened.parquet")

    df = pd.DataFrame(
        [
            {
                "id": "R1",
                "title": "Genomic cancer screening",
                "abstract": "genomic cancer screening improves survival",
                "authors": ["Author One"],
                "year": 2022,
                "venue": "Journal One",
                "doi": "",
                "url": "http://example.org/1",
                "source": "test",
                "score": 0.9,
                "reasons": [],
            },
            {
                "id": "R2",
                "title": "Manufacturing processes",
                "abstract": "manufacturing process optimization case study",
                "authors": ["Author Two"],
                "year": 2021,
                "venue": "Journal Two",
                "doi": "",
                "url": "http://example.org/2",
                "source": "test",
                "score": 0.1,
                "reasons": ["not-relevant"],
            },
            {
                "id": "R3",
                "title": "Editorial en espanol",
                "abstract": "editorial sobre resultados",
                "authors": ["Author Three"],
                "year": 2015,
                "venue": "Journal Three",
                "doi": "",
                "url": "http://example.org/3",
                "source": "test",
                "score": 0.2,
                "reasons": ["not-research"],
            },
            {
                "id": "R4",
                "title": "Lung cancer detection",
                "abstract": "lung cancer screening trial data",
                "authors": ["Author Four"],
                "year": 2019,
                "venue": "Journal Four",
                "doi": "",
                "url": "http://example.org/4",
                "source": "test",
                "score": 0.8,
                "reasons": [],
            },
        ]
    )
    df.to_parquet(input_path, index=False, schema=MERGED_SCHEMA)

    result = runner.invoke(
        cli_module.app,
        [
            "screen",
            "--in",
            str(input_path),
            "--out",
            str(output_path),
            "--recall",
            "0.8",
            "--year-min",
            "2018",
            "--drop-non-research",
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()

    screened = pd.read_parquet(output_path)
    assert {"id", "probability", "label", "reasons"}.issubset(screened.columns)
    assert screened["probability"].between(0.0, 1.0).all()
    included_reasons = screened.loc[screened["label"] == "included", "reasons"]
    assert included_reasons.apply(len).sum() == 0

    flagged = screened.loc[screened["id"] == "R3"].iloc[0]
    assert flagged["label"] == "excluded"
    assert len(flagged["reasons"]) > 0

    log_path = Path("data/cache/screen_log.csv")
    assert log_path.exists()
    log_df = pd.read_csv(log_path)
    last = log_df.iloc[-1]
    assert int(last["identified"]) == len(df)
    assert last["engine"] == "scikit"
    assert 0.0 <= float(last["threshold_used"]) <= 1.0
    assert int(last["excluded_rules"]) == 1
