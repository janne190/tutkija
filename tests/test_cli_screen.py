"""CLI screening command tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from la_pkg import cli as cli_module


def test_la_screen_creates_outputs(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)

    input_path = Path("merged.parquet")
    output_path = Path("screened.parquet")

    df = pd.DataFrame(
        [
            {
                "id": "R1",
                "title": "Genomic cancer screening",
                "abstract": "genomic cancer screening improves survival",
                "language": "en",
                "year": 2022,
                "type": "article",
                "gold_label": "included",
                "reasons": [],
            },
            {
                "id": "R2",
                "title": "Manufacturing processes",
                "abstract": "manufacturing process optimization case study",
                "language": "en",
                "year": 2021,
                "type": "article",
                "gold_label": "excluded",
                "reasons": [],
            },
            {
                "id": "R3",
                "title": "Editorial en espanol",
                "abstract": "editorial sobre resultados",
                "language": "es",
                "year": 2015,
                "type": "editorial",
                "gold_label": "excluded",
                "reasons": [],
            },
            {
                "id": "R4",
                "title": "Lung cancer detection",
                "abstract": "lung cancer screening trial data",
                "language": "en",
                "year": 2019,
                "type": "article",
                "gold_label": "included",
                "reasons": [],
            },
        ]
    )
    df.to_parquet(input_path, index=False)

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
            "--min-year",
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
    expected_cols = {
        "time",
        "identified",
        "screened",
        "excluded_rules",
        "excluded_model",
        "included",
        "engine",
        "recall_target",
        "threshold_used",
        "seeds_count",
        "version",
        "random_state",
        "fallback",
        "out_path",
    }
    assert expected_cols.issubset(log_df.columns)
    assert int(last["identified"]) == len(df)
    assert last["engine"] == "scikit"
    assert 0.0 <= float(last["threshold_used"]) <= 1.0
    assert int(last["excluded_rules"]) == 1
    assert float(last["screened"]) / float(last["identified"]) >= 0.7
    assert last["fallback"] in {"model", "default_prob_0.5", "seed_similarity"}
    assert Path(last["out_path"]).exists()
