"""CLI entrypoints for Tutkija."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .search import OpenAlexSearchResult, append_audit_log, query_openalex

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def main() -> None:
    """Tutkija CLI commands."""


def _load_env_example() -> str:
    env_path = Path(".env.example")
    if not env_path.exists():
        raise FileNotFoundError("Missing .env.example")
    return env_path.read_text(encoding="utf-8")


@app.command(name="hello")
def hello() -> None:
    """Print the configuration template."""
    typer.echo("Tutkija, konfiguraation malli alla")
    try:
        typer.echo(_load_env_example())
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc


@app.command(name="search")
def search(
    topic_arg: Optional[str] = typer.Argument(
        None,
        help="Hakusana OpenAlexille",
        metavar="TOPIC",
    ),
    topic_option: Optional[str] = typer.Option(
        None,
        "--topic",
        "-t",
        help="Hakusana OpenAlexille",
    ),
    out: Path = typer.Option(
        Path("data/cache/search.parquet"),
        "--out",
        help="Polku johon Parquet tiedosto tallennetaan",
        show_default=True,
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        min=1,
        help="Kuinka monta tulosta haetaan (oletus configista)",
    ),
    language: str = typer.Option(
        "auto",
        "--lang",
        help="Kielipreferenssi (auto|en|fi)",
    ),
) -> None:
    """Hae OpenAlexista ja tallenna Parquet tiedostona."""
    topic = topic_option or topic_arg
    if not topic:
        typer.secho(
            "Anna hakusana argumenttina tai --topic optiolla", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    result: OpenAlexSearchResult = query_openalex(topic, limit=limit, language=language)

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - guards CLI in broken envs
        typer.secho("pandas ei ole asennettu", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([paper.model_dump() for paper in result.papers])
    frame.to_parquet(out, index=False)

    append_audit_log(result.metrics, output_path=out)

    typer.echo(
        "OpenAlex search OK | "
        f"found={result.metrics.found} "
        f"unique={result.metrics.unique} "
        f"with_doi={result.metrics.with_doi} "
        f"path={out} "
        f"query='{result.metrics.query_used}' "
        f"fallback={result.metrics.fallback_used} "
        f"lang={result.metrics.language_used} "
        f"queries_tried={len(result.metrics.queries_tried)}"
    )


if __name__ == "__main__":
    app()
