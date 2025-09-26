"""CLI entrypoints for Tutkija."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd  # type: ignore[import-untyped]
import typer

from .screening import apply_rules, score_and_label
from .search.types import Paper
from .schema import SCREENED_SCHEMA, MERGED_SCHEMA

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
    topic: str | None = typer.Argument(
        None,
        help="Hakusana OpenAlexille (tai anna --topic optiolla)",
    ),
    topic_option: str | None = typer.Option(
        None,
        "--topic",
        help="Vaihtoehtoinen tapa antaa hakusana",
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

    selected_topic = topic_option or topic
    if not selected_topic:
        typer.secho(
            "Anna hakusana argumenttina tai --topic optiolla", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    from .search.openalex import OpenAlexSearchResult, append_audit_log, query_openalex

    result: OpenAlexSearchResult = query_openalex(
        selected_topic, limit=limit, language=language
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([paper.model_dump() for paper in result.papers])
    frame["reasons"] = [[] for _ in range(len(frame))]
    frame.to_parquet(out, index=False, schema=MERGED_SCHEMA)

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


@app.command(name="search-all")
def search_all(
    topic: str | None = typer.Argument(
        None,
        help="Hakusana kaikille tietolahteille",
    ),
    topic_option: str | None = typer.Option(
        None,
        "--topic",
        help="Vaihtoehtoinen tapa antaa hakusana",
    ),
    out: Path = typer.Option(
        Path("data/cache/merged.parquet"),
        "--out",
        help="Polku johon yhdistetty Parquet tallennetaan",
        show_default=True,
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        min=1,
        help="Kuinka monta tulosta haetaan lahteittain (oletus configista)",
    ),
    save_single: bool = typer.Option(
        False,
        "--save-single",
        help="Tallenna myos yksittaisen lahteen tulokset",
    ),
    language: str = typer.Option(
        "auto",
        "--lang",
        help="OpenAlexin kielipreferenssi (auto|en|fi)",
    ),
) -> None:
    """Hae OpenAlex, PubMed ja arXiv, yhdista ja dedupoi."""

    chosen_topic = topic_option or topic
    if not chosen_topic:
        typer.secho("Anna hakusana argumenttina", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    from .search.arxiv import query_arxiv
    from .search.merge import merge_and_filter
    from .search.openalex import query_openalex
    from .search.pubmed import query_pubmed

    oa_result = query_openalex(chosen_topic, limit=limit, language=language)
    pubmed_papers = query_pubmed(chosen_topic, max_results=limit or 200)
    arxiv_papers = query_arxiv(chosen_topic, max_results=limit or 200)

    merged_df, stats = merge_and_filter(oa_result.papers, pubmed_papers, arxiv_papers)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(out, index=False, schema=MERGED_SCHEMA)

    if save_single:
        _write_source_parquet(
            Path("data/cache/search_openalex.parquet"), oa_result.papers
        )
        _write_source_parquet(Path("data/cache/search_pubmed.parquet"), pubmed_papers)
        _write_source_parquet(Path("data/cache/search_arxiv.parquet"), arxiv_papers)

    log_path = Path("data/cache/merge_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # For PRISMA: identified = total from all sources before deduplication
    per_source_stats = stats["per_source"]
    identified = sum(per_source_stats.values())

    dup_doi = stats["dup_doi"]
    dup_title = stats["dup_title"]
    duplicates_removed = dup_doi + dup_title

    log_entry = {
        "topic": chosen_topic,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "identified": identified,
        "duplicates_removed": duplicates_removed,
        "per_source_counts": json.dumps(per_source_stats),
        "duplicates_by_doi": dup_doi,
        "duplicates_by_title": dup_title,
        "out_path": str(out),
    }
    pd.DataFrame([log_entry]).to_csv(
        log_path,
        mode="a",
        header=not log_path.exists(),
        index=False,
    )

    typer.echo(
        "Multisource OK | "
        f"per_source={stats['per_source']} "
        f"dup_doi={stats['dup_doi']} "
        f"dup_title={stats['dup_title']} "
        f"filtered={stats['filtered']} "
        f"final={len(merged_df)} "
        f"path={out}"
    )


@app.command(name="screen")
def screen(
    in_path: Path = typer.Option(
        ..., "--in", help="Path to merged.parquet", exists=True, dir_okay=False
    ),
    out_path: Path = typer.Option(
        ..., "--out", help="Path to save screened.parquet", dir_okay=False
    ),
    recall_target: float = typer.Option(
        0.9, "--recall", min=0.5, max=1.0, help="Target recall for model"
    ),
    engine: str = typer.Option(
        "scikit", "--engine", help="Screening engine: 'scikit' or 'asreview'"
    ),
    seeds: Optional[str] = typer.Option(
        None,
        "--seeds",
        help="Comma-separated list of known relevant paper IDs (e.g., DOIs)",
    ),
    year_min: Optional[int] = typer.Option(
        None, "--year-min", help="Minimum publication year"
    ),
    drop_non_research: bool = typer.Option(
        False, "--drop-non-research", help="Filter out editorials, letters, etc."
    ),
) -> None:
    """Screen articles using rules and a model, save to Parquet."""
    start_time = time.time()

    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Parse seeds string to list
    seeds_list = []
    if seeds:
        for tok in seeds.split(","):
            s = tok.strip()
            if s:
                seeds_list.append(s)

    df = pd.read_parquet(in_path)

    # 1. Apply rules
    df, rule_counts = apply_rules(
        df, year_min=year_min, drop_non_research=drop_non_research
    )

    # 2. Score and label with selected engine
    use_asreview = engine.lower() == "asreview"
    screened_df, screening_stats = score_and_label(
        df,
        target_recall=recall_target,
        use_asreview=use_asreview,
        seeds=seeds_list if seeds_list else None,
    )

    # 3. Ensure reasons consistency - empty list for included records
    # Double check reasons consistency
    # Verify that included records have empty reasons
    included_mask = screened_df["label"] == "included"
    bad_reasons = (
        screened_df[included_mask]["reasons"]
        .apply(lambda x: isinstance(x, list) and len(x) > 0)
        .sum()
    )
    if bad_reasons:
        raise RuntimeError(
            f"reasons must be empty for included records, found {bad_reasons} violations"
        )

    # Ensure reasons is list for included records
    screened_df.loc[screened_df["label"] == "included", "reasons"] = screened_df.loc[
        screened_df["label"] == "included", "reasons"
    ].apply(lambda x: [] if isinstance(x, list) else [])

    # Write screened data with schema control
    out_path.parent.mkdir(parents=True, exist_ok=True)
    screened_df.to_parquet(out_path, index=False, schema=SCREENED_SCHEMA)

    # 4. Write log
    log_path = Path("data/cache/screen_log.csv")

    # PRISMA fields
    excluded_count = (
        screening_stats["excluded_rules"] + screening_stats["excluded_model"]
    )

    log_entry = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_s": round(time.time() - start_time),
        "identified": screening_stats["identified"],
        "screened": screening_stats["screened"],
        "excluded": excluded_count,
        "included": screening_stats["included"],
        "excluded_rules": screening_stats["excluded_rules"],
        "excluded_model": screening_stats["excluded_model"],
        "engine": screening_stats["engine"],
        "recall_target": screening_stats["recall_target"],
        "threshold_used": screening_stats["threshold_used"],
        "seeds_count": screening_stats["seeds_count"],
        "version": screening_stats["version"],  # Added
        "random_state": screening_stats["random_state"],  # Added
        "fallback": screening_stats["fallback"],  # Added
        "in_path": str(in_path),
        "out_path": str(out_path),
    }

    log_df = pd.DataFrame([log_entry])
    log_df.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

    # 5. Print metrics
    typer.echo("Screening complete.")
    typer.echo("  Identified (pre-dedup): see merge log")
    typer.echo(f"  Screened (post-dedup): {screening_stats['screened']}")
    typer.echo(f"  Excluded: {excluded_count}")
    typer.echo(f"    - By rules: {screening_stats['excluded_rules']} {rule_counts}")
    typer.echo(f"    - By model: {screening_stats['excluded_model']}")
    typer.echo(f"  Included: {screening_stats['included']}")
    typer.echo(f"  Threshold used: {screening_stats['threshold_used']:.4f}")
    typer.echo(f"  Output written to: {out_path}")


@app.command(name="screen-metrics")
def screen_metrics(
    in_path: Path = typer.Option(
        Path("data/cache/screened.parquet"),
        "--in",
        help="Path to screened.parquet",
        exists=True,
        dir_okay=False,
    ),
) -> None:
    """Print screening metrics in a machine-readable format for audit."""
    df = pd.read_parquet(in_path)

    reasons_on_included = 0
    if "reasons" in df.columns and "label" in df.columns:
        included_df = df[df["label"] == "included"]
        reasons_on_included = included_df["reasons"].apply(lambda x: len(x) > 0).sum()

    metrics = {"reasons_on_included": int(reasons_on_included)}
    typer.echo(json.dumps(metrics))


def _write_source_parquet(path: Path, papers: Iterable[Paper]) -> None:
    records = [paper.model_dump() for paper in papers]
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df["reasons"] = [[] for _ in range(len(df))]
    df["score"] = [0.5 for _ in range(len(df))]
    df.to_parquet(path, index=False, schema=MERGED_SCHEMA)


if __name__ == "__main__":
    app()
