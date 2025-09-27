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

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def main() -> None:
    """Tutkija CLI commands."""


def _parse_seed_option(seeds: str | None) -> list[str]:
    return [s.strip() for s in (seeds or "").split(",") if s.strip()]


def _norm_reasons(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        return [] if not text else [text]
    try:  # Lazy import to avoid hard dependency at runtime
        import numpy as np  # type: ignore[import-untyped]

        if isinstance(value, np.ndarray):
            return [
                str(item)
                for item in value.tolist()
                if str(item).strip()
            ]
    except Exception:  # pragma: no cover - best effort normalisation
        pass
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    if hasattr(value, "tolist"):
        try:
            items = value.tolist()  # type: ignore[call-arg]
            return [str(item) for item in items if str(item).strip()]
        except Exception:  # pragma: no cover - defensive
            pass
    return [str(value)] if str(value).strip() else []


def _load_env_example() -> str:
    config_path = Path("config.example.toml")
    if not config_path.exists():
        raise FileNotFoundError("Missing config.example.toml")
    return config_path.read_text(encoding="utf-8")


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
    limit: int | None = typer.Option(
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
    merged_df.to_parquet(out, index=False)

    if save_single:
        _write_source_parquet(
            Path("data/cache/search_openalex.parquet"), oa_result.papers
        )
        _write_source_parquet(Path("data/cache/search_pubmed.parquet"), pubmed_papers)
        _write_source_parquet(Path("data/cache/search_arxiv.parquet"), arxiv_papers)

    log_path = Path("data/cache/merge_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_entry = {
        "topic": chosen_topic,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "per_source_counts": json.dumps(stats.get("per_source", {})),
        "duplicates_by_doi": stats.get("dup_doi", 0),
        "duplicates_by_title": stats.get("dup_title", 0),
        "filtered_by_rules": stats.get("filtered", 0),
        "final_count": len(merged_df),
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
        f"per_source={stats.get('per_source', {})} "
        f"dup_doi={stats.get('dup_doi', 0)} "
        f"dup_title={stats.get('dup_title', 0)} "
        f"filtered={stats.get('filtered', 0)} "
        f"final={len(merged_df)} "
        f"path={out}"
    )


@app.command(name="screen")
def screen(
    input_path: Path = typer.Option(
        Path("data/cache/merged.parquet"),
        "--in",
        help="Syote, yhdistetty kirjasto Parquet muodossa",
        show_default=True,
    ),
    output_path: Path = typer.Option(
        Path("data/cache/screened.parquet"),
        "--out",
        help="Tuloksen sijainti Parquet muodossa",
        show_default=True,
    ),
    recall: float = typer.Option(
        0.9,
        "--recall",
        min=0.0,
        max=1.0,
        help="Tavoiteltu recall raja",
        show_default=True,
    ),
    engine: str = typer.Option(
        "scikit",
        "--engine",
        help="Scorauksen moottori (scikit tai asreview)",
        show_default=True,
    ),
    seeds: str | None = typer.Option(
        None,
        "--seeds",
        help="Tunnetusti relevanttien julkaisujen id:t (pilkuilla eroteltu)",
    ),
    min_year: int | None = typer.Option(
        None,
        "--min-year",
        help="Pienin sallittu julkaisu vuosi (rules)",
    ),
    allowed_lang: list[str] = typer.Option(
        ["en", "fi"],
        "--allowed-lang",
        help="Sallitut kielet rules vaiheessa",
        metavar="LANG",
        show_default=True,
    ),
    drop_non_research: bool = typer.Option(
        False,
        "--drop-non-research",
        help="Merkitse editoriaalit/kirjeet uutiset rules vaiheessa",
        is_flag=True,
    ),
) -> None:
    """Suorita esiseulonta ja mallipohjainen luokittelu."""

    engine_name = engine.lower()
    if engine_name not in {"scikit", "asreview"}:
        raise typer.BadParameter(
            "engine tulee olla scikit tai asreview", param_hint="engine"
        )
    if recall <= 0 or recall > 1:
        raise typer.BadParameter(
            "recall tulee olla valilla (0, 1]", param_hint="recall"
        )
    if not input_path.exists():
        typer.secho(f"Syotetta ei loydy: {input_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    df = pd.read_parquet(input_path)

    rules_frame, rule_counts = apply_rules(
        df,
        year_min=min_year,
        allowed_lang=list(allowed_lang) if allowed_lang else None,
        drop_non_research=drop_non_research,
    )

    seeds_list = _parse_seed_option(seeds)
    try:
        scored_df, stats = score_and_label(
            rules_frame,
            target_recall=recall,
            seed=7,
            use_asreview=(engine_name == "asreview"),
            seeds=seeds_list,
        )
    except RuntimeError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    scored_df = scored_df.copy()
    if "reasons" not in scored_df.columns:
        scored_df["reasons"] = [[] for _ in range(len(scored_df))]

    scored_df["reasons"] = scored_df["reasons"].apply(_norm_reasons)
    included_mask = scored_df["label"].astype(str).str.lower().eq("included")
    scored_df.loc[included_mask, "reasons"] = scored_df.loc[
        included_mask, "reasons"
    ].apply(lambda _: [])
    bad = int(
        ((scored_df["label"].astype(str).str.lower() == "included")
         & (scored_df["reasons"].astype(str) != "[]")).sum()
    )
    if bad:
        raise RuntimeError(f"reasons must be empty for included, found {bad}")

    scored_df["probability"] = scored_df["probability"].astype(float)
    scored_df["label"] = scored_df["label"].astype(str)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_parquet(output_path, index=False)

    stats["identified"] = int(df.shape[0])
    stats["screened"] = int(scored_df.shape[0])
    stats["engine"] = engine_name
    stats["recall_target"] = float(recall)
    stats["seeds_count"] = len(seeds_list)
    stats["out_path"] = str(output_path)

    log_path = Path("data/cache/screen_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_columns = [
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
    ]
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_row = {"time": timestamp}
    log_row.update({key: stats[key] for key in log_columns[1:]})
    pd.DataFrame([[log_row[col] for col in log_columns]], columns=log_columns).to_csv(
        log_path,
        mode="a",
        header=not log_path.exists(),
        index=False,
    )

    typer.echo(
        "Screen OK | "
        f"identified={stats['identified']} "
        f"screened={stats['screened']} "
        f"excluded_rules={stats['excluded_rules']} "
        f"excluded_model={stats['excluded_model']} "
        f"included={stats['included']} "
        f"engine={stats['engine']} "
        f"threshold={stats['threshold_used']:.3f} "
        f"seeds={stats['seeds_count']} "
        f"fallback={stats['fallback']} "
        f"rules={json.dumps(rule_counts)}"
    )


def _write_source_parquet(path: Path, papers: Iterable[Paper]) -> None:
    records = [paper.model_dump() for paper in papers]
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(path, index=False)


if __name__ == "__main__":
    app()
