"""CLI entrypoints for Tutkija."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, Any, Mapping, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import typer

from dotenv import load_dotenv  # Import load_dotenv
from .pdf.fetchers import arxiv_pdf_url, pmc_pdf_url
from .screening import apply_rules, score_and_label
from .search.types import Paper
from .prisma.compute import compute_counts, validate_counts, write_counts_json_csv
from .prisma.render import render_prisma_python, render_prisma_r
from .prisma.attach import attach_to_qmd
from .runner.graph import DAGRunner
from .runner.schemas import RunConfig, Provider, RunState


# Load environment variables from .env file
load_dotenv()

app = typer.Typer(add_completion=False, no_args_is_help=True)
pdf_app = typer.Typer(add_completion=False, no_args_is_help=True)
parse_app = typer.Typer(add_completion=False, no_args_is_help=True)
rag_app = typer.Typer(add_completion=False, no_args_is_help=True)
write_app = typer.Typer(add_completion=False, no_args_is_help=True)
prisma_app = typer.Typer(add_completion=False, no_args_is_help=True)
run_app = typer.Typer(add_completion=False, no_args_is_help=True) # New typer for run command

app.add_typer(pdf_app, name="pdf")
app.add_typer(parse_app, name="parse")
app.add_typer(rag_app, name="rag")
app.add_typer(write_app, name="write")
app.add_typer(prisma_app, name="prisma")
app.add_typer(run_app, name="run") # Add run_app to main app

@app.callback()
def main() -> None:
    """Tutkija CLI commands."""


def is_empty(x: Any) -> bool:
    """Check if a value is empty, handling None, pandas/numpy objects, and sequences."""
    if x is None:
        return True
    if hasattr(x, "empty"):  # DataFrame/Series
        return x.empty
    if hasattr(x, "size"):  # ndarray
        return x.size == 0
    try:
        return len(x) == 0  # list, tuple, etc.
    except TypeError:
        return False


def has_reasons(v: Any) -> bool:
    """Check if a value contains any reasons, handling numpy arrays and other types."""
    if isinstance(v, np.ndarray):
        return v.size > 0
    try:
        return len(v) > 0
    except TypeError:
        return bool(v)


def _parse_seed_option(seeds: str | None) -> list[str]:
    return [s.strip() for s in (seeds or "").split(",") if s.strip()]


def _norm_reasons(value: object) -> list[str]:
    if is_empty(value):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if not is_empty(str(item).strip())]
    if isinstance(value, str):
        text = value.strip()
        return [] if is_empty(text) else [text]
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist() if not is_empty(str(item).strip())]
    if isinstance(value, tuple):
        return [str(item) for item in value if not is_empty(str(item).strip())]
    if hasattr(value, "tolist"):
        try:
            items = value.tolist()  # type: ignore[call-arg]
            return [str(item) for item in items if not is_empty(str(item).strip())]
        except Exception:  # pragma: no cover - defensive
            pass
    return [str(value)] if not is_empty(str(value).strip()) else []


def _load_env_example() -> str:
    env_path = Path(".env.example")
    if env_path.exists():
        return env_path.read_text(encoding="utf-8")
    # Fallback: varmistetaan että audit löytää avaimen vaikka tiedosto puuttuisi
    return "OPENAI_API_KEY=\n"


@app.command(name="hello")
def hello() -> None:
    """Tulosta .env.example -malli."""
    typer.echo("Tutkija, konfiguraation malli alla")
    typer.echo(_load_env_example())


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

    # Normaali muoto: lista merkkijonoja
    scored_df["reasons"] = scored_df["reasons"].apply(_norm_reasons)

    # Aseta included-riveille tyhjä lista rivikohtaisesti (.at)
    included_mask = scored_df["label"].astype(str).str.lower().eq("included")
    if getattr(included_mask, "any", None) and included_mask.any():
        for idx in scored_df.index[included_mask]:
            scored_df.at[idx, "reasons"] = []

    # Varmista ilman merkkijonovertailuja
    if getattr(included_mask, "any", None) and included_mask.any():
        bad_reasons = scored_df.loc[included_mask, "reasons"].apply(has_reasons)
        if bad_reasons.any():
            raise RuntimeError(
                f"reasons must be empty for included, found {bad_reasons.sum()}"
            )

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
    stats_map = cast(Mapping[str, Any], stats)
    log_row.update({key: stats_map.get(key, None) for key in log_columns[1:]})

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


def _pdf_provider_metadata(
    row: Mapping[str, object],
) -> tuple[Optional[str], Optional[str], bool]:
    """Return provider metadata for a row without performing network calls."""

    url = arxiv_pdf_url(row)
    if url:
        return "arxiv", url, False
    url = pmc_pdf_url(row)
    if url:
        return "pmc", url, False
    doi = row.get("doi") if isinstance(row, Mapping) else getattr(row, "doi", "")
    doi_str = str(doi or "").strip()
    if doi_str:
        return "unpaywall", None, True
    return None, None, False


@pdf_app.command(name="discover")
def discover_pdfs(
    input_path: Path = typer.Option(
        Path("data/cache/merged.parquet"),
        "--in",
        help="Syöte Parquet tiedosto (esim. la search-all tulos)",
        show_default=True,
    ),
    seed_csv: Path = typer.Option(
        Path("tools/seed_urls.csv"),
        "--seed-csv",
        help="Varapolku PDF-siementen CSV listalle",
        show_default=True,
    ),
    output_path: Path = typer.Option(
        Path("data/cache/pdf_index.parquet"),
        "--out",
        help="Kirjoitettava PDF-indeksi Parquet muodossa",
        show_default=True,
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Rajaa käsiteltävien rivien määrää",
    ),
) -> None:
    """Perusta PDF-indeksi ja rikasta se tarjoajatiedoilla."""

    source_kind = None
    frame: pd.DataFrame | None = None

    if input_path.exists():
        frame = pd.read_parquet(input_path)
        source_kind = "metadata"
    elif seed_csv.exists():
        frame = pd.read_csv(seed_csv)
        source_kind = "seeds"

    if frame is None:
        typer.secho(
            "Syöte puuttuu: odotin Parquet tiedostoa tai siemeniä CSV:stä",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    frame = frame.copy()
    if limit is not None:
        frame = frame.iloc[:limit].copy()

    if "source" not in frame.columns:
        frame["source"] = source_kind
    else:
        frame["source"] = frame["source"].fillna(source_kind)

    if "title" not in frame.columns:
        if "url" in frame.columns:
            fallback_titles = frame["url"].astype(str).fillna("")
        else:
            fallback_titles = pd.Series([""] * len(frame))
        frame["title"] = fallback_titles

    provider_rows = [
        _pdf_provider_metadata(row) for row in frame.to_dict(orient="records")
    ]
    provider_df = pd.DataFrame(
        provider_rows,
        columns=["pdf_provider", "pdf_provider_url", "pdf_needs_unpaywall_email"],
    )
    provider_df.index = frame.index
    frame = pd.concat([frame, provider_df], axis=1)

    if "pdf_path" not in frame.columns:
        frame["pdf_path"] = None
    if "pdf_license" not in frame.columns:
        frame["pdf_license"] = None
    if "has_fulltext" not in frame.columns:
        frame["has_fulltext"] = False

    frame["pdf_needs_unpaywall_email"] = (
        frame["pdf_needs_unpaywall_email"].fillna(False).astype(bool)
    )
    frame["pdf_discovery_source"] = source_kind

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)

    counts = Counter(frame["pdf_provider"].fillna("none"))
    typer.echo(
        "PDF discovery OK | "
        f"rows={len(frame)} "
        f"arxiv={counts.get('arxiv', 0)} "
        f"pmc={counts.get('pmc', 0)} "
        f"unpaywall={counts.get('unpaywall', 0)} "
        f"none={counts.get('none', 0)} "
        f"out={output_path}"
    )


@pdf_app.command(name="download")
def download_pdfs(
    index_path: Path = typer.Option(
        Path("data/cache/pdf_index.parquet"),
        "--in",
        help="PDF-indeksin Parquet polku",
        show_default=True,
    ),
    pdf_dir: Path = typer.Option(
        Path("data/pdfs"),
        "--pdf-dir",
        help="Hakemisto johon PDF:t tallennetaan",
        show_default=True,
    ),
    audit_log: Path = typer.Option(
        Path("data/logs/pdf_audit.csv"),
        "--audit",
        help="CSV loki lataustapahtumista",
        show_default=True,
    ),
    mailto: str = typer.Option(
        "",
        "--mailto",
        help="Unpaywall email parametri (asetetaan myös UNPAYWALL_EMAIL ympäristömuuttujasta)",
    ),
    timeout: int = typer.Option(
        30,
        "--timeout",
        help="HTTP aikakatkaisu sekunneissa",
        show_default=True,
    ),
    retries: int = typer.Option(
        2,
        "--retries",
        help="Kuinka monta uudelleenyritystä",
        show_default=True,
    ),
    throttle: int = typer.Option(
        200,
        "--throttle",
        help="Viive millisekunteina pyyntöjen välillä",
        show_default=True,
    ),
) -> None:
    """Lataa PDF:t ja päivitä indeksi audit-tiedoilla."""

    if not index_path.exists():
        typer.secho(f"Indeksi puuttuu: {index_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    from .pdf.download import download_all

    df = pd.read_parquet(index_path)
    result = download_all(
        df,
        pdf_dir,
        audit_log,
        timeout_s=timeout,
        retries=retries,
        throttle_ms=throttle,
        unpaywall_email=mailto or os.environ.get("UNPAYWALL_EMAIL", ""),
    )

    index_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(index_path, index=False)

    fetched = (
        int(result.get("has_fulltext", pd.Series(dtype=bool)).sum())
        if not result.empty
        else 0
    )
    typer.echo(
        "PDF download OK | "
        f"downloaded={fetched} "
        f"total={len(result)} "
        f"out={index_path} "
        f"audit={audit_log}"
    )


def _scan_pdf_directory(pdf_dir: Path) -> pd.DataFrame:
    """Return a DataFrame describing PDF artefacts under the directory."""

    pdf_files = sorted(p for p in pdf_dir.rglob("*.pdf") if p.is_file())
    records: list[dict[str, object]] = []
    for path in pdf_files:
        relative = path.relative_to(pdf_dir)
        identifier = relative.as_posix()
        if identifier.lower().endswith(".pdf"):
            identifier = identifier[:-4]
        identifier = identifier.replace("/", "_")
        records.append(
            {
                "id": identifier,
                "pdf_path": str(path),
            }
        )
    return pd.DataFrame.from_records(records)


@parse_app.command(name="run")
def parse_pdfs(
    pdf_dir: Path = typer.Option(
        Path("data/pdfs"),
        "--pdf-dir",
        help="Hakemisto josta PDF:t luetaan",
        show_default=True,
    ),
    out_dir: Path = typer.Option(
        Path("data/parsed"),
        "--out-dir",
        help="Hakemisto jonne TEI ja tekstit tallennetaan",
        show_default=True,
    ),
    index_out: Path = typer.Option(
        Path("data/cache/parsed_index.parquet"),
        "--index-out",
        help="Parquet polku jonne parse-metadata kirjoitetaan",
        show_default=True,
    ),
    grobid_url: str = typer.Option(
        "http://localhost:8070",
        "--grobid-url",
        help="GROBID-palvelun base URL",
        show_default=True,
    ),
    err_log: Path = typer.Option(
        Path("data/logs/parse_errors.csv"),
        "--err-log",
        help="CSV polku parse-virheille",
        show_default=True,
    ),
    sample: int | None = typer.Option(
        None,
        "--sample",
        help="Prosessoi vain ensimmäiset N riviä (debug)",
    ),
) -> None:
    """Jäsennä ladatut PDF:t GROBID-palvelulla."""

    if not pdf_dir.exists():
        typer.secho(f"PDF-hakemisto puuttuu: {pdf_dir}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    df = _scan_pdf_directory(pdf_dir)
    if df.empty:
        typer.secho(
            f"PDF-hakemistossa ei ole PDF-tiedostoja: {pdf_dir}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    from .parse.run import parse_all

    result = parse_all(
        df,
        out_dir,
        grobid_url,
        err_log,
        sample=sample,
    )

    index_out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(index_out, index=False)

    if sample is None:
        processed_indices = result.index
    else:
        processed_indices = result.index[: max(sample, 0)]
    processed_count = len(processed_indices)
    processed_display = processed_count if sample is not None else len(result)
    success = (
        int(result.loc[processed_indices, "parsed_ok"].sum())
        if processed_count and "parsed_ok" in result
        else 0
    )
    typer.echo(
        "Parse OK | "
        f"parsed={success} "
        f"processed={processed_display} "
        f"out-dir={out_dir} "
        f"index={index_out} "
        f"errors={err_log}"
    )


def _write_source_parquet(path: Path, papers: Iterable[Paper]) -> None:
    records = [paper.model_dump() for paper in papers]
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(path, index=False)


@rag_app.command(name="chunk")
def chunk_cli(
    parsed_index: Path = typer.Option(
        Path("data/cache/parsed_index.parquet"),
        "--parsed-index",
        help="Polku jäsennettyyn Parquet-indeksiin",
        show_default=True,
    ),
    out: Path = typer.Option(
        Path("data/cache/chunks.parquet"),
        "--out",
        help="Polku, johon chunkit tallennetaan",
        show_default=True,
    ),
    overlap: int = typer.Option(
        128,
        "--overlap",
        help="Chunkkien välinen overlap tokenien määrässä",
        show_default=True,
    ),
    max_tokens: int = typer.Option(
        1024,
        "--max-tokens",
        help="Maksimi tokenien määrä per chunk",
        show_default=True,
    ),
    min_tokens: int = typer.Option(
        50,
        "--min-tokens",
        help="Minimi tokenien määrä per chunk",
        show_default=True,
    ),
    include_front: bool = typer.Option(
        False,
        "--include-front",
        help="Pakota TEI front mukaan vaikka body puuttuisi",
        is_flag=True,
    ),
    use_text_txt: bool = typer.Option(
        False,
        "--use-text-txt",
        help="Fallback `parsed_txt_path`iin kun TEI/body ei tuota chunkkeja",
        is_flag=True,
    ),
    min_chars: int = typer.Option(
        0,
        "--min-chars",
        help="Minimi merkkien määrä per chunk (0 = ei käytössä)",
        show_default=True,
    ),
) -> None:
    """Paloittele TEI-dokumentit loogisiin osiin ja tokeneiksi."""
    from .rag.chunk import run_chunking

    try:
        run_chunking(
            parsed_index_path=parsed_index,
            out_path=out,
            max_tokens=max_tokens,
            overlap=overlap,
            min_tokens=min_tokens,
            include_front=include_front,
            use_text_txt=use_text_txt,
            min_chars=min_chars,
        )
    except FileNotFoundError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(1)
    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)


@rag_app.command(name="index")
def index_cli(
    chunks: Path = typer.Option(
        Path("data/cache/chunks.parquet"),
        "--chunks",
        help="Polku chunkkeihin",
        show_default=True,
    ),
    index_dir: Path = typer.Option(
        Path("data/index/chroma"),
        "--index-dir",
        help="Hakemisto, johon indeksi tallennetaan",
        show_default=True,
    ),
    embed_provider: str = typer.Option(
        "google",
        "--embed-provider",
        help="Embedding-mallin tarjoaja",
        show_default=True,
    ),
    embed_model: str = typer.Option(
        "text-embedding-004",
        "--embed-model",
        help="Käytettävä embedding-malli",
        show_default=True,
    ),
    batch: int = typer.Option(
        128,
        "--batch",
        help="Eräkoko indeksoinnissa",
        show_default=True,
    ),
) -> None:
    """Rakenna vektorikanta chunkeista."""
    from .rag.index import build_index

    try:
        build_index(
            chunks_path=chunks,
            index_dir=index_dir,
            embed_provider=embed_provider,
            embed_model=embed_model,
            batch_size=batch,
        )
    except FileNotFoundError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(1)
    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)


@rag_app.command(name="stats")
def stats_cli(
    index_dir: Path = typer.Option(
        Path("data/index/chroma"),
        "--index-dir",
        help="Hakemisto, josta indeksi ladataan",
        show_default=True,
    ),
) -> None:
    """Näytä statistiikkaa rakennetusta indeksistä."""
    from .rag.index import get_index_stats

    try:
        stats = get_index_stats(index_dir)
        typer.echo(stats.model_dump_json(indent=2))
    except FileNotFoundError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(1)


@rag_app.command(name="verify")
def verify_cli(
    index_dir: Path = typer.Option(
        Path("data/index/chroma"),
        "--index-dir",
        help="Hakemisto, josta indeksi ladataan",
        show_default=True,
    ),
    question_file: Path = typer.Option(
        Path("tests/data/rag_smoke_questions.json"),
        "--question-file",
        help="Polku JSON-tiedostoon, joka sisältää testikysymykset",
        show_default=True,
    ),
) -> None:
    """Aja smoketest-kysymykset indeksiä vasten."""
    from .rag.index import verify_index

    try:
        success = verify_index(index_dir, question_file)
        if not success:
            typer.secho("Index verification failed.", fg=typer.colors.RED)
            raise typer.Exit(1)
        else:
            typer.secho("Index verification passed.", fg=typer.colors.GREEN)
    except FileNotFoundError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(1)


@app.command(name="qa")
def qa_cli(
    question: str = typer.Option(
        ..., "--question", help="Kysymys, johon haetaan vastaus"
    ),
    index_dir: Path = typer.Option(
        Path("data/index/chroma"),
        "--index-dir",
        help="Hakemisto, josta indeksi ladataan",
        show_default=True,
    ),
    k: int = typer.Option(
        6, "--k", help="Kuinka monta dokumenttia haetaan", show_default=True
    ),
    llm_provider: str = typer.Option(
        "google", "--llm-provider", help="LLM-mallin tarjoaja", show_default=True
    ),
    llm_model: str = typer.Option(
        "gemini-1.5-flash",
        "--llm-model",
        help="Käytettävä LLM-malli",
        show_default=True,
    ),
    out: Path = typer.Option(
        Path("data/output/qa.jsonl"),
        "--out",
        help="Polku, johon QA-tulokset tallennetaan",
        show_default=True,
    ),
    chunks_path: Optional[Path] = typer.Option(
        None,
        "--chunks-path",
        help="Polku chunk-tiedostoon (prioriteetti: argumentti > index_meta.json -> data/cache/chunks.parquet)",
        show_default=False,
    ),
    audit_path: Optional[Path] = typer.Option(
        None,
        "--audit-path",
        help="Polku audit-lokiin (oletus: argumentti > index_dir/../logs/qa_audit.csv)",
        show_default=False,
    ),
) -> None:
    """Hae vastaus kysymykseen RAG-mallilla."""
    from .rag.qa import run_qa

    try:
        run_qa(
            question=question,
            index_dir=index_dir,
            k=k,
            llm_provider=llm_provider,
            llm_model=llm_model,
            out_path=out,
            chunks_path=chunks_path,
            audit_path=audit_path,
        )
    except FileNotFoundError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(1)
    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)


@write_app.command(name="init")
def write_init(
    out: Path = typer.Option(
        ...,
        "--out",
        help="Directory to initialize the report in.",
    ),
    style: Path = typer.Option(
        ...,
        "--style",
        help="Path to the CSL style file.",
    ),
    title: str = typer.Option(
        "Review title",
        "--title",
        help="The title of the report.",
    ),
    authors: str = typer.Option(
        ...,
        "--authors",
        help="A semicolon-separated list of authors (e.g., 'Lastname, First; Second, Third').",
    ),
) -> None:
    """Initialize a new report scaffold."""
    from .write.scaffold import create_report_scaffold

    author_list = [author.strip() for author in authors.split(";") if author.strip()]
    if not author_list:
        typer.secho("Authors cannot be empty.", fg=typer.colors.RED)
        raise typer.Exit(1)

    try:
        create_report_scaffold(out, title, author_list, style)
        typer.secho(f"Report scaffold created at: {out}", fg=typer.colors.GREEN)
    except FileNotFoundError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(1)
    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)


@write_app.command(name="bib")
def write_bib(
    in_path: Path = typer.Option(
        ...,
        "--in",
        help="Path to the input parquet file (e.g., parsed_index.parquet).",
    ),
    qa_path: Path = typer.Option(
        ...,
        "--qa",
        help="Path to the QA JSONL file.",
    ),
    out_path: Path = typer.Option(
        ...,
        "--out",
        help="Path to the output BibTeX file (e.g., references.bib).",
    ),
    missing_log_path: Path = typer.Option(
        Path("data/logs/bib_missing.csv"),
        "--missing-log",
        help="Path to the log file for missing BibTeX entries.",
    ),
) -> None:
    """Generate a BibTeX file from DOIs and PMIDs."""
    from .write.refs import collect_and_write_references

    try:
        collect_and_write_references(in_path, qa_path, out_path, missing_log_path)
        typer.secho(f"BibTeX file created at: {out_path}", fg=typer.colors.GREEN)
    except FileNotFoundError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(1)
    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)


@write_app.command(name="fill")
def write_fill(
    qmd_path: Path = typer.Option(
        ...,
        "--qmd",
        help="Path to the QMD report file.",
    ),
    qa_path: Path = typer.Option(
        ...,
        "--qa",
        help="Path to the QA JSONL file.",
    ),
    logs_dir: Path = typer.Option(
        ...,
        "--logs-dir",
        help="Directory containing the log files.",
    ),
    out_path: Path = typer.Option(
        ...,
        "--out",
        help="Path to the output QMD file.",
    ),
    parsed_index_path: Path = typer.Option(
        Path("data/cache/parsed_index.parquet"),
        "--parsed-index-path",
        help="Path to the parsed index parquet file.",
    ),
) -> None:
    """Fill the QMD template with data from QA and logs."""
    from .write.compose import fill_qmd_template

    try:
        fill_qmd_template(qmd_path, qa_path, logs_dir, parsed_index_path, out_path)
        typer.secho(f"QMD file filled and saved to: {out_path}", fg=typer.colors.GREEN)
    except FileNotFoundError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(1)
    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)


@write_app.command(name="render")
def write_render(
    dir_path: Path = typer.Option(
        ...,
        "--dir",
        help="Directory containing the Quarto report.",
    ),
    formats: str = typer.Option(
        "html,pdf",
        "--format",
        help="Comma-separated list of output formats (e.g., html,pdf).",
    ),
) -> None:
    """Render the Quarto report."""
    from .write.render import render_report

    format_list = [f.strip() for f in formats.split(",")]
    try:
        render_report(dir_path, format_list)
        typer.secho(f"Report rendered successfully in: {dir_path}", fg=typer.colors.GREEN)
    except (FileNotFoundError, RuntimeError) as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(1)
    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)


@write_app.command(name="check")
def write_check(
    bib_path: Path = typer.Option(
        ...,
        "--bib",
        help="Path to the BibTeX file to check.",
    ),
    log_path: Path = typer.Option(
        Path("data/logs/linkcheck_failures.csv"),
        "--log-path",
        help="Path to the output log file for link check results.",
    ),
) -> None:
    """Run quality checks on the report files (e.g., link checking)."""
    from .write.linkcheck import run_linkcheck

    try:
        run_linkcheck(bib_path, log_path)
        typer.secho(f"Link check complete. Log saved to: {log_path}", fg=typer.colors.GREEN)
    except FileNotFoundError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(1)
    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)


@prisma_app.command(name="compute")
def prisma_compute(
    search_audit: Path = typer.Option(
        Path("data/logs/search_audit.csv"),
        "--search-audit",
        help="Path to the search audit CSV file.",
        show_default=True,
    ),
    merged: Path = typer.Option(
        Path("data/cache/merged.parquet"),
        "--merged",
        help="Path to the merged Parquet file.",
        show_default=True,
    ),
    screened: Optional[Path] = typer.Option(
        None,
        "--screened",
        help="Path to the screened Parquet file (optional).",
        show_default=True,
    ),
    parsed_index: Optional[Path] = typer.Option(
        None,
        "--parsed-index",
        help="Path to the parsed index Parquet file (optional).",
        show_default=True,
    ),
    qa: Optional[Path] = typer.Option(
        None,
        "--qa",
        help="Path to the QA JSONL file (optional).",
        show_default=True,
    ),
    out_json: Path = typer.Option(
        Path("data/cache/prisma_counts.json"),
        "--out-json",
        help="Path to the output JSON file for PRISMA counts.",
        show_default=True,
    ),
    out_csv: Path = typer.Option(
        Path("data/cache/prisma_counts.csv"),
        "--out-csv",
        help="Path to the output CSV file for PRISMA counts.",
        show_default=True,
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit with error if validation fails (otherwise warn).",
        is_flag=True,
    ),
    validation_log: Path = typer.Option(
        Path("data/logs/prisma_validation.csv"),
        "--validation-log",
        help="Path to the validation log CSV file.",
        show_default=True,
    ),
) -> None:
    """Compute PRISMA 2020 counts from pipeline artifacts."""
    try:
        counts = compute_counts(
            search_audit_path=search_audit,
            merged_path=merged,
            screened_path=screened,
            parsed_index_path=parsed_index,
            qa_path=qa,
        )
        issues = validate_counts(counts)
        
        if issues:
            validation_log.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(issues).to_csv(validation_log, index=False)
            for issue in issues:
                msg = f"Validation {issue['status']}: {issue['rule']} - {issue['details']}"
                if issue['status'] == "FAIL":
                    typer.secho(msg, fg=typer.colors.RED)
                else:
                    typer.secho(msg, fg=typer.colors.YELLOW)
            if strict and any(issue['status'] == "FAIL" for issue in issues):
                raise typer.Exit(code=1)
        else:
            typer.secho("PRISMA counts validated successfully.", fg=typer.colors.GREEN)

        write_counts_json_csv(counts, out_json, out_csv)
        typer.secho("PRISMA compute OK.", fg=typer.colors.GREEN)

    except FileNotFoundError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(1)
    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)


@prisma_app.command(name="render")
def prisma_render(
    counts: Path = typer.Option(
        Path("data/cache/prisma_counts.json"),
        "--counts",
        help="Path to the PRISMA counts JSON file.",
        show_default=True,
    ),
    out_dir: Path = typer.Option(
        Path("output/report"),
        "--out-dir",
        help="Directory to save the rendered diagram.",
        show_default=True,
    ),
    engine: str = typer.Option(
        "python",
        "--engine",
        help="Rendering engine to use (python or r).",
        show_default=True,
    ),
    formats: str = typer.Option(
        "svg,png",
        "--formats",
        help="Comma-separated list of output formats (svg,png).",
        show_default=True,
    ),
) -> None:
    """Render the PRISMA 2020 diagram."""
    try:
        format_list = tuple(f.strip() for f in formats.split(','))
        if engine.lower() == "python":
            render_prisma_python(counts_path=counts, out_dir=out_dir, formats=format_list)
        elif engine.lower() == "r":
            render_prisma_r(counts_path=counts, out_dir=out_dir)
        else:
            typer.secho(f"Unknown rendering engine: {engine}. Use 'python' or 'r'.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        typer.secho("PRISMA render OK.", fg=typer.colors.GREEN)
    except FileNotFoundError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(1)
    except RuntimeError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(1)
    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)


@prisma_app.command(name="attach")
def prisma_attach(
    qmd: Path = typer.Option(
        Path("output/report/report.qmd"),
        "--qmd",
        help="Path to the Quarto markdown file.",
        show_default=True,
    ),
    image: Path = typer.Option(
        Path("output/report/prisma.svg"),
        "--image",
        help="Path to the PRISMA image file (SVG or PNG).",
        show_default=True,
    ),
) -> None:
    """Attach the PRISMA diagram to a Quarto markdown report."""
    try:
        attach_to_qmd(qmd_path=qmd, image_path=image)
        typer.secho("PRISMA attach OK.", fg=typer.colors.GREEN)
    except FileNotFoundError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(1)
    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)


@prisma_app.command(name="all")
def prisma_all(
    search_audit: Path = typer.Option(
        Path("data/logs/search_audit.csv"),
        "--search-audit",
        help="Path to the search audit CSV file.",
        show_default=True,
    ),
    merged: Path = typer.Option(
        Path("data/cache/merged.parquet"),
        "--merged",
        help="Path to the merged Parquet file.",
        show_default=True,
    ),
    screened: Optional[Path] = typer.Option(
        None,
        "--screened",
        help="Path to the screened Parquet file (optional).",
        show_default=True,
    ),
    parsed_index: Optional[Path] = typer.Option(
        None,
        "--parsed-index",
        help="Path to the parsed index Parquet file (optional).",
        show_default=True,
    ),
    qa: Optional[Path] = typer.Option(
        None,
        "--qa",
        help="Path to the QA JSONL file (optional).",
        show_default=True,
    ),
    out_json: Path = typer.Option(
        Path("data/cache/prisma_counts.json"),
        "--out-json",
        help="Path to the output JSON file for PRISMA counts.",
        show_default=True,
    ),
    out_csv: Path = typer.Option(
        Path("data/cache/prisma_counts.csv"),
        "--out-csv",
        help="Path to the output CSV file for PRISMA counts.",
        show_default=True,
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit with error if validation fails (otherwise warn).",
        is_flag=True,
    ),
    validation_log: Path = typer.Option(
        Path("data/logs/prisma_validation.csv"),
        "--validation-log",
        help="Path to the validation log CSV file.",
        show_default=True,
    ),
    out_dir: Path = typer.Option(
        Path("output/report"),
        "--out-dir",
        help="Directory to save the rendered diagram.",
        show_default=True,
    ),
    engine: str = typer.Option(
        "python",
        "--engine",
        help="Rendering engine to use (python or r).",
        show_default=True,
    ),
    formats: str = typer.Option(
        "svg,png",
        "--formats",
        help="Comma-separated list of output formats (svg,png).",
        show_default=True,
    ),
    qmd: Path = typer.Option(
        Path("output/report/report.qmd"),
        "--qmd",
        help="Path to the Quarto markdown file.",
        show_default=True,
    ),
    image: Path = typer.Option(
        Path("output/report/prisma.svg"),
        "--image",
        help="Path to the PRISMA image file (SVG or PNG).",
        show_default=True,
    ),
) -> None:
    """Run the full PRISMA 2020 diagram generation pipeline (compute -> render -> attach)."""
    try:
        # 1. Compute
        typer.secho("Computing PRISMA counts...", fg=typer.colors.BLUE)
        counts = compute_counts(
            search_audit_path=search_audit,
            merged_path=merged,
            screened_path=screened,
            parsed_index_path=parsed_index,
            qa_path=qa,
        )
        issues = validate_counts(counts)
        
        if issues:
            validation_log.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(issues).to_csv(validation_log, index=False)
            for issue in issues:
                msg = f"Validation {issue['status']}: {issue['rule']} - {issue['details']}"
                if issue['status'] == "FAIL":
                    typer.secho(msg, fg=typer.colors.RED)
                else:
                    typer.secho(msg, fg=typer.colors.YELLOW)
            if strict and any(issue['status'] == "FAIL" for issue in issues):
                raise typer.Exit(code=1)
        else:
            typer.secho("PRISMA counts validated successfully.", fg=typer.colors.GREEN)

        write_counts_json_csv(counts, out_json, out_csv)
        typer.secho("PRISMA compute OK.", fg=typer.colors.GREEN)

        # 2. Render
        typer.secho("Rendering PRISMA diagram...", fg=typer.colors.BLUE)
        format_list = tuple(f.strip() for f in formats.split(','))
        if engine.lower() == "python":
            render_prisma_python(counts_path=out_json, out_dir=out_dir, formats=format_list)
        elif engine.lower() == "r":
            render_prisma_r(counts_path=out_json, out_dir=out_dir)
        else:
            typer.secho(f"Unknown rendering engine: {engine}. Use 'python' or 'r'.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        typer.secho("PRISMA render OK.", fg=typer.colors.GREEN)

        # 3. Attach
        typer.secho("Attaching PRISMA diagram to report...", fg=typer.colors.BLUE)
        attach_to_qmd(qmd_path=qmd, image_path=image)
        typer.secho("PRISMA attach OK.", fg=typer.colors.GREEN)

        typer.secho("PRISMA all pipeline completed successfully.", fg=typer.colors.GREEN)

    except FileNotFoundError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(1)
    except RuntimeError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(1)
    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)


@run_app.command(name="start")
def run_cli(
    topic: str = typer.Option(..., "--topic", help="The research topic."),
    provider: Provider = typer.Option(
        Provider.gemini, "--provider", help="LLM provider to use."
    ),
    model: str = typer.Option(
        "gemini-1.5-pro", "--model", help="LLM model to use."
    ),
    budget_usd: float = typer.Option(
        3.0, "--budget", help="Maximum USD budget for LLM calls."
    ),
    max_iterations: int = typer.Option(
        2, "--max-iter", help="Maximum iterations for critic feedback loop."
    ),
    top_k: int = typer.Option(6, "--top-k", help="Top K results for retrieval."),
    bm25_k: int = typer.Option(20, "--bm25-k", help="BM25 K value for retrieval."),
    questions: Optional[Path] = typer.Option(
        None,
        "--questions",
        help="Path to a questions.yaml file, or a comma-separated list of questions.",
    ),
    require_sources: int = typer.Option(
        2, "--require-sources", help="Minimum unique sources required per answer."
    ),
    enforce_pages: bool = typer.Option(
        True, "--enforce-pages", help="Enforce page number validation for citations."
    ),
    output_dir: Path = typer.Option(
        Path(f"data/runs/{time.strftime('%Y%m%d-%H%M%S')}"),
        "--output-dir",
        help="Directory to store run artifacts.",
    ),
    skip_search: bool = typer.Option(False, "--skip-search", help="Skip search node."),
    skip_screen: bool = typer.Option(False, "--skip-screen", help="Skip screen node."),
    skip_ingest: bool = typer.Option(False, "--skip-ingest", help="Skip ingest node."),
    skip_index: bool = typer.Option(False, "--skip-index", help="Skip index node."),
    skip_qa: bool = typer.Option(False, "--skip-qa", help="Skip QA node."),
    skip_write: bool = typer.Option(False, "--skip-write", help="Skip write node."),
    skip_prisma: bool = typer.Option(False, "--skip-prisma", help="Skip prisma node."),
    skip_critic: bool = typer.Option(False, "--skip-critic", help="Skip critic node."),
    resume: Optional[Path] = typer.Option(
        None, "--resume", help="Resume a suspended run from a given run directory."
    ),
) -> None:
    """Run the full research pipeline."""
    run_config = RunConfig(
        topic=topic,
        provider=provider,
        model=model,
        budget_usd=budget_usd,
        max_iterations=max_iterations,
        top_k=top_k,
        bm25_k=bm25_k,
        questions=(
            list(questions.read_text().splitlines())
            if questions and questions.is_file()
            else (questions.name.split(",") if questions else None)
        ),
        require_sources=require_sources,
        enforce_pages=enforce_pages,
        output_dir=output_dir,
    )

    initial_state: Optional[RunState] = None
    if resume:
        # TODO: Implement state loading from resume directory
        typer.secho(f"Resuming run from {resume} is not yet implemented.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    runner = DAGRunner(run_config)
    final_state = runner.run(initial_state=initial_state)

    typer.echo("\n--- Run Summary ---")
    typer.echo(f"Run ID: {final_state.run_id}")
    typer.echo(f"Total LLM Cost: ${final_state.total_llm_cost_usd:.4f}")
    typer.echo(f"Total Duration: {final_state.total_duration_s:.2f} seconds")
    typer.echo(f"Final Status: {final_state.final_status}")
    if final_state.errors:
        typer.secho("Errors encountered:", fg=typer.colors.RED)
        for error in final_state.errors:
            typer.secho(f"- {error}", fg=typer.colors.RED)
    if final_state.warnings:
        typer.secho("Warnings encountered:", fg=typer.colors.YELLOW)
        for warning in final_state.warnings:
            typer.secho(f"- {warning}", fg=typer.colors.YELLOW)
    typer.secho(f"Full report available at: {final_state.config.output_dir / final_state.run_id}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
