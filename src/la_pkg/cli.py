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

# Load environment variables from .env file
load_dotenv()

app = typer.Typer(add_completion=False, no_args_is_help=True)
pdf_app = typer.Typer(add_completion=False, no_args_is_help=True)
parse_app = typer.Typer(add_completion=False, no_args_is_help=True)
rag_app = typer.Typer(add_completion=False, no_args_is_help=True)

app.add_typer(pdf_app, name="pdf")
app.add_typer(parse_app, name="parse")
app.add_typer(rag_app, name="rag")


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
        help="Polku chunk-tiedostoon (oletus: index_dir/../chunks.parquet)",
        show_default=False,
    ),
    audit_path: Optional[Path] = typer.Option(
        None,
        "--audit-path",
        help="Polku audit-lokiin (oletus: index_dir/../logs/qa_audit.csv)",
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


if __name__ == "__main__":
    app()
