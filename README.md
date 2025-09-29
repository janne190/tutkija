[![CI](https://github.com/janne190/tutkija/actions/workflows/ci.yml/badge.svg)](https://github.com/janne190/tutkija/actions/workflows/ci.yml)

# Tutkija

Tutkija is a starter template for a multi-agent literature review workflow. The repository documents shared working practices, developer tooling and a phased delivery plan before feature work starts.

## Documentation
- `docs/pelisaannot.md` contains the team working agreement plus DoR/DoD
- `docs/ARCHITECTURE.md` summarises the system on a single page
- `docs/mittarit.md` lists the metrics and guard-rails
- `docs/adr/` tracks architectural decisions

## Quick start
```powershell
# set up the environment
. .\.venv\Scripts\Activate.ps1
uv pip install -e ".[dev,parse]"

# configuration smoke test
la hello

# OpenAlex metadata search (tries language fallbacks automatically)
la search "genomic screening" --out data\cache\search.parquet
la search --topic "genominen seulonta sy\u00f6v\u00e4ss\u00e4" --lang auto --out data\cache\search.parquet

# Multisource merge and metrics
la search-all --topic "genomic screening cancer" --out data\cache\merged.parquet --save-single

# Screening workflow (comma separated seeds)
la screen --in data\cache\merged.parquet --out data\cache\screened.parquet --recall 0.9 --seeds "doi:10.1038/xyz , pmid:12345"

# Fetch PDFs and parse with GROBID (set UNPAYWALL_EMAIL in .env)
la pdf discover --in data\cache\merged.parquet --out data\cache\pdf_index.parquet
la pdf download --in data\cache\pdf_index.parquet --pdf-dir data\pdfs --audit data\logs\pdf_audit.csv --mailto $Env:UNPAYWALL_EMAIL
docker run -d --name grobid -p 8070:8070 ghcr.io/kermitt2/grobid:latest
la parse run --pdf-dir data\pdfs --out-dir data\parsed --index-out data\cache\parsed_index.parquet --grobid-url http://localhost:8070 --err-log data\logs\parse_errors.csv --sample 20
```
> Windows: wrap the topic in quotes so UTF-8 characters survive.

## RAG & QA

The project includes a Retrieval Augmented Generation (RAG) pipeline for answering questions from the parsed TEI documents. The process involves three main steps: chunking the documents, building a vector index, and running the question-answering command.

```powershell
# 1) Chunk TEI documents into smaller pieces
la rag chunk --parsed-index data/cache/parsed_index.parquet `
             --out data/cache/chunks.parquet `
             --include-front true --use-text-txt true `
             --min-tokens 50 --overlap 64 --max-tokens 1024

# 2) Build a vector index (using ChromaDB and Google embeddings by default)
la rag index build --chunks data/cache/chunks.parquet `
                   --index-dir data/index/chroma `
                   --embed-provider google --embed-model text-embedding-004

# 3) Ask a question
la qa --question "Mitkä menetelmät ovat aineistossa yleisiä?" `
      --index-dir data/index/chroma --k 6 `
      --llm-provider openai --llm-model gpt-4o-mini `
      --audit-path data/logs/qa_audit.csv `
      --out data/output/qa.jsonl
```

## Writing and Rendering

The project includes a pipeline for generating a final report in Quarto (`.qmd`) format, including BibTeX references and rendered outputs (HTML, PDF).

```powershell
# 1) Initialize a new report scaffold
la write init --out output/report `
              --style docs/styles/apa.csl `
              --title "My Awesome Review" `
              --authors "Doe, John; Smith, Jane"

# 2) Collect references and generate a BibTeX file
la write bib --in data/cache/parsed_index.parquet `
             --qa data/qa/qa.jsonl `
             --out output/report/references.bib

# 3) Fill the QMD template with claims and method summaries
la write fill --qmd output/report/report.qmd `
              --qa data/qa/qa.jsonl `
              --logs-dir data/logs `
              --out output/report/report.qmd

# 4) Render the report to HTML and PDF (requires Quarto installation)
la write render --dir output/report --format html,pdf

# 5) (Optional) Check for broken links in the bibliography
la write check --bib output/report/references.bib `
               --log-path data/logs/linkcheck_failures.csv
```

## PRISMA 2020 Diagram

The project includes a pipeline for generating and attaching a PRISMA 2020 flow diagram to the report.

```powershell
# 1) Compute PRISMA counts
la prisma compute `
  --search-audit data/logs/search_audit.csv `
  --merged data/cache/merged.parquet `
  --screened data/cache/screened.parquet `
  --parsed-index data/cache/parsed_index.parquet `
  --qa data/qa/qa.jsonl `
  --out-json data/cache/prisma_counts.json `
  --out-csv data/cache/prisma_counts.csv

# 2) Render the diagram (Python engine)
la prisma render `
  --counts data/cache/prisma_counts.json `
  --out-dir output/report `
  --engine python `
  --formats svg,png

# 3) Attach the diagram to the report
la prisma attach `
  --qmd output/report/report.qmd `
  --image output/report/prisma.svg

# 4) Run the full pipeline (compute -> render -> attach)
la prisma all --engine python
```

### Provider Configuration

You can switch the embedding and language model providers via the command-line options. For example, to use a different embedding model, you would change the `--embed-provider` and `--embed-model` flags on the `la rag index` command. The `la qa` command will then automatically read these from the index metadata. The `--llm-provider` and `--llm-model` flags on the `la qa` command control the language model used for generating answers, and can be different from the embedding provider.

**`chunks.parquet` path resolution for `la qa`:**
The `la qa` command resolves the `chunks.parquet` path with the following priority:
1. Explicitly provided `--chunks-path` argument.
2. `chunks_path` stored in `index_meta.json` (created by `la rag index build`).
3. Fallback to `data/cache/chunks.parquet`.

**LLM provider vs. Embedding provider:**
The `la qa` command uses the `embed_provider` and `embed_model` stored in the `index_meta.json` for retrieval. The `--llm-provider` and `--llm-model` arguments for `la qa` specify the model used for generating the answer, which can be different from the embedding model.

**OpenAI Stub:**
For offline smoke testing, the OpenAI provider is currently stubbed (`src/la_pkg/rag/qa.py::MockOpenAIModel`) to return a valid JSON response with at least two distinct sources, without requiring an actual API call. This allows `la qa --llm-provider openai` to run successfully offline.

**Index meta:**
The `la rag index build` command now always writes `index_meta.json` even for empty `chunks.parquet` files, ensuring it contains `embed_provider`, `embed_model`, and `chunks_path`.

**Environment Variables:**
Set `GEMINI_API_KEY` (or `GOOGLE_API_KEY` as a fallback) in your `.env` file for Google models.
Set `OPENAI_API_KEY` in your `.env` file for OpenAI models.

### Interpreting the Output

The `la qa` command produces a `qa.jsonl` file, where each line is a JSON object representing a question-answering result. The structure is defined by the `QAResult` model and includes the original question, the generated answer, a list of claims made in the answer, and the sources used to generate the answer. Each claim is supported by one or more citations, which include a quote from the source text and, where available, page numbers.

### Installing dependencies behind a proxy or offline

Some corporate environments block direct access to PyPI. Two proven options are below; adapt the hostnames and credentials to your network.

**Option A – internal mirror / proxy-aware pip**

```powershell
# teach pip to use the internal simple index and trust its TLS certificate
pip config set global.index-url http://<mirror-host>/simple
pip config set global.trusted-host <mirror-host>

# optionally configure HTTP(S) proxies for pip/uv
$env:HTTP_PROXY="http://user:pass@proxy:8080"
$env:HTTPS_PROXY="http://user:pass@proxy:8080"

# install all development extras from the mirror
uv pip install -e ".[dev,parse]"
```

**Option B – offline wheelhouse via GitHub Actions**

```bash
# trigger the matrix build for the Python version you need (e.g. 3.11)
gh workflow run Wheelhouse --ref main -f pyver=3.11
gh run watch
# download the Windows artifact; swap to manylinux for Linux hosts
gh run download --name wheelhouse-win_amd64-py3.11 -D wheelhouse
```

Then, inside the restricted environment:

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install --no-index --find-links=wheelhouse pip setuptools wheel build
pip install --no-index --find-links=wheelhouse -r tools/requirements-wheelhouse.txt
pip install --no-index --find-links=wheelhouse -e . --no-build-isolation
```

The curated `tools/requirements-wheelhouse.txt` file is limited to wheel-only dependencies so `pip download` never attempts to reach PyPI.

All first-party tests and commands run without reaching the public internet once the wheels are available locally.

`la search-all` performs live HTTP requests to OpenAlex (`https://api.openalex.org/works`), PubMed (`https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`) and arXiv (`https://export.arxiv.org/api/query`). No API keys are required. Set the optional `TUTKIJA_CONTACT_EMAIL` environment variable to add a polite User-Agent/`mailto` to the calls.

`la pdf discover` scans the merged metadata or the optional `tools/seed_urls.csv` list and records which provider (arXiv, PubMed Central or Unpaywall) can serve each paper. `la pdf download` reads that index, writes PDFs under `data/pdfs/`, augments the Parquet file with `pdf_path`, `pdf_license` and `has_fulltext` columns and appends an audit trail to `data/logs/pdf_audit.csv`. Provide your registered Unpaywall email via `.env` (`UNPAYWALL_EMAIL=`) or the `--mailto` flag so the API call is accepted.

`la parse run` expects a running GROBID container. It scans the PDF directory, sends each file to the service, stores the returned TEI XML under `data/parsed/<id>/tei.xml` and writes a lightweight Markdown summary to `text.txt`. Errors are recorded in `data/logs/parse_errors.csv` and the metadata index lands in `data/cache/parsed_index.parquet`.

### Phase 4 helper script

The repository ships with a convenience wrapper for Windows operators. `tools/run_vaihe4.ps1` provisions the virtual environment, checks for the required environment variables and executes the discovery, download and parsing commands in sequence. Run it from the project root:

```powershell
pwsh -ExecutionPolicy Bypass -File tools/run_vaihe4.ps1
```

Use `-SkipDiscover` when you already have an up-to-date `data/cache/pdf_index.parquet`. Override the default paths with the corresponding parameters, for example:

```powershell
pwsh -File tools/run_vaihe4.ps1 -PdfDir D:\\Tutkija\\pdfs -ParsedDir D:\\Tutkija\\parsed -GrobidUrl http://grobid.internal:8070 -SkipDiscover
```

## Testing

```bash
# fast unit suite – skips anything marked as network or integration
pytest -m "not network"

# nightly or local smoke that exercises the end-to-end pipeline
pytest -m "network or integration"
```

The PDF, Unpaywall and GROBID interactions are fully mocked in unit tests. Mark longer or network-bound checks with `@pytest.mark.integration` or `@pytest.mark.network` so that CI jobs can exclude them when running in a sandboxed environment.

Running `la search` appends a row to `data\cache\search_log.csv`. `la search-all` writes the merged Parquet plus an audit row in `data\cache\merge_log.csv` so you can track sources, duplicates and counts per topic. `la screen` records `screened.parquet` and a corresponding `screen_log.csv` entry containing the PRISMA-style tallies, engine metadata, random seed, fallback used and resolved output path. Seeds must be provided as a comma-separated list (e.g. `--seeds "doi:10.1038/... , pmid:12345 , id:W1111"`).

Tutkija ships with a scikit-learn baseline (TF-IDF + logistic regression) and an optional ASReview integration. Install the latter with `pip install tutkija[asreview]` or `pipx install asreview`. When gold labels are present the classifier chooses the smallest probability threshold that reaches the requested recall. Without gold labels the engine falls back to seed-similarity scoring (TF-IDF + cosine) or, if no seeds are provided, assigns a uniform probability of 0.5 so every record can continue through manual review.

**Index meta:**
The `la rag index build` command now always writes `index_meta.json` even for empty `chunks.parquet` files, ensuring it contains `embed_provider`, `embed_model`, and `chunks_path`.

## CLI metrics
The multisource command reports the same statistics that land in the merge log. These guard-rails help keep the dataset healthy when the pipeline evolves.

| Metric | Description |
| --- | --- |
| `per_source` | How many records arrived from OpenAlex, PubMed and arXiv before filtering |
| `dup_doi` | Number of rows removed via exact DOI matches |
| `dup_title` | Number of rows removed via near-identical titles (RapidFuzz token sort ratio \>= 90) |
| `filtered` | Future hook for additional screening rules (currently zero) |
| `final_count` | Records that remain after dedupe and filters |

## Delivery phases
- Phase 0: foundation, CLI smoke test (`la hello`)
- Phase 1: search + metadata with golden snapshot tests and metrics
- Phase 2: screening, PDF handling and automated reporting

Always run `make setup` (or the equivalent commands) before your first change and verify the pre-commit hooks via `pre-commit run --all-files`.

## Release
Tag versions as `vX.Y.Z`, for example `v0.0.1`. The Release workflow builds artifacts and publishes a GitHub Release automatically.
