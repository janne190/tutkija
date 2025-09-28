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
la pdf --in data\cache\merged.parquet --out data\cache\with_pdfs.parquet --pdf-dir data\pdfs --log data\logs\pdf_audit.csv --mailto $Env:UNPAYWALL_EMAIL
docker run -d --name grobid -p 8070:8070 ghcr.io/kermitt2/grobid:latest
la parse --in data\cache\with_pdfs.parquet --out data\cache\parsed.parquet --parsed-dir data\parsed --grobid-url http://localhost:8070 --err-log data\logs\parse_errors.csv --sample 20
```
> Windows: wrap the topic in quotes so UTF-8 characters survive.

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

`la pdf` looks for arXiv, PubMed Central and Unpaywall download locations in that order. The command writes PDFs under `data/pdfs/`, augments the Parquet file with `pdf_path`, `pdf_license` and `has_fulltext` columns and appends an audit trail to `data/logs/pdf_audit.csv`. Provide your registered Unpaywall email via `.env` (`UNPAYWALL_EMAIL=`) so the API call is accepted.

`la parse` expects a running GROBID container. It reads the PDF-enriched parquet, sends each file to the service, stores the returned TEI XML under `data/parsed/<id>/tei.xml` and writes a lightweight Markdown summary to `text.txt`. Errors are recorded in `data/logs/parse_errors.csv`.

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
