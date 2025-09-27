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
uv pip install -e .

# configuration smoke test
la hello

# OpenAlex metadata search (tries language fallbacks automatically)
la search "genomic screening" --out data\cache\search.parquet
la search --topic "genominen seulonta sy\u00f6v\u00e4ss\u00e4" --lang auto --out data\cache\search.parquet

# Multisource merge and metrics
la search-all --topic "genomic screening cancer" --out data\cache\merged.parquet --save-single

# Screening workflow (comma separated seeds)
la screen --in data\cache\merged.parquet --out data\cache\screened.parquet --recall 0.9 --seeds "doi:10.1038/xyz , pmid:12345"
```
> Windows: wrap the topic in quotes so UTF-8 characters survive.

`la search-all` performs live HTTP requests to OpenAlex (`https://api.openalex.org/works`), PubMed (`https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`) and arXiv (`https://export.arxiv.org/api/query`). No API keys are required. Set the optional `TUTKIJA_CONTACT_EMAIL` environment variable to add a polite User-Agent/`mailto` to the calls.

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
