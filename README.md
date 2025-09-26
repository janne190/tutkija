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

# Screening
la screen --in data\cache\merged.parquet --out data\cache\screened.parquet --recall 0.9 --engine scikit

### Seeds (optional)

You can provide seed papers to help prioritize similar papers, as a comma-separated string.
Supported ID prefixes: doi:, pmid:, arxiv:.

```powershell
la screen --in data\cache\merged.parquet --out data\cache\screened.parquet --recall 0.9 --seeds "doi:10.1038/xxxx,pmid:123456,arxiv:2101.01234"
```

# Live smoke verification (hits OpenAlex, PubMed, arXiv)
la search-all --topic "genomic screening cancer" --limit 40
# expected: per_source, dup_doi, dup_title, filtered, final all report values > 0
```
> Windows: wrap the topic in quotes so UTF-8 characters survive.

`la search-all` performs live HTTP requests to OpenAlex (`https://api.openalex.org/works`), PubMed (`https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`) and arXiv (`https://export.arxiv.org/api/query`). No API keys are required. Set the optional `TUTKIJA_CONTACT_EMAIL` environment variable to add a polite User-Agent/`mailto` to the calls.

Running `la search` appends a row to `data\cache\search_log.csv`. `la search-all` writes the merged Parquet plus an audit row in `data\cache\merge_log.csv` so you can track sources, duplicates and counts per topic. `la screen` produces `screened.parquet` and a `screen_log.csv` with screening metrics.

The screening command supports two engines:
- `scikit` (default): Built-in engine using scikit-learn, always available
- `asreview` (optional): ASReview engine, requires additional installation

To use ASReview, install it first with either:
```powershell
uv pip install asreview
# or
pip install tutkija[asreview]
```

For both engines, you can provide seed papers to improve the model's accuracy. Seeds are known relevant papers that help train the model. You can provide them as multiple --seeds flags OR as a single comma-separated string:

```powershell
# Method 1: Multiple --seeds flags (preferred)
la screen --in data\cache\merged.parquet --out data\cache\screened.parquet --seeds "doi:10.1234/xyz" --seeds "pmid:56789"

# Method 2: Comma-separated string
la screen --in data\cache\merged.parquet --out data\cache\screened.parquet --seeds "doi:10.1234/xyz,pmid:56789"
```

Each seed must be in the format `doi:...` or `pmid:...`. Both formats are supported with either engine (`scikit` or `asreview`).

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
