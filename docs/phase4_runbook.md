# Phase 4 sample extraction

The Phase 4 workflow (PDF discovery, download, parsing) produces three primary artefacts under `data/`:

- `data/cache/pdf_index.parquet` – discovery metadata with provider hints and file paths.
- `data/logs/pdf_audit.csv` – download audit trail covering provider, status, and byte counts.
- `data/cache/parsed_index.parquet` – parse results that link PDFs to generated TEI/text artefacts.

To help operators validate integrations without rerunning the entire pipeline, you can generate a synthetic snapshot with the helper script described below. The script emits CSV representations of the discovery, audit, and parsing tables plus a curated subset of PDF and TEI files that mirror a ~20 row run. Binary artefacts (PDFs and the packaged zip archive) are intentionally git-ignored so they can be created locally without bloating the repository.

## Using the packaged sample

1. Run the helper script to (re)build the dataset:
   ```bash
   python scripts/create_phase4_samples.py
   ```
   This command creates the cached CSVs, TEI/text artefacts, lightweight PDF stubs, and `data/samples/vaihe4_sample.zip`.
2. Unpack `data/samples/vaihe4_sample.zip` preserving its directory structure:
   ```bash
   unzip data/samples/vaihe4_sample.zip -d data/samples/extracted
   ```
3. Copy the cached CSVs to the expected cache and logs locations:
   ```bash
   cp data/samples/extracted/cache/pdf_index.csv data/cache/pdf_index.csv
   cp data/samples/extracted/logs/pdf_audit.csv data/logs/pdf_audit.csv
   cp data/samples/extracted/cache/parsed_index.csv data/cache/parsed_index.csv
   ```
4. Copy the bundled PDFs and parsed artefacts into the live directories:
   ```bash
   cp -R data/samples/extracted/pdfs/* data/pdfs/
   cp -R data/samples/extracted/parsed/* data/parsed/
   ```
5. When Parquet outputs are required, convert the CSVs using your preferred tooling (for example `pandas` or `pyarrow`) once dependencies are available:
   ```bash
   python - <<'PY'
   import pandas as pd
   pd.read_csv('data/cache/pdf_index.csv').to_parquet('data/cache/pdf_index.parquet', index=False)
   pd.read_csv('data/cache/parsed_index.csv').to_parquet('data/cache/parsed_index.parquet', index=False)
   PY
   ```

The synthetic rows emulate a discovery run that mixes arXiv, PubMed Central, and Unpaywall candidates. PDF stubs and TEI outputs are intentionally lightweight and generated on demand so that the repository can avoid tracking binary artefacts.

## Regenerating the snapshot

Run `python scripts/create_phase4_samples.py` whenever you need to refresh the dataset. The script uses deterministic seed data to rebuild the CSV caches, PDF placeholders, TEI/text summaries, and the `data/samples/vaihe4_sample.zip` archive.
