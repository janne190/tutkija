import pandas as pd

pdf_frame = pd.read_parquet("data/cache/pdf_index.parquet")
parsed_frame = pd.read_parquet("data/cache/parsed_index.parquet")

pdf_ok = int(pdf_frame.get("has_fulltext", pd.Series(dtype=int)).sum())
parsed_ok = int(parsed_frame.get("parsed_ok", pd.Series(dtype=int)).sum())

print("PDF ok:", pdf_ok, "/", len(pdf_frame))
print("Parsed ok:", parsed_ok, "/", len(parsed_frame))

if len(parsed_frame) > 0:
    ratio = parsed_ok / len(parsed_frame)
    if ratio < 0.8:
        raise SystemExit(f"parsed_ok ratio too low: {ratio:.2f}")
