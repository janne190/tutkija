import csv
from pathlib import Path

log_path = Path("data/cache/screen_log.csv")
if not log_path.exists():
    raise SystemExit("screen_log.csv puuttuu")

rows = list(csv.DictReader(log_path.open(encoding="utf-8")))
if not rows:
    raise SystemExit("screen_log.csv tyhja")

last = rows[-1]
required = [
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
missing = [field for field in required if field not in last]
if missing:
    raise SystemExit(f"screen_log viimeinen rivi puuttuu kentat: {missing}")


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


identified = _as_float(last["identified"])
screened = _as_float(last["screened"])
if identified <= 0:
    raise SystemExit("identified oltava positiivinen")
ratio = screened / identified
if ratio < 0.7:
    raise SystemExit(f"screened/identified liian pieni: {ratio:.2f}")

out_path = Path(last["out_path"])
if not out_path.exists():
    raise SystemExit(f"out_path ei osoita tiedostoon: {out_path}")

print("screen_log viimeinen rivi OK", last)
