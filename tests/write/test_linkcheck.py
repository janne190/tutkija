# tests/write/test_linkcheck.py
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from la_pkg.write.linkcheck import run_linkcheck

MOCK_BIBTEX_CONTENT = """@article{test1,
    doi = {10.1234/good},
    url = {http://example.com/good}
}
@article{test2,
    doi = {10.1234/bad}
}
"""


def test_run_linkcheck(tmp_path: Path):
    bib_path = tmp_path / "refs.bib"
    log_path = tmp_path / "linkcheck.csv"
    bib_path.write_text(MOCK_BIBTEX_CONTENT)

    with patch("la_pkg.write.linkcheck.httpx.Client.head") as mock_head:
        # Simulate one good and one bad link
        mock_head.side_effect = [
            (tmp_path / "good_response").touch(),  # Mock a successful response object
            (tmp_path / "good_response").touch(),
            Exception("Request failed"),
        ]

        run_linkcheck(bib_path, log_path)

    assert log_path.exists()
    df = pd.read_csv(log_path)
    assert len(df) == 3
    assert df["status_code"].isin([200, 0]).all()
    assert df[df["status_code"] == 0].iloc[0]["reason"].startswith("Request Error")
