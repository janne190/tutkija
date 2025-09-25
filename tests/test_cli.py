"""Smoke tests for the la CLI."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def _resolve_cli() -> str:
    candidates = [
        Path(sys.executable).with_name("la.exe"),
        Path(sys.executable).with_name("la"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    which_path = shutil.which("la")
    if which_path:
        return which_path
    raise FileNotFoundError("la entrypoint puuttuu testiajosta")


def test_la_hello_runs() -> None:
    """CLI should emit Tutkija banner and config template."""
    cli_path = _resolve_cli()
    out = subprocess.check_output([cli_path, "hello"], text=True)
    assert "Tutkija, konfiguraation malli alla" in out
