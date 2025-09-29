# src/la_pkg/write/render.py
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def get_quarto_bin() -> str:
    """Get the path to the Quarto binary."""
    return os.environ.get("QUARTO_BIN", "quarto")


def is_quarto_installed() -> bool:
    """Check if Quarto is installed and available on the PATH."""
    return shutil.which(get_quarto_bin()) is not None


def render_report(report_dir: Path, formats: list[str]) -> None:
    """
    Renders the Quarto report to the specified formats.
    """
    if not is_quarto_installed():
        raise RuntimeError(
            "Quarto is not installed or not found in PATH. "
            "Please install it from https://quarto.org/docs/get-started/"
        )

    qmd_file = report_dir / "report.qmd"
    if not qmd_file.exists():
        raise FileNotFoundError(f"Report file not found: {qmd_file}")

    command = [get_quarto_bin(), "render", str(qmd_file)]
    for fmt in formats:
        command.extend(["--to", fmt])

    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            cwd=report_dir,
        )
        print(process.stdout)
    except FileNotFoundError:
        raise RuntimeError(f"Quarto binary not found at: {get_quarto_bin()}")
    except subprocess.CalledProcessError as e:
        print("Quarto stdout:", e.stdout)
        print("Quarto stderr:", e.stderr)
        raise RuntimeError(f"Quarto rendering failed with exit code {e.returncode}") from e
