# tests/write/test_cli_smoke.py
from __future__ import annotations

from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import patch

from la_pkg.cli import app

runner = CliRunner()


def test_write_init_smoke(tmp_path: Path) -> None:
    """
    Smoke test for the `la write init` command.
    """
    report_dir = tmp_path / "report"
    style_dir = tmp_path / "styles"
    style_dir.mkdir()
    style_file = style_dir / "apa.csl"
    style_file.touch()

    result = runner.invoke(
        app,
        [
            "write",
            "init",
            "--out",
            str(report_dir),
            "--style",
            str(style_file),
            "--title",
            "My Test Report",
            "--authors",
            "Doe, John; Smith, Jane",
        ],
    )

    assert result.exit_code == 0
    assert f"Report scaffold created at: {report_dir}" in result.stdout
    assert (report_dir / "report.qmd").exists()
    assert (report_dir / "references.bib").exists()
    assert (report_dir / "_quarto.yml").exists()
    assert (report_dir / "style.csl").exists()

    qmd_content = (report_dir / "report.qmd").read_text()
    assert 'title: "My Test Report"' in qmd_content
    assert 'author: ["Doe, John", "Smith, Jane"]' in qmd_content


@patch("la_pkg.write.render.subprocess.run")
@patch("la_pkg.write.render.is_quarto_installed", return_value=True)
def test_write_render_smoke(mock_is_installed, mock_run, tmp_path: Path) -> None:
    """
    Smoke test for the `la write render` command.
    """
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "report.qmd").touch()

    result = runner.invoke(
        app,
        [
            "write",
            "render",
            "--dir",
            str(report_dir),
            "--format",
            "html,pdf",
        ],
    )

    assert result.exit_code == 0
    assert f"Report rendered successfully in: {report_dir}" in result.stdout
    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    assert "quarto" in args[0]
    assert "render" in args[0]
    assert str(report_dir / "report.qmd") in args[0]
    assert "--to" in args[0]
    assert "html" in args[0]
    assert "pdf" in args[0]
