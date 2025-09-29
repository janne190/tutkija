"""
Renders the PRISMA diagram in SVG and PNG formats.
"""
import json
import logging
import subprocess
from pathlib import Path
from typing import Tuple, Dict, Any

import jinja2

from .schema import PrismaCounts

logger = logging.getLogger(__name__)

def render_prisma_python(counts_path: Path, out_dir: Path, formats: Tuple[str, ...] = ("svg", "png")):
    """
    Renders the PRISMA diagram using a Jinja2 template.
    Converts SVG to PNG if cairosvg is installed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(counts_path, 'r') as f:
        counts_data = json.load(f)

    # The template expects a flat dictionary
    template_vars = {
        "identified": counts_data["identified"],
        "duplicates_removed": counts_data["duplicates_removed"],
        "records_screened": counts_data["records_screened"],
        "records_excluded": counts_data["records_excluded"],
        "full_text_assessed": counts_data["full_text_assessed"],
        "studies_included": counts_data["studies_included"],
    }

    # Load Jinja2 template
    template_path = Path(__file__).parent.parent.parent.parent / "docs" / "assets" / "prisma_template.svg.j2"
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found at {template_path}")

    template_str = template_path.read_text()
    template = jinja2.Template(template_str)
    
    # Render SVG
    svg_content = template.render(template_vars)
    svg_path = out_dir / "prisma.svg"
    svg_path.write_text(svg_content)
    logger.info(f"Successfully rendered SVG to {svg_path}")

    # Render PNG if requested and possible
    if "png" in formats:
        try:
            import cairosvg
            png_path = out_dir / "prisma.png"
            cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), write_to=str(png_path))
            logger.info(f"Successfully rendered PNG to {png_path}")
        except ImportError:
            logger.warning("`cairosvg` is not installed. Skipping PNG generation. Install with `pip install cairosvg`.")

def render_prisma_r(counts_path: Path, out_dir: Path, rscript: str = "Rscript"):
    """
    Renders the PRISMA diagram using an R script.
    Checks for the Rscript executable first.
    """
    # Check if Rscript is available
    try:
        subprocess.run([rscript, "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error(f"'{rscript}' executable not found. Cannot render with R.")
        raise RuntimeError(f"Rscript not found at '{rscript}'")

    r_script_path = Path(__file__).parent.parent.parent.parent / "scripts" / "prisma" / "render_prisma.R"
    if not r_script_path.exists():
        raise FileNotFoundError(f"R script not found at {r_script_path}")

    cmd = [
        rscript,
        str(r_script_path),
        "--counts", str(counts_path),
        "--out-dir", str(out_dir)
    ]
    
    logger.info(f"Executing R script: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info(f"R script executed successfully. Output should be in {out_dir}")
