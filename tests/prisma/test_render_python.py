"""
Unit tests for PRISMA diagram rendering (Python).
"""
import pytest
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET

from src.la_pkg.prisma.render import render_prisma_python
from src.la_pkg.prisma.schema import PrismaCounts, IdentificationCounts

@pytest.fixture
def mock_counts_json(tmp_path: Path) -> Path:
    """Creates a mock prisma_counts.json file."""
    counts_data = PrismaCounts(
        identified=IdentificationCounts(total=300, by_source={'PubMed': 100, 'OpenAlex': 200}),
        duplicates_removed=50,
        records_screened=200,
        records_excluded=50,
        full_text_assessed=140,
        studies_included=100,
    )
    json_path = tmp_path / "prisma_counts.json"
    with open(json_path, 'w') as f:
        f.write(counts_data.model_dump_json(indent=2))
    return json_path

def test_render_prisma_python_svg_output(mock_counts_json: Path, tmp_path: Path):
    """Tests that render_prisma_python generates a valid SVG."""
    out_dir = tmp_path / "output"
    render_prisma_python(counts_path=mock_counts_json, out_dir=out_dir, formats=("svg",))

    svg_path = out_dir / "prisma.svg"
    assert svg_path.exists()
    
    svg_content = svg_path.read_text()
    
    # Parse SVG content as XML to check text elements
    root = ET.fromstring(svg_content)
    
    # Find all text elements and check their content
    text_elements = [text.text for text in root.findall(".//{http://www.w3.org/2000/svg}text")]
    
    assert "Records identified: 300" in text_elements
    assert "(Sources: {\"PubMed\": 100, \"OpenAlex\": 200})" in text_elements
    assert "Duplicates removed: 50" in text_elements
    assert "Records screened: 200" in text_elements
    assert "Records excluded: 50" in text_elements
    assert "Full-text assessed: 140" in text_elements
    assert "Studies included in synthesis: 100" in text_elements


def test_render_prisma_python_png_output(mock_counts_json: Path, tmp_path: Path):
    """Tests that render_prisma_python generates a PNG if cairosvg is available."""
    out_dir = tmp_path / "output"
    
    try:
        import cairosvg
        cairosvg_available = True
    except ImportError:
        cairosvg_available = False

    render_prisma_python(counts_path=mock_counts_json, out_dir=out_dir, formats=("svg", "png"))

    png_path = out_dir / "prisma.png"
    if cairosvg_available:
        assert png_path.exists()
    else:
        assert not png_path.exists()
        # Check if a warning was logged (requires capturing logs, which is more complex for a simple test)

def test_render_prisma_python_missing_template_raises_error(mock_counts_json: Path, tmp_path: Path, monkeypatch):
    """Tests that missing template raises FileNotFoundError."""
    # Temporarily change the template path to a non-existent one
    monkeypatch.setattr('src.la_pkg.prisma.render.Path.exists', lambda x: False)
    
    out_dir = tmp_path / "output"
    with pytest.raises(FileNotFoundError, match="Template not found"):
        render_prisma_python(counts_path=mock_counts_json, out_dir=out_dir)
