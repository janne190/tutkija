"""
Attaches the PRISMA diagram to a Quarto markdown file.
"""
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

def attach_to_qmd(qmd_path: Path, image_path: Path):
    """
    Attaches the PRISMA diagram to the given Quarto markdown file.
    Finds the '# PRISMA' anchor or '# References' and inserts the image link.
    If the image link already exists, it will be replaced.
    """
    if not qmd_path.exists():
        raise FileNotFoundError(f"Quarto file not found at {qmd_path}")

    content = qmd_path.read_text(encoding='utf-8')
    
    # Use relative path for the image in the markdown file
    image_rel_path = image_path.relative_to(qmd_path.parent)
    
    markdown_link = f'![PRISMA 2020]({image_rel_path}){{fig-cap="PRISMA 2020 flow diagram"}}'
    
    # Regex to find an existing PRISMA diagram link
    prisma_regex = re.compile(r"!\[PRISMA 2020\]\(.*\)\{.*\}")

    if prisma_regex.search(content):
        # If link exists, replace it
        new_content = prisma_regex.sub(markdown_link, content)
        logger.info(f"Replaced existing PRISMA diagram link in {qmd_path}")
    else:
        # Otherwise, insert it before '# References' or at the end of the file
        insertion_point = content.find("\n# References")
        if insertion_point != -1:
            new_content = content[:insertion_point] + f"\n## PRISMA\n\n{markdown_link}\n" + content[insertion_point:]
            logger.info(f"Inserted PRISMA diagram link before '# References' in {qmd_path}")
        else:
            new_content = content + f"\n\n## PRISMA\n\n{markdown_link}\n"
            logger.info(f"Appended PRISMA diagram link to {qmd_path}")
            
    qmd_path.write_text(new_content, encoding='utf-8')
