# 7. PRISMA 2020 Diagram Generation

* Status: proposed
* Deciders:
* Date: 2025-09-29

## Context and Problem Statement

The project requires a standard PRISMA 2020 flow diagram to visualize the study selection process. This diagram is a critical component of systematic reviews and must be generated automatically and reproducibly based on the project's data pipeline. The process should be integrated into the existing CLI and CI/CD workflows.

## Decision Drivers

* **Reproducibility:** The diagram must be generated from a single source of truth for counts.
* **Automation:** The generation process should be fully automated via CLI commands.
* **CI/CD Integration:** The process must run in a CI environment without heavy dependencies (like a full R installation).
* **Flexibility:** Allow for both a lightweight, dependency-free rendering option and a more advanced option if specific tools (like R) are available.
* **Validation:** The counts must be internally consistent, and any discrepancies should raise warnings.

## Considered Options

1.  **R-only solution:** Use an R package like `PRISMA2020` or `DiagrammeR` to generate the diagram. This would require R and its dependencies to be installed in all environments, including CI.
2.  **Python-only solution:** Use a Python library to generate the diagram from scratch (e.g., `matplotlib`, `graphviz`). This gives full control but can be complex to implement and maintain the specific PRISMA layout.
3.  **Hybrid Python/R solution with Python as fallback:** Implement the primary rendering logic in Python using a template-based approach (SVG+Jinja2) for CI compatibility. Provide an optional R-based renderer for local use or for users who prefer the output of R packages.

## Decision Outcome

Chosen option: **"Hybrid Python/R solution with Python as fallback"**, because it provides the best balance of CI/CD compatibility, ease of implementation, and flexibility.

### Positive Consequences

*   The CI environment does not need a full R installation, keeping it lightweight.
*   The Python fallback using an SVG template is robust and has minimal dependencies (`jinja2`, `cairosvg` for PNG conversion).
*   The calculation logic is centralized in Python, providing a single source of truth (`data/cache/prisma_counts.json`).
*   Developers can still use a potentially higher-fidelity R renderer locally if they have R installed.
*   The CLI design cleanly separates computation, rendering, and attachment steps.

### Negative Consequences

*   Maintaining two rendering paths (Python and R) adds some complexity, though the R path is optional.
*   The SVG template is less flexible than a programmatic R solution if the PRISMA layout itself needs to be changed.

## Implementation Details

*   **Data Source of Truth:** A new file, `data/cache/prisma_counts.json`, will store the computed numbers. A CSV version will also be generated.
*   **Calculation Logic:** A new `la_pkg.prisma.compute` module will contain functions to calculate counts from `search_audit.csv`, `merged.parquet`, `screened.parquet`, etc. It will include fallback logic for missing files and validation rules to check for inconsistencies.
*   **Python Rendering:** A Jinja2 template (`docs/assets/prisma_template.svg.j2`) will define the SVG structure. The `la_pkg.prisma.render.render_prisma_python` function will inject the counts into this template. `cairosvg` will be used to convert the resulting SVG to PNG.
*   **R Rendering (Optional):** The `la_pkg.prisma.render.render_prisma_r` function will be a wrapper around an R script (`scripts/prisma/render_prisma.R`), which can be implemented using a package like `PRISMA2020`.
*   **CLI:** The `la_pkg.cli` module will have a new `prisma` command group with subcommands `compute`, `render`, `attach`, and `all`.
*   **Validation:** Validation errors will be logged to `data/logs/prisma_validation.csv` and will print a warning by default. A `--strict` flag will turn these warnings into errors.
