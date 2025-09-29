# ADR 0006: Writing and References Pipeline

## Status

Accepted

## Context

The project requires a standardized and automated way to generate a final report from the processed data, including analysis results, logs, and external citations. The process should be reproducible, version-controllable, and support academic citation standards.

## Decision

We will adopt the following toolchain for the writing and rendering pipeline:

1.  **Format**: [Quarto](https://quarto.org/) (`.qmd`) will be the primary authoring format. It provides a superset of Pandoc Markdown, allowing for the inclusion of code, tables, and complex layouts. It can render to multiple outputs (HTML, PDF) from a single source.
2.  **Reference Management**: [BibTeX](http://www.bibtex.org/) (`.bib`) will be used for managing bibliographic references. It is a well-established standard compatible with a vast ecosystem of tools.
3.  **Citation Styling**: [Citation Style Language (CSL)](https://citationstyles.org/) (`.csl`) files will be used to control the formatting of citations and bibliographies. This provides flexibility to switch between different academic styles (e.g., APA, Nature).
4.  **CLI Integration**: All operations will be exposed via a `la write` command group within the existing `la_pkg` CLI, ensuring consistent user interaction.
5.  **Artifact Paths**: All generated report artifacts will be stored in `output/report/` to keep outputs separate from source data and code.

## Consequences

### Positive

*   **Reproducibility**: The entire report generation process is scriptable and can be run from the CLI, making it easy to reproduce.
*   **Version Control**: Plain text formats (`.qmd`, `.bib`, `.csl`) are friendly to version control systems like Git.
*   **Flexibility**: Quarto and CSL allow for easy changes in output format and citation style without altering the source content.
*   **Automation**: The pipeline automates the tedious tasks of collecting references, inserting results, and formatting the document.

### Negative

*   **Dependency**: The rendering step introduces a dependency on a Quarto installation. This will be managed in CI using a dedicated setup action or a Docker image. For local development, users will be prompted with clear installation instructions if Quarto is not found.

### CI Strategy

*   Unit and integration tests for the Python code will be run on every commit, mocking the Quarto binary to avoid the installation dependency.
*   A full `quarto render` will be executed only on tag releases or manual triggers to generate the final HTML and PDF artifacts, which will be attached to the release.
