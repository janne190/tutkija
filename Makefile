    .PHONY: setup lint test format ci

    setup:
	pipx install uv || true
	uv venv
	. .venv/bin/activate && uv pip install -r requirements.txt
	pre-commit install || true

    lint:
	. .venv/bin/activate && ruff check . && mypy --ignore-missing-imports .

    test:
	. .venv/bin/activate && pytest -q

    format:
	. .venv/bin/activate && ruff format .
