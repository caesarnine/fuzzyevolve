# Repository Guidelines

## Project Structure & Module Organization
- Source: `fuzzyevolve/` (CLI in `cli.py`; core in `evolution/`, `llm/`, `utils/`).
- Tests: `tests/` with `test_*.py` for archive, diff, driver, and parsers.
- Config: `pyproject.toml` (project metadata, Ruff); lockfile: `uv.lock`.
- Artifacts: default output `best.txt`; optional logs via `--log-file`.

## Build, Test, and Development Commands (uv)
- Create env + install deps:
  ```bash
  uv venv && source .venv/bin/activate
  uv sync --extra dev         # dev tools (pytest, ruff, mypy, build)
  # Optional extras: uv sync --extra vertex
  ```
- Lint/format (Ruff):
  ```bash
  uv run ruff format . && uv run ruff check .
  ```
- Type checks:
  ```bash
  uv run mypy fuzzyevolve
  ```
- Tests (pytest):
  ```bash
  uv run pytest -q
  uv run pytest -q -k driver
  ```
- Build package:
  ```bash
  uv run python -m build
  ```
- Run CLI locally:
  ```bash
  uv run fuzzyevolve "Seed text" -g "goal" -o best.txt
  uv run fuzzyevolve --help
  ```

## Coding Style & Naming Conventions
- Python ≥ 3.10; 4-space indents; line length 88; double quotes.
- Type hints and brief docstrings for public APIs.
- Names: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`.
- Use Ruff for formatting and linting; keep ignores aligned with `pyproject.toml`.

## Testing Guidelines
- Framework: `pytest`. Place tests in `tests/` named `test_*.py`; functions `test_*`.
- Add focused unit tests for changes in `evolution/`, `llm/`, and `utils/`.
- Mock external LLM calls in unit tests; keep tests deterministic.

## Commit & Pull Request Guidelines
- Commits: imperative, scoped summaries (e.g., `evolution: improve driver scoring`).
- PRs: problem/solution, linked issues, CLI examples (before/after), risks/rollbacks.
- Verify locally: `uv run ruff check`, `uv run mypy`, and `uv run pytest` all pass.

## Security & Configuration Tips
- LLM via LiteLLM: export provider credentials (e.g., `OPENAI_API_KEY` or Vertex AI auth) before running.
- Do not commit secrets or large artifacts; prefer `--log-file` and keep logs untracked.
