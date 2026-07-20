# Repository Guidelines

## Project Structure & Module Organization

The repository contains two Python packages. Retrieval code lives in `src/retrieval/`; benchmark entry points, index builders, reports, and plotting utilities live in `scripts/`. Retrieval tests are in `tests/test_retrieval.py`.

`latex-parser/` is a standalone chunking package with its own `src/chunking/`, `scripts/`, `tests/`, and `pyproject.toml`. Design notes and implementation plans are kept under `docs/superpowers/`. Keep generated datasets, caches, indexes, and benchmark outputs outside source directories; large `*.parquet` and `*.jsonl` files are intentionally ignored.

## Build, Test, and Development Commands

- `python -m pip install -e ".[dev,eval]"` installs retrieval code, pytest, dataset adapters, and plotting support.
- `python -m pip install -e "./latex-parser[dev]"` installs the chunking package and its test dependencies.
- `pytest` runs both suites using the root `pyproject.toml` paths.
- `pytest tests/test_retrieval.py -v` or `pytest latex-parser/tests -v` runs one area while iterating.
- `python scripts/run_public_bench.py --benchmark litsearch --mode sparse --cache-dir <hf-cache>` runs an offline sparse benchmark against a staged Hugging Face cache.
- `./scripts/run_all_benchmarks.sh` runs the full sparse/dense/hybrid sweep. It requires cached datasets, Qdrant, and the configured embedding endpoint; use `BUILD_INDEXES=0` to reuse collections.

## Coding Style & Naming Conventions

Target Python 3.10 or newer. Use four-space indentation, standard-library-first imports, type hints for public interfaces, and small, focused modules. Name functions and files with `snake_case`, classes with `PascalCase`, and constants with `UPPER_SNAKE_CASE`. Match the surrounding code's direct, dependency-light style. No formatter or linter is configured, so keep imports tidy and avoid unrelated reformatting. Shell scripts should remain Bash-compatible and retain strict error handling (`set -euo pipefail`).

## Testing Guidelines

Tests use pytest. Name files `test_*.py` and test functions `test_<behavior>`. Add a regression test for behavior changes; prefer small in-memory fixtures, `tmp_path`, and `monkeypatch` over network or service calls. There is no declared coverage threshold, but changed branches and failure paths should be exercised. Run `pytest` before opening a pull request.

## Commit & Pull Request Guidelines

History favors short, task-focused subjects such as `added progress tracking` and `Fixed scholar gym cache`; Conventional Commit prefixes are not required. Prefer an imperative subject that names the affected behavior, and keep each commit scoped.

Pull requests should explain the motivation and implementation, list verification commands, and link any issue or design document. For benchmark changes, include the dataset, mode, model/configuration, and a concise before/after metric sample. Call out new cache, Qdrant, or embedding-service requirements explicitly.
