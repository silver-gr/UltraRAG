# Testing Guide

## Quick Commands

```bash
# From repo root
cd /Users/silver/Projects/UltraRAG

# All tests (unit + integration)
python -m pytest

# Unit-only (no external API calls expected)
python -m pytest -m unit

# Integration-only (may require real API keys / external resources)
python -m pytest -m integration
```

## Marker Policy

- `-m unit`: fast tests, no external LLM/embedding API dependency expected.
- Full `pytest`: runs unit and integration tests.
- `-m integration`: can require API keys, model/network availability, and larger local fixtures.

## Current Test Structure

`tests/` currently contains:

- `tests/conftest.py`
- `tests/test_books.py`
- `tests/test_calibre_metadata.py`
- `tests/test_chunking.py`
- `tests/test_citations.py`
- `tests/test_config.py`
- `tests/test_indexing.py`
- `tests/test_integration.py`
- `tests/test_loader.py`
- `tests/test_obsession_radar.py`
- `tests/test_retrieval.py`
- `tests/fixtures/` (sample vault fixtures)

New hardening tests:

- `tests/test_auth.py`
- `tests/test_user_storage.py`
- `tests/test_rate_limit.py`

## Useful Debug Runs

```bash
# Stop on first failure
python -m pytest -x

# Verbose traceback
python -m pytest --tb=long

# Coverage report
python -m pytest --cov=. --cov-report=term-missing --cov-report=html
```

## Notes

- Run `python -m pytest -m unit` before opening PRs.
- Integration tests should be run when touching indexing/retrieval pipelines or external service integrations.
