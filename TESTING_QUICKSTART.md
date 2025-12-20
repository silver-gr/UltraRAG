# Testing Quick Start Guide

## Installation

```bash
# Install test dependencies
pip install -r requirements.txt
```

This will install:
- `pytest>=7.0.0` - Testing framework
- `pytest-cov>=4.0.0` - Coverage reporting

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_loader.py
pytest tests/test_chunking.py
pytest tests/test_config.py
```

### Coverage Reports

```bash
# Terminal coverage report
pytest --cov=.

# HTML coverage report (detailed)
pytest --cov=. --cov-report=html
open htmlcov/index.html  # macOS
# or
xdg-open htmlcov/index.html  # Linux

# XML coverage report (for CI/CD)
pytest --cov=. --cov-report=xml
```

### Test Filtering

```bash
# Run specific test class
pytest tests/test_loader.py::TestWikilinkExtraction

# Run specific test
pytest tests/test_loader.py::TestWikilinkExtraction::test_extract_simple_wikilinks

# Run tests matching a pattern
pytest -k "wikilink"
pytest -k "config"

# Run only fast tests (skip slow tests)
pytest -m "not slow"
```

### Debugging

```bash
# Stop on first failure
pytest -x

# Show local variables on failure
pytest -l

# Drop into debugger on failure
pytest --pdb

# Verbose traceback
pytest --tb=long
```

## Test Structure

```
tests/
├── __init__.py              # Package init
├── conftest.py              # Shared fixtures
├── test_loader.py           # ObsidianLoader tests (30 tests)
├── test_chunking.py         # ObsidianChunker tests (25 tests)
└── test_config.py           # Configuration tests (40 tests)
```

## Example Test Run

```bash
$ pytest tests/test_loader.py -v

tests/test_loader.py::TestWikilinkExtraction::test_extract_simple_wikilinks PASSED
tests/test_loader.py::TestWikilinkExtraction::test_extract_wikilinks_with_aliases PASSED
tests/test_loader.py::TestTagExtraction::test_extract_simple_tags PASSED
...
======================== 30 passed in 2.5s =========================
```

## Common Issues

### Import Errors
Make sure you're running from the project root:
```bash
cd /Users/silver/Projects/UltraRAG
pytest
```

### Missing Dependencies
Install all dependencies:
```bash
pip install -r requirements.txt
```

### Environment Variables
Tests use temporary directories and mock API keys by default.
No .env file is needed for unit tests.

## What's Tested

### Loader Tests (test_loader.py)
- Wikilink extraction (all formats)
- Tag extraction (inline and frontmatter)
- Note loading (with/without frontmatter)
- Vault loading (subdirectories, error handling)
- Document conversion
- Wikilink graph building
- Path security

### Chunking Tests (test_chunking.py)
- Token counting approximation
- All chunking strategies (simple, markdown, semantic, hybrid)
- Chunk size and overlap
- Code block preservation
- Parent document context
- Error handling

### Config Tests (test_config.py)
- Embedding config validation
- LLM config validation (temperature 0-2)
- chunk_overlap < chunk_size validation
- Vault path validation
- API key validation
- Environment variable loading
- Edge cases and boundary values

## CI/CD Integration

For GitHub Actions:
```yaml
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest --cov=. --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Coverage Goals

Current: Aiming for 80%+ coverage
- loader.py: Comprehensive coverage
- chunking.py: All strategies covered
- config.py: All validators covered

## Next Steps

1. Run tests: `pytest -v`
2. Check coverage: `pytest --cov=. --cov-report=html`
3. Fix any failing tests
4. Add tests for new features
5. Maintain 80%+ coverage

## Getting Help

- Test documentation: `tests/README.md`
- Test summary: `TEST_SUMMARY.md`
- Pytest docs: https://docs.pytest.org/
- Coverage docs: https://coverage.readthedocs.io/
