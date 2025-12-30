# Testing Guide

Comprehensive testing guide for UltraRAG.

## Quick Start

```bash
# Install test dependencies (included in requirements.txt)
pip install -r requirements.txt

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=. --cov-report=html
open htmlcov/index.html  # View coverage report
```

## Test Structure

```
tests/
├── __init__.py          # Test package initialization
├── conftest.py          # Shared pytest fixtures
├── test_loader.py       # Tests for ObsidianLoader (30 tests)
├── test_chunking.py     # Tests for ObsidianChunker (25 tests)
└── test_config.py       # Tests for configuration validation (40 tests)
```

**Total: 95+ unit tests**

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_loader.py
pytest tests/test_chunking.py
pytest tests/test_config.py

# Run specific test class
pytest tests/test_loader.py::TestWikilinkExtraction
pytest tests/test_config.py::TestEmbeddingConfig

# Run specific test function
pytest tests/test_loader.py::TestWikilinkExtraction::test_extract_simple_wikilinks
```

### Test Filtering

```bash
# Run tests matching a pattern
pytest -k "wikilink"
pytest -k "config"

# Run only fast tests (skip slow tests)
pytest -m "not slow"

# Run only integration tests
pytest -m integration

# Run only unit tests
pytest -m unit
```

### Coverage Reports

```bash
# Terminal coverage report
pytest --cov=.

# HTML coverage report (detailed)
pytest --cov=. --cov-report=html
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux

# XML coverage report (for CI/CD)
pytest --cov=. --cov-report=xml
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

## Test Coverage by Module

### test_loader.py (30 tests)
Tests for the ObsidianLoader module:

| Test Class | Description | Tests |
|-----------|-------------|-------|
| `TestWikilinkExtraction` | Wikilink parsing (simple, aliases, spaces, nested) | 6 |
| `TestTagExtraction` | Tag extraction (inline, frontmatter, combined) | 7 |
| `TestNoteLoading` | Note loading (frontmatter, unicode, errors) | 4 |
| `TestVaultLoading` | Vault loading (subdirs, filtering, errors) | 6 |
| `TestDocumentConversion` | LlamaIndex Document conversion | 2 |
| `TestWikilinkGraph` | Graph building from wikilinks | 3 |
| `TestPathSecurity` | Path traversal protection | 2 |

### test_chunking.py (25 tests)
Tests for the ObsidianChunker module:

| Test Class | Description | Tests |
|-----------|-------------|-------|
| `TestTokenCounting` | Token count approximation | 4 |
| `TestChunkingStrategy` | Strategy selection and execution | 4 |
| `TestSimpleChunking` | Basic sentence chunking | 3 |
| `TestMarkdownChunking` | Header-based splitting | 3 |
| `TestMarkdownSemanticChunking` | Hybrid parsing | 3 |
| `TestParentDocumentContext` | Context metadata | 4 |
| `TestChunkingErrorHandling` | Error propagation | 1 |
| `TestCodeBlockPreservation` | Code block integrity | 3 |

### test_config.py (40 tests)
Tests for configuration validation:

| Test Class | Description | Tests |
|-----------|-------------|-------|
| `TestEmbeddingConfig` | chunk_overlap validation, defaults | 6 |
| `TestLLMConfig` | Temperature validation (0-2 range) | 7 |
| `TestVectorDBConfig` | Vector DB defaults | 2 |
| `TestGraphDBConfig` | Graph DB defaults | 2 |
| `TestRetrievalConfig` | Retrieval defaults | 2 |
| `TestRAGConfig` | Vault path, API key validation | 9 |
| `TestLoadConfig` | Environment variable loading | 4 |
| `TestConfigEdgeCases` | Boundary values, special chars | 8 |

## Shared Fixtures

Located in `tests/conftest.py`:

| Fixture | Description |
|---------|-------------|
| `temp_vault` | Temporary vault directory for testing |
| `sample_note_content` | Sample markdown with frontmatter |
| `sample_note_no_frontmatter` | Simple markdown without frontmatter |
| `create_test_note` | Factory to create test notes |
| `mock_embed_model` | Mock embedding model for fast tests |
| `sample_wikilink_content` | Various wikilink formats |
| `sample_tag_content` | Various tag formats |
| `sample_code_block_content` | Code blocks for testing |

## Writing New Tests

### Test Naming Convention
- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Example Test

```python
def test_feature_description(temp_vault, create_test_note):
    """Test description of what is being tested."""
    # Arrange
    note_path = create_test_note("test.md", "content")
    loader = ObsidianLoader(temp_vault)

    # Act
    result = loader.some_method()

    # Assert
    assert result == expected_value
```

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

Target: **80%+ code coverage** across:
- loader.py
- chunking.py
- config.py

## Troubleshooting

### Import Errors
Ensure you're running from the project root:
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
Tests use temporary directories and mock API keys by default. No `.env` file is needed for unit tests.
