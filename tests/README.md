# UltraRAG Test Suite

Comprehensive unit tests for the UltraRAG system.

## Test Structure

```
tests/
├── __init__.py          # Test package initialization
├── conftest.py          # Shared pytest fixtures
├── test_loader.py       # Tests for ObsidianLoader
├── test_chunking.py     # Tests for ObsidianChunker
└── test_config.py       # Tests for configuration validation
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_loader.py
pytest tests/test_chunking.py
pytest tests/test_config.py
```

### Run specific test class
```bash
pytest tests/test_loader.py::TestWikilinkExtraction
pytest tests/test_config.py::TestEmbeddingConfig
```

### Run specific test function
```bash
pytest tests/test_loader.py::TestWikilinkExtraction::test_extract_simple_wikilinks
```

### Run with coverage report
```bash
pytest --cov=. --cov-report=html
# Open htmlcov/index.html to view detailed coverage report
```

### Run with verbose output
```bash
pytest -v
```

### Run only fast tests (skip slow tests)
```bash
pytest -m "not slow"
```

## Test Coverage

### test_loader.py
Tests for the ObsidianLoader module:
- **Wikilink extraction**: Simple links, aliases, spaces, nested paths
- **Tag extraction**: Simple tags, nested tags, hyphens, frontmatter tags
- **Note loading**: With/without frontmatter, unicode content, error handling
- **Vault loading**: Empty vaults, multiple notes, subdirectories, error handling
- **Document conversion**: Metadata preservation, custom fields
- **Wikilink graph**: Graph building, unresolved links
- **Security**: Path traversal protection

### test_chunking.py
Tests for the ObsidianChunker module:
- **Token counting**: Simple text, longer text, empty strings
- **Chunking strategies**: markdown_semantic, semantic, markdown, simple
- **Simple chunking**: Basic chunking, chunk size, overlap
- **Markdown chunking**: Header splitting, code block preservation
- **Semantic chunking**: Large section splitting, small section preservation
- **Parent context**: Context addition, multiple documents, summary truncation
- **Error handling**: Error propagation
- **Code preservation**: Python and JavaScript code blocks

### test_config.py
Tests for configuration validation:
- **EmbeddingConfig**: chunk_overlap validation, default values
- **LLMConfig**: Temperature validation (0-2 range), default values
- **VectorDBConfig**: Default configuration
- **GraphDBConfig**: Default configuration
- **RetrievalConfig**: Default configuration
- **RAGConfig**: Vault path validation, API key validation, nested configs
- **load_config**: Environment variable loading, type conversion
- **Edge cases**: Boundary values, special characters, unicode paths

## Test Fixtures

### Shared Fixtures (conftest.py)
- `temp_vault`: Temporary vault directory for testing
- `sample_note_content`: Sample markdown with frontmatter
- `sample_note_no_frontmatter`: Simple markdown without frontmatter
- `create_test_note`: Factory to create test notes
- `mock_embed_model`: Mock embedding model
- `sample_wikilink_content`: Various wikilink formats
- `sample_tag_content`: Various tag formats
- `sample_code_block_content`: Code blocks for testing

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

### Using Fixtures
```python
def test_with_fixture(temp_vault, mock_embed_model):
    """Test using shared fixtures."""
    config = EmbeddingConfig()
    chunker = ObsidianChunker(config, mock_embed_model)
    # Test logic here
```

## Coverage Goals

Target: 80%+ code coverage across all modules

Current coverage areas:
- ObsidianLoader: Comprehensive coverage of all methods
- ObsidianChunker: All chunking strategies and utilities
- Configuration: All validation rules and edge cases

## Continuous Integration

These tests are designed to run in CI/CD pipelines:
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests with coverage
pytest --cov=. --cov-report=xml

# Generate coverage badge
coverage-badge -o coverage.svg
```

## Troubleshooting

### Import Errors
Ensure you're running tests from the project root:
```bash
cd /Users/silver/Projects/UltraRAG
pytest
```

### Missing Dependencies
Install test dependencies:
```bash
pip install pytest>=7.0.0 pytest-cov>=4.0.0
```

### Environment Variables
Some tests may require environment variables. Create a `.env.test` file:
```bash
OBSIDIAN_VAULT_PATH=/path/to/test/vault
VOYAGE_API_KEY=test-key
```
