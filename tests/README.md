# UltraRAG Tests

Unit tests for UltraRAG components.

## Quick Start

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html
```

## Full Documentation

See [docs/TESTING.md](../docs/TESTING.md) for complete testing documentation including:
- Test structure and organization
- Running specific tests
- Coverage reports
- Writing new tests
- CI/CD integration
- Troubleshooting

## Test Files

| File | Module | Tests |
|------|--------|-------|
| `test_loader.py` | ObsidianLoader | 30 |
| `test_chunking.py` | ObsidianChunker | 25 |
| `test_config.py` | Configuration | 40 |
