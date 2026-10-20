#!/bin/bash
# Script to run UltraRAG tests

echo "Running UltraRAG Test Suite..."
echo "================================"
echo ""

# Check if uv is installed (uv run provides pytest via the default dev group)
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed"
    echo "Install with: brew install uv"
    exit 1
fi

# Run tests
echo "Running all tests..."
uv run pytest tests/ -v

echo ""
echo "Test run complete!"
echo ""
echo "To run specific tests:"
echo "  uv run pytest tests/test_loader.py"
echo "  uv run pytest tests/test_chunking.py"
echo "  uv run pytest tests/test_config.py"
echo ""
echo "Coverage report (written on every run, see pytest.ini):"
echo "  open htmlcov/index.html"
