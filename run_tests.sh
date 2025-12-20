#!/bin/bash
# Script to run UltraRAG tests

echo "Running UltraRAG Test Suite..."
echo "================================"
echo ""

# Check if pytest is installed
if ! python -m pytest --version &> /dev/null; then
    echo "Error: pytest is not installed"
    echo "Install with: pip install pytest pytest-cov"
    exit 1
fi

# Run tests
echo "Running all tests..."
python -m pytest tests/ -v

echo ""
echo "Test run complete!"
echo ""
echo "To run specific tests:"
echo "  pytest tests/test_loader.py"
echo "  pytest tests/test_chunking.py"
echo "  pytest tests/test_config.py"
echo ""
echo "To generate coverage report:"
echo "  pytest --cov=. --cov-report=html"
echo "  open htmlcov/index.html"
