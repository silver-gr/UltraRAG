#!/bin/bash
# Setup script for UltraRAG

set -e

echo "🚀 UltraRAG Setup Script"
echo "========================"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version || { echo "❌ Python 3 not found. Please install Python 3.10+"; exit 1; }

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file with your settings:"
    echo "   - Set OBSIDIAN_VAULT_PATH to your vault location"
    echo "   - Add API keys (VOYAGE_API_KEY, GOOGLE_API_KEY)"
    echo ""
fi

# Create data directory
mkdir -p data/lancedb

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Activate the virtual environment: source venv/bin/activate"
echo "3. Run the system: python main.py"
echo "   OR launch web UI: streamlit run app.py"
echo ""
