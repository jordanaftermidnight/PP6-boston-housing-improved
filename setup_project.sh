#!/bin/bash

echo "🏠 Boston Housing Project - Automated Setup"
echo "=========================================="

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p models results visualizations

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "🚀 To run the project:"
echo "   source venv/bin/activate"
echo "   python boston_housing_improved.py"
echo ""
echo "📊 Or use Jupyter:"
echo "   jupyter notebook boston_housing_improved.ipynb"