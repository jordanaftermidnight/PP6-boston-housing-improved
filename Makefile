.PHONY: all setup run clean help

# Default target
all: setup run

# Setup virtual environment and install dependencies
setup:
	@echo "📦 Setting up virtual environment..."
	python3 -m venv venv
	@echo "📚 Installing dependencies..."
	venv/bin/pip install --upgrade pip
	venv/bin/pip install -r requirements.txt
	@echo "📁 Creating directories..."
	mkdir -p models results visualizations
	@echo "✅ Setup complete!"

# Run the main script
run:
	@echo "🚀 Running Boston Housing analysis..."
	venv/bin/python boston_housing_improved.py

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf models/*.h5 models/*.pkl
	rm -rf visualizations/*.png
	rm -rf results/*.json results/*.txt
	@echo "✅ Clean complete!"

# Show help
help:
	@echo "Available targets:"
	@echo "  all     - Setup and run (default)"
	@echo "  setup   - Create venv and install dependencies"
	@echo "  run     - Run the main analysis"
	@echo "  clean   - Remove generated files"
	@echo "  help    - Show this help message"