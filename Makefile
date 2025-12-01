# Makefile for Fusion ID Project

.PHONY: help quickstart setup test lint clean experiments report sync install dev-install specs \
        uv-install venv-create install-deps verify test-setup clean-venv clean-all \
        baseline score-fusion feature-fusion check-env notebook docs watch profile count \
        link-dataset sync-results format test-file \
        embeddings embedding-experiment fusion-experiment embedding-all

# Variables
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
BLACK := $(VENV)/bin/black
FLAKE8 := $(VENV)/bin/flake8
UV := uv
REMOTE_HOST := UbuntuSungoddess
REMOTE_PATH := ~/fusion_id

# Detect OS for platform-specific commands
UNAME := $(shell uname -s)
ifeq ($(UNAME), Darwin)
    BREW_CHECK := $(shell which brew 2>/dev/null)
endif

# Default target
help:
	@echo "Fusion ID Project - Available Commands"
	@echo "======================================"
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make quickstart   - Complete setup from scratch (installs uv, Python, dependencies)"
	@echo ""
	@echo "📦 Setup & Installation:"
	@echo "  make setup        - Set up the project environment"
	@echo "  make uv-install   - Install uv package manager"
	@echo "  make venv-create  - Create virtual environment with uv"
	@echo "  make install-deps - Install project dependencies"
	@echo "  make dev-install  - Install development dependencies"
	@echo ""
	@echo "🧪 Testing & Verification:"
	@echo "  make test-setup   - Run setup verification test"
	@echo "  make test         - Run all unit tests"
	@echo "  make check-env    - Check environment and dependencies"
	@echo "  make verify       - Verify installation"
	@echo ""
	@echo "🔬 Experiments:"
	@echo "  make baseline     - Run baseline experiments"
	@echo "  make experiments  - Run all experiments"
	@echo "  make report       - Generate research report"
	@echo ""
	@echo "🧬 Embedding Experiments:"
	@echo "  make embeddings           - Generate text embeddings from LM Studio"
	@echo "  make embedding-experiment - Run embedding similarity identification"
	@echo "  make fusion-experiment    - Run input fusion ML experiments"
	@echo "  make embedding-all        - Run full embedding pipeline (all experiments)"
	@echo ""
	@echo "🧹 Maintenance:"
	@echo "  make clean        - Clean cache and temporary files"
	@echo "  make clean-venv   - Remove virtual environment"
	@echo "  make clean-all    - Deep clean (including venv)"
	@echo "  make lint         - Run code quality checks"
	@echo "  make format       - Format code with black"
	@echo ""
	@echo "🔄 Sync & Remote:"
	@echo "  make sync         - Sync to remote GPU server"
	@echo "  make sync-results - Sync results from remote"
	@echo ""
	@echo "📊 Other:"
	@echo "  make specs        - View specifications"
	@echo "  make notebook     - Start Jupyter notebook"
	@echo "  make docs         - Build documentation"
	@echo "  make count        - Count lines of code"

# Complete quickstart from scratch
quickstart: uv-install venv-create install-deps setup verify
	@echo "🎉 Quickstart complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Edit .env with your LM Studio configuration"
	@echo "2. Activate virtual environment: source .venv/bin/activate"
	@echo "3. Run test: make test-setup"
	@echo "4. Start experimenting: make baseline"

# Install uv package manager
uv-install:
	@echo "Checking for uv package manager..."
	@if ! command -v uv &> /dev/null; then \
		echo "Installing uv..."; \
		if [ "$(UNAME)" = "Darwin" ]; then \
			if [ -n "$(BREW_CHECK)" ]; then \
				brew install uv; \
			else \
				curl -LsSf https://astral.sh/uv/install.sh | sh; \
			fi \
		elif [ "$(UNAME)" = "Linux" ]; then \
			curl -LsSf https://astral.sh/uv/install.sh | sh; \
		else \
			echo "Please install uv manually from https://github.com/astral-sh/uv"; \
			exit 1; \
		fi \
	else \
		echo "✓ uv is already installed"; \
	fi

# Create virtual environment with uv
venv-create:
	@echo "Creating virtual environment with Python 3.11..."
	@if [ ! -d "$(VENV)" ]; then \
		$(UV) venv --python 3.11 $(VENV) || $(UV) venv $(VENV); \
		echo "✓ Virtual environment created"; \
	else \
		echo "✓ Virtual environment already exists"; \
	fi
	@echo "Bootstrapping pip..."
	@$(UV) pip install --python $(VENV) pip setuptools wheel
	@echo "✓ pip bootstrapped"

# Install dependencies with uv
install-deps: venv-create
	@echo "Installing project dependencies with uv..."
	@$(UV) pip install --python $(VENV) -r requirements.txt
	@echo "Installing project in editable mode..."
	@$(UV) pip install --python $(VENV) -e .
	@echo "✓ Dependencies installed"

# Setup project environment
setup: install-deps
	@echo "Setting up project directories..."
	@mkdir -p data/datasets data/cache
	@mkdir -p results/experiments results/reports
	@mkdir -p specs docs scripts
	@echo "Creating .env file from template..."
	@cp -n .env.example .env || true
	@echo "Creating dataset symlink..."
	@ln -sf $(DATASET_ROOT)/$(CURRENT_DATASET) data/datasets/current 2>/dev/null || true
	@echo "✓ Project structure created"

# Verify installation
verify:
	@echo "Verifying installation..."
	@$(PYTHON) --version
	@$(PYTHON) -c "import numpy; print(f'  NumPy: {numpy.__version__}')" 2>/dev/null || echo "  NumPy: not installed"
	@$(PYTHON) -c "import pandas; print(f'  Pandas: {pandas.__version__}')" 2>/dev/null || echo "  Pandas: not installed"
	@$(PYTHON) -c "import sklearn; print(f'  Scikit-learn: {sklearn.__version__}')" 2>/dev/null || echo "  Scikit-learn: not installed"
	@echo "✓ Basic verification complete"

# Test setup
test-setup:
	@echo "Running setup test..."
	@$(PYTHON) test_setup.py

# Install dependencies (legacy)
install: install-deps

# Install development dependencies
dev-install: install-deps
	@echo "Installing development dependencies..."
	@$(UV) pip install --python $(VENV) -e .

# Run tests
test:
	@echo "Running unit tests..."
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing

# Run specific test file
test-file:
	@echo "Running test file: $(FILE)"
	$(PYTEST) $(FILE) -v

# Code quality checks
lint:
	@echo "Running code formatters and linters..."
	$(BLACK) src/ experiments/ tests/ --check
	$(FLAKE8) src/ experiments/ tests/ --max-line-length=100

# Format code
format:
	@echo "Formatting code..."
	$(BLACK) src/ experiments/ tests/

# Clean temporary files and cache
clean:
	@echo "Cleaning temporary files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*~" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache
	@rm -rf .coverage
	@rm -rf htmlcov
	@echo "Cleaning data cache..."
	@rm -rf data/cache/*
	@echo "Clean complete!"

# Clean virtual environment
clean-venv:
	@echo "Removing virtual environment..."
	@rm -rf $(VENV)
	@echo "Virtual environment removed"

# Deep clean (everything including venv)
clean-all: clean clean-venv
	@echo "Deep clean complete!"

# Run baseline experiments
baseline: venv-create
	@echo "Running baseline experiments..."
	@$(PYTHON) experiments/run_baseline.py

# Run score fusion experiments
score-fusion: venv-create
	@echo "Running score fusion experiments..."
	@$(PYTHON) experiments/run_score_fusion.py

# Run feature fusion experiments
feature-fusion: venv-create
	@echo "Running feature fusion experiments..."
	@$(PYTHON) experiments/run_feature_fusion.py

# Run all experiments
experiments: baseline score-fusion feature-fusion
	@echo "All experiments complete!"
	@echo "Results saved in results/experiments/"

# Generate research report
report: venv-create
	@echo "Generating research report..."
	@$(PYTHON) scripts/generate_report.py
	@echo "Report saved in results/reports/"

# Sync to remote GPU server
sync:
	@echo "Syncing to remote server $(REMOTE_HOST)..."
	rsync -avz --exclude-from=.gitignore . $(REMOTE_HOST):$(REMOTE_PATH)/
	@echo "Sync complete!"

# Sync results from remote
sync-results:
	@echo "Syncing results from remote server..."
	rsync -avz $(REMOTE_HOST):$(REMOTE_PATH)/results/ results/
	@echo "Results synced!"

# Create dataset symlink
link-dataset:
	@echo "Creating dataset symlink..."
	@ln -sf $(DATASET_ROOT)/$(CURRENT_DATASET) data/datasets/current
	@echo "Dataset linked!"

# View specifications
specs:
	@echo "Available specifications:"
	@ls -la specs/
	@echo ""
	@echo "View with: cat specs/<filename>"

# Check environment
check-env: venv-create
	@echo "Checking environment setup..."
	@$(PYTHON) -c "import sys; print(f'Python: {sys.version}')"
	@$(PYTHON) -c "import numpy; print(f'NumPy: {numpy.__version__}')"
	@$(PYTHON) -c "import pandas; print(f'Pandas: {pandas.__version__}')"
	@$(PYTHON) -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
	@echo "Checking LM Studio connection..."
	@curl -s http://$(LM_STUDIO_HOST):1234/v1/models | head -n 1 || echo "LM Studio not accessible"

# Run jupyter notebook
notebook: venv-create
	@echo "Starting Jupyter notebook..."
	@$(VENV)/bin/jupyter notebook notebooks/

# Build documentation
docs: venv-create
	@echo "Building documentation..."
	@mkdir -p docs
	@$(PYTHON) scripts/build_docs.py

# Watch for changes and run tests
watch:
	@echo "Watching for changes..."
	@while true; do \
		$(MAKE) test; \
		echo "Waiting for changes..."; \
		sleep 5; \
	done

# Profile code
profile: venv-create
	@echo "Profiling code..."
	@$(PYTHON) -m cProfile -s cumulative experiments/run_baseline.py

# Count lines of code
count:
	@echo "Counting lines of code..."
	@find src experiments -name "*.py" | xargs wc -l | sort -n

# ============================================================================
# Embedding Experiment Targets
# ============================================================================

# Generate text embeddings using LM Studio
embeddings: venv-create
	@echo "Generating text embeddings..."
	@$(PYTHON) experiments/run_embedding_experiments.py --generate-embeddings-only

# Run embedding similarity identification experiment
embedding-experiment: venv-create
	@echo "Running embedding similarity experiment..."
	@$(PYTHON) experiments/run_embedding_experiments.py --embedding-only

# Run input fusion ML experiment
fusion-experiment: venv-create
	@echo "Running fusion ML experiment..."
	@$(PYTHON) experiments/run_embedding_experiments.py --fusion-only

# Run full embedding pipeline (embeddings + both experiments)
embedding-all: venv-create
	@echo "Running full embedding experiment pipeline..."
	@$(PYTHON) experiments/run_embedding_experiments.py --all