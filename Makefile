# Makefile for keystroke-scripts
# Compatible with Slurm cluster environments

SHELL := /bin/bash
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
UV := uv

# Detect OS for platform-specific commands
UNAME := $(shell uname -s)
ifeq ($(UNAME), Darwin)
    BREW_CHECK := $(shell which brew 2>/dev/null)
endif

.PHONY: all quickstart help venv install clean reinstall check \
        run-scenario run-runner uv-install venv-create install-deps verify

# Default target
help:
	@echo "Keystroke Scripts - Available Commands"
	@echo "======================================="
	@echo ""
	@echo "Quick Start:"
	@echo "  make quickstart   - Complete setup from scratch"
	@echo "  source .venv/bin/activate"
	@echo ""
	@echo "Setup:"
	@echo "  make uv-install   - Install uv package manager"
	@echo "  make venv-create  - Create virtual environment"
	@echo "  make install-deps - Install dependencies"
	@echo "  make verify       - Verify installation"
	@echo ""
	@echo "Run Experiments:"
	@echo "  make run-scenario - Run scenario-based experiments"
	@echo "  make run-runner   - Run standard ML experiments"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean        - Remove venv and cache"
	@echo "  make reinstall    - Clean and reinstall"
	@echo "  make check        - Check all imports work"

# Complete quickstart from scratch
quickstart: uv-install venv-create install-deps verify
	@echo ""
	@echo "============================================"
	@echo "Setup complete!"
	@echo "Run: source .venv/bin/activate"
	@echo "============================================"

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
		echo "uv is already installed"; \
	fi

# Create virtual environment with uv
venv-create:
	@echo "Creating virtual environment..."
	@if [ ! -d "$(VENV)" ]; then \
		$(UV) venv --python 3.11 $(VENV) || $(UV) venv $(VENV); \
		echo "Virtual environment created"; \
	else \
		echo "Virtual environment already exists"; \
	fi
	@echo "Bootstrapping pip..."
	@$(UV) pip install --python $(VENV) pip setuptools wheel

# Install dependencies with uv
install-deps: venv-create
	@echo "Installing project dependencies..."
	@$(UV) pip install --python $(VENV) -e .
	@echo "Dependencies installed"

# Verify installation
verify:
	@echo "Verifying installation..."
	@$(PYTHON) --version
	@$(PYTHON) -c "import numpy; print(f'  NumPy: {numpy.__version__}')"
	@$(PYTHON) -c "import pandas; print(f'  Pandas: {pandas.__version__}')"
	@$(PYTHON) -c "import polars; print(f'  Polars: {polars.__version__}')"
	@$(PYTHON) -c "import sklearn; print(f'  Scikit-learn: {sklearn.__version__}')"
	@$(PYTHON) -c "import xgboost; print(f'  XGBoost: {xgboost.__version__}')"
	@$(PYTHON) -c "import catboost; print(f'  CatBoost: {catboost.__version__}')"
	@$(PYTHON) -c "import lightgbm; print(f'  LightGBM: {lightgbm.__version__}')"
	@$(PYTHON) -c "import torch; print(f'  PyTorch: {torch.__version__}')"
	@$(PYTHON) -c "import bob.measure; print('  bob.measure: OK')" 2>/dev/null || echo "  bob.measure: not installed (optional)"
	@echo "Verification complete"

# Full import check
check:
	@echo "Checking all imports..."
	@$(PYTHON) -c "\
import sys; \
print(f'Python: {sys.version}'); \
import ml_core; print('ml_core: OK'); \
import ml_utils; print('ml_utils: OK'); \
import ml_visualizer; print('ml_visualizer: OK'); \
import ml_scenario_runner; print('ml_scenario_runner: OK'); \
import scenarios; print('scenarios: OK'); \
print('All imports successful!'); \
"

# Clean up
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV)
	rm -rf __pycache__ */__pycache__
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf build dist
	@echo "Cleanup complete"

# Reinstall from scratch
reinstall: clean quickstart

# Run scenario-based experiments
run-scenario:
	@echo "Running scenario-based experiments..."
	$(PYTHON) ml_scenario_runner.py $(ARGS)

# Run standard ML experiments
run-runner:
	@echo "Running ML experiments..."
	$(PYTHON) ml_runner.py $(ARGS)
