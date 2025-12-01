# Makefile for keystroke-scripts
# Compatible with Slurm environments

SHELL := /bin/bash
PYTHON := python3
VENV := .venv
VENV_BIN := $(VENV)/bin
PIP := $(VENV_BIN)/pip
PYTHON_VENV := $(VENV_BIN)/python

.PHONY: all quickstart install clean venv reinstall help check run-scenario run-runner

# Default target
all: install

# Quickstart - one command setup
quickstart: venv
	@echo "Upgrading pip..."
	@$(PIP) install --upgrade pip setuptools wheel
	@echo "Installing project dependencies..."
	@$(PIP) install -e .
	@echo ""
	@echo "============================================"
	@echo "Setup complete! Run: source .venv/bin/activate"
	@echo "============================================"

# Help target
help:
	@echo "Available targets:"
	@echo "  make quickstart   - One-command setup (recommended)"
	@echo "  make install      - Create venv and install all dependencies"
	@echo "  make venv         - Create virtual environment only"
	@echo "  make reinstall    - Remove venv and reinstall from scratch"
	@echo "  make clean        - Remove virtual environment and cache files"
	@echo "  make check        - Verify installation by importing key modules"
	@echo "  make run-scenario - Run scenario-based experiments"
	@echo "  make run-runner   - Run standard ML experiments"
	@echo ""
	@echo "Quick start:"
	@echo "  make quickstart"
	@echo "  source .venv/bin/activate"

# Create virtual environment
venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created at $(VENV)"

# Install all dependencies
install: venv
	@echo "Upgrading pip..."
	$(PIP) install --upgrade pip setuptools wheel
	@echo "Installing project dependencies..."
	$(PIP) install -e .
	@echo ""
	@echo "Installation complete!"
	@echo "To activate: source $(VENV)/bin/activate"

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
reinstall: clean install

# Check installation
check:
	@echo "Checking installation..."
	@$(PYTHON_VENV) -c "\
import sys; \
print(f'Python: {sys.version}'); \
import pandas; print(f'pandas: {pandas.__version__}'); \
import numpy; print(f'numpy: {numpy.__version__}'); \
import polars; print(f'polars: {polars.__version__}'); \
import sklearn; print(f'scikit-learn: {sklearn.__version__}'); \
import xgboost; print(f'xgboost: {xgboost.__version__}'); \
import catboost; print(f'catboost: {catboost.__version__}'); \
import lightgbm; print(f'lightgbm: {lightgbm.__version__}'); \
import matplotlib; print(f'matplotlib: {matplotlib.__version__}'); \
import seaborn; print(f'seaborn: {seaborn.__version__}'); \
import plotly; print(f'plotly: {plotly.__version__}'); \
import torch; print(f'torch: {torch.__version__}'); \
import bob.measure; print('bob.measure: OK'); \
print(''); \
print('All imports successful!'); \
"

# Run scenario-based experiments
run-scenario:
	@echo "Running scenario-based experiments..."
	$(PYTHON_VENV) ml_scenario_runner.py $(ARGS)

# Run standard ML experiments
run-runner:
	@echo "Running ML experiments..."
	$(PYTHON_VENV) ml_runner.py $(ARGS)
