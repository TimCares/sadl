# =============================================================================
# SADL Makefile
# =============================================================================
#
# Usage:
#   make help        - Show available commands
#   make install     - Install all dependencies
#   make check       - Run all checks (lint, typecheck, test)
#   make format      - Format code
#   make build       - Build package
#   make publish     - Publish to PyPI
#
# =============================================================================

.PHONY: help install install-dev sync lint format typecheck test test-cov \
        check clean pre-commit run build publish publish-test \
        bump bump-patch bump-minor bump-major changelog version commit

.DEFAULT_GOAL := help

# Colors for terminal output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# =============================================================================
# Help
# =============================================================================

help: ## Show this help message
	@echo "$(BLUE)LLM Project$(RESET) - Available commands:\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# =============================================================================
# Installation & Setup
# =============================================================================

install: ## Install production dependencies
	uv sync

install-dev: ## Install all dependencies including dev tools
	uv sync --extra dev

sync: install-dev ## Alias for install-dev

bootstrap: install-dev pre-commit-install ## Full project setup (deps + pre-commit hooks)
	@echo "$(GREEN)✓ Project bootstrapped successfully!$(RESET)"

pre-commit-install: ## Install pre-commit hooks
	uv run pre-commit install

# =============================================================================
# Code Quality
# =============================================================================



lint: ## Run linter (ruff)
	uv run ruff check .

lint-fix: ## Run linter and auto-fix issues
	uv run ruff check --fix .

format: ## Format code (ruff)
	uv run ruff format .

format-check: ## Check code formatting without changes
	uv run ruff format --check .

typecheck: ## Run type checker (mypy)
	uv run mypy .

# =============================================================================
# Testing
# =============================================================================

test: ## Run tests
	uv run pytest

test-v: ## Run tests with verbose output
	uv run pytest -v

test-cov: ## Run tests with coverage report
	uv run pytest --cov=src --cov-report=term-missing

test-cov-html: ## Run tests with HTML coverage report
	uv run pytest --cov=src --cov-report=html
	@echo "$(GREEN)Coverage report: htmlcov/index.html$(RESET)"

test-fast: ## Run tests in parallel (faster)
	uv run pytest -n auto

# =============================================================================
# All-in-One Commands
# =============================================================================

check: format lint typecheck test ## Run all checks (format, lint, typecheck, test)
	@echo "$(GREEN)✓ All checks passed!$(RESET)"

ci: format-check lint typecheck test-cov ## Run CI checks (stricter)
	@echo "$(GREEN)✓ CI checks passed!$(RESET)"

# =============================================================================
# Build & Publish
# =============================================================================

build: check clean ## Build package (runs checks first)
	uv build
	@echo "$(GREEN)✓ Package built successfully!$(RESET)"
	@echo "$(BLUE)Distribution files in dist/$(RESET)"

build-only: ## Build package without running checks
	uv build

publish-test: build ## Publish to TestPyPI (runs checks and build first)
	uv publish --publish-url https://test.pypi.org/legacy/
	@echo "$(GREEN)✓ Published to TestPyPI!$(RESET)"
	@echo "$(BLUE)Test install: pip install -i https://test.pypi.org/simple/ sadl$(RESET)"

publish: build ## Publish to PyPI (runs checks and build first)
	@echo "$(YELLOW)Publishing to PyPI...$(RESET)"
	uv publish
	@echo "$(GREEN)✓ Published to PyPI!$(RESET)"

# =============================================================================
# Version Management (Semantic Release & Commitizen)
# =============================================================================

commit: ## Create a conventional commit (interactive)
	uv run cz commit

version: ## Show current version
	@uv run cz version --project

version-next: ## Show next version (without applying)
	@uv run semantic-release version --print

bump: ## Auto-bump version based on commits (dry-run preview)
	@echo "$(BLUE)Current version:$(RESET)"
	@uv run cz version --project
	@echo "$(BLUE)Next version would be:$(RESET)"
	@uv run semantic-release version --print || echo "$(YELLOW)No release needed$(RESET)"
	@echo "$(YELLOW)Run 'make bump-apply' to apply changes$(RESET)"

bump-apply: ## Apply version bump locally (no push, no remote release)
	uv run semantic-release version --no-push --no-vcs-release

bump-patch: ## Bump patch version (x.x.X) using commitizen
	uv run cz bump --increment PATCH

bump-minor: ## Bump minor version (x.X.0) using commitizen
	uv run cz bump --increment MINOR

bump-major: ## Bump major version (X.0.0) using commitizen
	uv run cz bump --increment MAJOR

changelog: ## Generate/update changelog
	uv run semantic-release changelog

release: check ## Full release: bump, tag, push, create release (requires GH_TOKEN)
	uv run semantic-release version
	@echo "$(GREEN)✓ Release created!$(RESET)"

release-local: check ## Local release: bump and tag only (no push)
	uv run semantic-release version --no-push --no-vcs-release
	@echo "$(GREEN)✓ Local release created!$(RESET)"

release-dry: ## Preview release (shows what would happen)
	@echo "$(BLUE)=== Release Preview ===$(RESET)"
	@echo "$(BLUE)Last released version:$(RESET)"
	@uv run semantic-release version --print-last-released || echo "No previous release"
	@echo "$(BLUE)Next version:$(RESET)"
	@uv run semantic-release version --print || echo "$(YELLOW)No release needed$(RESET)"
	@echo "$(YELLOW)This was a preview. No changes were made.$(RESET)"

fix: lint-fix format ## Auto-fix linting issues and format code
	@echo "$(GREEN)✓ Code fixed and formatted!$(RESET)"

pre-commit: ## Run pre-commit on all files
	uv run pre-commit run --all-files

# =============================================================================
# Run Application
# =============================================================================

run: ## Run the main application
	uv run python main.py

run-debug: ## Run with debug logging
	uv run python main.py hydra.verbose=true

# =============================================================================
# Cleanup
# =============================================================================

clean: ## Remove build artifacts and cache files
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned!$(RESET)"

clean-outputs: ## Remove Hydra output directories
	rm -rf outputs
	rm -rf multirun
	find . -type f -name "*.log" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Output directories cleaned!$(RESET)"

clean-all: clean clean-outputs ## Remove all generated files
	rm -rf .venv
	@echo "$(GREEN)✓ All cleaned!$(RESET)"
