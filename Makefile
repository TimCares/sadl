# SADL Makefile

.PHONY: help install install-dev sync lint format typecheck ci-lint test \
        test-install test-all check clean pre-commit run build publish publish-test \
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
	@echo "$(BLUE)SADL$(RESET) - Available commands:\n"
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
	@echo "$(GREEN) Project bootstrapped successfully!$(RESET)"

pre-commit-install: ## Install pre-commit hooks
	uv run pre-commit install

# =============================================================================
# Code Quality
# =============================================================================

format: ## Format code (ruff)
	uv run ruff format .

format-check: ## Check code formatting without changes
	uv run ruff format --check .

lint: ## Run linter (ruff)
	uv run ruff check .

lint-fix: ## Run linter and auto-fix issues
	uv run ruff check --fix .

typecheck: ## Run type checker (mypy)
	uv run mypy .

ci-lint: format-check lint typecheck ## Run all static checks (CI)
	@echo "$(GREEN) All static checks passed!$(RESET)"

# =============================================================================
# Testing
# =============================================================================

test: ## Run tests
	uv run pytest

test-v: ## Run tests with verbose output
	uv run pytest -v

test-fast: ## Run tests in parallel (faster)
	uv run pytest -n auto

test-install: build-only ## Test installation from built wheel
	@chmod +x tests/test_install.sh
	@./tests/test_install.sh

test-all: test test-install ## Run all tests including installation test
	@echo "$(GREEN) All tests passed!$(RESET)"

# =============================================================================
# All-in-One Commands
# =============================================================================

check: format lint typecheck test-all ## Run all checks (format, lint, typecheck, test, test install)
	@echo "$(GREEN) All checks passed!$(RESET)"

ci: format-check lint typecheck ## Run CI checks (stricter)
	@echo "$(GREEN) CI checks passed!$(RESET)"

# =============================================================================
# Build & Publish
# =============================================================================

build: check clean ## Build package (runs checks first)
	uv build
	@echo "$(GREEN) Package built successfully!$(RESET)"
	@echo "$(BLUE)Distribution files in dist/$(RESET)"

build-only: ## Build package without running checks
	uv build

publish-test: build ## Publish to TestPyPI (runs checks and build first)
	uv publish --publish-url https://test.pypi.org/legacy/
	@echo "$(GREEN) Published to TestPyPI!$(RESET)"
	@echo "$(BLUE)Test install: pip install -i https://test.pypi.org/simple/ py-sadl$(RESET)"

publish: build ## Publish to PyPI (runs checks and build first)
	@echo "$(YELLOW)Publishing to PyPI...$(RESET)"
	uv publish
	@echo "$(GREEN) Published to PyPI!$(RESET)"

# =============================================================================
# Version Management (Semantic Release & Commitizen)
# =============================================================================

commit: ## Create a conventional commit
	uv run cz commit

version: ## Show current version
	@uv run cz version --project

bump-dry:
	@uv run semantic-release -v --noop version

# =============================================================================
# Cleanup
# =============================================================================

clean: ## Remove everything
	rm -rf .venv
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN) Cleaned!$(RESET)"
