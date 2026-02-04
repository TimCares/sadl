#!/bin/bash
# =============================================================================
# Installation Test Script
# =============================================================================
# Tests that the built wheel can be installed and imported correctly.
# Uses uv for fast, isolated testing.
#
# Usage: ./tests/test_install.sh
# =============================================================================

set -e

# Colors
GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'
RESET='\033[0m'

# Check if uv is available, install if not
if ! command -v uv &> /dev/null; then
    printf "${YELLOW}uv not found, installing...${RESET}\n"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the updated PATH
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &> /dev/null; then
        printf "${RED}Failed to install uv. Please install manually: https://docs.astral.sh/uv/${RESET}\n"
        exit 1
    fi
    printf "${GREEN}uv installed successfully${RESET}\n"
fi

# Check if wheel exists
if ! ls dist/*.whl 1> /dev/null 2>&1; then
    printf "${RED}No wheel found in dist/. Run 'make build-only' first.${RESET}\n"
    exit 1
fi

TEST_VENV=".venv-install-test"

# Ensure cleanup on exit, error, or interrupt
cleanup() {
    rm -rf "$TEST_VENV"
}
trap cleanup EXIT

printf "${YELLOW}Creating isolated test environment...${RESET}\n"

# Clean up any existing test venv
rm -rf "$TEST_VENV"

# Create isolated venv and install the wheel
uv venv "$TEST_VENV" --quiet
uv pip install dist/*.whl --quiet --python "$TEST_VENV/bin/python"

printf "${YELLOW}Running installation tests...${RESET}\n"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the verification script
"$TEST_VENV/bin/python" "$SCRIPT_DIR/verify_install.py"

printf "${GREEN} Installation test completed successfully!${RESET}\n"
