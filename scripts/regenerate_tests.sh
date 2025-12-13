#!/bin/bash
# Regenerate IEEE 754 test suite for Noir
#
# This script downloads test cases from the IBM FPgen suite and generates
# chunked Noir test files with a CI matrix for GitHub Actions.
#
# Usage:
#   ./scripts/regenerate_tests.sh              # Generate all tests
#   ./scripts/regenerate_tests.sh --operation add  # Generate only addition tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Regenerating IEEE 754 test suite..."
echo "Project root: $PROJECT_ROOT"
echo ""

# Default arguments
ARGS=(
    --all
    --split
    --output-dir src/ieee754_tests
    --ci-matrix .github/ci-matrix.json
)

# Append any additional arguments passed to this script
ARGS+=("$@")

echo "Running: python3 scripts/generate_tests.py ${ARGS[*]}"
echo ""

python3 scripts/generate_tests.py "${ARGS[@]}"

echo ""
echo "Done! Generated tests are in src/ieee754_tests/"
echo "CI matrix is at .github/ci-matrix.json"
