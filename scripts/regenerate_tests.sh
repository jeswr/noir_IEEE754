#!/bin/bash
# Regenerate IEEE 754 test suite for Noir
#
# This script downloads test cases from the IBM FPgen suite and generates
# separate Noir test packages for CI.
#
# Usage:
#   ./scripts/regenerate_tests.sh              # Generate all tests as packages
#   ./scripts/regenerate_tests.sh --operation add  # Generate only addition tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Regenerating IEEE 754 test suite..."
echo "Project root: $PROJECT_ROOT"
echo ""

# Default arguments - use packages mode for separate test packages
ARGS=(
    --all
    --packages
    --output-dir test_packages
    --ci-matrix .github/test-matrix.json
)

# Append any additional arguments passed to this script
ARGS+=("$@")

echo "Running: python3 scripts/generate_tests.py ${ARGS[*]}"
echo ""

python3 scripts/generate_tests.py "${ARGS[@]}"

echo ""
echo "Done! Generated test packages are in test_packages/"
echo "CI matrix is at .github/test-matrix.json"
