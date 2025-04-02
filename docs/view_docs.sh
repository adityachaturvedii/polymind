#!/bin/bash
# Simple script to open the documentation in a web browser

# Get the absolute path to the HTML documentation
DOCS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/_build/html" && pwd)/index.html"

# Check if the documentation exists
if [ ! -f "$DOCS_PATH" ]; then
    echo "Documentation not found. Please build it first with 'cd docs && make html'"
    exit 1
fi

# Try to open the documentation with the default browser
if command -v xdg-open > /dev/null; then
    xdg-open "$DOCS_PATH"
elif command -v open > /dev/null; then
    open "$DOCS_PATH"
elif command -v start > /dev/null; then
    start "$DOCS_PATH"
else
    echo "Could not open the documentation. Please open it manually at: $DOCS_PATH"
fi
