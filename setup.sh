#!/bin/bash
# setup.sh - Installs Python dependencies and creates data/out folder structure

# Exit immediately if a command fails
set -e

echo "Installing Python dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    python3 -m pip install --upgrade pip
    python3 -m pip install -r requirements.txt
else
    echo "Error: requirements.txt not found in the current directory."
    exit 1
fi

echo "Creating folder structure: data/out ..."
mkdir -p data/out

echo "Setup complete!"
