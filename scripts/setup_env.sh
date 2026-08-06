#!/usr/bin/env bash
set -e

echo "=== Setting up atmosIQ Development Environment ==="

# Check Python version
python3 --version

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment and installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Python Environment Setup Complete ==="
