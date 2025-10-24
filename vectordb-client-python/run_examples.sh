#!/bin/bash

# Vector Database Setup and Test Script
# This script sets up the environment and runs the vector database examples

echo "Vector Database CRUD Example Setup"
echo "=================================="

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

echo "Python version:"
python --version

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed or not in PATH"
    exit 1
fi

echo ""
echo "Installing required packages..."
pip install -r requirements.txt

echo ""
echo "Running simple example..."
python simple_example.py

echo ""
echo "Running comprehensive CRUD example..."
python vector_db_crud.py

echo ""
echo "Running advanced features example..."
python advanced_example.py

echo ""
echo "All examples completed successfully!"
echo "Check the generated files:"
echo "- chroma_db/ (persistent database from main example)"
echo "- advanced_chroma_db/ (persistent database from advanced example)"
echo "- vector_db_export.json (exported data)"