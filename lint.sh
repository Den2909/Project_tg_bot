#!/bin/bash
echo "Running Black..."
black --check .
echo "Running Flake8..."
flake8 .
echo "Running Pylint..."
pylint *.py
echo "Running Mypy..."
mypy .
echo "Linting completed!"
