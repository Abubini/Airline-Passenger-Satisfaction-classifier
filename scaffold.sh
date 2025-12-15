#!/bin/bash

# Create directory structure for machine learning project

echo "Creating project directory structure..."

# Create main directories
mkdir -p data/processed
mkdir -p data/raw
mkdir -p src
mkdir -p models
mkdir -p visuals

# Create Python files in src directory
touch src/preprocess.py
touch src/train.py
touch src/evaluate.py
touch src/classify.py

# Create root-level files
touch main.py
touch README.md
touch requirements.txt

# Create empty __init__.py files to make directories importable as packages
touch src/__init__.py
touch data/__init__.py
touch models/__init__.py
touch visuals/__init__.py

echo "Directory structure created successfully!"
echo ""
echo "Project structure:"
echo "- data/"
echo "  - processed/"
echo "  - raw/"
echo "- src/"
echo "  - preprocess.py"
echo "  - train.py"
echo "  - evaluate.py"
echo "  - classify.py"
echo "  - __init__.py"
echo "- models/"
echo "  - __init__.py"
echo "- visuals/"
echo "  - __init__.py"
echo "- main.py"
echo "- README.md"
echo "- requirements.txt"
