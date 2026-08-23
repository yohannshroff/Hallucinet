#!/usr/bin/env bash
# Create a venv and install Part 1-2 dependencies. Neo4j/Ollama are NOT
# touched by this script -- see docs/manual_setup_neo4j.md and
# docs/manual_setup_ollama.md for those.
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Environment ready. Activate later with: source .venv/bin/activate"
