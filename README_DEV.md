# Development: Running the Working Python Server

This file explains how to create a Python 3.11 virtual environment and run the `working_python_server.py` for local development.

Prerequisites
- Windows with the `py` launcher installed.

Quick start (one-liner)

Open PowerShell in the repository root and run:

    py -3.11 -m venv .venv311; .\.venv311\Scripts\Activate.ps1; python -m pip install --upgrade pip setuptools wheel; python -m pip install -r requirements.txt; python -u 'Documents/game101/Downloads/recovered_nucleus_eye/world-engine-feat-v3-1-advanced-math/working_python_server.py'

Recommended steps (explained):

1. Install Python 3.11 if you don't have it:

    winget install --id Python.Python.3.11 -e --silent

2. Create a venv and activate it:

    py -3.11 -m venv .venv311
    .\.venv311\Scripts\Activate.ps1

3. Install dependencies:

    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r requirements.txt

4. Run the server:

    python -u 'Documents/game101/Downloads/recovered_nucleus_eye/world-engine-feat-v3-1-advanced-math/working_python_server.py'

Notes
- The repo contains a `.venv311` created during verification; you can recreate it as above.
- If you prefer a different venv path, update the commands accordingly.
