<#
Dev helper: create a Python 3.11 venv and run the working server.

Usage: Open PowerShell in the repo root and run
    .\dev_run_server.ps1

#>

Write-Host "Checking for Python 3.11 via py launcher..."
$py311 = (& py -3.11 -c "import sys; print(sys.executable)") 2>$null
if (-not $py311) {
    Write-Host "Python 3.11 not found. Attempting to install via winget..."
    winget install --id Python.Python.3.11 -e --silent
}

Write-Host "Creating venv at .venv311 (if missing)..."
py -3.11 -m venv .venv311

Write-Host "Activating venv and installing requirements..."
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

Write-Host "Starting working_python_server.py..."
python -u 'Documents/game101/Downloads/recovered_nucleus_eye/world-engine-feat-v3-1-advanced-math/working_python_server.py'
