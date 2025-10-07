Write-Host "Activating .venv311 and running smoke test..."
.\.venv311\Scripts\Activate.ps1
python -u tests\smoke_test_ws.py
