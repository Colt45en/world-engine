# Nucleus System Test Runner (PowerShell)

Write-Host "Nucleus System Test Suite Runner" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green

# Check if Node.js is available
try {
    $nodeVersion = node --version
    Write-Host "Node.js found: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "Node.js is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Get current directory
$currentDir = Get-Location
Write-Host "Current directory: $currentDir" -ForegroundColor Yellow
Write-Host ""

# Run integration test
Write-Host "Running integration tests..." -ForegroundColor Cyan
try {
    node integration_test.js
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Integration tests completed successfully" -ForegroundColor Green
    } else {
        Write-Host "Integration tests failed" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Failed to run integration tests: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "All nucleus tests completed successfully!" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green
