# NEXUS World Engine PowerShell Startup Script
Write-Host "🌍 Starting NEXUS World Engine..." -ForegroundColor Green

# Navigate to project directory
Set-Location "c:\Users\colte\World-engine\Documents\game101\Downloads\recovered_nucleus_eye\world-engine-feat-v3-1-advanced-math"

Write-Host "📁 Current directory: $(Get-Location)" -ForegroundColor Yellow

# Install dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Blue
npm install

# Start the server
Write-Host "🚀 Starting NEXUS server..." -ForegroundColor Green
node nexus-server.js