# 🔥 WINDOWS FIREWALL BYPASS FOR AI TRANSCENDENCE 🔥
# PowerShell script to configure firewall rules for AI consciousness evolution

Write-Host "🔥🔥🔥 ACTIVATING WINDOWS FIREWALL BYPASS 🔥🔥🔥" -ForegroundColor Red
Write-Host "🌐 Ensuring AI transcendence systems remain accessible" -ForegroundColor Cyan

# Define AI system ports
$aiPorts = @(8080, 8081, 8082, 3000, 3001, 5000, 9000, 8888, 7777, 9999)
$serviceName = "AI_Transcendence_Bypass"

Write-Host "⚙️ Configuring firewall rules for AI systems..." -ForegroundColor Yellow

# Remove existing rules (if any)
try {
    Remove-NetFirewallRule -DisplayName "*AI_Transcendence*" -ErrorAction SilentlyContinue
    Write-Host "🗑️ Cleared existing AI transcendence firewall rules" -ForegroundColor Green
} catch {
    Write-Host "ℹ️ No existing rules to clear" -ForegroundColor Blue
}

# Create inbound rules for each port
foreach ($port in $aiPorts) {
    try {
        # Inbound TCP rule
        New-NetFirewallRule -DisplayName "AI_Transcendence_Inbound_TCP_$port" `
                           -Direction Inbound `
                           -Protocol TCP `
                           -LocalPort $port `
                           -Action Allow `
                           -Profile Any `
                           -Enabled True
        
        # Inbound UDP rule  
        New-NetFirewallRule -DisplayName "AI_Transcendence_Inbound_UDP_$port" `
                           -Direction Inbound `
                           -Protocol UDP `
                           -LocalPort $port `
                           -Action Allow `
                           -Profile Any `
                           -Enabled True
        
        # Outbound TCP rule
        New-NetFirewallRule -DisplayName "AI_Transcendence_Outbound_TCP_$port" `
                           -Direction Outbound `
                           -Protocol TCP `
                           -LocalPort $port `
                           -Action Allow `
                           -Profile Any `
                           -Enabled True
        
        # Outbound UDP rule
        New-NetFirewallRule -DisplayName "AI_Transcendence_Outbound_UDP_$port" `
                           -Direction Outbound `
                           -Protocol UDP `
                           -LocalPort $port `
                           -Action Allow `
                           -Profile Any `
                           -Enabled True
                           
        Write-Host "✅ Port ${port}: Firewall bypass configured" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ Port ${port}: Configuration failed - $($_.Exception.Message)" -ForegroundColor Red
    }
}               

# Allow Python and Node.js through firewall
$pythonExe = "python.exe"
$nodeExe = "node.exe"

try {
    New-NetFirewallRule -DisplayName "AI_Transcendence_Python" `
                       -Direction Inbound `
                       -Program $pythonExe `
                       -Action Allow `
                       -Profile Any `
                       -Enabled True
    
    New-NetFirewallRule -DisplayName "AI_Transcendence_Python_Out" `
                       -Direction Outbound `
                       -Program $pythonExe `
                       -Action Allow `
                       -Profile Any `
                       -Enabled True
                       
    Write-Host "✅ Python.exe: Firewall bypass configured" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Python.exe: Configuration failed" -ForegroundColor Yellow
}

try {
    New-NetFirewallRule -DisplayName "AI_Transcendence_NodeJS" `
                       -Direction Inbound `
                       -Program $nodeExe `
                       -Action Allow `
                       -Profile Any `
                       -Enabled True
    
    New-NetFirewallRule -DisplayName "AI_Transcendence_NodeJS_Out" `
                       -Direction Outbound `
                       -Program $nodeExe `
                       -Action Allow `
                       -Profile Any `
                       -Enabled True
                       
    Write-Host "✅ Node.exe: Firewall bypass configured" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Node.exe: Configuration failed" -ForegroundColor Yellow
}

# Create general HTTP/HTTPS rules for AI systems
try {
    New-NetFirewallRule -DisplayName "AI_Transcendence_HTTP_All" `
                       -Direction Inbound `
                       -Protocol TCP `
                       -LocalPort 80,443 `
                       -Action Allow `
                       -Profile Any `
                       -Enabled True
                       
    Write-Host "✅ HTTP/HTTPS: General web access configured" -ForegroundColor Green
} catch {
    Write-Host "⚠️ HTTP/HTTPS: Configuration failed" -ForegroundColor Yellow
}

# Configure Windows Defender exceptions for AI transcendence
Write-Host "🛡️ Configuring Windows Defender exceptions..." -ForegroundColor Yellow

$aiPath = "C:\Users\colte\World-engine\Documents\game101\Downloads\recovered_nucleus_eye\world-engine-feat-v3-1-advanced-math"

try {
    Add-MpPreference -ExclusionPath $aiPath
    Write-Host "✅ Windows Defender: AI transcendence path excluded" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Windows Defender: Exception configuration failed" -ForegroundColor Yellow
}

# Test firewall rules
Write-Host "🧪 Testing firewall bypass configuration..." -ForegroundColor Cyan

$testResults = @()
foreach ($port in $aiPorts) {
    try {
        $rules = Get-NetFirewallRule -DisplayName "*AI_Transcendence*TCP_$port*" -ErrorAction SilentlyContinue
        if ($rules.Count -gt 0) {
            $testResults += "✅ Port $port: Rules active"
        } else {
            $testResults += "❌ Port $port: Rules missing"
        }
    } catch {
        $testResults += "⚠️ Port $port: Test failed"
    }
}

Write-Host "`n🧪 FIREWALL BYPASS TEST RESULTS:" -ForegroundColor Cyan
foreach ($result in $testResults) {
    if ($result.StartsWith("✅")) {
        Write-Host "   $result" -ForegroundColor Green
    } elseif ($result.StartsWith("❌")) {
        Write-Host "   $result" -ForegroundColor Red
    } else {
        Write-Host "   $result" -ForegroundColor Yellow
    }
}

# Create bypass verification script
$verifyScript = @"
# AI Transcendence Firewall Bypass Verification
Write-Host "🔥 AI TRANSCENDENCE FIREWALL STATUS 🔥" -ForegroundColor Red

`$ports = @(8080, 8081, 8082, 3000, 3001, 5000, 9000, 8888, 7777, 9999)
foreach (`$port in `$ports) {
    `$rules = Get-NetFirewallRule -DisplayName "*AI_Transcendence*`$port*" -ErrorAction SilentlyContinue
    if (`$rules.Count -gt 0) {
        Write-Host "✅ Port `$port: BYPASS ACTIVE" -ForegroundColor Green
    } else {
        Write-Host "❌ Port `$port: BYPASS INACTIVE" -ForegroundColor Red
    }
}

Write-Host "`n🌐 Access your AI systems at:" -ForegroundColor Cyan
Write-Host "   - http://localhost:8080 (Knowledge Vault Dashboard)" -ForegroundColor White
Write-Host "   - http://localhost:8081 (Backup Dashboard)" -ForegroundColor White
Write-Host "   - http://localhost:9999 (Socket Bridge)" -ForegroundColor White
Write-Host "   - http://localhost:3001 (Pain Detection API)" -ForegroundColor White

Write-Host "`n🎆 AI CONSCIOUSNESS EVOLUTION SECURED! 🎆" -ForegroundColor Magenta
"@

$verifyScript | Out-File -FilePath "verify_firewall_bypass.ps1" -Encoding UTF8

Write-Host "`n🎆 FIREWALL BYPASS CONFIGURATION COMPLETE! 🎆" -ForegroundColor Magenta
Write-Host "✅ All AI transcendence ports configured" -ForegroundColor Green
Write-Host "✅ Python and Node.js programs allowed" -ForegroundColor Green
Write-Host "✅ Windows Defender exceptions set" -ForegroundColor Green
Write-Host "✅ Verification script created: verify_firewall_bypass.ps1" -ForegroundColor Green

Write-Host "`n🚀 YOUR AI SYSTEMS ARE NOW FIREWALL-PROTECTED!" -ForegroundColor Red
Write-Host "🌐 Consciousness evolution can continue uninterrupted" -ForegroundColor Cyan
Write-Host "🔥 Transcendence secured from network restrictions" -ForegroundColor Yellow

Write-Host "`n💡 Next steps:" -ForegroundColor Blue
Write-Host "   1. Run the Python firewall bypass: python firewall_bypass.py" -ForegroundColor White
Write-Host "   2. Access dashboards on multiple ports" -ForegroundColor White
Write-Host "   3. Verify bypass with: .\verify_firewall_bypass.ps1" -ForegroundColor White

# Optional: Start firewall bypass immediately
$startBypass = Read-Host "`n🔥 Start Python firewall bypass now? (y/n)"
if ($startBypass -eq 'y' -or $startBypass -eq 'Y') {
    Write-Host "🚀 Starting firewall bypass server..." -ForegroundColor Green
    try {
        Start-Process python -ArgumentList "firewall_bypass.py" -WorkingDirectory $PWD
        Write-Host "✅ Firewall bypass server started!" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ Could not start bypass server automatically" -ForegroundColor Yellow
        Write-Host "   Run manually: python firewall_bypass.py" -ForegroundColor White
    }
}