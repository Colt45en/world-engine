# Stop OneDrive Bullshit Script
# This script disables OneDrive's aggressive behavior and prevents it from interfering with your files

Write-Host "🛑 Stopping OneDrive Bullshit..." -ForegroundColor Yellow

# Stop OneDrive processes
Write-Host "  • Stopping OneDrive processes..." -ForegroundColor Cyan
Get-Process -Name "OneDrive" -ErrorAction SilentlyContinue | Stop-Process -Force
Write-Host "    ✅ OneDrive processes stopped" -ForegroundColor Green

# Remove from startup
Write-Host "  • Removing OneDrive from startup..." -ForegroundColor Cyan
try {
    Remove-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" -Name "OneDrive" -Force -ErrorAction SilentlyContinue
    Write-Host "    ✅ OneDrive removed from startup" -ForegroundColor Green
} catch {
    Write-Host "    ⚠️  OneDrive not found in startup (already removed)" -ForegroundColor Yellow
}

# Disable sync
Write-Host "  • Disabling OneDrive sync..." -ForegroundColor Cyan
New-Item -Path "HKCU:\Software\Microsoft\OneDrive" -Force | Out-Null
Set-ItemProperty -Path "HKCU:\Software\Microsoft\OneDrive" -Name "DisablePersonalSync" -Value 1 -Force
Set-ItemProperty -Path "HKCU:\Software\Microsoft\OneDrive" -Name "PreventNetworkTrafficPreUserSignIn" -Value 1 -Force
Write-Host "    ✅ OneDrive sync disabled" -ForegroundColor Green

# Remove from Explorer
Write-Host "  • Removing OneDrive from Windows Explorer..." -ForegroundColor Cyan
New-Item -Path "HKCU:\Software\Classes\CLSID\{018D5C66-4533-4307-9B53-224DE2ED1FE6}" -Force | Out-Null
Set-ItemProperty -Path "HKCU:\Software\Classes\CLSID\{018D5C66-4533-4307-9B53-224DE2ED1FE6}" -Name "System.IsPinnedToNameSpaceTree" -Value 0 -Force
Write-Host "    ✅ OneDrive removed from Explorer" -ForegroundColor Green

# Disable OneDrive notifications
Write-Host "  • Disabling OneDrive notifications..." -ForegroundColor Cyan
$notificationPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Notifications\Settings\Microsoft.SkyDrive.Desktop"
if (Test-Path $notificationPath) {
    Set-ItemProperty -Path $notificationPath -Name "Enabled" -Value 0 -Force
    Write-Host "    ✅ OneDrive notifications disabled" -ForegroundColor Green
} else {
    Write-Host "    ⚠️  OneDrive notification settings not found" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎉 OneDrive has been told to FUCK OFF!" -ForegroundColor Green
Write-Host "   • OneDrive won't start automatically" -ForegroundColor White
Write-Host "   • OneDrive sync is disabled" -ForegroundColor White
Write-Host "   • OneDrive is removed from Explorer" -ForegroundColor White
Write-Host "   • Use your 1TB Google Drive instead! 💪" -ForegroundColor Magenta
Write-Host ""
Write-Host "⚠️  You may need to restart Windows for all changes to take effect" -ForegroundColor Yellow
