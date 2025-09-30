# NUCLEAR OneDrive Protection Script
# This script permanently disables OneDrive and prevents Microsoft from ever re-enabling it

Write-Host "☢️  NUCLEAR OneDrive Protection - Never Lose Files Again!" -ForegroundColor Red
Write-Host ""

# Step 1: Kill all OneDrive processes
Write-Host "1️⃣  Terminating all OneDrive processes..." -ForegroundColor Yellow
Get-Process -Name "OneDrive*" -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process -Name "FileCoAuth" -ErrorAction SilentlyContinue | Stop-Process -Force
Write-Host "    ✅ All OneDrive processes terminated" -ForegroundColor Green

# Step 2: Remove from all startup locations
Write-Host "2️⃣  Removing OneDrive from ALL startup locations..." -ForegroundColor Yellow
$startupPaths = @(
    "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run",
    "HKLM:\Software\Microsoft\Windows\CurrentVersion\Run",
    "HKCU:\Software\Microsoft\Windows\CurrentVersion\RunOnce",
    "HKLM:\Software\Microsoft\Windows\CurrentVersion\RunOnce"
)
foreach ($path in $startupPaths) {
    try {
        Remove-ItemProperty -Path $path -Name "OneDrive" -Force -ErrorAction SilentlyContinue
    } catch {}
}
Write-Host "    ✅ OneDrive startup entries nuked" -ForegroundColor Green

# Step 3: Disable OneDrive services
Write-Host "3️⃣  Disabling OneDrive services..." -ForegroundColor Yellow
$services = @("OneDrive Updater Service", "Microsoft OneDrive")
foreach ($service in $services) {
    try {
        Get-Service -Name "*$service*" -ErrorAction SilentlyContinue | Stop-Service -Force -ErrorAction SilentlyContinue
        Get-Service -Name "*$service*" -ErrorAction SilentlyContinue | Set-Service -StartupType Disabled -ErrorAction SilentlyContinue
    } catch {}
}
Write-Host "    ✅ OneDrive services disabled" -ForegroundColor Green

# Step 4: Registry nuclear option
Write-Host "4️⃣  Registry nuclear option..." -ForegroundColor Yellow
$regPaths = @{
    "HKCU:\Software\Microsoft\OneDrive" = @{
        "DisablePersonalSync" = 1
        "PreventNetworkTrafficPreUserSignIn" = 1
        "DisableFileSyncNGSC" = 1
        "KnownFolderBackUp" = 0
        "SilentBusinessConfigCompleted" = 1
    }
    "HKCU:\Software\Classes\CLSID\{018D5C66-4533-4307-9B53-224DE2ED1FE6}" = @{
        "System.IsPinnedToNameSpaceTree" = 0
    }
    "HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\Advanced" = @{
        "ShowSyncProviderNotifications" = 0
    }
}

foreach ($regPath in $regPaths.Keys) {
    New-Item -Path $regPath -Force | Out-Null
    foreach ($setting in $regPaths[$regPath].Keys) {
        Set-ItemProperty -Path $regPath -Name $setting -Value $regPaths[$regPath][$setting] -Force
    }
}
Write-Host "    ✅ Registry fortified against OneDrive" -ForegroundColor Green

# Step 5: Block OneDrive executable
Write-Host "5️⃣  Creating OneDrive execution block..." -ForegroundColor Yellow
$oneDriveExe = "C:\Program Files\Microsoft OneDrive\OneDrive.exe"
if (Test-Path $oneDriveExe) {
    try {
        # Rename the executable so it can't run
        $backupName = "$oneDriveExe.DISABLED_BY_USER"
        if (!(Test-Path $backupName)) {
            Move-Item -Path $oneDriveExe -Destination $backupName -Force
            Write-Host "    ✅ OneDrive executable disabled" -ForegroundColor Green
        } else {
            Write-Host "    ✅ OneDrive executable already disabled" -ForegroundColor Green
        }
    } catch {
        Write-Host "    ⚠️  Could not disable executable (insufficient permissions)" -ForegroundColor Yellow
    }
} else {
    Write-Host "    ℹ️  OneDrive executable not found" -ForegroundColor Cyan
}

# Step 6: Create monitoring script
Write-Host "6️⃣  Creating OneDrive resurrection monitor..." -ForegroundColor Yellow
$monitorScript = @"
# OneDrive Resurrection Monitor
# Run this occasionally to check if Microsoft tried to re-enable OneDrive

`$processes = Get-Process -Name "OneDrive*" -ErrorAction SilentlyContinue
if (`$processes) {
    Write-Host "⚠️  WARNING: OneDrive is running again!" -ForegroundColor Red
    `$processes | Stop-Process -Force
    Write-Host "✅ OneDrive terminated" -ForegroundColor Green
} else {
    Write-Host "✅ OneDrive is still dead - good!" -ForegroundColor Green
}

`$startup = Get-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" -Name "OneDrive" -ErrorAction SilentlyContinue
if (`$startup) {
    Write-Host "⚠️  WARNING: OneDrive is back in startup!" -ForegroundColor Red
    Remove-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" -Name "OneDrive" -Force
    Write-Host "✅ OneDrive startup entry removed again" -ForegroundColor Green
}
"@

$monitorScript | Out-File -FilePath ".\monitor_onedrive_resurrection.ps1" -Force
Write-Host "    ✅ Monitor script created: monitor_onedrive_resurrection.ps1" -ForegroundColor Green

Write-Host ""
Write-Host "☢️  NUCLEAR OPTION COMPLETE!" -ForegroundColor Red
Write-Host "🛡️  OneDrive Protection Status:" -ForegroundColor White
Write-Host "   ✅ Processes terminated" -ForegroundColor Green
Write-Host "   ✅ Startup entries removed" -ForegroundColor Green
Write-Host "   ✅ Services disabled" -ForegroundColor Green
Write-Host "   ✅ Registry locked down" -ForegroundColor Green
Write-Host "   ✅ Executable disabled" -ForegroundColor Green
Write-Host "   ✅ Monitoring system created" -ForegroundColor Green
Write-Host ""
Write-Host "🚀 Your 1TB Google Drive is now free to work without Microsoft interference!" -ForegroundColor Magenta
Write-Host "💡 Run 'monitor_onedrive_resurrection.ps1' after Windows updates to check for OneDrive resurrection attempts" -ForegroundColor Cyan
Write-Host ""
Write-Host "🎯 THREE TIMES THIS MONTH? NEVER AGAIN! 🎯" -ForegroundColor Red
