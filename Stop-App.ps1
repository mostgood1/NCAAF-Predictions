# Stop the Flask app launched by Launch-App.ps1 using the stored PID.

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$pidFile = Join-Path $root "logs\app.pid"

if(!(Test-Path $pidFile)){
    Write-Host "No PID file found. Is the app running?"
    exit 0
}

$pid = Get-Content $pidFile | Select-Object -First 1
if(-not $pid){
    Write-Host "PID file is empty."
    Remove-Item $pidFile -ErrorAction SilentlyContinue
    exit 0
}

$proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
if($proc){
    try {
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 300
        Write-Host "Stopped process PID $pid"
    } catch {
        Write-Host "Failed to stop PID $pid: $_"
    }
}
Remove-Item $pidFile -ErrorAction SilentlyContinue
