# Run the daily data update pipeline (auto-detect weeks) with logging and single-instance lock.
# Usage examples:
#   .\Run-DailyUpdate.ps1
#   .\Run-DailyUpdate.ps1 -PrintJson

param(
    [switch]$PrintJson
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

# Ensure logs directory exists
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Use a named Mutex to avoid overlapping runs
$mutexName = 'Global/NCAAFDailyUpdate'
$mutex = New-Object System.Threading.Mutex($false, $mutexName)
$hasLock = $false
try {
    $hasLock = $mutex.WaitOne(0)
} catch {}
if(-not $hasLock){
    Write-Host "Another daily update is already running. Exiting." -ForegroundColor Yellow
    exit 0
}

try {
    $py = Join-Path $root '.venv\Scripts\python.exe'
    if(!(Test-Path $py)) { $py = 'python' }
    $script = Join-Path $root 'weekly_update.py'
    if(!(Test-Path $script)) { throw "weekly_update.py not found at $script" }

    $argsList = @($script)
    if($PrintJson){ $argsList += @('--print-json') }

    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $log = Join-Path $logDir "daily_update.$stamp.log"

    Write-Host "Running: $py $($argsList -join ' ')" -ForegroundColor Cyan
    & $py @argsList *>&1 | Tee-Object -FilePath $log
    Write-Host "Log: $log" -ForegroundColor Green
}
finally {
    if($hasLock){ $mutex.ReleaseMutex() | Out-Null }
    $mutex.Dispose()
}
