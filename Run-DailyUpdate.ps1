# Run the daily data update pipeline (auto-detect weeks) with logging and single-instance lock.
# Usage examples:
#   .\Run-DailyUpdate.ps1
#   .\Run-DailyUpdate.ps1 -PrintJson

param(
    [switch]$PrintJson,
    [switch]$LogRecommendations,
    [int]$LogWeek,
    [double]$Bankroll = 1000,
    [double]$KellyFactor = 0.5,
    [double]$EvThreshold = 0.02,
    [switch]$SkipScoreCheck
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

    # Lightweight daily scores finalization pass (prior + current week) unless skipped
    if(-not $SkipScoreCheck){
        try {
            Write-Host "Running daily_scores_check.py" -ForegroundColor Cyan
            & $py (Join-Path $root 'daily_scores_check.py') *>&1 | Tee-Object -FilePath $log -Append
        } catch {
            Write-Host "daily_scores_check.py failed: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }

    if($LASTEXITCODE -ne 0){ Write-Host "weekly_update.py failed (exit $LASTEXITCODE). Skipping rec logging." -ForegroundColor Yellow }
    elseif($LogRecommendations){
        try {
            $recUrl = "http://127.0.0.1:5051/api/recommendations/simple?log=true&bankroll=$Bankroll&kelly_factor=$KellyFactor&ev_threshold=$EvThreshold"
            if($LogWeek){ $recUrl += "&week=$LogWeek" }
            Write-Host "Attempting to log recommendations via $recUrl" -ForegroundColor Cyan
            # Use curl if available; fallback to Invoke-WebRequest
            if(Get-Command curl -ErrorAction SilentlyContinue){
                curl -s $recUrl | Out-Null
            } else {
                Invoke-WebRequest -Uri $recUrl -UseBasicParsing | Out-Null
            }
            Write-Host "Recommendations logging request sent." -ForegroundColor Green
        }
        catch {
            Write-Host "Failed to log recommendations: $($_.Exception.Message)" -ForegroundColor Red
        }
    }
    Write-Host "Log: $log" -ForegroundColor Green
}
finally {
    if($hasLock){ $mutex.ReleaseMutex() | Out-Null }
    $mutex.Dispose()
}
