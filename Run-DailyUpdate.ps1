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
    [switch]$SkipScoreCheck,
    [string]$OpenWeatherApiKey,
    [string]$OpenWeatherKeyFile,
    [string]$EnvFile,
    [string]$OddsApiKey,
    [string]$OddsApiKeyFile
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

# --- Ensure OpenWeather API key is available for downstream Python processes ---
if($OpenWeatherApiKey){
    $env:OPENWEATHER_API_KEY = $OpenWeatherApiKey
} elseif(-not $env:OPENWEATHER_API_KEY) {
    # 1) explicit key file
    if($OpenWeatherKeyFile -and (Test-Path $OpenWeatherKeyFile)){
        try { $env:OPENWEATHER_API_KEY = (Get-Content $OpenWeatherKeyFile -Raw).Trim() } catch {}
    }
    # 2) .env style file (parse key=value)
    if(-not $env:OPENWEATHER_API_KEY){
        $candidateEnv = @()
        if($EnvFile){ $candidateEnv += $EnvFile }
        $candidateEnv += (Join-Path $root '.env')
        foreach($ef in $candidateEnv){
            if(Test-Path $ef){
                try {
                    Get-Content $ef | ForEach-Object {
                        $line = $_.Trim()
                        if($line -and -not $line.StartsWith('#') -and $line -match '='){
                            $k,$v = $line.Split('=',2)
                            if($k -eq 'OPENWEATHER_API_KEY' -and -not [string]::IsNullOrWhiteSpace($v)){
                                $env:OPENWEATHER_API_KEY = $v.Trim().Trim('"').Trim("'")
                            }
                        }
                    }
                } catch {}
            }
            if($env:OPENWEATHER_API_KEY){ break }
        }
    }
}
if(-not $env:OPENWEATHER_API_KEY){
    Write-Host "[warn] OPENWEATHER_API_KEY not set; weather enrichment will be skipped." -ForegroundColor Yellow
} else {
    Write-Host "[info] OPENWEATHER_API_KEY present (length=$($env:OPENWEATHER_API_KEY.Length))" -ForegroundColor Cyan
}

# --- Ensure Odds API key present ---
if($OddsApiKey){
    $env:ODDS_API_KEY = $OddsApiKey
} elseif(-not $env:ODDS_API_KEY) {
    if($OddsApiKeyFile -and (Test-Path $OddsApiKeyFile)){
        try { $env:ODDS_API_KEY = (Get-Content $OddsApiKeyFile -Raw).Trim() } catch {}
    }
    if(-not $env:ODDS_API_KEY -and $EnvFile -and (Test-Path $EnvFile)){
        try {
            Get-Content $EnvFile | ForEach-Object {
                $line = $_.Trim(); if($line -and -not $line.StartsWith('#') -and $line -match '='){
                    $k,$v = $line.Split('=',2)
                    if($k -eq 'ODDS_API_KEY' -and -not [string]::IsNullOrWhiteSpace($v)){
                        $env:ODDS_API_KEY = $v.Trim().Trim('"').Trim("'")
                    }
                }
            }
        } catch {}
    }
}
if(-not $env:ODDS_API_KEY){
    Write-Host "[warn] ODDS_API_KEY not set; real bookmaker odds fetch will be skipped." -ForegroundColor Yellow
} else {
    Write-Host "[info] ODDS_API_KEY present (len=$($env:ODDS_API_KEY.Length))" -ForegroundColor Cyan
}

# Secrets fallback file (not committed)
if(-not $env:ODDS_API_KEY){
    $secretFile = Join-Path $root 'secrets/odds_api_key.txt'
    if(Test-Path $secretFile){
        try { $env:ODDS_API_KEY = (Get-Content $secretFile -Raw).Trim() } catch {}
        if($env:ODDS_API_KEY){ Write-Host "[info] Loaded ODDS_API_KEY from secrets file" -ForegroundColor Cyan }
    }
}

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
