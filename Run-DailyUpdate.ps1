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
    [string]$OddsApiKeyFile,
    [string]$CfbdApiKey,
    [switch]$DisableGitPush,
    [string]$GitCommitMessage
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

# --- Helper: parse .env style key=value files into env vars (only keys we care about) ---
function Set-EnvFromFileIfPresent {
    param(
        [string]$FilePath,
        [string[]]$Keys
    )
    if(-not $FilePath -or -not (Test-Path $FilePath)){ return }
    try {
        Get-Content $FilePath | ForEach-Object {
            $line = $_.Trim()
            if($line -and -not $line.StartsWith('#') -and $line -match '='){
                $k,$v = $line.Split('=',2)
                if($Keys -contains $k){
                    $val = $v.Trim().Trim('"').Trim("'")
                    if(-not [string]::IsNullOrWhiteSpace($val)){
                        Set-Item -Path "Env:$k" -Value $val -ErrorAction SilentlyContinue | Out-Null
                    }
                }
            }
        }
    } catch {}
}

# --- Optional: dot-source a secrets script if provided ---
$secretsPs1 = Join-Path $root 'secrets\secrets.ps1'
if(Test-Path $secretsPs1){
    try { . $secretsPs1 } catch {}
}

# --- Ensure OpenWeather API key is available for downstream Python processes ---
if($OpenWeatherApiKey){
    $env:OPENWEATHER_API_KEY = $OpenWeatherApiKey
    $env:OWM_API_KEY = $OpenWeatherApiKey
}
if(-not $env:OPENWEATHER_API_KEY){
    # Allow alternate env var names to satisfy the requirement
    if($env:OWM_API_KEY){ $env:OPENWEATHER_API_KEY = $env:OWM_API_KEY }
    elseif($env:OPENWEATHERMAP_API_KEY){ $env:OPENWEATHER_API_KEY = $env:OPENWEATHERMAP_API_KEY }
}
if(-not $env:OPENWEATHER_API_KEY){
    # 1) explicit key file
    if($OpenWeatherKeyFile -and (Test-Path $OpenWeatherKeyFile)){
        try { $env:OPENWEATHER_API_KEY = (Get-Content $OpenWeatherKeyFile -Raw).Trim() } catch {}
        if($env:OPENWEATHER_API_KEY){ $env:OWM_API_KEY = $env:OPENWEATHER_API_KEY }
    }
}
if(-not $env:OPENWEATHER_API_KEY){
    # 2) .env style files
    $candidateEnv = @()
    if($EnvFile){ $candidateEnv += $EnvFile }
    $candidateEnv += (Join-Path $root '.env')
    foreach($ef in $candidateEnv){
        Set-EnvFromFileIfPresent -FilePath $ef -Keys @('OPENWEATHER_API_KEY','OWM_API_KEY','OPENWEATHERMAP_API_KEY')
        if(-not $env:OPENWEATHER_API_KEY){
            if($env:OWM_API_KEY){ $env:OPENWEATHER_API_KEY = $env:OWM_API_KEY }
            elseif($env:OPENWEATHERMAP_API_KEY){ $env:OPENWEATHER_API_KEY = $env:OPENWEATHERMAP_API_KEY }
        }
        if($env:OPENWEATHER_API_KEY){ break }
    }
}
if(-not $env:OPENWEATHER_API_KEY){
    # 3) secrets fallback file
    $owFiles = @(
        (Join-Path $root 'secrets\openweather_api_key.txt'),
        (Join-Path $root 'secrets\owm_api_key.txt'),
        (Join-Path $root 'secrets\openweathermap_api_key.txt')
    )
    foreach($f in $owFiles){
        if(Test-Path $f){
            try { $env:OPENWEATHER_API_KEY = (Get-Content $f -Raw).Trim() } catch {}
            if($env:OPENWEATHER_API_KEY){
                $env:OWM_API_KEY = $env:OPENWEATHER_API_KEY
                Write-Host "[info] Loaded OPENWEATHER_API_KEY from secrets file ($([System.IO.Path]::GetFileName($f)))" -ForegroundColor Cyan
                break
            }
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
}
if(-not $env:ODDS_API_KEY){
    if($OddsApiKeyFile -and (Test-Path $OddsApiKeyFile)){
        try { $env:ODDS_API_KEY = (Get-Content $OddsApiKeyFile -Raw).Trim() } catch {}
    }
}
if(-not $env:ODDS_API_KEY){
    # .env style files (support both provided EnvFile and default .env like OpenWeather)
    $candidateEnv2 = @()
    if($EnvFile){ $candidateEnv2 += $EnvFile }
    $candidateEnv2 += (Join-Path $root '.env')
    foreach($ef in $candidateEnv2){
        Set-EnvFromFileIfPresent -FilePath $ef -Keys @('ODDS_API_KEY')
        if($env:ODDS_API_KEY){ break }
    }
}
if(-not $env:ODDS_API_KEY){
    # Secrets fallback file (not committed)
    $secretFile = Join-Path $root 'secrets\odds_api_key.txt'
    if(Test-Path $secretFile){
        try { $env:ODDS_API_KEY = (Get-Content $secretFile -Raw).Trim() } catch {}
        if($env:ODDS_API_KEY){ Write-Host "[info] Loaded ODDS_API_KEY from secrets file" -ForegroundColor Cyan }
    }
}
if(-not $env:ODDS_API_KEY){
    Write-Host "[warn] ODDS_API_KEY not set; real bookmaker odds fetch will be skipped." -ForegroundColor Yellow
} else {
    Write-Host "[info] ODDS_API_KEY present (len=$($env:ODDS_API_KEY.Length))" -ForegroundColor Cyan
}

# --- Ensure CFBD API key present for CFBD-backed scripts ---
if($CfbdApiKey){ $env:CFBD_API_KEY = $CfbdApiKey }
if(-not $env:CFBD_API_KEY){
    # Try .env files
    $candidateEnvCFBD = @()
    if($EnvFile){ $candidateEnvCFBD += $EnvFile }
    $candidateEnvCFBD += (Join-Path $root '.env')
    foreach($ef in $candidateEnvCFBD){
        Set-EnvFromFileIfPresent -FilePath $ef -Keys @('CFBD_API_KEY','CFBD_TOKEN','CFBD')
        if($env:CFBD_API_KEY -or $env:CFBD_TOKEN -or $env:CFBD){ break }
    }
    # Secrets fallback
    if(-not ($env:CFBD_API_KEY -or $env:CFBD_TOKEN -or $env:CFBD)){
        foreach($f in @('secrets\cfbd_api_key.txt','secrets\cfbd_token.txt','secrets\cfbd.txt')){
            $p = Join-Path $root $f
            if(Test-Path $p){
                try {
                    $val = (Get-Content $p -Raw).Trim()
                    if($val){ $env:CFBD_API_KEY = $val }
                } catch {}
                break
            }
        }
    }
}
if($env:CFBD_API_KEY){ Write-Host "[info] CFBD_API_KEY present" -ForegroundColor Cyan }
else { Write-Host "[warn] CFBD_API_KEY not set; postseason appends may be skipped." -ForegroundColor Yellow }

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

    # Ensure postseason games (conf championships + bowls) are present in enhanced schedule BEFORE weekly_update
    try {
        Write-Host "Appending postseason games to enhanced schedule (wk15 + bowls)" -ForegroundColor Cyan
        $postPath = (Join-Path $root 'scripts' 'add_postseason_games_2025.py')
        $prevEA = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        & $py $postPath --from-week 15 --to-week 15 --postseason-weeks 1 2 3 4 5 6 *>&1 | Tee-Object -FilePath $log -Append
        $ErrorActionPreference = $prevEA
    } catch {
        Write-Host "[warn] postseason append step encountered an issue: $($_.Exception.Message)" -ForegroundColor Yellow
    }

    Write-Host "Running: $py $($argsList -join ' ')" -ForegroundColor Cyan
    $wkCmd = if($PrintJson){ '"' + $py + '" ' + '"' + $script + '" --print-json 2>&1' } else { '"' + $py + '" ' + '"' + $script + '" 2>&1' }
    $wkOut = & cmd /c $wkCmd
    if($wkOut){ $wkOut | Tee-Object -FilePath $log }

    # Lightweight daily scores finalization pass (prior + current week) unless skipped
    if(-not $SkipScoreCheck){
        try {
            Write-Host "Running daily_scores_check.py (scan today & yesterday + last 7 days)" -ForegroundColor Cyan
            & $py (Join-Path $root 'daily_scores_check.py') --scan-today-yesterday --scan-last-days 7 *>&1 | Tee-Object -FilePath $log -Append
        } catch {
            Write-Host "daily_scores_check.py failed: $($_.Exception.Message)" -ForegroundColor Yellow
        }
        # Always re-freeze pregame predictions for finals using nearest pre-kickoff snapshot
        try {
            Write-Host "Restoring pregame predictions for finalized games" -ForegroundColor Cyan
            & $py (Join-Path $root 'scripts' 'restore_pregame_predictions.py') --write *>&1 | Tee-Object -FilePath $log -Append
        } catch {
            Write-Host "restore_pregame_predictions.py failed: $($_.Exception.Message)" -ForegroundColor Yellow
        }
        # Generate postseason recommendations snapshots (weeks 1-6)
        try {
            Write-Host "Generating postseason recommendations snapshots (weeks 1-6)" -ForegroundColor Cyan
            & $py (Join-Path $root 'scripts' 'postseason_recs_all.py') *>&1 | Tee-Object -FilePath $log -Append
        } catch {
            Write-Host "postseason_recs_all.py failed: $($_.Exception.Message)" -ForegroundColor Yellow
        }
        # Best-effort: ask local web app (if running) to reload predictions so cards settle now
        try {
            $svcPort = if($env:PORT){ $env:PORT } else { 5051 }
            $urls = @(
                "http://127.0.0.1:$svcPort/api/refresh-data?quick=1",
                "http://127.0.0.1:$svcPort/api/admin/reload"
            )
            foreach($u in $urls){
                try {
                    Write-Host "Attempting app reload via $u" -ForegroundColor DarkCyan
                    Invoke-RestMethod -Method GET -Uri $u -TimeoutSec 10 -ErrorAction Stop | Out-Null
                    Write-Host "App reload request sent." -ForegroundColor DarkGreen
                    break
                } catch {}
            }
        } catch {}
    }

    if($LASTEXITCODE -ne 0){ Write-Host "weekly_update.py failed (exit $LASTEXITCODE). Skipping rec logging." -ForegroundColor Yellow }
    elseif($LogRecommendations){
        try {
            # Use configured PORT if present; default to 5051
            $svcPort = if($env:PORT){ $env:PORT } else { 5051 }
            $recUrl = "http://127.0.0.1:$svcPort/api/recommendations/simple?log=true&bankroll=$Bankroll&kelly_factor=$KellyFactor&ev_threshold=$EvThreshold"
            if($LogWeek){ $recUrl += "&week=$LogWeek" }
            Write-Host "Attempting to log recommendations via $recUrl" -ForegroundColor Cyan
            # Use PowerShell-native web cmdlets; add small retry with longer timeout for heavier computations
            $ok = $false
            $attempts = 3
            for($i=1; $i -le $attempts; $i++){
                try {
                    Invoke-RestMethod -Method GET -Uri $recUrl -TimeoutSec 20 -ErrorAction Stop | Out-Null
                    $ok = $true; break
                } catch {
                    try {
                        Invoke-WebRequest -Uri $recUrl -UseBasicParsing -TimeoutSec 20 -ErrorAction Stop | Out-Null
                        $ok = $true; break
                    } catch {
                        if($i -lt $attempts){ Start-Sleep -Seconds 2 }
                    }
                }
            }
            if(-not $ok){ throw "recommendations logging request timed out after $attempts attempts" }
            Write-Host "Recommendations logging request sent." -ForegroundColor Green
        }
        catch {
            Write-Host "Failed to log recommendations: $($_.Exception.Message)" -ForegroundColor Yellow
            # Fallback: call Python helper directly to append recommendations without HTTP
            try {
                $py = Join-Path $root '.venv\Scripts\python.exe'
                if(!(Test-Path $py)) { $py = 'python' }
                $weekLiteral = if($LogWeek){ [string]$LogWeek } else { 'None' }
                $code = "import app, json; print(json.dumps(app.log_recommendations_cli(week=$weekLiteral, bankroll=$Bankroll, kelly_factor=$KellyFactor, ev_threshold=$EvThreshold)))"
                Write-Host "Attempting CLI fallback to log recommendations..." -ForegroundColor Cyan
                & $py @('-c', $code) | Tee-Object -FilePath $log -Append
                Write-Host "CLI fallback executed." -ForegroundColor Green
            } catch {
                Write-Host "CLI fallback failed: $($_.Exception.Message)" -ForegroundColor Red
            }
        }
    }
    Write-Host "Log: $log" -ForegroundColor Green

    # --- Optional: auto git commit & push of newly generated data/code changes ---
    $doGit = $true
    if($DisableGitPush){ $doGit = $false }
    if($env:DAILY_UPDATE_DISABLE_GIT -and $env:DAILY_UPDATE_DISABLE_GIT -eq '1'){ $doGit = $false }
    if($doGit){
        try {
            $gitCmd = Get-Command git -ErrorAction SilentlyContinue
            if(-not $gitCmd){ Write-Host "[git] git not found in PATH; skipping auto push." -ForegroundColor Yellow }
            else {
                git rev-parse --is-inside-work-tree 2>$null | Out-Null
                if($LASTEXITCODE -ne 0){ Write-Host "[git] Not inside a git work tree; skipping." -ForegroundColor Yellow }
                else {
                    # Stage only tracked modifications and key data artifacts; avoid committing logs & secrets.
                    # First stage tracked modified/deleted files.
                    git add -u 2>$null
                    # Then explicitly add updated prediction / odds data artifacts including offline caches
                    if(Test-Path 'data/recommendations_cache'){ git add 'data/recommendations_cache' 2>$null }
                    if(Test-Path 'data/odds_coverage'){ git add 'data/odds_coverage' 2>$null }
                    if(Test-Path 'data/recommendations_summary'){ git add 'data/recommendations_summary' 2>$null }
                    # Only add patterns that actually match files to avoid fatal pathspec errors
                    foreach($pat in @('data/*.csv','data/*.json','recommendations_*.json')){
                        if(Test-Path $pat){ git add $pat 2>$null }
                    }
                    # Exclude logs (in case someone previously tracked) by resetting them.
                    if(Test-Path .git){
                        foreach($l in (git ls-files logs 2>$null)) { git restore --staged $l 2>$null }
                    }
                    # Check if there is anything to commit
                    git diff --cached --quiet 2>$null
                    if($LASTEXITCODE -eq 0){
                        Write-Host "[git] No staged changes to commit." -ForegroundColor DarkGray
                    } else {
                        if(-not $GitCommitMessage -or [string]::IsNullOrWhiteSpace($GitCommitMessage)){
                            $GitCommitMessage = "daily update auto-commit: $stamp"
                        }
                        git commit -m "$GitCommitMessage" | Out-Null
                        if($LASTEXITCODE -ne 0){ throw "Commit failed (exit $LASTEXITCODE)" }
                        # Capture push output explicitly to avoid NativeCommandError throwing under ErrorActionPreference=Stop
                        # Use cmd /c to avoid PowerShell NativeCommandError behavior and capture output reliably
                        $pushOut = & cmd /c "git push 2>&1"
                        $pushCode = $LASTEXITCODE
                        if($pushOut){ $pushOut | ForEach-Object { Write-Host "[git] $_" } }
                        if($pushCode -ne 0){ throw "Push failed (exit $pushCode)" }
                        Write-Host "[git] Auto push complete." -ForegroundColor Green
                    }
                }
            }
        } catch {
            Write-Host "[git] Auto push encountered an error: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "[git] Auto push disabled (switch or env var)." -ForegroundColor DarkGray
    }
}
finally {
    if($hasLock){ $mutex.ReleaseMutex() | Out-Null }
    $mutex.Dispose()
}
