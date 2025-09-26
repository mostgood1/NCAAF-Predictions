# Run weekly update automation.
# Examples:
#   .\Run-WeeklyUpdate.ps1 -PriorWeek 1 -UpcomingWeek 2 -PrintJson
#   .\Run-WeeklyUpdate.ps1  # auto-detect weeks

param(
    [int]$PriorWeek,
    [int]$UpcomingWeek,
    [switch]$PrintJson,
    [switch]$DisableGitPush,
    [string]$GitCommitMessage
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
if(!(Test-Path $py)) { $py = "python" }
$script = Join-Path $root "weekly_update.py"
if(!(Test-Path $script)) { throw "weekly_update.py not found at $script" }

$argsList = @($script)
if($PriorWeek) { $argsList += @("--prior-week", "$PriorWeek") }
if($UpcomingWeek) { $argsList += @("--upcoming-week", "$UpcomingWeek") }
if($PrintJson) { $argsList += @("--print-json") }

$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$log = Join-Path $logDir "weekly_update.$stamp.log"

Write-Host "Running: $py $($argsList -join ' ')"
& $py @argsList *>&1 | Tee-Object -FilePath $log
Write-Host "Log: $log"

# --- Optional: auto git commit & push of newly generated data/code changes ---
$doGit = $true
if($DisableGitPush){ $doGit = $false }
if($env:WEEKLY_UPDATE_DISABLE_GIT -and $env:WEEKLY_UPDATE_DISABLE_GIT -eq '1'){ $doGit = $false }
if($doGit){
    try {
        $gitCmd = Get-Command git -ErrorAction SilentlyContinue
        if(-not $gitCmd){ Write-Host "[git] git not found in PATH; skipping auto push." -ForegroundColor Yellow }
        else {
            git rev-parse --is-inside-work-tree 2>$null | Out-Null
            if($LASTEXITCODE -ne 0){ Write-Host "[git] Not inside a git work tree; skipping." -ForegroundColor Yellow }
            else {
                # Stage only tracked modifications and key data artifacts; avoid committing logs & secrets.
                git add -u 2>$null
                foreach($pat in @('data/*.csv','data/*.json','recommendations_*.json')){
                    if(Test-Path $pat){ git add $pat 2>$null }
                }
                if(Test-Path .git){
                    foreach($l in (git ls-files logs 2>$null)) { git restore --staged $l 2>$null }
                }
                git diff --cached --quiet 2>$null
                if($LASTEXITCODE -eq 0){
                    Write-Host "[git] No staged changes to commit." -ForegroundColor DarkGray
                } else {
                    if(-not $GitCommitMessage -or [string]::IsNullOrWhiteSpace($GitCommitMessage)){
                        $GitCommitMessage = "weekly update auto-commit: $stamp"
                    }
                    git commit -m "$GitCommitMessage" | Out-Null
                    if($LASTEXITCODE -ne 0){ throw "Commit failed (exit $LASTEXITCODE)" }
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
