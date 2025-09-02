# Run weekly update automation.
# Examples:
#   .\Run-WeeklyUpdate.ps1 -PriorWeek 1 -UpcomingWeek 2 -PrintJson
#   .\Run-WeeklyUpdate.ps1  # auto-detect weeks

param(
    [int]$PriorWeek,
    [int]$UpcomingWeek,
    [switch]$PrintJson
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
