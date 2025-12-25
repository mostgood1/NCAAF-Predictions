# Register a Windows Scheduled Task to run the daily updater at a specific local time.
# Usage examples:
#   .\Register-DailyUpdateTask.ps1 -At '06:15' -User 'DOMAIN\\user' -Password '***'
#   .\Register-DailyUpdateTask.ps1 -At '06:15'   # runs under current user context (will prompt if needed)

param(
    [Parameter(Mandatory=$true)][string]$At,
    [string]$User,
    [string]$Password
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

$scriptPath = Join-Path $root 'Run-DailyUpdate.ps1'
if(!(Test-Path $scriptPath)){
    throw "Run-DailyUpdate.ps1 not found at $scriptPath"
}

# Validate time format
try {
    [void][DateTime]::ParseExact($At,'HH:mm',$null)
} catch {
    throw "-At must be in 24h HH:mm format"
}

$taskName = 'NCAAF Daily Update'
$psExe = (Get-Command powershell.exe).Source
$quotedScript = '"' + $scriptPath + '"'
$tr = '"' + ($psExe + ' -NoProfile -ExecutionPolicy Bypass -File ' + $quotedScript) + '"'

# Build schtasks.exe args
# Detect elevation; request highest only if running elevated
function Test-IsElevated {
    try {
        $wi = [Security.Principal.WindowsIdentity]::GetCurrent()
        $wp = New-Object Security.Principal.WindowsPrincipal($wi)
        return $wp.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    } catch { return $false }
}

$args = @('/Create','/SC','DAILY','/TN',$taskName,'/TR',$tr,'/ST',$At,'/F')
if(Test-IsElevated){ $args += @('/RL','HIGHEST') } else { Write-Host '[info] Not elevated; registering without /RL HIGHEST' -ForegroundColor Yellow }
if($User){ $args += @('/RU',$User) }
if($Password){ $args += @('/RP',$Password) }

Write-Host ("schtasks " + ($args -join ' ')) -ForegroundColor Cyan
& schtasks @args | Write-Host
Write-Host "Scheduled task '$taskName' registered to run at $At daily." -ForegroundColor Green
