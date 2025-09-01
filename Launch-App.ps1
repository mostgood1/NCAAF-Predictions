# Launch the Flask app without a console window and open the browser.
# - Uses the repo's .venv\Scripts\pythonw.exe when available (falls back to python.exe hidden)
# - Writes a PID file so you can stop it later via Stop-App.ps1
# - Optional parameters: -Port 5051 -OpenBrowser

param(
    [int]$Port = 5051,
    [switch]$OpenBrowser = $true
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$pythonw = Join-Path $root ".venv\Scripts\pythonw.exe"
if(!(Test-Path $pythonw)){
    $pythonw = Join-Path $root ".venv\Scripts\python.exe"
}
$app = Join-Path $root "app.py"
if(!(Test-Path $app)){
    throw "app.py not found at: $app"
}

$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdout = Join-Path $logDir "app.out.log"
$stderr = Join-Path $logDir "app.err.log"
$pidFile = Join-Path $logDir "app.pid"

# Stop prior instance if PID file exists
if(Test-Path $pidFile){
    try {
        $oldPid = Get-Content $pidFile | Select-Object -First 1
        if($oldPid -and (Get-Process -Id $oldPid -ErrorAction SilentlyContinue)){
            Stop-Process -Id $oldPid -Force -ErrorAction SilentlyContinue
            Start-Sleep -Milliseconds 300
        }
    } catch {}
    Remove-Item $pidFile -ErrorAction SilentlyContinue
}

# Respect an existing PORT, otherwise set it for convenience
if(-not $env:PORT){ $env:PORT = $Port }

# Launch hidden (pythonw shows no console; python.exe is hidden via WindowStyle)
$proc = Start-Process -FilePath $pythonw \
    -ArgumentList "`"$app`"" \
    -WorkingDirectory $root \
    -WindowStyle Hidden \
    -RedirectStandardOutput $stdout \
    -RedirectStandardError $stderr \
    -PassThru

$proc.Id | Out-File -FilePath $pidFile -Encoding ascii -Force

# Optionally wait for the server and open the browser
if($OpenBrowser){
    $url = "http://127.0.0.1:$($env:PORT)/"
    $maxTries = 30
    for($i=0; $i -lt $maxTries; $i++){
        try {
            $wc = New-Object Net.WebClient
            $null = $wc.DownloadString($url)
            Start-Process $url | Out-Null
            break
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
}

Write-Host "Started app (PID $($proc.Id)). Logs: $stdout, $stderr"
