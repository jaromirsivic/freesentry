param(
    [ValidateSet('start', 'stop')][string]$Action = 'start',
    [string]$LogDir,
    [ValidateRange(1024, 65535)][int]$Port = 6006,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$appRoot = Split-Path $PSScriptRoot -Parent
$stateDir = Join-Path $appRoot '.tensorboard'
$stateFile = Join-Path $stateDir 'server.json'
$python = Join-Path $appRoot '.venv\Scripts\python.exe'

try {
    $server = $null
    $state = $null
    if (Test-Path -LiteralPath $stateFile) {
        $state = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
        $candidate = Get-Process -Id $state.pid -ErrorAction SilentlyContinue
        # A reused PID must never stop an unrelated process or the trainer.
        if ($candidate -and $candidate.StartTime.ToUniversalTime().Ticks.ToString() -eq $state.started) {
            $server = $candidate
        }
    }

    if ($Action -eq 'stop') {
        if ($server) {
            Stop-Process -InputObject $server
            $server.WaitForExit(10000) | Out-Null
            if (-not $server.HasExited) { throw 'TensorBoard did not stop.' }
            Write-Host 'TensorBoard stopped. Training continues.'
        } else {
            Write-Host 'No TensorBoard launched by this script is running.'
        }
        if (Test-Path -LiteralPath $stateFile) { Remove-Item -LiteralPath $stateFile }
        exit 0
    }

    if ($server) {
        Write-Host "TensorBoard is already running: $($state.url) (logs: $($state.logDir))"
        if (-not $NoBrowser) { Start-Process $state.url }
        exit 0
    }

    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw 'uv was not found. Install uv, then run this script again.'
    }
    if (-not (Test-Path -LiteralPath $python)) {
        & uv venv (Join-Path $appRoot '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the virtual environment.' }
    }
    & $python -c "import importlib.util, sys; sys.exit(importlib.util.find_spec('tensorboard') is None)"
    if ($LASTEXITCODE -ne 0) {
        & uv pip install --python $python 'tensorboard>=2.20,<3'
        if ($LASTEXITCODE -ne 0) { throw 'TensorBoard installation failed.' }
    }
    if (-not $LogDir) { $LogDir = Join-Path $appRoot 'runs' }
    $LogDir = [System.IO.Path]::GetFullPath($LogDir)
    New-Item -ItemType Directory -Force -Path $stateDir, $LogDir | Out-Null
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
    try { $listener.Start() } finally { $listener.Stop() }

    $url = "http://127.0.0.1:$Port/#scalars"
    $stderr = Join-Path $stateDir 'stderr.log'
    $stdout = Join-Path $stateDir 'stdout.log'
    $arguments = '-m tensorboard.main --logdir "{0}" --host 127.0.0.1 --port {1} --reload_interval 5 --load_fast false' -f $LogDir.TrimEnd('\'), $Port
    $server = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $appRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    @{
        pid = $server.Id
        started = $server.StartTime.ToUniversalTime().Ticks.ToString()
        url = $url
        logDir = $LogDir
    } | ConvertTo-Json | Set-Content -LiteralPath $stateFile -Encoding UTF8

    $ready = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        if ($server.HasExited) { throw "TensorBoard exited. See $stderr" }
        try {
            $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 1
            if ($response.StatusCode -eq 200) { $ready = $true; break }
        } catch { }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) {
        Stop-Process -InputObject $server
        Remove-Item -LiteralPath $stateFile
        throw "TensorBoard did not become ready. See $stderr"
    }
    Write-Host "TensorBoard: $url"
    Write-Host "Logs: $LogDir"
    if (-not $NoBrowser) { Start-Process $url }
} catch {
    Write-Host "TensorBoard error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
