# Start, or resume, the multi-day training run on the RTX 3070 natively on Windows.
# Run scripts\setup_windows.ps1 first (or double-click scripts\start_3070.cmd).
#
#   pwsh -File scripts\start_3070.ps1                 # start or resume runs\full-01
#   pwsh -File scripts\start_3070.ps1 -Run runs\x     # another run folder
#   .venv\Scripts\python.exe scripts\status.py runs\full-01   # progress
#
# Training runs in its own minimized window titled "bg-tavern-agent training" and logs to
# runs\<name>.log. Close that window to stop it; run this again to resume from the last
# checkpoint. The window is started through WMI so it does not belong to whatever terminal
# (or Claude Code session) ran this script, and survives it closing.
param(
    [string]$Run = "runs\full-01",
    # Stop after this many; raise it to extend (start_3070.cmd -Iterations 1000). ~33 s each
    # on a 3070 plus evals: 500 is about 5 hours.
    [int]$Iterations = 500,
    [int]$ScheduleIterations = 4000,  # learning-rate schedule length, independent of stopping
    # Extra config for a new run, e.g. -Set 'opponents=library','learner_seats=(0,)'
    [string[]]$Set = @()
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Repo ".venv\Scripts\python.exe"
$Bench = Join-Path $HOME "AutoTavern\tavernbench"
if (-not (Test-Path $Py)) { throw "No .venv: run scripts\setup_windows.ps1 first." }

$runName = (Split-Path -Leaf $Run)
$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'scripts[\\/]train\.py' -and $_.CommandLine -match [regex]::Escape($runName) }
if ($running) {
    Write-Host "Already training $Run (process $($running.ProcessId -join ', ')). Not starting a second copy."
    exit 0
}

$mbFile = Join-Path $Repo "runs\.micro_batch"
$mb = if (Test-Path $mbFile) { (Get-Content $mbFile -Raw).Trim() } else { "512" }
$items = @("bench_dir=$Bench", "micro_batch=$mb", "device=cuda", "iterations=$Iterations",
           "lr_schedule_iters=$ScheduleIterations") + $Set
# each value single-quoted, so tuples like (0,) reach train.py as text
$set = ($items | ForEach-Object { "'" + ($_ -replace "'", "''") + "'" }) -join ' '
$log = Join-Path $Repo "$Run.log"
New-Item -ItemType Directory -Force (Join-Path $Repo "runs") | Out-Null

# The window's own command: train with output to the window and the log, then say so.
$inner = @"
`$host.UI.RawUI.WindowTitle = 'bg-tavern-agent training ($Run)'
Set-Location '$Repo'
`$env:PYTHONUNBUFFERED = '1'
`$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new(`$false)
& '$Py' scripts\train.py --preset 3070 --run '$Run' --set $set 2>&1 | Tee-Object -FilePath '$log' -Append
Write-Host ''
Write-Host 'Training stopped. Run scripts\start_3070.cmd to resume.'
"@
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($inner))
$pwsh = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
if (-not $pwsh) { $pwsh = (Get-Command powershell).Source }
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]7 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "`"$pwsh`" -NoProfile -NoExit -EncodedCommand $encoded"
    CurrentDirectory = $Repo
    ProcessStartupInformation = $startup
}
if ($r.ReturnValue -ne 0) { throw "Could not start the training window (WMI error $($r.ReturnValue))." }
Write-Host "Training started: minimized window 'bg-tavern-agent training ($Run)', log $log"
Write-Host "Progress: .venv\Scripts\python.exe scripts\status.py $Run"
