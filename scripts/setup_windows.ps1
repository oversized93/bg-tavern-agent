# One-time setup natively on Windows (no WSL, no admin): Rust in the user folder, TavernGym
# built from AutoTavern at the pinned commit, CUDA PyTorch in .venv, then checks.
# Safe to rerun: finished steps are skipped.
#
#   git clone https://github.com/oversized93/bg-tavern-agent.git
#   cd bg-tavern-agent
#   pwsh -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
#   scripts\start_3070.cmd        # then start the run
#
# Needs: Python 3.10+ (python.org build), git, an NVIDIA driver new enough for CUDA 13.
# No Visual Studio or Windows SDK: TavernGym is built with Rust's GNU toolchain, whose own
# linker needs no MSVC. One wrinkle: pyo3 links Python through a "raw-dylib" import, for which
# rustc calls `dlltool`; the toolchain's bundled GNU dlltool needs an assembler it doesn't
# ship, so we use LLVM's dlltool instead (Rust's llvm-tools component: llvm-ar.exe, run under
# the name dlltool.exe, behaves as llvm-dlltool).
#
# Writes to: %USERPROFILE%\.cargo, %USERPROFILE%\.rustup, %USERPROFILE%\AutoTavern and this
# repo's .venv / runs. Changes no system setting and not PATH.
param([string]$TorchIndex = "https://download.pytorch.org/whl/cu130", [string]$Torch = "torch==2.14.1")
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$AT = Join-Path $HOME "AutoTavern"
$ATCommit = "fb61201a557ddd36a411cf44aac51ce4d9b606ce"  # the commit pyproject.toml pins
$Host_ = "x86_64-pc-windows-gnu"
$Toolchain = "1.98.1-$Host_"  # AutoTavern's rust-toolchain.toml pins 1.98.1
function Step($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Check($ok, $msg) { if (-not $ok) { throw $msg } }

Step "1/7 GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
Check ($LASTEXITCODE -eq 0) "nvidia-smi failed: install or update the NVIDIA driver."

Step "2/7 Rust (user folder only)"
$cargoBin = Join-Path $HOME ".cargo\bin"
if (-not (Test-Path "$cargoBin\rustup.exe")) {
    $dl = Join-Path $env:TEMP "rustup-init-$PID"
    New-Item -ItemType Directory -Force $dl | Out-Null
    $url = "https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe"
    Invoke-WebRequest $url -OutFile "$dl\rustup-init.exe" -UseBasicParsing
    $sum = Invoke-WebRequest "$url.sha256" -UseBasicParsing
    $sum = if ($sum.Content -is [byte[]]) { [Text.Encoding]::ASCII.GetString($sum.Content) } else { [string]$sum.Content }
    $want = ($sum.Trim() -split '\s+')[0]
    $got = (Get-FileHash "$dl\rustup-init.exe" -Algorithm SHA256).Hash.ToLower()
    Check ($want -eq $got) "rustup-init.exe checksum mismatch ($got vs $want)"
    & "$dl\rustup-init.exe" -y --default-host $Host_ --default-toolchain none --profile minimal --no-modify-path
    Check ($LASTEXITCODE -eq 0) "rustup-init failed"
}
$env:PATH = "$cargoBin;$env:PATH"
rustup toolchain install $Toolchain --profile minimal
rustup component add llvm-tools --toolchain $Toolchain
$tools = Join-Path $HOME ".rustup\toolchains\$Toolchain\lib\rustlib\$Host_\bin"
$dlltool = Join-Path $HOME ".cargo\llvm-dlltool"
New-Item -ItemType Directory -Force $dlltool | Out-Null
Copy-Item "$tools\llvm-ar.exe" "$dlltool\dlltool.exe" -Force
Copy-Item "$tools\libgcc_s_seh-1.dll", "$tools\libwinpthread-1.dll" $dlltool -Force -ErrorAction SilentlyContinue
$env:PATH = "$dlltool;$env:PATH"

Step "3/7 AutoTavern (TavernGym + TavernBench) at the pinned commit"
if (-not (Test-Path "$AT\.git")) { git clone https://github.com/MCG-NJU/AutoTavern $AT }
git -C $AT fetch -q origin
git -C $AT -c advice.detachedHead=false checkout -q $ATCommit
Check ($LASTEXITCODE -eq 0) "could not check out AutoTavern $ATCommit"

Step "4/7 Python venv and CUDA PyTorch (~3 GB download the first time)"
Set-Location $Repo
if (-not (Test-Path ".venv\Scripts\python.exe")) { python -m venv .venv }
$py = Join-Path $Repo ".venv\Scripts\python.exe"
& $py -m pip install -q --upgrade pip
& $py -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>$null
if ($LASTEXITCODE -ne 0) { & $py -m pip install $Torch --index-url $TorchIndex }
& $py -m pip install -q "maturin>=1.15,<2" numpy tensorboard pytest

Step "5/7 Build TavernGym (a few minutes)"
& $py -c "import taverngym" 2>$null
if ($LASTEXITCODE -ne 0) {
    Push-Location $AT
    & (Join-Path $Repo ".venv\Scripts\maturin.exe") build --release -i $py --out "$AT\dist"
    Check ($LASTEXITCODE -eq 0) "TavernGym build failed"
    Pop-Location
    $wheel = Get-ChildItem "$AT\dist\taverngym-*-win_amd64.whl" | Sort-Object LastWriteTime | Select-Object -Last 1
    & $py -m pip install --force-reinstall --no-deps $wheel.FullName
}
& $py -m pip install -q -e $Repo --no-deps  # deps installed above; taverngym from the wheel

Step "6/7 Checks: GPU from PyTorch, tests, TavernBench"
& $py -c "import torch, taverngym; assert torch.cuda.is_available(); print('PyTorch', torch.__version__, 'on', torch.cuda.get_device_name(0)); print('TavernGym', taverngym.version(), taverngym.game_version())"
Check ($LASTEXITCODE -eq 0) "PyTorch can't see the GPU"
$env:TAVERNBENCH_DIR = "$AT\tavernbench"
& $py -m pytest -q
Check ($LASTEXITCODE -eq 0) "tests failed"

Step "7/7 Memory check: one full-size iteration on the GPU (picks micro_batch)"
New-Item -ItemType Directory -Force "$Repo\runs" | Out-Null
if (-not (Test-Path "$Repo\runs\.micro_batch")) {
    # 512 is the measured fit on an 8 GB 3070 (3.4 GB peak, same speed as 1024 at 6.4 GB).
    foreach ($mb in 512, 256) {
        $probe = Join-Path $env:TEMP "bg-probe-$PID-$mb"
        & $py scripts\train.py --preset 3070 --run $probe --set iterations=1 eval_every=0 device=cuda "micro_batch=$mb" *> "$probe.log"
        if ($LASTEXITCODE -eq 0) { Set-Content "$Repo\runs\.micro_batch" $mb -NoNewline; Write-Host "micro_batch=$mb fits"; break }
        if (-not (Select-String -Path "$probe.log" -Pattern "out of memory" -Quiet)) { Get-Content "$probe.log" -Tail 20; throw "memory check failed" }
        Write-Host "micro_batch=$mb ran out of GPU memory; trying smaller"
    }
    Check (Test-Path "$Repo\runs\.micro_batch") "even micro_batch=256 ran out of memory"
}
Step "Setup finished. Start the run: scripts\start_3070.cmd"
