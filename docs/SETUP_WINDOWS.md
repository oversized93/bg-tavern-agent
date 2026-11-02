# Setting up natively on Windows (no WSL, no admin)

This is how the RTX 3070 machine runs: plain Windows 10/11, no WSL, no Visual Studio C++
tools, nothing that needs an administrator. (The WSL route in SETUP_WSL2.md also works; it
needs admin rights and a reboot, and turns on Windows' hypervisor, which AMD Ryzen Master and
some anti-cheats dislike.)

Needs: Python 3.10+ from python.org, git, an NVIDIA driver recent enough for CUDA 13.

```powershell
git clone https://github.com/oversized93/bg-tavern-agent.git
cd bg-tavern-agent
pwsh -ExecutionPolicy Bypass -File scripts\setup_windows.ps1   # 10-20 min the first time
scripts\start_3070.cmd                                          # or double-click it
```

`setup_windows.ps1` (safe to rerun; finished steps are skipped):
1. Rust in your user folder (`%USERPROFILE%\.cargo`, `.rustup`) with the **GNU** toolchain,
   checksum-verified, PATH untouched. The GNU toolchain links with its own bundled linker, so
   no MSVC or Windows SDK is needed.
2. A `dlltool` for pyo3's Python import library. The toolchain's bundled GNU dlltool needs an
   assembler it doesn't ship, so the script copies Rust's `llvm-ar.exe` (rustup component
   `llvm-tools`) to `%USERPROFILE%\.cargo\llvm-dlltool\dlltool.exe`, where it acts as
   llvm-dlltool.
3. AutoTavern at the pinned commit in `%USERPROFILE%\AutoTavern` (TavernGym source and
   TavernBench).
4. `.venv` with CUDA PyTorch from the pytorch.org index (`torch==2.14.1`, cu130; PyPI's
   Windows torch is CPU-only).
5. TavernGym built into a wheel with maturin and installed.
6. Checks: PyTorch sees the GPU, `pytest` (16 tests, including the TavernBench oracle check).
7. Memory check: one full-size iteration at `micro_batch=512`, written to `runs\.micro_batch`.

## Running
- `scripts\start_3070.cmd` starts or resumes `runs\full-01` (stops at 500 iterations, about 5 hours;
  `start_3070.cmd -Iterations 1000` extends it, and the learning-rate schedule spans 4,000
  either way, so extending doesn't make it jump) in its own minimized window titled **bg-tavern-agent training**, logging to
  `runs\full-01.log`. It refuses to start a second copy of a run that's already training.
- Close that window to stop. Run `start_3070.cmd` again to resume from the last checkpoint
  (written every 5 iterations).
- Another run: `-Run runs\<name>`, and a new run's config as trailing `KEY=VALUE` words. The
  library-opponent run (resume it with just `start_3070.cmd -Run runs\lib-01 -Iterations 300`):
  ```
  start_3070.cmd -Run runs\lib-01 -Iterations 300 opponents=library library_path=data/library/train.jsonl.gz exam_path=data/library/exam.jsonl.gz "learner_seats=(0,)" init_from=runs/full-01/policy_00500.pt n_games=256 lr=2e-4
  ```
- Progress: `.venv\Scripts\python.exe scripts\status.py runs\full-01`.
- The window is started through WMI, so it isn't tied to the terminal (or Claude Code
  session) that launched it.

## GPU memory on Windows
When VRAM runs out, the NVIDIA driver on Windows doesn't raise an error: it spills into
system RAM ("shared GPU memory") and everything slows down ~5x. Measured on the 3070 with the
full model: `micro_batch=2048` used 7.4 GB dedicated + 5.8 GB shared and the update took 237 s
instead of 22 s. So training caps PyTorch at 80% of VRAM (`gpu_mem_fraction`), which turns
an oversized batch into a normal out-of-memory error. `micro_batch=512` peaks at 3.4 GB and
is as fast as 1024 (6.4 GB), leaving room to play Hearthstone on the same card.

Check for spill while training (should be near 0):
```powershell
(Get-Counter '\GPU Process Memory(*)\Shared Usage').CounterSamples | Sort-Object CookedValue | Select-Object -Last 3
```

## Before leaving it for days
- Settings > System > Power & sleep: Sleep **Never** (when plugged in).
- Settings > Update & Security > Windows Update: **Pause updates**, so it doesn't reboot
  mid-run. If it does reboot, run `start_3070.cmd` again.
