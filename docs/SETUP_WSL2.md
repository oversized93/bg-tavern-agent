# Setting up on Windows (WSL2) with an RTX 3070

TavernGym's own CI only runs on Ubuntu, and WSL2 is the easiest way to give PyTorch the GPU,
so everything runs inside WSL2.

## 1. WSL2 and the GPU
1. In an administrator PowerShell: `wsl --install -d Ubuntu-24.04`, then reboot.
2. Keep the NVIDIA driver on **Windows** up to date (Game Ready or Studio). WSL2 uses it
   directly; do **not** install a Linux NVIDIA driver inside Ubuntu.
3. In Ubuntu, `nvidia-smi` should list the 3070.

Give WSL2 enough memory. Create `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=24GB      # about 3/4 of your RAM; the default is half
processors=8     # leave a couple of cores for Windows
```

Then `wsl --shutdown` in PowerShell and reopen Ubuntu.

## 2. Toolchain
```bash
sudo apt update && sudo apt install -y build-essential git python3-venv python3-dev tmux
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env
rustup toolchain install 1.98.1   # the version TavernGym pins
```

## 3. The project
Clone into the Linux filesystem (`~/`), not `/mnt/c/...`, which is many times slower.

```bash
cd ~ && git clone https://github.com/oversized93/bg-tavern-agent.git && cd bg-tavern-agent
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[test]"          # builds TavernGym from source: a few minutes
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pytest -q
python scripts/train.py --preset smoke --run runs/smoke
```

TavernBench (the paper's test set) lives in the AutoTavern repository, not in the pip
package. Clone it at the pinned commit to score policies on it and to track it during training:

```bash
git clone https://github.com/MCG-NJU/AutoTavern ~/AutoTavern
git -C ~/AutoTavern checkout fb61201a557ddd36a411cf44aac51ce4d9b606ce
python scripts/tavernbench.py runs/smoke/latest.pt --bench ~/AutoTavern/tavernbench
# during training: add  --set bench_dir=~/AutoTavern/tavernbench  to train.py
```

## 4. Before leaving it running for days
- **Windows Update:** Settings > Windows Update > Pause updates (or set active hours), so it
  doesn't reboot mid-run.
- **Sleep:** Settings > System > Power > Screen and sleep: set sleep to *Never* while plugged in.
- **Run in tmux** so closing the terminal doesn't kill training:
  ```bash
  tmux new -s train
  cd ~/bg-tavern-agent && source .venv/bin/activate
  python scripts/train.py --preset 3070 --run runs/full-01 --set bench_dir=~/AutoTavern/tavernbench
  # detach: Ctrl-b then d      reattach later: tmux attach -t train
  ```
- **If anything stops it** (reboot, crash), rerun the same command: it resumes from
  `latest.pt`, which is written atomically every few iterations.
- **Checking from your phone:** `python scripts/status.py runs/<name>` prints a one-screen
  summary. For curves: `tensorboard --logdir runs`.

## 5. If the GPU runs out of memory
Lower `micro_batch` (gradient accumulation keeps the effective minibatch the same):
`python scripts/train.py --run runs/<name> --set micro_batch=1024`
