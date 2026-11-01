#!/usr/bin/env bash
# One-time setup inside WSL2 Ubuntu on the RTX 3070 machine: tools, this project, TavernBench,
# and checks that the GPU path works end to end. Safe to rerun: finished steps are skipped.
#
#   git clone https://github.com/oversized93/bg-tavern-agent.git ~/bg-tavern-agent
#   bash ~/bg-tavern-agent/scripts/setup_wsl.sh
#
# Then start the long run with:  bash ~/bg-tavern-agent/scripts/start_3070.sh
#
# What it downloads: Ubuntu packages (apt), Rust from rustup.rs, this repo and AutoTavern
# (pinned commit) from GitHub, Python packages from PyPI (torch, numpy, tensorboard, pytest,
# maturin; ~3 GB, mostly torch's CUDA libraries). Everything lands in ~/bg-tavern-agent,
# ~/AutoTavern, ~/.cargo and ~/.rustup. Nothing touches Windows or Hearthstone.
set -eo pipefail

REPO="$HOME/bg-tavern-agent"
AT="$HOME/AutoTavern"
AT_COMMIT=fb61201a557ddd36a411cf44aac51ce4d9b606ce  # the commit pyproject.toml pins
step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

step "1/7 GPU visible from WSL"
export PATH="$PATH:/usr/lib/wsl/lib"
if ! nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; then
  echo "No GPU in WSL. Update the NVIDIA driver in Windows (never inside Ubuntu), run"
  echo "'wsl --shutdown' in PowerShell, reopen Ubuntu and run this script again."
  exit 1
fi

step "2/7 Ubuntu packages (asks for your Ubuntu password)"
sudo apt-get update -y
sudo apt-get install -y build-essential git curl python3-venv python3-dev tmux

step "3/7 Rust (builds the simulator)"
if [ ! -x "$HOME/.cargo/bin/rustup" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
source "$HOME/.cargo/env"
rustup toolchain install 1.98.1 --profile minimal  # the version TavernGym pins

step "4/7 This project and AutoTavern (TavernBench) at the pinned commit"
[ -d "$REPO/.git" ] || git clone https://github.com/oversized93/bg-tavern-agent.git "$REPO"
git -C "$REPO" pull --ff-only
[ -d "$AT/.git" ] || git clone https://github.com/MCG-NJU/AutoTavern "$AT"
git -C "$AT" fetch -q origin
git -C "$AT" -c advice.detachedHead=false checkout -q "$AT_COMMIT"

step "5/7 Python packages (builds the simulator from source: several minutes)"
cd "$REPO"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[test]"

step "6/7 Checks: GPU from PyTorch, tests, a short training run, TavernBench"
python -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot see the GPU'; \
print('PyTorch', torch.__version__, 'on', torch.cuda.get_device_name(0))"
TAVERNBENCH_DIR="$AT/tavernbench" pytest -q
python scripts/train.py --preset smoke --run runs/smoke --set device=cuda
python scripts/tavernbench.py runs/smoke/latest.pt --bench "$AT/tavernbench" --device cuda \
  --out runs/smoke/tavernbench | tail -3

step "7/7 Memory check: one full-size update on the 3070 (picks micro_batch)"
mkdir -p runs
for mb in 2048 1024 512; do
  rm -rf runs/_probe
  if python scripts/train.py --preset 3070 --run runs/_probe \
       --set iterations=1 n_games=16 decisions_per_iter=8192 eval_every=0 micro_batch=$mb \
       > runs/_probe.log 2>&1; then
    echo "$mb" > runs/.micro_batch
    echo "micro_batch=$mb fits in GPU memory"
    break
  fi
  if ! grep -q "out of memory" runs/_probe.log; then
    echo "The memory check failed for another reason:"; tail -20 runs/_probe.log; exit 1
  fi
  echo "micro_batch=$mb ran out of GPU memory; trying smaller"
done
rm -rf runs/_probe runs/_probe.log
[ -f runs/.micro_batch ] || { echo "Even micro_batch=512 ran out of memory."; exit 1; }

step "Setup finished. Start the long run with:  bash ~/bg-tavern-agent/scripts/start_3070.sh"
