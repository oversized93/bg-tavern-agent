#!/usr/bin/env bash
# Start, or resume, the multi-day training run on the RTX 3070, in a tmux session named
# "train" so it keeps going when the terminal is closed. Run scripts/setup_wsl.sh first.
#
#   bash ~/bg-tavern-agent/scripts/start_3070.sh          # start or resume runs/full-01
#   tmux attach -t train                                   # watch it (leave: Ctrl-b then d)
#   cd ~/bg-tavern-agent && .venv/bin/python scripts/status.py runs/full-01   # progress
#
# After a reboot or crash, run this again: it resumes from the last checkpoint.
set -eo pipefail
cd "$HOME/bg-tavern-agent"
RUN="${1:-runs/full-01}"
MB="$(cat runs/.micro_batch 2>/dev/null || echo 2048)"  # chosen by setup_wsl.sh's memory check
# The recipe on top of --preset 3070 (decided from the CPU tests in NOTES.md).
SET="bench_dir=$HOME/AutoTavern/tavernbench micro_batch=$MB device=cuda"

if tmux has-session -t train 2>/dev/null; then
  echo "Already running. Watch it with: tmux attach -t train"
  exit 0
fi
mkdir -p runs
tmux new-session -d -s train \
  ". .venv/bin/activate && python scripts/train.py --preset 3070 --run $RUN --set $SET \
   2>&1 | tee -a $RUN.log; echo; echo 'Training stopped. Rerun scripts/start_3070.sh to resume.'; \
   exec bash"
sleep 5
echo "Training started in tmux session 'train' (run folder: ~/bg-tavern-agent/$RUN)."
echo "Watch: tmux attach -t train    Progress: .venv/bin/python scripts/status.py $RUN"
tail -5 "$RUN.log" 2>/dev/null || true
