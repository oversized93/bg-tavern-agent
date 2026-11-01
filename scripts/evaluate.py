"""Evaluate a policy against TavernGym's seven scripted bots.

    python scripts/evaluate.py runs/full-01/policy_00100.pt --games 2000
    python scripts/evaluate.py random --games 2000          # baseline: random legal actions
    python scripts/evaluate.py runs/full-01 --games 500     # every snapshot in a run folder

Reports average placement (1 best, 8 worst) with a 95% interval, and top-1 / top-4 rates.
By default the policy takes its most likely action, which is what a recommendation would show;
--sample samples from it instead. Evaluation seeds are fixed, so checkpoints are comparable.
For reference, uniformly random play places about 7.95, and a scripted bot 4.5 by symmetry.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bgagent.evaluation import evaluate  # noqa: E402
from bgagent.policy import load_policy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", help="policy_XXXXX.pt, latest.pt, a run folder, or 'random'")
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--parallel", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.policy == "random":
        targets = [("random", None)]
    else:
        p = Path(args.policy)
        files = sorted(p.glob("policy_*.pt")) if p.is_dir() else [p]
        targets = [(str(f), f) for f in files]
    for name, f in targets:
        model = load_policy(f, args.device)[0] if f is not None else None
        r = evaluate(model, args.games, args.parallel, args.threads, args.sample, args.device)
        print(json.dumps({"policy": name, **r}), flush=True)


if __name__ == "__main__":
    main()
