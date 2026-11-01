"""Behaviour statistics for a policy in games against the scripted bots.

    python scripts/diagnose.py runs/x/policy_00010.pt runs/x/policy_00050.pt --games 64

Placement against the bots moves late, because a weak policy finishes 8th whatever it does.
These numbers move early, so they show whether training is teaching anything: at each End
Turn it records the warband size, Tavern Tier and gold left unspent, by turn.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import taverngym
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bgagent.policy import load_policy  # noqa: E402


@torch.inference_mode()
def diagnose(policy, games, seed=20_000_003, sample=True):
    env = taverngym.BatchEnv(min(games, 64), seed=seed, controlled=(0,))
    rng = np.random.default_rng(seed)
    by_turn = defaultdict(lambda: defaultdict(list))
    actions_per_turn, finished = [], 0
    count = np.zeros(env.n_slots, int)
    o = env.observe()
    while finished < games:
        r = o["ready"]
        acts = np.zeros(env.n_slots, np.int64)
        idx = np.nonzero(r)[0]
        if len(idx):
            m = o["action_mask"][idx]
            if policy is None:
                a = np.array([rng.choice(np.nonzero(row)[0]) for row in m])
            else:
                lg, _, _ = policy(torch.from_numpy(o["features"][idx]),
                                  torch.from_numpy(o["cards"][idx]), torch.from_numpy(m))
                a = (torch.multinomial(torch.softmax(lg, -1), 1)[:, 0] if sample
                     else lg.argmax(-1)).numpy()
            acts[idx] = a
            for j, i in enumerate(idx):
                f = o["features"][i]
                count[i] += 1
                if a[j] == 0 and f[1] == 0:  # End Turn, not hero select
                    t = int(round(f[0] * 20))
                    by_turn[t]["board"].append(f[33] * 7)
                    by_turn[t]["tier"].append(f[10] * 6)
                    by_turn[t]["gold_left"].append(f[7] * 10)
                    actions_per_turn.append(count[i])
                    count[i] = 0
        _, dones, _ = env.step(acts)
        finished += int(dones.sum())
        o = env.observe()
    out = {"actions_per_turn": round(float(np.mean(actions_per_turn)), 1)}
    for t in (2, 4, 6, 8):
        if by_turn[t]["board"]:
            out[f"turn{t}"] = {k: round(float(np.mean(v)), 2) for k, v in by_turn[t].items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policies", nargs="+", help="policy files, or 'random'")
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()
    for p in args.policies:
        model = None if p == "random" else load_policy(p)[0]
        print(json.dumps({"policy": p, **diagnose(model, args.games, sample=not args.greedy)}),
              flush=True)


if __name__ == "__main__":
    main()
