"""Harvest an opponent library from simulated games (see bgagent/library.py).

    python scripts/build_library.py --out data/library/train.jsonl.gz --games 1000 --seed 1 \
        --policy runs/full-01/policy_00500.pt:0,1,2,3 --policy runs/full-01/policy_00250.pt:4,5
    python scripts/build_library.py --out data/library/exam.jsonl.gz --games 300 --seed 2 \
        --policy runs/full-01/policy_00475.pt:0,1,2,3 --policy runs/full-01/policy_00375.pt:4,5

Seats without a --policy are TavernGym's scripted bots. Every alive seat's board is recorded
at each turn. The exam library must come from other seeds and policies than the training
one: it is never trained on.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import taverngym
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bgagent.filters import ActionFilter  # noqa: E402
from bgagent.library import harvest  # noqa: E402
from bgagent.policy import load_policy  # noqa: E402
from bgagent.ppo import act  # noqa: E402
from bgagent.rules import rules_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--parallel", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--policy", action="append", default=[],
                    help="path:seat,seat,...  (repeatable; other seats are bots)")
    ap.add_argument("--per-turn", type=int, default=2000)
    ap.add_argument("--keep-top", type=int, default=400)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    seat_model = {}
    for spec in args.policy:
        path, seats = spec.rsplit(":", 1)
        model = load_policy(path)[0].to(device).eval()
        for s in seats.split(","):
            seat_model[int(s)] = model
    controlled = tuple(sorted(seat_model)) or (0,)
    env = taverngym.BatchEnv(args.parallel, seed=args.seed, controlled=controlled,
                             config=rules_config("full"), threads=4)
    filters = {id(m): ActionFilter.for_model(env.n_slots, m) for m in set(seat_model.values())}
    rng = np.random.default_rng(args.seed)

    def act_fn(model, feats, cards, mask, idx):
        if model is None:
            return np.array([rng.choice(np.nonzero(r)[0]) for r in mask])
        filt = filters[id(model)]
        m = filt.apply(feats, mask, idx)
        a, _, _ = act(model, feats, cards, m, device, device.type == "cuda")
        filt.record(idx, a)
        return a

    t0 = time.time()
    lib = harvest(env, lambda i: seat_model.get(env.slot(i)[1]), args.games, act_fn,
                  per_turn_cap=args.per_turn, keep_top=args.keep_top, seed=args.seed)
    lib.save(args.out)
    med = lib.median_curve()
    print(f"{len(lib)} entries over turns {lib.turns()[0]}-{lib.turns()[-1]} from {args.games} "
          f"games in {time.time() - t0:.0f}s -> {args.out}")
    print("median strength by turn:", {t: round(v) for t, v in med.items() if t <= 16})
    best = {t: int(max(s for s, _ in lib.by_turn[t])) for t in lib.turns() if t <= 16}
    print("strongest by turn:", best)


if __name__ == "__main__":
    main()
