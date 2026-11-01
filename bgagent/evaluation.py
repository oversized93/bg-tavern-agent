"""Evaluate a policy (or random play) against TavernGym's seven scripted bots."""

import numpy as np
import taverngym
import torch

from .filters import ActionFilter
from .rules import rules_config

EVAL_SEED = 10_000_019  # fixed and far from training seeds, so checkpoints are comparable


def _choose(model, feats, cards, mask, sample, device, amp, rng):
    if model is None:
        return np.array([rng.choice(np.nonzero(row)[0]) for row in mask])
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        logits, _, _ = model(torch.from_numpy(feats).to(device), torch.from_numpy(cards).to(device),
                             torch.from_numpy(mask).to(device))
    if sample:
        return torch.multinomial(torch.softmax(logits.float(), -1), 1)[:, 0].cpu().numpy()
    return logits.argmax(-1).cpu().numpy()


@torch.inference_mode()
def evaluate(policy, games, n_parallel=64, threads=4, sample=False, device="cpu",
             seed=EVAL_SEED, rules="full", opponent=None):
    """Average placement (1 best, 8 worst) with a 95% interval, top-1 and top-4 rates.

    `policy` sits in seat 0. By default the other seven seats are TavernGym's scripted bots;
    with `opponent` (a model) they are seven copies of it, a head-to-head test where 4.5 means
    "as strong as the opponent". `policy=None` plays uniformly random legal actions (about
    7.95 against the bots). `rules` names a rule set in bgagent/rules.py; "full" is the real
    game with a per-turn action cap. Each model uses the action filter it was trained with."""
    device = torch.device(device)
    amp = device.type == "cuda"
    controlled = (0,) if opponent is None else tuple(range(8))
    k = len(controlled)
    n_games = max(1, min(n_parallel, games))
    env = taverngym.BatchEnv(n_games, seed=seed, controlled=controlled,
                             config=rules_config(rules), threads=threads)
    rng = np.random.default_rng(seed)
    mine = np.arange(env.n_slots) % k == 0  # slot i is seat controlled[i % k] of game i // k
    places, started = [], n_games
    active = np.ones(env.n_slots, bool)
    filt = ActionFilter.for_model(env.n_slots, policy)  # the rules the policy trained with
    opp_filt = ActionFilter.for_model(env.n_slots, opponent)
    o = env.observe()
    while len(places) < games:
        ready = o["ready"] & active
        actions = np.zeros(env.n_slots, np.int64)
        for model, f, sel in ((policy, filt, mine), (opponent, opp_filt, ~mine)):
            idx = np.nonzero(ready & sel)[0]
            if not len(idx):
                continue
            m = f.apply(o["features"][idx], o["action_mask"][idx], idx)
            actions[idx] = _choose(model, o["features"][idx], o["cards"][idx], m, sample,
                                   device, amp, rng)
            f.record(idx, actions[idx])
        _, dones, info = env.step(actions)
        for i in np.nonzero(dones & active & mine)[0]:
            if info["placement"][i] > 0:
                places.append(int(info["placement"][i]))
            if started >= games:
                active[i : i + k] = False  # enough games started; this game idles
            else:
                started += 1
        if not active.any() and len(places) < games:
            break  # some games were stopped by a simulator guard
        o = env.observe()
    p = np.array(places[:games])
    ci = 1.96 * p.std(ddof=1) / np.sqrt(len(p)) if len(p) > 1 else float("nan")
    return {"games": int(len(p)), "avg_placement": round(float(p.mean()), 3),
            "ci95": round(float(ci), 3), "top1": round(float((p == 1).mean()), 3),
            "top4": round(float((p <= 4).mean()), 3)}
