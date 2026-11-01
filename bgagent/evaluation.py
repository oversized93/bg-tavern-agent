"""Evaluate a policy (or random play) against TavernGym's seven scripted bots."""

import numpy as np
import taverngym
import torch

from .filters import ActionFilter

EVAL_SEED = 10_000_019  # fixed and far from training seeds, so checkpoints are comparable


@torch.inference_mode()
def evaluate(policy, games, n_parallel=64, threads=4, sample=False, device="cpu",
             seed=EVAL_SEED):
    """Average placement (1 best, 8 worst) with a 95% interval, top-1 and top-4 rates.

    `policy=None` plays uniformly random legal actions (the floor: about 7.95)."""
    device = torch.device(device)
    amp = device.type == "cuda"
    env = taverngym.BatchEnv(min(n_parallel, games), seed=seed, controlled=(0,),
                             threads=threads)
    rng = np.random.default_rng(seed)
    places, started = [], env.n_slots
    active = np.ones(env.n_slots, bool)
    filt = ActionFilter.for_model(env.n_slots, policy)  # the rules the policy trained with
    o = env.observe()
    while len(places) < games:
        ready = o["ready"] & active
        actions = np.zeros(env.n_slots, np.int64)
        idx = np.nonzero(ready)[0]
        if len(idx):
            m = filt.apply(o["features"][idx], o["action_mask"][idx], idx)
            if policy is None:
                actions[idx] = [rng.choice(np.nonzero(row)[0]) for row in m]
            else:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    logits, _, _ = policy(torch.from_numpy(o["features"][idx]).to(device),
                                          torch.from_numpy(o["cards"][idx]).to(device),
                                          torch.from_numpy(m).to(device))
                if sample:
                    a = torch.multinomial(torch.softmax(logits.float(), -1), 1)[:, 0]
                else:
                    a = logits.argmax(-1)
                actions[idx] = a.cpu().numpy()
            filt.record(idx, actions[idx])
        _, dones, info = env.step(actions)
        for i in np.nonzero(dones & active)[0]:
            if info["placement"][i] > 0:
                places.append(int(info["placement"][i]))
            if started >= games:
                active[i] = False  # enough games started; this slot idles
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
