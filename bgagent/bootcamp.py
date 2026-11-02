"""The levelling bootcamp (NOTES.md, 2026-11-02).

Without levelling, the AI never learns to play the higher Tiers, and without that skill
levelling doesn't pay, so it never levels: in scratch-01 the Tier at turn 4 fell from 1.8 to 1.1
while everything else improved, and a scripted-bot test showed levelling costs a simple player
about half a place against the library seats. The bootcamp breaks the loop: while a learner
seat is below `curve` for the turn and can afford Upgrade, its action is Upgrade instead of the
policy's sample. Everything else is the policy's own choice, and the forcing is meant to be
handed back (TrainConfig.level_force_decay), as OpenAI Five's scripted item purchases were and
as Jump-Start RL's guide policy is.

The forced steps are not the policy's choices: they train the value head (what a higher Tier
is worth) but not the policy (build_batch's `train_pi`).

Cranny's decision (2026-11-02, "Do A"): the schedule is real winners' average Tier by turn, a
temporary training aid. It is the one exception to "no human moves in training"; nothing else
human-derived steers the policy without asking.
"""

import numpy as np

from . import layout as L

# Turn -> Tier to have reached by then. Real winners (Firestone's Perfect Games): Tier 2 by turn
# 2-3, 2.45 at turn 4, 3.6 at 6, 4.9 at 8, 5.6 at 10.
WINNERS = ((2, 2), (5, 3), (6, 4), (8, 5), (10, 6))

F_TURN, F_HERO_SELECT, F_TIER = 0, 1, 10  # global floats: turn / 20, hero select flag, Tier / 6


def check_curve(curve):
    turns = [int(t) for t, _ in curve]
    tiers = [int(k) for _, k in curve]
    if turns != sorted(set(turns)) or tiers != sorted(tiers) or not all(1 <= k <= 6 for k in tiers):
        raise ValueError(f"level_curve needs (turn, tier) pairs, turns increasing, tiers 1-6 "
                         f"and not falling; got {curve}")


def target_tier(curve, turns):
    """The Tier `curve` asks for at each turn in `turns` (1 before its first entry)."""
    turns = np.asarray(turns)
    out = np.ones(turns.shape, np.int64)
    for t, k in curve:
        out[turns >= t] = np.maximum(out[turns >= t], k)
    return out


class LevelCurve:
    """Forces Upgrade for a BatchEnv's slots. Each episode of each slot is on the curve with
    probability `p`, drawn when the episode's first decision is seen."""

    def __init__(self, env, curve, p=1.0, seed=0):
        check_curve(curve)
        self.env, self.curve, self.p = env, tuple(tuple(c) for c in curve), float(p)
        self.rng = np.random.default_rng(seed)
        self.episode = np.full(env.n_slots, -1, np.int64)
        self.on = np.zeros(env.n_slots, bool)

    def _flags(self, slots):
        for i in slots:
            ep = self.env.episode_index(self.env.slot(int(i))[0])
            if self.episode[i] != ep:
                self.episode[i] = ep
                self.on[i] = self.rng.random() < self.p
        return self.on[slots]

    def apply(self, feats, mask, actions, slots):
        """Replace `actions` (in place) with Upgrade where the curve says so; returns which.
        `mask` is the legal mask the policy chose from (after the action filter)."""
        slots = np.asarray(slots)
        if not len(slots) or self.p <= 0:
            return np.zeros(len(slots), bool)
        turn = np.rint(feats[:, F_TURN] * 20).astype(np.int64)
        tier = np.rint(feats[:, F_TIER] * 6).astype(np.int64)
        force = (self._flags(slots) & (feats[:, F_HERO_SELECT] == 0)
                 & (tier < target_tier(self.curve, turn)) & np.asarray(mask, bool)[:, L.A_UPGRADE])
        actions[force] = L.A_UPGRADE
        return force
