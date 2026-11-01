"""Action filters that stop a policy looping on free, reversible actions.

Taking the policy's single most likely action (what a recommendation shows) can cycle:
Freeze toggles the shop frozen and unfrozen, moving minion 1 to slot 0 twice swaps them
back, and playing a minion with a targeted Battlecry then cancelling the target puts it
back in hand. Each step returns to a state where the same action is again most likely, and
TavernGym has no per-turn limit of its own (bgagent/rules.py adds one). Measured before
these filters: 97.5% of greedy decisions were moves (f2), and a policy trained without the
cancel cap drifted into Play/Cancel loops of 400-1,000 actions per turn (f3). NOTES.md.

The filter narrows the legal mask, so it is applied identically in training, evaluation
and live advice, and the policy is trained on exactly the choices it will be offered.
  - no_refreeze: Freeze is unavailable while the shop is already frozen.
  - move_cap: at most this many minion moves per turn (6 rearranges any 7-minion board).
  - cancel_cap: at most this many Cancels per turn (backing out of a target or a choice).
"""

import numpy as np

from . import layout as L

F_TURN = 0  # global floats: turn / 20
F_FROZEN = 15  # global floats: shop is frozen


def filter_mask(feats, mask, moves_this_turn, move_cap, no_refreeze, cancels_this_turn=0,
                cancel_cap=0):
    """Rows of `mask` narrowed by the rules above. Never removes every legal action."""
    out = mask.copy()
    if no_refreeze:
        out[feats[:, F_FROZEN] > 0.5, L.A_FREEZE] = False
    if move_cap:
        out[np.asarray(moves_this_turn) >= move_cap, L.A_MOVE :] = False
    if cancel_cap:
        rows = np.broadcast_to(np.asarray(cancels_this_turn) >= cancel_cap, (len(out),))
        out[rows, L.A_CANCEL] = False
    empty = ~out.any(1)
    out[empty] = mask[empty]
    return out


class ActionFilter:
    """Per-slot move and cancel counters for a batch of games."""

    def __init__(self, n_slots, move_cap=0, no_refreeze=False, cancel_cap=0):
        self.move_cap, self.no_refreeze, self.cancel_cap = move_cap, no_refreeze, cancel_cap
        self.turn = np.full(n_slots, -1.0)
        self.moves = np.zeros(n_slots, int)
        self.cancels = np.zeros(n_slots, int)

    @classmethod
    def for_model(cls, n_slots, model):
        cfg = getattr(model, "cfg", None)
        return cls(n_slots, getattr(cfg, "move_cap", 0), getattr(cfg, "no_refreeze", False),
                   getattr(cfg, "cancel_cap", 0))

    @property
    def active(self):
        return bool(self.move_cap or self.no_refreeze or self.cancel_cap)

    def apply(self, feats, mask, idx):
        """Filtered mask for the slots `idx` (feats and mask are rows for those slots)."""
        if not self.active:
            return mask
        turn = feats[:, F_TURN]
        new = turn != self.turn[idx]
        self.moves[idx[new]] = 0
        self.cancels[idx[new]] = 0
        self.turn[idx] = turn
        return filter_mask(feats, mask, self.moves[idx], self.move_cap, self.no_refreeze,
                           self.cancels[idx], self.cancel_cap)

    def record(self, idx, actions):
        a = np.asarray(actions)
        self.moves[idx] += a >= L.A_MOVE
        self.cancels[idx] += a == L.A_CANCEL
