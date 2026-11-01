"""Load a trained policy and ask it for recommendations on any TavernGym game state.

This is the interface the live bridge will call: Power.log -> TavernGym game state ->
`Advisor.advise(state_json, seat)` -> ranked actions with probabilities. It never acts on the
real game; it only returns suggestions.
"""

import json
from pathlib import Path

import numpy as np
import taverngym
import torch

from . import layout as L
from .filters import filter_mask
from .model import ModelConfig, TavernNet


def model_config_from(ck) -> ModelConfig:
    """The model config saved in a checkpoint. Keys added later default to what older
    checkpoints were trained with: no factored head, no action filter."""
    old = {"factored": False, "move_cap": 0, "no_refreeze": False, "cancel_cap": 0,
           "combat_head": False}
    return ModelConfig(**{**old, **ck["model_config"]})


def load_policy(path, device="cpu"):
    ck = torch.load(Path(path), map_location=device, weights_only=False)
    model = TavernNet(model_config_from(ck)).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), ck.get("iteration")


def observe_state(state_json: str, seat: int):
    """Observation and legal mask for `seat` in a saved game.

    Accepts either `Game.state_json()` output or a `BatchEnv.save()` episode. TavernGym has no
    single-game observe call in Python, so the game is loaded into a one-game BatchEnv that
    controls only `seat`.
    """
    obj = json.loads(state_json)
    game = obj["game"] if "game" in obj else obj
    episode = json.dumps({"game": game, "index": 0, "finished": [False]})
    env = taverngym.BatchEnv(1, seed=0, controlled=(seat,))
    env.load(0, episode)
    o = env.observe()
    return o["features"], o["cards"], o["action_mask"], bool(o["ready"][0])


def placement_from_value(v: float) -> float:
    """The value head predicts the placement reward (1st = +1, 8th = -1, linear)."""
    return 1 + (1 - v) * 3.5


class Advisor:
    def __init__(self, policy_path, device="cpu"):
        self.device = torch.device(device)
        self.model, self.iteration = load_policy(policy_path, self.device)

    @torch.inference_mode()
    def advise(self, state_json: str, seat: int = 0, top_k: int = 5, moves_this_turn: int = 0,
               cancels_this_turn: int = 0):
        """`moves_this_turn` / `cancels_this_turn`: minion moves and cancelled targets already
        this turn (the live bridge counts them from the log); the policy was trained with
        per-turn caps on both."""
        f, c, m, ready = observe_state(state_json, seat)
        if not ready or not m[0].any():
            return {"ready": False, "actions": []}
        cfg = self.model.cfg
        m = filter_mask(f, m, [moves_this_turn], cfg.move_cap, cfg.no_refreeze,
                        [cancels_this_turn], cfg.cancel_cap)
        logits, value, place = self.model(
            torch.from_numpy(f).to(self.device),
            torch.from_numpy(c).to(self.device),
            torch.from_numpy(m).to(self.device),
        )
        probs = torch.softmax(logits[0], -1).cpu().numpy()
        place_p = torch.softmax(place[0], -1).cpu().numpy()
        order = np.argsort(-probs)[: min(top_k, int(m[0].sum()))]
        return {
            "ready": True,
            "expected_placement": round(placement_from_value(float(value[0])), 2),
            "placement_probs": [round(float(p), 3) for p in place_p],
            "actions": [
                {"action": int(a), "name": taverngym.action_name(int(a)),
                 "prob": round(float(probs[a]), 3)}
                for a in order
            ],
        }
