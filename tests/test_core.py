"""Fast checks that the observation layout, model and PPO math still line up with TavernGym.

Run with `pytest -q`. They take well under a minute on CPU.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import taverngym
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bgagent import layout as L  # noqa: E402
from bgagent.model import ModelConfig, TavernNet  # noqa: E402
from bgagent.policy import Advisor, observe_state  # noqa: E402
from bgagent.ppo import build_batch  # noqa: E402


def random_observations(steps=300, seed=3):
    env = taverngym.BatchEnv(n_games=8, seed=seed, controlled=(0,))
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        o = env.observe()
        yield o
        m, r = o["action_mask"], o["ready"]
        env.step([rng.choice(np.nonzero(m[i])[0]) if r[i] else 0 for i in range(len(r))])


def test_layout_matches_engine():
    """Slot counts read through our offsets agree with the engine's own counters."""
    checked = 0
    for o in random_observations():
        f, c = o["features"], o["cards"]
        for i in np.nonzero(o["ready"])[0]:
            if f[i, 1] == 1:  # hero select: board/hand are empty and counters trivial
                continue
            nb = round(f[i, 33] * 7)  # global: board.len() / 7
            nh = round(f[i, 34] * 10)  # global: hand.len() / 10
            b0, b1 = L.F["board"]
            h0, h1 = L.F["hand"]
            assert f[i, b0:b1].reshape(L.N_BOARD, L.MINION_F)[:, 0].sum() == nb
            assert (c[i, L.C["board"][0] : L.C["board"][1]] > 0).sum() == nb
            assert f[i, h0:h1].reshape(L.N_HAND, L.ITEM_F)[:, 0].sum() == nh
            assert (c[i, L.C["hand"][0] : L.C["hand"][1]] > 0).sum() == nh
            assert c[i, L.C["hero"][0]] > 0
            checked += 1
    assert checked > 100


def test_buy_mask_only_on_present_shop_slots():
    for o in random_observations(150):
        f, m = o["features"], o["action_mask"]
        s0, s1 = L.F["shop"]
        for i in np.nonzero(o["ready"])[0]:
            present = f[i, s0:s1].reshape(L.N_SHOP, L.ITEM_F)[:, 0] > 0
            assert not (m[i, L.A_BUY : L.A_BUY + L.N_SHOP] & ~present).any()


def test_model_masks_illegal_actions():
    o = next(iter(random_observations(1)))
    net = TavernNet(ModelConfig.preset("tiny"))
    m = torch.from_numpy(o["action_mask"])
    logits, value, place = net(torch.from_numpy(o["features"]), torch.from_numpy(o["cards"]), m)
    assert logits.shape == (8, L.N_ACTIONS)
    assert torch.isfinite(logits[m]).all()
    assert torch.isinf(logits[~m]).all()
    assert value.abs().max() <= 1 and place.shape == (8, 8)


def test_factored_head_is_type_times_target():
    from bgagent.model import ACTION_GROUPS

    torch.manual_seed(0)
    tgt = torch.randn(4, L.N_ACTIONS)
    mask = torch.rand(4, L.N_ACTIONS) < 0.3
    mask[:, L.A_END] = True
    mask[0, L.A_MOVE :] = False  # a state with no legal move
    tgt = tgt.masked_fill(~mask, float("-inf"))
    typ = torch.randn(4, len(ACTION_GROUPS))
    p = torch.softmax(TavernNet.factor(tgt, typ), -1)
    assert torch.allclose(p.sum(-1), torch.ones(4))
    assert (p[~mask] == 0).all()
    legal_types = torch.stack([mask[:, s:e].any(-1) for s, e in ACTION_GROUPS], 1)
    pt = torch.softmax(typ.masked_fill(~legal_types, float("-inf")), -1)
    for k, (s, e) in enumerate(ACTION_GROUPS):
        assert torch.allclose(p[:, s:e].sum(-1), pt[:, k], atol=1e-6)
        within = torch.softmax(tgt[1, s:e], -1)
        if mask[1, s:e].any():
            assert torch.allclose(p[1, s:e] / pt[1, k], within, atol=1e-5)


def test_fresh_factored_policy_does_not_favour_moves():
    """With a full board, an untrained flat head puts most of its mass on the 42 moves; the
    factored head gives moves about one share per legal action type."""
    torch.manual_seed(0)
    for o in random_observations(400, seed=11):
        f = o["features"]
        full = np.nonzero(o["ready"] & (np.round(f[:, 33] * 7) >= 6))[0]
        if len(full):
            break
    i = full[:1]
    args = (torch.from_numpy(f[i]), torch.from_numpy(o["cards"][i]),
            torch.from_numpy(o["action_mask"][i]))
    move_share = {}
    for fac in (False, True):
        net = TavernNet(ModelConfig(d=64, layers=2, heads=4, factored=fac))
        with torch.no_grad():
            move_share[fac] = torch.softmax(net(*args)[0], -1)[0, L.A_MOVE :].sum().item()
    assert move_share[False] > 0.5 and move_share[True] < 0.3


def test_action_names_match_pointer_layout():
    assert taverngym.action_name(L.A_PLAY + 3 * L.N_POS + 2).startswith("Play { hand: 3, pos: 2")
    assert taverngym.action_name(L.A_MAGNETIZE + 4 * L.N_BOARD + 5) == "Magnetize { hand: 4, board: 5 }"
    assert taverngym.action_name(L.A_MOVE + 6 * L.N_BOARD + 1) == "Move { from: 6, to: 1 }"


def test_gae_with_terminal_reward():
    """gamma = lambda = 1: every advantage is the final reward minus that step's value."""
    f = np.zeros(L.OBS_FLOATS, np.float16)
    c = np.zeros(L.OBS_CARDS, np.int16)
    mk = np.packbits(np.ones(L.N_ACTIONS, bool))
    vals = [0.1, -0.2, 0.3]
    traj = [[f, c, mk, 0, -1.0, v, 0.0] for v in vals]
    traj[-1][6] = 0.5
    b = build_batch([(traj, 3)], gamma=1.0, lam=1.0)
    assert np.allclose(b["adv"], [0.5 - v for v in vals])
    assert np.allclose(b["ret"], 0.5)
    assert (b["place"] == 2).all() and b["mask"].shape == (3, L.N_ACTIONS)


def test_advisor_on_saved_state(tmp_path):
    g = taverngym.Game(seed=5)
    while not (g.turn >= 3 and g.needs_action(0)):
        for p in range(g.num_players):
            if g.needs_action(p):
                g.step(p, g.bot_action(p))
    state = g.state_json()
    _, _, mask, ready = observe_state(state, 0)
    assert ready and mask.any()
    net = TavernNet(ModelConfig.preset("tiny"))
    path = tmp_path / "p.pt"
    torch.save({"model": net.state_dict(), "model_config": vars(net.cfg), "iteration": 0}, path)
    advice = Advisor(path).advise(state, 0, top_k=3)
    assert advice["ready"] and len(advice["actions"]) == 3
    assert all(mask[0, a["action"]] for a in advice["actions"])
    assert 1 <= advice["expected_placement"] <= 8
