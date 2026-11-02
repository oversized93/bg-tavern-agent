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


def test_action_filter_rules():
    from bgagent.filters import F_FROZEN, F_TURN, ActionFilter

    feats = np.zeros((2, L.OBS_FLOATS), np.float32)
    feats[:, F_TURN] = 0.25
    feats[0, F_FROZEN] = 1.0
    mask = np.zeros((2, L.N_ACTIONS), bool)
    mask[:, [L.A_END, L.A_FREEZE, L.A_MOVE + 1, L.A_MOVE + 7]] = True
    filt = ActionFilter(2, move_cap=2, no_refreeze=True)
    idx = np.array([0, 1])
    m = filt.apply(feats, mask, idx)
    assert not m[0, L.A_FREEZE] and m[1, L.A_FREEZE]  # frozen shop: no re-freeze
    filt.record(idx, [L.A_MOVE + 1, L.A_MOVE + 1])
    filt.record(idx, [L.A_MOVE + 7, L.A_END])
    m = filt.apply(feats, mask, idx)
    assert not m[0, L.A_MOVE :].any() and m[1, L.A_MOVE :].any()  # cap hit for slot 0 only
    feats[:, F_TURN] = 0.30  # next turn: counters reset
    assert filt.apply(feats, mask, idx)[0, L.A_MOVE :].any()
    only_moves = np.zeros((1, L.N_ACTIONS), bool)
    only_moves[0, L.A_MOVE + 1] = True
    f1 = feats[:1].copy()
    filt.moves[0] = 5
    assert filt.apply(f1, only_moves, np.array([0])).any()  # never empties the mask


def test_cancel_cap():
    from bgagent.filters import F_TURN, ActionFilter

    feats = np.zeros((1, L.OBS_FLOATS), np.float32)
    feats[:, F_TURN] = 0.25
    mask = np.zeros((1, L.N_ACTIONS), bool)
    mask[0, [L.A_CANCEL, L.A_CHOOSE]] = True
    filt, idx = ActionFilter(1, cancel_cap=2), np.array([0])
    for _ in range(2):
        assert filt.apply(feats, mask, idx)[0, L.A_CANCEL]
        filt.record(idx, [L.A_CANCEL])
    m = filt.apply(feats, mask, idx)
    assert not m[0, L.A_CANCEL] and m[0, L.A_CHOOSE]  # must pick a target now
    feats[:, F_TURN] = 0.30
    assert filt.apply(feats, mask, idx)[0, L.A_CANCEL]  # new turn


def _longest_turn(rules, steps=3000):
    """Seat 0 moves minions whenever it can, else takes any action but End Turn: the loop a
    move-loving policy falls into. Returns the most actions it took in one turn."""
    from bgagent.rules import rules_config

    env = taverngym.BatchEnv(1, seed=4, controlled=(0,), config=rules_config(rules))
    rng = np.random.default_rng(0)
    o, turn, n, longest = env.observe(), -1, 0, 0
    for _ in range(steps):
        if o["ready"][0]:
            t = round(float(o["features"][0, 0]) * 20)
            n, turn = (0, t) if t != turn else (n, turn)
            legal = np.nonzero(o["action_mask"][0])[0]
            moves, other = legal[legal >= L.A_MOVE], legal[legal != L.A_END]
            a = rng.choice(moves) if len(moves) else rng.choice(other) if len(other) else L.A_END
            n += 1
            longest = max(longest, n)
        else:
            a = 0
        _, done, _ = env.step([a])
        if done[0] or longest > 200:
            break
        o = env.observe()
    return longest


def test_action_cap_ends_move_loops():
    """TavernGym has no per-turn limit by default; our rule sets add one."""
    from bgagent.rules import ACTION_CAP

    assert _longest_turn("uncapped") > 200
    assert _longest_turn("full") <= ACTION_CAP + 1  # the cap, then End Turn


def test_head_to_head_eval():
    from bgagent.evaluation import evaluate

    torch.manual_seed(0)
    net = TavernNet(ModelConfig.preset("tiny")).eval()
    r = evaluate(net, 6, n_parallel=3, threads=1, sample=True, opponent=net)
    assert r["games"] == 6 and 1 <= r["avg_placement"] <= 8


def test_hybrid_opponents_seat_layout():
    from bgagent.ppo import Collector, TrainConfig

    cfg = TrainConfig(model="tiny", opponents="hybrid", learner_seats=(0, 1, 2, 3),
                      snapshot_seats=(4, 5), n_games=2, threads=1, decisions_per_iter=1)
    col = Collector(cfg, torch.device("cpu"))
    assert col.env.n_slots == 2 * 6  # seats 0-5 controlled, 6-7 scripted bots
    assert col.learner.sum() == 2 * 4 and not col.learner[[4, 5, 10, 11]].any()
    net = TavernNet(ModelConfig.preset("tiny")).eval()
    trajs, _ = col.collect(net)  # stops after the first finished learner game
    assert trajs and all(1 <= place <= 8 for _, place in trajs)


def _bench_dir():
    import os

    for d in (os.environ.get("TAVERNBENCH_DIR", ""), "~/AutoTavern/tavernbench",
              "/root/deps/AutoTavern/tavernbench"):
        if d and (Path(d).expanduser() / "tavernbench" / "engine.py").exists():
            return Path(d).expanduser()
    return None


@pytest.mark.skipif(_bench_dir() is None, reason="no AutoTavern checkout (TAVERNBENCH_DIR)")
def test_tavernbench_options_reproduce_labels(tmp_path):
    """The adapter builds each option the way the labels were made: scoring the positions it
    builds with TavernGym's own combat preview recovers the labelled best answer."""
    from bgagent import tavernbench as TB

    bench = _bench_dir()
    engine, report = TB.import_bench(bench)
    _, labels, states = TB.load_set(bench / "data" / "eval")
    net = TavernNet(ModelConfig.preset("tiny"))
    path = tmp_path / "p.pt"
    torch.save({"model": net.state_dict(), "model_config": vars(net.cfg), "iteration": 0}, path)
    sc = TB.Scorer(path, engine)
    for tid in ("E3-000", "P3-000", "O4-000", "T2-000", "T5-000"):
        opts = sc.options(states[tid])
        assert len(opts) == len(states[tid].get("options", states[tid].get("perms")))
        win = {json.dumps(o.answer): engine.preview(engine.from_state(o.states[-1]), 0, 1, n=512)
               ["win"] for o in opts}
        best = json.loads(max(win, key=win.get))
        row = {"response": json.dumps({"answer": best})}
        assert report.score_answer(labels[tid], row)[0] == 1, tid
        assert sc.by_policy(opts) in [o.answer for o in opts]


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


def test_lr_schedule_independent_of_stopping():
    from bgagent.ppo import TrainConfig, Trainer

    t = Trainer.__new__(Trainer)
    t.cfg = TrainConfig(lr=3e-4, lr_final=1e-4, iterations=500, lr_schedule_iters=4000)
    assert abs(t.lr_at(0) - 3e-4) < 1e-12 and abs(t.lr_at(3999) - 1e-4) < 1e-12
    assert 2.7e-4 < t.lr_at(499) < 2.8e-4  # stopping at 500 leaves lr high, as on the 4000 path
    t.cfg.lr_schedule_iters = 0
    assert abs(t.lr_at(499) - 1e-4) < 1e-12  # default: the schedule spans `iterations`


def test_combat_labels():
    from bgagent.ppo import combat_labels

    f = np.zeros((7, L.OBS_FLOATS), np.float16)
    turns = [0, 1, 1, 2, 2, 2, 3]
    hp = [40, 30, 30, 25, 25, 26, 25]  # turn 1 -> 2 loses 5; turn 2 -> 3 loses 1 (armor ends at 26)
    for i, (t, h) in enumerate(zip(turns, hp)):
        f[i, 0], f[i, 5] = t / 20, h / 40
    f[0, 1] = 1  # hero select
    lost, dmg, w = combat_labels(f, place=3)
    assert np.allclose(dmg, [10, 5, 5, 1, 1, 1, 25], atol=0.05)
    assert (lost == 1).all() and w.tolist() == [0, 1, 1, 1, 1, 1, 1]
    _, dmg1, _ = combat_labels(f, place=1)
    assert dmg1[-1] == 0  # 1st place won its last combat


def test_combat_head_trains():
    from bgagent.ppo import TrainConfig, build_batch, ppo_update

    cfg = TrainConfig(model="tiny", minibatch=16, micro_batch=8, combat_coef=1.0)
    net = TavernNet(ModelConfig(d=64, layers=2, heads=4, combat_head=True))
    o = next(iter(random_observations(1)))
    mk = np.packbits(o["action_mask"][0])
    traj = [[o["features"][0].astype(np.float16), o["cards"][0].astype(np.int16), mk, 0, -1.0,
             0.0, 0.0] for _ in range(16)]
    b = build_batch([(traj, 4)], 1.0, 0.995)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    stats = ppo_update(net, opt, b, cfg, torch.device("cpu"))
    assert np.isfinite(stats["combat"]) and stats["updates"] >= 1


def test_huge_stats_stay_finite_and_match():
    # full-01 died at it 268: a minion past 655,040 attack or health (TavernGym encodes stat/10,
    # unbounded) overflowed the float16 rollout storage to inf, and one update made every
    # weight NaN. Storage must stay finite, and the network must see the stored value and the
    # raw one alike, so rollout and update agree.
    from bgagent.model import FEATURE_MAX
    from bgagent.ppo import store_features

    o = next(iter(random_observations(1)))
    f = o["features"][:1].copy()
    atk = L.F["board"][0] + 1  # first warband minion's attack / 10
    f[0, atk] = 1e6
    stored = store_features(f)
    assert stored.dtype == np.float16 and np.isfinite(stored).all()
    assert stored[0, atk] == np.float16(FEATURE_MAX)
    net = TavernNet(ModelConfig.preset("tiny")).eval()
    clipped = f.copy()
    clipped[0, atk] = FEATURE_MAX
    c = torch.from_numpy(o["cards"][:1])
    with torch.no_grad():
        raw_out, clip_out = net(torch.from_numpy(f), c), net(torch.from_numpy(clipped), c)
    assert torch.equal(raw_out[0], clip_out[0]) and torch.equal(raw_out[1], clip_out[1])


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast")
def test_update_skips_nonfinite_step():
    # Second guard: whatever the source, a NaN loss must not reach the weights.
    from bgagent.ppo import TrainConfig, build_batch, ppo_update

    cfg = TrainConfig(model="tiny", minibatch=16, micro_batch=8)
    net = TavernNet(ModelConfig(d=64, layers=2, heads=4))
    o = next(iter(random_observations(1)))
    mk = np.packbits(o["action_mask"][0])
    bad = o["features"][0].astype(np.float16)
    bad[0] = np.nan
    traj = [[bad, o["cards"][0].astype(np.int16), mk, 0, -1.0, 0.0, 0.0] for _ in range(16)]
    b = build_batch([(traj, 4)], 1.0, 0.995)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    stats = ppo_update(net, opt, b, cfg, torch.device("cpu"))
    assert stats["skipped"] >= 1
    assert all(torch.isfinite(p).all() for p in net.parameters())


def _random_until(env, turn, seed=0):
    rng = np.random.default_rng(seed)
    o = env.observe()
    while True:
        r = np.nonzero(o["ready"])[0]
        if len(r) and o["features"][r[0]][1] == 0 and round(o["features"][r[0]][0] * 20) >= turn:
            return o
        acts = np.zeros(env.n_slots, np.int64)
        for i in r:
            acts[i] = rng.choice(np.nonzero(o["action_mask"][i])[0])
        env.step(acts)
        o = env.observe()


def test_library_inject_roundtrip():
    from bgagent.library import board_strength, inject, make_entry

    env = taverngym.BatchEnv(1, seed=21, controlled=tuple(range(8)))
    _random_until(env, 6)
    s = json.loads(env.save(0))
    g = s["game"]
    src = max(range(8), key=lambda p: len(g["players"][p]["board"]))
    entry = make_entry(g["players"][src], 5)
    assert entry["board"] and entry["strength"] == board_strength(entry["board"])
    dst = (src + 1) % 8
    inject(g, dst, entry, scale=3.0)
    env.load(0, json.dumps(s))
    g2 = json.loads(env.save(0))["game"]
    board = g2["players"][dst]["board"]
    want = sum(max(1, round(m["atk"] * 3)) + max(1, round(m["hp"] * 3)) for m in entry["board"])
    assert board_strength(board) == want and g2["players"][dst]["tier"] == entry["tier"]
    assert all(m["pool_copies"] == 0 for m in board)
    eids = [m["eid"] for p in g2["players"] for m in p["board"]]
    assert len(eids) == len(set(eids))
    _random_until(env, 9)  # the game keeps running with the injected board


def test_library_seat_with_huge_board_wins():
    from bgagent.library import Difficulty, Library, LibrarySeats

    env = taverngym.BatchEnv(1, seed=5, controlled=tuple(range(8)))
    o = _random_until(env, 2)
    g = json.loads(env.save(0))["game"]
    m = next(m for p in g["players"] for m in p["board"])
    big = dict(m, atk=500, hp=500, max_hp=500, peak_atk=500, peak_hp=500)
    lib = Library([{"turn": t, "strength": 7000, "tier": 1, "board": [big] * 7, "hand": [],
                    "stats": {}, "counters": []} for t in range(1, 40)])
    seats = LibrarySeats(env, (3,), lib, Difficulty([{1: 7000}]), sigma=0.0)
    rng = np.random.default_rng(1)
    final = {}
    while len(final) < 8:
        if seats.prepare(o):
            o = env.observe()
        acts = np.zeros(env.n_slots, np.int64)
        idx = np.nonzero(o["ready"])[0]
        lib_idx = idx[seats.mask[idx]]
        acts[lib_idx] = seats.actions(o, lib_idx)
        for i in idx[~seats.mask[idx]]:
            acts[i] = rng.choice(np.nonzero(o["action_mask"][i])[0])
        _, dones, info = env.step(acts)
        for i in np.nonzero(dones)[0]:
            final[int(i)] = int(info["placement"][i])
        o = env.observe()
    assert final[3] == 1
    assert all(abs(given - 7000) <= 7 for rows in seats.given.values() for _, given in rows)


def test_library_game_restarts_when_learner_is_out():
    # Library seats are controlled slots, so BatchEnv would play them on against each other
    # after the learner is eliminated (every turn with a save/load) unless told they're done.
    from bgagent.library import Difficulty, Library, LibrarySeats

    env = taverngym.BatchEnv(1, seed=5, controlled=tuple(range(8)))
    o = _random_until(env, 2)
    g = json.loads(env.save(0))["game"]
    m = next(m for p in g["players"] for m in p["board"])
    big = dict(m, atk=500, hp=500, max_hp=500, peak_atk=500, peak_hp=500)
    lib = Library([{"turn": t, "strength": 7000, "tier": 1, "board": [big] * 7, "hand": [],
                    "stats": {}, "counters": []} for t in range(1, 40)])
    seats = LibrarySeats(env, tuple(range(1, 8)), lib, Difficulty([{1: 7000}]), sigma=0.0)
    rng = np.random.default_rng(1)
    while True:
        if seats.prepare(o):
            o = env.observe()
        acts = np.zeros(env.n_slots, np.int64)
        idx = np.nonzero(o["ready"])[0]
        lib_idx = idx[seats.mask[idx]]
        acts[lib_idx] = seats.actions(o, lib_idx)
        for i in idx[~seats.mask[idx]]:
            acts[i] = rng.choice(np.nonzero(o["action_mask"][i])[0])
        _, dones, info = env.step(acts)
        seats.end_finished(dones)
        o = env.observe()
        if dones[0]:
            break
    assert info["placement"][0] == 8 and env.episode_index(0) == 0
    env.step(np.zeros(env.n_slots, np.int64))
    assert env.episode_index(0) == 1


def test_library_pick_and_difficulty():
    from bgagent.library import Difficulty, Library

    lib = Library([{"turn": 5, "strength": s, "tier": 1, "board": [], "hand": [], "stats": {},
                    "counters": [], "tag": s} for s in (10, 100, 1000)])
    entry, scale = lib.pick(5, 120, np.random.default_rng(0), k=64)
    assert entry["tag"] == 100 and abs(scale - 1.2) < 1e-9
    entry, _ = lib.pick(9, 900, np.random.default_rng(0), k=64)  # nearest turn the library has
    assert entry["tag"] == 1000
    d = Difficulty([{5: 100}, {5: 400}, {5: 1600}])
    assert abs(d.target(0.5, 5) - 200) < 1e-6 and abs(d.target(2.0, 5) - 1600) < 1e-6
    assert abs(d.target(9.0, 5) - 1600) < 1e-6  # clamped to the top rung


def test_harvest_collects_entries():
    from bgagent.library import board_strength, harvest

    env = taverngym.BatchEnv(2, seed=8, controlled=(0, 1))
    rng = np.random.default_rng(0)

    def random_act(model, feats, cards, mask, idx):
        return np.array([rng.choice(np.nonzero(m)[0]) for m in mask])

    lib = harvest(env, lambda i: None, 2, random_act, per_turn_cap=50, keep_top=10)
    assert len(lib) > 0 and min(lib.turns()) >= 1
    for t in lib.turns():
        for strength, js in lib.by_turn[t]:
            e = json.loads(js)
            assert e["turn"] == t and board_strength(e["board"]) == strength


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


def test_placement_utility():
    from bgagent.ppo import linear_reward, utility_adjustment

    assert linear_reward(1) == 1.0 and linear_reward(8) == -1.0
    assert all(utility_adjustment(p, ()) == 0.0 for p in range(1, 9))
    u = (3.0, 1.0, 0.5, 0.0, -0.25, -0.5, -0.75, -1.0)
    for p in range(1, 9):
        assert abs(linear_reward(p) + utility_adjustment(p, u) - u[p - 1]) < 1e-9


def test_anchor_kl():
    import copy

    from bgagent.ppo import TrainConfig, build_batch, ppo_update

    cfg = TrainConfig(model="tiny", minibatch=16, micro_batch=8)
    torch.manual_seed(0)
    net = TavernNet(ModelConfig(d=64, layers=2, heads=4))
    o = next(iter(random_observations(1)))
    mk = np.packbits(o["action_mask"][0])
    traj = [[o["features"][0].astype(np.float16), o["cards"][0].astype(np.int16), mk, 0, -1.0,
             0.0, 0.0] for _ in range(16)]
    b = build_batch([(traj, 4)], 1.0, 0.995)
    same = copy.deepcopy(net).eval()
    opt = torch.optim.AdamW(net.parameters(), lr=0.0)  # lr 0: weights stay equal to the anchor
    s = ppo_update(net, opt, b, cfg, torch.device("cpu"), anchor=same, anchor_coef=0.1)
    assert abs(s["anchor_kl"]) < 1e-5
    other = TavernNet(ModelConfig(d=64, layers=2, heads=4)).eval()  # different weights
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    s = ppo_update(net, opt, b, cfg, torch.device("cpu"), anchor=other, anchor_coef=0.1)
    assert s["anchor_kl"] > 1e-4 and np.isfinite(s["anchor_kl"])
    assert all(torch.isfinite(p).all() for p in net.parameters())


def test_difficulty_below_zero_scales_the_median():
    from bgagent.library import Difficulty

    lad = Difficulty([{8: 200.0}, {8: 1000.0}])
    assert lad.target(0.0, 8) == 200.0 and lad.target(1.0, 8) == 1000.0
    assert abs(lad.target(-1.0, 8) - 20.0) < 1e-9 and abs(lad.target(-2.0, 8) - 2.0) < 1e-9
    assert lad.target(0.5, 8) > lad.target(0.0, 8) > lad.target(-0.5, 8) > lad.target(-1.0, 8)
