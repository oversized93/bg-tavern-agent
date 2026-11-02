"""Opponent libraries: the paper's library seats (arXiv 2610.33845, Section 6 and Appendix C).

"Rather than self-play, the learner faces seven library seats." "A library entry is one seat's
board at one turn, with its hand and hidden counters", and "each turn a library seat takes an
entry that matches a target strength curve." The paper publishes no more than that, so the
rest is ours (NOTES.md):
- A library seat never recruits. It is a controlled slot that only presses End Turn (a random
  legal option when a choice is forced, e.g. hero select), and at the start of each of its
  turns its board, hand, Tavern Tier and buff counters are replaced with an entry, through
  BatchEnv save/load. Its hero, Health and elimination follow the normal rules.
- Strength is the warband's total Attack + Health, the same as Firestone's `warbandStats`.
- An entry is picked near the target in log space and its stats scaled to hit the target
  (the simulated games that fill a library rarely reach top players' late-game boards).
- Difficulty d runs over a ladder of curves: 0 is the library's own median (as strong as the
  games it was harvested from), then Firestone's MMR brackets from all players to the top 1%
  (`human.MMR_BRACKETS`); between two rungs the target is their geometric mean.
- Training and exam libraries are harvested from different seeds and policies ("training uses
  twins built the same way from other sources").
"""

import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Buff counters that cards read in combat; copied with an entry. Game history (streaks, wins,
# last damage) stays with the seat that receives it.
COPY_STATS = (
    "baller_bonus", "battlecries_triggered", "beetle_atk", "beetle_hp", "cards_played",
    "deathrattles_triggered", "elem_give_atk", "elem_give_hp", "elementals_played",
    "eternal_knight_deaths", "gem_atk", "gem_hp", "golden_played", "knight_bonus",
    "lobster_bonus", "magnetized", "spell_atk", "spell_hp", "spellcraft_cast", "spells_cast",
    "tavern_atk", "tavern_elemental_atk", "tavern_elemental_hp", "tavern_hp",
    "tavern_spells_cast", "undead_atk", "undead_hp", "volumizer_atk", "volumizer_hp",
)
SCALED = ("atk", "hp", "max_hp", "peak_atk", "peak_hp")


def board_strength(board):
    return int(sum(m["atk"] + m["hp"] for m in board))


def make_entry(player, turn):
    """The entry for `player` (a player dict from a saved game) as it fought on `turn`."""
    return {
        "turn": int(turn),
        "strength": board_strength(player["board"]),
        "tier": int(player["tier"]),
        "board": player["board"],
        "hand": player["hand"],
        "stats": {k: player["stats"][k] for k in COPY_STATS if k in player["stats"]},
        "counters": player["counters"],
    }


def _fresh_minion(game, m, scale):
    m = json.loads(json.dumps(m))  # deep copy
    game["next_eid"] += 1
    m["eid"] = game["next_eid"]
    m["pool_copies"] = 0  # not from the shared pool: nothing returns to it (engine convention)
    m["killed_by"] = 0
    if scale != 1.0:
        for k in SCALED:
            m[k] = max(1, int(round(m[k] * scale)))
    return m


def inject(game, seat, entry, scale=1.0, hand=True):
    """Give `seat` the entry's board (stats times `scale`), hand, Tier and buff counters.
    `game` is the "game" object of a BatchEnv save, edited in place."""
    p = game["players"][seat]
    p["board"] = [_fresh_minion(game, m, scale) for m in entry["board"]]
    if hand:
        items = []
        for it in entry["hand"]:
            it = json.loads(json.dumps(it))
            inner = it.get("item", {})
            if "Minion" in inner:
                inner["Minion"] = _fresh_minion(game, inner["Minion"], 1.0)
            items.append(it)
        p["hand"] = items
    p["tier"] = entry["tier"]
    p["stats"].update(entry["stats"])
    p["counters"] = json.loads(json.dumps(entry["counters"]))


class Library:
    """Entries grouped by turn. Kept as JSON strings (parsed on pick) to stay small."""

    def __init__(self, entries=()):
        self.by_turn = defaultdict(list)  # turn -> [(strength, json)]
        for e in entries:
            self.add(e)
        self._index = None

    def add(self, e):
        self.by_turn[e["turn"]].append((e["strength"], json.dumps(e, separators=(",", ":"))))
        self._index = None

    def __len__(self):
        return sum(len(v) for v in self.by_turn.values())

    def _build(self):
        idx = {}
        for t, rows in self.by_turn.items():
            rows.sort(key=lambda r: r[0])
            idx[t] = np.array([r[0] for r in rows], np.float64)
        self._index = idx

    def turns(self):
        return sorted(self.by_turn)

    def median_curve(self):
        if self._index is None:
            self._build()
        return {t: float(np.median(s)) for t, s in sorted(self._index.items()) if len(s)}

    def pick(self, turn, target, rng, k=8, max_scale=50.0):
        """(entry, scale): of k random entries for this turn (the nearest turn the library
        has), the one closest to `target` in log strength, and the factor that scales its
        stats onto the target."""
        if self._index is None:
            self._build()
        turns = self.turns()
        t = min(turns, key=lambda x: (abs(x - turn), -x))
        s, rows = self._index[t], self.by_turn[t]
        cand = rng.integers(0, len(rows), size=min(k, len(rows)))
        logs = np.log(np.maximum(s[cand], 1.0))
        j = int(cand[np.argmin(np.abs(logs - np.log(max(target, 1.0))))])
        entry = json.loads(rows[j][1])
        strength = rows[j][0]
        scale = 1.0 if strength <= 0 else float(np.clip(target / strength, 1 / max_scale, max_scale))
        return entry, scale

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for t in self.turns():
                for _, js in self.by_turn[t]:
                    f.write(js + "\n")

    @classmethod
    def load(cls, path):
        lib = cls()
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                lib.by_turn[e["turn"]].append((e["strength"], line.rstrip("\n")))
        return lib


class Difficulty:
    """Target strength by turn on a ladder of curves; d in [0, len(curves) - 1]."""

    def __init__(self, curves):
        self.curves = [dict(c) for c in curves]

    @property
    def max(self):
        return len(self.curves) - 1

    @staticmethod
    def _at(curve, turn):
        if turn in curve:
            return curve[turn]
        turns = sorted(curve)
        return curve[min(turns, key=lambda x: (abs(x - turn), -x))]

    def target(self, d, turn):
        d = float(np.clip(d, 0, self.max))
        lo = int(np.floor(d))
        hi = min(lo + 1, self.max)
        a, b = max(self._at(self.curves[lo], turn), 1.0), max(self._at(self.curves[hi], turn), 1.0)
        w = d - lo
        return float(np.exp((1 - w) * np.log(a) + w * np.log(b)))


def ladder(library, human_curves=None):
    """The difficulty ladder: the library's own median, then the human curves (easiest
    first; default Firestone's MMR brackets from data/firestone)."""
    if human_curves is None:
        from .human import MMR_BRACKETS, warband_curve

        human_curves = [warband_curve(m) for m in MMR_BRACKETS]
    return Difficulty([library.median_curve(), *human_curves])


class LibrarySeats:
    """Drives the library seats of a BatchEnv: a new entry at each of their turn starts, then
    End Turn. Call `prepare(o)` before choosing actions; if it returns True, observe again."""

    def __init__(self, env, seats, library, difficulty, d=0.0, sigma=0.25, seed=0, hand=True):
        self.env, self.library, self.difficulty = env, library, difficulty
        self.d, self.sigma, self.hand = float(d), float(sigma), hand
        self.rng = np.random.default_rng(seed)
        self.mask = np.array([env.slot(i)[1] in set(seats) for i in range(env.n_slots)])
        self.done = {}  # slot -> (episode, turn) already given an entry
        self.stats_reset()

    def stats_reset(self):
        self.given = defaultdict(list)  # turn -> [(target, strength given)]

    def prepare(self, o):
        need = defaultdict(list)
        for i in np.nonzero(o["ready"] & self.mask)[0]:
            f = o["features"][i]
            if f[1] != 0:  # hero select
                continue
            turn = int(round(f[0] * 20))
            g, seat = self.env.slot(int(i))
            key = (self.env.episode_index(g), turn)
            if self.done.get(int(i)) != key:
                need[g].append((int(i), seat, turn, key))
        for g, items in need.items():
            s = json.loads(self.env.save(g))
            game = s["game"]
            for i, seat, turn, key in items:
                if not game["players"][seat]["alive"]:
                    continue
                target = self.difficulty.target(self.d, turn)
                target *= float(np.exp(self.rng.normal(0.0, self.sigma)))
                entry, scale = self.library.pick(turn, target, self.rng)
                inject(game, seat, entry, scale, hand=self.hand)
                self.done[i] = key
                self.given[turn].append((target, board_strength(game["players"][seat]["board"])))
            self.env.load(g, json.dumps(s))
        return bool(need)

    def actions(self, o, idx):
        """End Turn where legal, else a random legal action, for slots `idx`."""
        out = np.zeros(len(idx), np.int64)
        for j, i in enumerate(idx):
            m = o["action_mask"][i]
            out[j] = 0 if m[0] else self.rng.choice(np.nonzero(m)[0])
        return out


def harvest(env, policy_for_slot, games, act_fn, per_turn_cap=2000, keep_top=400, seed=0):
    """Play `games` games in `env` and collect every alive seat's board at each turn.

    `policy_for_slot(i)` gives the model for controlled slot i (None = random legal);
    `act_fn(model, feats, cards, mask)` returns actions. Uncontrolled seats are TavernGym's
    bots. At each turn start the game is saved and every alive player's board is recorded as
    the board it fought with on the previous turn. Per turn the strongest `keep_top` entries
    are always kept, the rest by reservoir sampling up to `per_turn_cap`."""
    rng = np.random.default_rng(seed)
    top = defaultdict(list)  # turn -> [(strength, entry)] strongest
    res = defaultdict(list)  # turn -> reservoir
    seen = defaultdict(int)
    last_turn = {}
    finished = 0  # complete games
    groups = defaultdict(list)
    for i in range(env.n_slots):
        groups[id(policy_for_slot(i))].append(i)
    models = {k: policy_for_slot(v[0]) for k, v in groups.items()}
    start = [env.episode_index(g) for g in range(env.n_games)]
    o = env.observe()
    while finished < games:
        for i in np.nonzero(o["ready"])[0]:
            g = env.slot(int(i))[0]
            f = o["features"][i]
            if f[1] != 0:
                continue
            turn = int(round(f[0] * 20))
            key = (env.episode_index(g), turn)
            prev = last_turn.get(g)
            if prev is None or prev[0] != key[0] or prev[1] < turn:
                last_turn[g] = key
                if prev is not None and prev[0] == key[0] and turn > 1:
                    game = json.loads(env.save(g))["game"]
                    for p in game["players"]:
                        if p["alive"] and p["board"]:
                            _keep(make_entry(p, turn - 1), top, res, seen, rng,
                                  per_turn_cap - keep_top, keep_top)
        acts = np.zeros(env.n_slots, np.int64)
        for k, slots in groups.items():
            idx = np.array([i for i in slots if o["ready"][i]], np.int64)
            if not len(idx):
                continue
            acts[idx] = act_fn(models[k], o["features"][idx], o["cards"][idx],
                               o["action_mask"][idx], idx)
        env.step(acts)
        finished = sum(env.episode_index(g) - start[g] for g in range(env.n_games))
        o = env.observe()
    lib = Library()
    for t in set(top) | set(res):
        for _, e in top[t]:
            lib.add(e)
        for e in res[t]:
            lib.add(e)
    return lib


def _keep(e, top, res, seen, rng, cap, keep_top):
    t = e["turn"]
    tl = top[t]
    if len(tl) < keep_top:
        tl.append((e["strength"], e))
        tl.sort(key=lambda x: x[0])
        return
    if e["strength"] > tl[0][0]:
        e, tl[0] = tl[0][1], (e["strength"], e)
        tl.sort(key=lambda x: x[0])
    seen[t] += 1
    r = res[t]
    if len(r) < cap:
        r.append(e)
    else:
        j = int(rng.integers(0, seen[t]))
        if j < cap:
            r[j] = e
