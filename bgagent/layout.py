"""Offsets into TavernGym's flat observation, derived from crates/taverngym/src/obs.rs.

The engine exposes block *sizes* through `taverngym.layout()` but not offsets, so this module
rebuilds the order `obs::observe_view` writes them in and checks the totals against the
installed build at import time. If TavernGym changes its encoding, the assertion fails loudly
instead of the model silently reading the wrong features.
"""

import taverngym

_L = dict(taverngym.layout())

MINION_F = _L["MINION_F"]  # 31
ITEM_F = _L["ITEM_F"]  # 40
GLOBAL_F = _L["GLOBAL_F"]  # 53
N_QUESTS = _L["N_QUESTS"]
QUEST_F = _L["QUEST_F"]
N_SECRETS = _L["N_SECRETS"]
PENDING_KINDS = _L["PENDING_KINDS"]
OPP_F = _L["OPP_F"]
N_OPP = _L["N_OPP"]
N_SHOP = _L["N_SHOP"]
N_BOARD = _L["N_BOARD"]
N_HAND = _L["N_HAND"]
N_CHOICE = _L["N_CHOICE"]
N_POS = _L["N_POS"]
N_POWERS = _L["N_POWERS"]
N_TRINKETS = 8  # fixed in obs.rs

OBS_FLOATS = taverngym.OBS_FLOATS
OBS_CARDS = taverngym.OBS_CARDS
N_ACTIONS = taverngym.N_ACTIONS
N_CARDS = taverngym.N_CARDS

# Action ranges (actions.rs)
A_END, A_REFRESH, A_FREEZE, A_UPGRADE = 0, 1, 2, 3
A_HERO_POWER = _L["A_HERO_POWER"]
A_DARK_GIFT = _L["A_DARK_GIFT"]
A_CANCEL = _L["A_CANCEL"]
A_BUY = _L["A_BUY"]
A_SELL = _L["A_SELL"]
A_ACTIVATE = _L["A_ACTIVATE"]
A_CHOOSE = _L["A_CHOOSE"]
A_PLAY = _L["A_PLAY"]  # + hand * N_POS + pos
A_MAGNETIZE = _L["A_MAGNETIZE"]  # + hand * N_BOARD + board
A_MOVE = _L["A_MOVE"]  # + from * N_BOARD + to
N_GLOBAL_ACTIONS = A_BUY  # end, refresh, freeze, upgrade, 2 hero powers, dark gift, cancel


def _spans(sizes):
    out, at = {}, 0
    for name, n in sizes:
        out[name] = (at, at + n)
        at += n
    return out, at


# ---- float blocks, in the order obs.rs writes them
F, _f_total = _spans(
    [
        ("global", GLOBAL_F),
        ("quests", N_QUESTS * QUEST_F),
        ("pending", PENDING_KINDS),
        ("board", N_BOARD * MINION_F),
        ("hand", N_HAND * ITEM_F),
        ("shop", N_SHOP * ITEM_F),
        ("choice", N_CHOICE * ITEM_F),
        ("opp", N_OPP * OPP_F),
        ("opp_board", N_OPP * N_BOARD * MINION_F),
    ]
)

# ---- card-index blocks, in the order obs.rs writes them
C, _c_total = _spans(
    [
        ("hero", 1),
        ("powers", N_POWERS),
        ("deity", 1),
        ("pending_src", 1),
        ("quests", 3 * N_QUESTS),  # (quest, its reward, owned reward) per slot
        ("secrets", N_SECRETS),
        ("board", N_BOARD),
        ("hand", N_HAND),
        ("shop", N_SHOP),
        ("choice", N_CHOICE),
        ("trinkets", N_TRINKETS),
        ("opp_hero", N_OPP),
        ("opp_board", N_OPP * N_BOARD),
    ]
)

assert _f_total == OBS_FLOATS, f"float layout {_f_total} != OBS_FLOATS {OBS_FLOATS}; obs.rs changed"
assert _c_total == OBS_CARDS, f"card layout {_c_total} != OBS_CARDS {OBS_CARDS}; obs.rs changed"
assert A_MOVE + N_BOARD * N_BOARD == N_ACTIONS, "action layout changed"

# Context cards pooled into the global token: hero, powers, deity, pending source, quests,
# secrets (one contiguous run) and the owned trinkets.
GLOBAL_CARD_SPANS = [(C["hero"][0], C["secrets"][1]), C["trinkets"]]
N_GLOBAL_CARDS = sum(b - a for a, b in GLOBAL_CARD_SPANS)
GLOBAL_TOKEN_F = F["pending"][1]  # global + quests + pending floats are contiguous from 0

# Token order fed to the transformer.
TOKENS = [
    ("global", 1),
    ("board", N_BOARD),
    ("hand", N_HAND),
    ("shop", N_SHOP),
    ("choice", N_CHOICE),
    ("opp", N_OPP),
]
T, N_TOKENS = _spans(TOKENS)
