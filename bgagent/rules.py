"""TavernGym rule sets used for training and evaluation.

Keys are TavernGym `RulesConfig` fields; an unknown key raises inside TavernGym.

TavernGym has no limit on actions per turn unless `turn_action_caps` is set (its
`max_actions_per_turn` = 200 is only the ceiling for those entries). Moving minions, freezing,
and playing a minion then cancelling its target never advance the game, so a policy that
likes them can stay in one turn forever. A real turn has a timer, so we cap every turn at
ACTION_CAP actions; past it only End Turn is legal. The scripted bots stop at 60 by
themselves, and a trained policy takes about 18 per turn (NOTES.md).
"""

ACTION_CAP = 50

RULES = {
    "full": {"turn_action_caps": [ACTION_CAP]},
    "simple": {  # the paper's first curriculum stage: no trinkets, no Dark Gifts
        "turn_action_caps": [ACTION_CAP],
        "lesser_trinket_turn": 99,
        "greater_trinket_turn": 99,
        "dark_gift_unlock_turn": 99,
    },
    "uncapped": {},  # TavernGym's defaults: every run before 2026-11-01 trained and evaluated here
}


def rules_config(name):
    """The `config` argument for taverngym.BatchEnv / Game: a dict, or None for defaults."""
    return RULES[name] or None
