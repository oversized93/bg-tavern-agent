"""Ask a trained policy what to do in a game state.

    python scripts/advise.py runs/full-01/policy_00100.pt --state saved_state.json --seat 0
    python scripts/advise.py runs/full-01/policy_00100.pt --demo --turn 6   # simulated game

--state takes `Game.state_json()` output or a `BatchEnv.save()` episode. Later, the live
bridge will produce the same JSON from Power.log. --demo plays a bot game to the given turn
and asks for advice for seat 0, which is handy for eyeballing whether suggestions look sane.
"""

import argparse
import json
import sys
from pathlib import Path

import taverngym

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bgagent.policy import Advisor  # noqa: E402


def demo_state(seed, turn, seat=0):
    g = taverngym.Game(seed=seed)
    while not g.is_over():
        if g.turn >= turn and g.needs_action(seat):
            return g.state_json(), json.loads(g.view(seat))
        for p in range(g.num_players):
            if g.needs_action(p):
                g.step(p, g.bot_action(p))
    raise SystemExit(f"seat {seat} was out before turn {turn}; try another seed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("--state", help="file with a saved game state")
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--turn", type=int, default=5)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--show-view", action="store_true", help="print the seat's view too")
    args = ap.parse_args()

    if args.demo:
        state, view = demo_state(args.seed, args.turn, args.seat)
        if args.show_view:
            print(json.dumps(view, indent=1, ensure_ascii=False)[:4000])
    elif args.state:
        state = Path(args.state).read_text()
    else:
        ap.error("give --state FILE or --demo")
    advice = Advisor(args.policy).advise(state, args.seat, args.top)
    print(json.dumps(advice, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
