"""Train the policy with PPO in TavernGym.

    python scripts/train.py --preset smoke --run runs/smoke          # CPU plumbing check
    python scripts/train.py --preset 3070 --run runs/full-01           # the multi-day run
    python scripts/train.py --run runs/full-01                         # resume after a crash
    python scripts/train.py --run runs/full-01 --set iterations=600    # extend a finished run

A run folder holds config.json, metrics.jsonl, tb/ (TensorBoard), latest.pt (full resumable
state, rewritten atomically) and policy_XXXXX.pt snapshots. Re-running with the same --run
resumes from latest.pt with the saved config; --set overrides individual fields.
"""

import argparse
import ast
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bgagent.ppo import TrainConfig, Trainer  # noqa: E402


def parse_value(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder (created or resumed)")
    ap.add_argument("--preset", choices=["smoke", "small", "3070"],
                    help="starting config for a new run")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config fields, e.g. n_games=64 lr=5e-5")
    args = ap.parse_args()

    run = Path(args.run)
    saved = run / "config.json"
    if saved.exists():
        cfg = TrainConfig(**json.loads(saved.read_text()))
        if args.preset:
            print(f"note: {run} already exists; resuming with its saved config, --preset ignored")
    else:
        cfg = TrainConfig.preset(args.preset) if args.preset else TrainConfig()
    names = {f.name for f in fields(TrainConfig)}
    for kv in args.set:
        k, _, v = kv.partition("=")
        if k not in names:
            ap.error(f"unknown config field {k!r}")
        setattr(cfg, k, parse_value(v))
    if isinstance(cfg.learner_seats, list):
        cfg.learner_seats = tuple(cfg.learner_seats)
    run.mkdir(parents=True, exist_ok=True)
    saved.write_text(json.dumps(asdict(cfg), indent=2))
    Trainer(cfg, run).run()


if __name__ == "__main__":
    main()
