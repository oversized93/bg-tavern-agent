# bg-tavern-agent

A Hearthstone Battlegrounds **recommendation** agent. A policy network is trained with PPO in
TavernGym (MCG-NJU/AutoTavern, a faithful BG simulator), then fed live game state parsed from
Power.log to suggest moves. Cranny plays every game himself.

## Hard rules
- **Recommendations only.** Never write code that sends input to Hearthstone, clicks, types,
  moves the mouse, writes game memory, edits game files, or plays/queues games. Reading
  Power.log and other local logs is fine. If a task seems to need any of that, stop and ask.
- Never claim a result without a number from a run. Every experiment goes in NOTES.md.

## Layout
- `bgagent/layout.py`: offsets into TavernGym's flat observation, rebuilt from
  `crates/taverngym/src/obs.rs` and asserted against `taverngym.layout()` at import. If
  TavernGym is upgraded and the assert fires, re-derive the order from obs.rs; never just
  bump numbers.
- `bgagent/model.py`: entity transformer (one token per minion / hand card / shop offer /
  choice option / opponent, plus a global token), pointer action head over the 245 actions,
  value head, auxiliary placement head. Presets: tiny, small (~2.9M), medium (~6.3M),
  full (~16.9M).
- `bgagent/ppo.py`: rollout collection over BatchEnv (whole episodes, reward at the end),
  GAE, PPO update with bf16 autocast and gradient accumulation, checkpoints, league
  snapshots, periodic eval vs scripted bots.
- `bgagent/policy.py`: `Advisor` - load a policy, give ranked actions for any saved game
  state. The live bridge will call this.
- `scripts/`: train.py, evaluate.py, advise.py, status.py.

## Commands
- Tests: `pytest -q` (fast; run before every commit)
- Smoke run: `python scripts/train.py --preset smoke --run runs/smoke`
- Real run: `python scripts/train.py --preset 3070 --run runs/<name>` (resume = same command)
- Progress: `python scripts/status.py runs/<name>`
- Evaluate: `python scripts/evaluate.py runs/<name>/policy_00100.pt --games 2000`
- Advice demo: `python scripts/advise.py runs/<name>/latest.pt --demo --turn 6`

## Conventions
- Training from scratch against scripted bots gives no signal (a fresh policy places 8th
  nearly every game). Start in league/self-play; the bots are an evaluation yardstick.
- Placement: 1 best, 8 worst. Reward: 1st = +1, 8th = -1, linear (TavernGym's).
- TavernGym is pinned in pyproject.toml. Its card data is patch 36.6.3; live patches drift.
- Long runs live in tmux inside WSL2. Never delete a run folder without asking.
