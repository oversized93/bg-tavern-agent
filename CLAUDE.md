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
- `bgagent/filters.py`: narrows the legal mask so the top action can't loop (no Freeze while
  frozen, at most 6 moves and 2 Cancels per turn). Used identically in training, eval and
  advice; the settings are saved with the model.
- `bgagent/rules.py`: TavernGym rule sets (`full`, `simple`, `uncapped`). TavernGym has no
  per-turn action limit unless `turn_action_caps` is set; ours cap every turn at 50.
- `bgagent/policy.py`: `Advisor` - load a policy, give ranked actions for any saved game
  state. The live bridge will call this.
- `bgagent/tavernbench.py`: answers TavernBench's 23 decision scenarios with a policy (value
  head or constrained policy pick) and scores them with the benchmark's own code. Needs an
  AutoTavern checkout at the pinned commit (`--bench`, `bench_dir`, `TAVERNBENCH_DIR`).
- `bgagent/library.py`: the paper's library seats. Opponent seats that never recruit; each
  turn their board, hand, Tier and buff counters are replaced (BatchEnv save/load) with an
  entry matched to a target strength curve, on a difficulty ladder from the library's own
  median up to Firestone's top-1% curve. `opponents=library`, `library_path`, `exam_path`.
- `bgagent/replay.py`: real games. Reads HSReplay XML (Firestone's Perfect Games replays): at
  each combat the main player's board as it entered and the opponent's board as the game
  created it (stats read when that block ends: minions are created with base stats), Tiers,
  heroes. `EntryBuilder` turns boards into library entries through TavernGym's debug
  commands. Player names and account ids are never read (names are blanked before parsing).
- `bgagent/bootcamp.py`: the levelling bootcamp. While a learner seat is below a (turn, tier)
  schedule and can afford Upgrade, it upgrades instead of acting; forced steps train the value
  head, not the policy. `level_curve`, `level_force`, `level_force_decay` (the hand-back).
  Evaluations use the same forcing.
- `bgagent/dashboard.py` + `dashboard.html`: the live training dashboard. A standard-library web
  server that reads runs/*/metrics.jsonl (newest attempt per iteration, like status.py) and one
  self-contained page (hand-drawn SVG charts, no CDN) that compares two runs and redraws every 30 s.
- `bgagent/human.py`: Firestone reference curves (warband strength by turn per MMR bracket),
  read from `data/firestone` (fetched by `scripts/fetch_firestone.py`, not in git).
- `scripts/`: train.py, evaluate.py, advise.py, status.py, tavernbench.py, fetch_firestone.py,
  build_library.py; setup and start scripts for Windows (`setup_windows.ps1`, `start_3070.cmd`)
  and WSL2 (`setup_wsl.sh`, `start_3070.sh`).

## Commands
- Tests: `pytest -q` (fast; run before every commit)
- Smoke run: `python scripts/train.py --preset smoke --run runs/smoke`
- Real run: `python scripts/train.py --preset 3070 --run runs/<name>` (resume = same command)
- Progress: `python scripts/status.py runs/<name>`
- Dashboard: `scripts\dashboard.cmd` (double-click) or `python scripts/dashboard.py` →
  http://localhost:8050/#run:compare-run; `--lan` for a phone on the same Wi-Fi
- Evaluate: `python scripts/evaluate.py runs/<name>/policy_00100.pt --games 2000`
- Advice demo: `python scripts/advise.py runs/<name>/latest.pt --demo --turn 6`
- TavernBench: `python scripts/tavernbench.py runs/<name>/policy_00100.pt --bench ~/AutoTavern/tavernbench`
- Head to head: `python scripts/evaluate.py runs/b/policy_00200.pt --vs runs/a/policy_00100.pt`
- Is it learning? `python scripts/diagnose.py random runs/<name>/policy_00050.pt`
  (placement vs bots lags far behind; warband size and tier by turn move first)
- Human data: `python scripts/fetch_firestone.py` (cached 12 h; data/firestone)
- Real games: `python scripts/fetch_perfect_games.py` (Firestone's Perfect Games: first-place
  replays, ~1 MB each, one request per 2 s; data/firestone/perfect)
- Real-board libraries: `python scripts/build_real_library.py` (data/library/real-train and
  real-exam, split by game; `--min-rank 6814` for top-25% games)
- Library: `python scripts/build_library.py --out data/library/train.jsonl.gz --games 1000
  --seed 1 --policy runs/full-01/policy_00500.pt:0,1,2,3` (exam: other seed and policies)

## Conventions
- Training from scratch against scripted bots alone gives no signal (a fresh policy places
  8th nearly every game). Default is `opponents=mixed`: 4 learner seats + 4 scripted bots.
- Default policy head is factored (action type, then target). The flat head drowns in the
  42 legal minion moves of a full board. See NOTES.md for the numbers.
- Train on `rules=full`. The simplified-rules curriculum stage lost badly when evaluated
  on the real game (it never saw trinkets or Dark Gifts). Evaluation is always on `full`.
- Beating the scripted bots is not playing well: the 2.94-vs-bots policy is near chance on
  TavernBench. Report both.
- Placement: 1 best, 8 worst. Reward: 1st = +1, 8th = -1, linear (TavernGym's).
- TavernGym is pinned in pyproject.toml. Its card data is patch 36.6.3; live patches drift.
- The 3070 run is native Windows (docs/SETUP_WINDOWS.md): `scripts\start_3070.cmd` starts or
  resumes it in its own window. GPU memory: Windows spills into system RAM instead of failing
  (5x slower); training caps PyTorch at 80% of VRAM and the 3070 uses micro_batch=512.
- Never delete a run folder without asking.
- Human data (Firestone and HSReplay stats, Cranny's own game logs) only calibrates opponents and
  evaluation, as the paper did. Never train the policy on human moves (no imitation learning):
  Cranny's decision, to stay with the paper's method. One exception he approved (2026-11-02):
  the levelling bootcamp may force Upgrade on real winners' average Tier-by-turn schedule
  (`bootcamp.WINNERS`) as a temporary training aid. Ask before any other human-derived forcing.
- Firestone data: its developer, 2026-11-01 (relayed by Cranny): "So long as the data is not
  used commercially, all data available on the application itself can be fully utilized by
  any other developer". Its stats endpoints (`static.zerotoheroes.com/api/bgs/...`) are the files
  the app loads. Non-commercial only: selling or paywalling the tool would need new permission.
  Keep request rates human-like, cache downloads under `data/` (not in git), never redistribute
  the raw files, and credit Firestone.
