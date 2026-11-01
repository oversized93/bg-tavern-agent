# Roadmap

Goal: on every Battlegrounds decision, show the moves a strong policy would make, with
probabilities, while Cranny plays. Never automate play.

## Phase 1: training pipeline (done)
- TavernGym observation layout verified against the engine on 9,600 observations.
- Entity transformer with pointer action head, value head and placement head.
- PPO over BatchEnv: whole-episode rollouts, GAE (gamma 1.0, lambda 0.995), bf16, gradient
  accumulation, atomic checkpoints with resume, league snapshots, eval vs scripted bots.
- `Advisor`: ranked recommendations for any saved game state.

## Phase 2: train on the 3070
1. `--preset small` (simple rules, ~2.9M params, pure self-play): hours. Proves learning
   end to end; tune lr / entropy here because it's cheap.
2. `--preset 3070` (full rules, ~16.9M params, league): the multi-day run. Restarting a
   stage from a stage-1 checkpoint needs matching model size, so stage 2 starts fresh unless
   stage 1 used `model=full`.
3. Track `eval vs bots` (random = 7.95, scripted bot = 4.50). Beating 4.5 clearly is the
   first milestone; the paper's agents were then trained against stronger opponents.

## Phase 3: measure against the paper (TavernBench)
Adapter from TavernBench play tasks (`tavernbench/data/eval`, saved states via
`Game.from_json`) to `Advisor`, mapping its top action into each task's answer format, then
`tavernbench.report`. Gives a number directly comparable to TavernAgent's 91% and the LLMs'
44-63%.

## Phase 4: live bridge (needs real logs)
Power.log -> TavernGym game state -> `Advisor`.
- Card IDs: TavernGym's `cards.json` uses Blizzard card IDs and DBF IDs, the same ones
  Power.log prints.
- Target format: a `Game.state_json()` document (or edit a template game with `Game.debug`).
- Information limits match the client: opponent boards = last seen in combat, never hands.
- Must be estimated: the shared minion pool; some running counters (spells cast, etc.) are
  tallied from log history.
- Verification loop, per turn: (1) log -> state -> `Game.view()` matches the log field by
  field; (2) replaying the actions actually taken reproduces the next turn's state;
  (3) each real combat's outcome is plausible under a few thousand simulated combats, and
  deterministic effects match exactly. Every mismatch is a converter fix or a card-data fix.

## Phase 5: opponents from real games
The paper's strongest opponents ("library seats") were built from top-player statistics and
recalibrated to opponents met live. Not released. Substitute: boards seen in Cranny's own
games (Power.log, Firestone game-history export), turn by turn, plus league snapshots.

## Phase 6: patch drift
TavernGym is pinned to 36.6.3. After a balance patch, diff combats against Firestone's
continuously updated simulator (`@firestone-hs/simulate-bgs-battle`; check its license
before reusing code) to find stale cards, and update the card data.

## Phase 7: overlay
Show the top three recommendations and the expected placement next to the game window. A
separate always-on-top window that only reads; it never touches the game.
