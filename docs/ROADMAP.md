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
1. `--preset small` (full rules, ~2.9M params, 4 learners + 4 bots): hours. Proves learning
   end to end; tune lr / entropy here because it's cheap. (A simplified-rules first stage
   did worse in the CPU tests: see NOTES.md.)
2. `--preset 3070` (full rules, ~16.9M params, 4 learners + 4 bots): the multi-day run.
3. Track `eval vs bots` (random = 7.95, scripted bot = 4.50). The tiny CPU model already
   passes 4.5 (NOTES.md), so the bots stop being a useful yardstick soon. Next yardsticks:
   past snapshots of our own policy (league) and TavernBench (phase 3). The bots are weak:
   TavernGym describes them as making games "varied, not strong".

## Phase 3: measure against the paper (TavernBench) - adapter done
`bgagent/tavernbench.py` / `scripts/tavernbench.py` answer the 23 decision scenarios (232 of
the 368 evaluation tasks) from the network alone and score them with TavernBench's own code.
Checked: the same options scored by TavernGym's combat preview get 100%, so the options are
built right. Training logs it with `bench_dir=...`. First results (NOTES.md, 2026-11-01): the
CPU models are near chance (0.24-0.31 vs chance 0.24; TavernAgent 0.89, best LLM 0.49).
Still open: the seven combat-fact scenarios (B1-B6, R2), which need the simulator, not the
policy.

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
