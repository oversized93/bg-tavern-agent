# Research log

Newest entries at the bottom. Every claim gets a number and the command that produced it.

## 2026-10-31: pipeline built and checked

**TavernGym build.** AutoTavern commit `fb61201a` builds with Rust 1.97 using
`--ignore-rust-version` (it pins 1.98.1). On the 3070 machine, install 1.98.1 as the setup doc
says. Bot-only games: ~150 games/s on one busy CPU core here (the paper reports 190-280).

**Observation layout.** obs.rs writes floats as: global 53, quests 6, pending 9, board 7x31,
hand 10x40, shop 8x40, choices 16x40, opponents 7x9, opponents' last-seen boards 49x31 = 3,227.
Cards: hero, 2 powers, deity, pending source, quests 9, secrets 5, board 7, hand 10, shop 8,
choices 16, trinkets 8, opponent heroes 7, opponent boards 49 = 124. Checked on 9,600
observations from random play: board/hand slot counts read through these offsets match the
engine's own counters, and buy actions are never legal on empty shop slots. 0 mismatches.

**Model sizes** (`ModelConfig.preset`): tiny 0.40M, small 2.85M, medium 6.29M,
full 16.86M parameters. The paper's sizes were 3.5M / 8M / 18M; ours are close enough.

**Baselines vs seven scripted bots.** Random legal play: 7.94 avg placement over 200 games
(paper: 7.97). A scripted bot in a seat is 4.50 by symmetry.

**Finding: training from scratch against the bots gives no signal.** A fresh policy placed
7.95-7.99 for 8 straight iterations (8k decisions each) and the value loss collapsed to
0.001: it finishes 8th nearly every game, the reward is a constant -1, and the advantages
are noise. Fix: start in league / self-play, where opponents are as weak as the learner and
placement spreads over 1-8. The bots become an evaluation yardstick (every `eval_every`
iterations). The paper also began on simplified rules; `rules=simple` turns off trinkets and
Dark Gifts.

**Evaluation detail.** An early greedy policy can repeat a no-op (moving a minion back and
forth) until the 200-actions-per-turn cap, which makes greedy evaluation crawl. Training-time
evaluation samples from the policy; `scripts/evaluate.py` is greedy by default.

**First learning check (CPU, tiny model, simple rules, pure self-play, 8k decisions/iter,
lr 3e-4, 2 epochs).** Placement against the bots doesn't move this early (7.95 at it 10,
7.92 at it 20, 128 sampled games each): a weak policy is 8th whatever it does. Behaviour does
move. `scripts/diagnose.py`, 32 games each, warband size at End Turn:

| policy | turn 4 | turn 6 | turn 8 |
|---|---|---|---|
| random | 0.78 | 1.50 | 2.25 |
| it 10 (82k decisions) | 1.02 | 2.43 | 3.22 |
| it 20 (164k decisions) | 1.64 | 3.48 | 5.28 |

So the reward reaches the policy and it is learning to fill its board, after about 0.05% of
the paper's per-stage budget. Tier at turn 8 hasn't risen yet (2.6-2.9), and gold is still
left unspent; those are the next things to watch.

The same run kept going to iteration 140 (~1.2M decisions, ~35 min on 2 CPU cores). Placement
against the bots, 128 sampled games every 10 iterations:
7.95, 7.92, 7.56, 7.54, 7.67, 7.58, 7.66, 7.40, 7.22, 7.42, 7.27, 7.20, 7.35, 7.32 (±0.2).
Final eval at iteration 150: 7.06 ±0.24 (top-4 5%).

Behaviour at iteration 150 vs random (`diagnose.py`, 32 games): warband at turn 8 is 6.97
(random 2.25) and unspent gold at turn 8 is 1.24 (random 5.25), so it buys and fills the
board. But Tavern Tier at turn 8 is 2.31 (random 2.62): it has not learned to level, and it
takes 41 actions per turn (random 8.5), mostly free minion moves. Leveling is the obvious
next skill; watch tier-by-turn on the 3070 runs, and if it stays flat, consider a small
per-action cost on moves.

Self-play alone moves it off the floor, slowly. The 3070 runs use a 40x bigger model and
~50x more decisions per hour, so this is only a lower bound on the plumbing working.

## 2026-10-31 (evening): why it stalls, and the action-space fix

Three CPU runs, same settings as e0 except one change each, run in parallel (stopped early
once the answer was clear). Eval vs bots, 128 sampled games every 10 iterations:

| run | it 10 | 20 | 30 | 40 | 50 | notes |
|---|---|---|---|---|---|---|
| e0 baseline (lr 3e-4) | 7.95 | 7.92 | 7.56 | 7.54 | 7.67 | continued: 7.06, 7.15, 7.10, 7.03 at it 160-190 |
| e1 lr 1e-3 | 7.80 | 7.78 | 7.40 | 7.35 | 7.39 | it 54: 35 actions/turn, 73% moves, tier 3.10 at turn 8 |
| e2 move_cost 0.0005 | 7.95 | 7.81 | 7.58 | 7.66 | 7.59 | it 57: 27 actions/turn, 68% moves |

e0 continued past 150 plateaued at ~7.05-7.15 while its minion shuffling grew to 88-116
actions per turn, 82-88% of them moves. e1's higher learning rate is slightly ahead (within
noise). A small move cost trims the shuffling but doesn't fix it.

**Diagnosis.** With a full board there are 42 legal moves (7 x 6) against a handful of
other actions, so a fresh policy puts most of its probability on moves, and the habit
compounds. That breaks credit assignment: the reward comes only at game end, and with
lambda = 0.995 the advantage reaches ~200 decisions back, which is two turns at 100 actions
per turn. Leveling on turn 4 can't be connected to the result.

**Fix: factored action head.** The policy now picks an action type first (end turn,
refresh, freeze, upgrade, hero power, Dark Gift, cancel, buy, sell, activate, choose,
play, magnetize, move), then the target. Same 245-way categorical, reparameterised, so PPO
is unchanged. Test: a fresh policy on a full board puts >50% on moves with the flat head
and <30% with the factored head (in practice ~1.5% of decisions are moves in early
rollouts). Also added opponents=mixed: learner seats plus TavernGym's scripted bots.

**Result: factored head + mixed opponents.** Same CPU settings, lr 1e-3, 100 iterations.
Eval vs bots, 128 sampled games every 10 iterations (±0.2-0.3):

| run | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|
| e1 flat head, self-play | 7.80 | 7.78 | 7.40 | 7.35 | 7.39 | | | | | |
| f1 factored, self-play | 7.43 | 7.27 | 7.15 | 7.22 | 7.02 | 6.89 | 6.93 | 6.79 | 6.92 | 7.00 |
| f2 factored, mixed (4 learners + 4 bots) | 7.57 | 6.99 | 6.41 | 6.40 | 6.41 | 6.38 | 5.94 | 5.92 | 5.82 | 5.63 |

f2 at it 100, 512 sampled games: 5.81 ±0.16, top-4 22.5%. f1 levels (tier 3.5-3.7 at turn 8)
but plateaus near 6.9; f2 fills the board and levels less (tier 1.9-2.3 at turn 8) and is
still improving at it 100. Presets now default to factored + mixed.

**Greedy loops.** Taking f2's single top action every time (what a recommendation shows)
loops: Freeze toggled again and again, and Move 1->0 repeated (a swap back and forth),
each turn running to the 200-action cap. 6,000 greedy decisions: 97.5% moves, 5 End Turns.
Fix: `bgagent/filters.py`, no Freeze while frozen and at most 6 moves per turn, applied the
same way in training, evaluation, diagnostics and advice. Applied at inference only to f2
(trained without it): greedy 5.78 ±0.21 over 256 games, matching sampled play.

## 2026-10-31 (night): train on the full rules; TavernGym has no turn limit

Same CPU settings as f2 (tiny model, factored head, 4 learners + 4 bots, lr 1e-3, 8k
decisions/iter, 100 iterations), now trained with the action filter. Eval vs bots, 128
sampled games every 10 iterations, always on the full rules (TavernGym defaults):

| run | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|
| f2 simple rules, no filter | 7.57 | 6.99 | 6.41 | 6.40 | 6.41 | 6.38 | 5.94 | 5.92 | 5.82 | 5.63 |
| f3 simple rules, filter | 7.68 | 6.86 | 6.70 | 6.74 | 6.80 | 6.66 | 6.59 | 6.91 | 7.16 | 6.90 |
| f4 full rules, filter | 6.83 | 5.77 | 5.34 | 4.96 | 4.67 | 4.55 | 3.98 | 3.66 | 3.90 | 3.78 |

**Full rules win by a mile.** f4 passes a scripted bot's 4.50 at iteration 70 and ends at
3.78 ±0.34 (top-1 13%, top-4 66%). In training it averages 3.39 in lobbies with 3 copies of
itself and 4 bots; 18 actions per turn, 27% of them moves; 6.9 minions and tier 3.4 at
turn 8. The simplified rules (no trinkets, no Dark Gifts) were also the rules every earlier
run trained on, and its evaluations were on the full game: the policy met trinket and Dark
Gift choices for the first time at test time. So f2's 5.63 understated what the setup could
do, and the curriculum stage cost more than it saved at this scale. Presets now train on
full rules, and evaluation says which rules it used.

The scripted bots are weak (TavernGym: "varied, not strong"), so past ~4 they stop being a
useful yardstick. Next ones: our own past snapshots and TavernBench.

**f3 found a new loop.** With moves capped, it learned to play a minion with a targeted
Battlecry (or start a targeted Hero Power) and Cancel, over and over: 400-1,071 actions
per turn by iteration 98-100; 60,000 sampled decisions in 5 games were 49% Cancel and 44%
Play. Its placement fell from 6.59 to 6.90-7.16 as the loop grew.

**TavernGym has no per-turn action limit.** `max_actions_per_turn` = 200 is only the
ceiling for `turn_action_caps`, which is empty by default (`config.rs`: "a policy without a
cap can act forever: whoever drives one sets this"). Earlier notes here that said turns ran
"to the 200-action cap" were wrong. Checked: a driver that always moves a minion when it
can spent 2,867 actions in one turn under the defaults, and 50 with our cap. Fixes, both
tested:
- `bgagent/rules.py`: every rule set caps each turn at 50 actions (past it, only End
  Turn). `rules=uncapped` is the old behaviour.
- `filters.py`: at most 2 Cancels per turn (`cancel_cap`, saved with the model).

A greedy evaluation of f4 under the old, uncapped rules ran for an hour on 512 games
without finishing (killed); 16 greedy games on another seed finished normally (15.3
actions per turn), so a few games must have hit an endless loop that the cap now ends.

**f5: f4 plus the caps, 150 iterations.** Eval vs bots (128 sampled games, capped full
rules): 7.09, 6.27, 5.91, 5.19, 4.84, 5.02, 4.84, 4.12, 4.34, 4.12, 4.08, 4.16, 3.91,
3.59, 4.08. Same curve as f4 within noise (a different random path); it flattens around
3.6-4.1 from iteration 80. 16-18 actions per turn, 30% moves. At this model size and
learning rate the bots are close to exhausted as opponents.

**Greedy (the top recommendation every time), 512 games, capped full rules, cancel cap 2:**

| policy | avg placement | top-1 | top-4 |
|---|---|---|---|
| random | 7.94 | | |
| f2 it 100 (simple rules, filter at inference only) | 5.78 ±0.21 (256 games, uncapped) | | |
| f4 it 100 (cancel cap at inference only) | 3.35 ±0.16 | 19.1% | 72.9% |
| f5 it 150 | 3.58 ±0.16 | 11.7% | 72.3% |

`python scripts/evaluate.py runs/<run>/policy_XXXXX.pt --games 512 --threads 1 --device cpu`
(f4 with `--cancel-cap 2`). Greedy beats sampled by 0.3-0.5 places, so the single top
recommendation is the policy's best play, which is what the overlay will show. Both 512-game
runs finished in minutes once turns were capped.

## 2026-11-01: 100 more iterations from f4; harder opponents vs control

Both continue f4 (it 100, 3.35 greedy vs bots) to it 200 with the same settings (tiny model,
lr 1e-3, 8k decisions/iter), plus the cancel cap and 50-action turn cap (`runs/launch-g.sh`):
- g1 control: 4 learners + 4 scripted bots, as before.
- g2 hybrid: 4 learners + 2 seats played by a past snapshot (league of the last 4, seeded
  with f4's it 70-100) + 2 bots. `opponents=hybrid`, new.

Eval vs bots, 128 sampled games every 10 iterations (it 110-200):

| run | 110 | 120 | 130 | 140 | 150 | 160 | 170 | 180 | 190 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|
| g1 control | 3.83 | 3.69 | 3.38 | 3.66 | 3.34 | 3.35 | 3.07 | 3.12 | 3.01 | 3.09 |
| g2 hybrid | 3.95 | 3.73 | 3.22 | 3.34 | 3.41 | 3.30 | 3.14 | 3.15 | 3.08 | 3.10 |

Greedy, 512 games, capped full rules. "vs f4" puts seven copies of f4 it 100 around the
tested policy (`--vs`); 4.5 = equal strength. Check: f4 vs itself, 4.72 ±0.50 (64 sampled).

| policy | vs bots | top-1 | top-4 | vs f4 it 100 | top-1 | top-4 |
|---|---|---|---|---|---|---|
| f4 it 100 | 3.35 ±0.16 | 19.1% | 72.9% | (4.5) | | |
| g1 it 200 | 2.94 ±0.15 | 21.9% | 80.9% | 3.80 ±0.20 | 19.1% | 61.7% |
| g2 it 200 | 3.01 ±0.15 | 20.3% | 81.4% | 3.97 ±0.20 | 17.4% | 61.9% |

Head to head between them: g1 among seven g2s 4.48 ±0.19; g2 among seven g1s 4.44 ±0.20.

- **More training still pays.** 100 more iterations made it clearly stronger than f4: 0.4
  places better against the bots, and 0.5-0.7 better than 4.5 against f4 itself.
- **Harder opponents made no difference yet.** g1 and g2 are equal head to head. g2's
  rollouts are ~35% slower (1,300-1,400 vs 2,000-2,250 decisions/s) because the snapshot
  seats need the network too. Mixed (4 + 4 bots) stays the default; revisit hybrid when
  the bots really are exhausted.
- Behaviour drift, it 100 -> 200 (training rollouts): actions per turn 18 -> 21-23, entropy
  1.79 -> 1.46-1.54, tier at turn 8 3.39 -> 2.94 (g1) / 3.19 (g2), warband 6.8-6.9.
- The bots aren't exhausted: it improved by 0.4 against them in these 100 iterations.

## 2026-11-01: TavernBench, the outside check

`bgagent/tavernbench.py` answers TavernBench's 23 decision scenarios (232 of the 368
evaluation tasks: hero, trinket and discover picks, buys, arrangement, action order, targets,
turn plans) from the network alone, and scores them with the benchmark's own `report` code
(a category = mean of its scenarios; a total = mean over scenarios). Two ways to pick:
- value: apply each option to a copy of the game, take the best value-head position.
- policy: what the policy does with only the task's options open (its most likely next action
  among those that keep an option alive). Hero choice and arrangement fall back to value.

**The adapter is right.** Scoring the same option positions with TavernGym's combat preview
(512 combats each) instead of the network gets 1.00 on all 18 next-combat scenarios (D3, D5,
E3, O1-5, P1-5, T1-5), 104/104 option sets applied. `tests/test_core.py` keeps a 5-task version
of this check (skipped without an AutoTavern checkout).

**Results** (`python scripts/tavernbench.py <policy> --bench <AutoTavern>/tavernbench`):

| system | decision (23) | select | econ | position | order | targets | strategy |
|---|---|---|---|---|---|---|---|
| chance | 0.24 | 0.30 | 0.12 | 0.20 | 0.21 | 0.25 | 0.25 |
| TavernAgent (paper) | 0.89 | 0.64 | 1.00 | 0.97 | 1.00 | 0.97 | 0.75 |
| Claude Sonnet 5.5 | 0.49 | 0.39 | 0.50 | 0.45 | 0.68 | 0.53 | 0.29 |
| Claude Opus 5.5 | 0.46 | 0.41 | 0.50 | 0.33 | 0.60 | 0.53 | 0.38 |
| Jev | 0.30 | 0.40 | 0.12 | 0.25 | 0.20 | 0.45 | 0.17 |
| ours untrained, value / policy | 0.16 / 0.16 | | | | | | |
| f4 it 100, value / policy | 0.31 / 0.27 | 0.30 / 0.31 | 0.12 / 0.12 | 0.17 / 0.17 | 0.50 / 0.35 | 0.35 / 0.25 | 0.21 / 0.36 |
| g1 it 200, value / policy | 0.27 / 0.24 | 0.31 / 0.30 | 0.12 / 0.00 | 0.15 / 0.15 | 0.45 / 0.25 | 0.23 / 0.28 | 0.16 / 0.34 |
| g2 it 200, value / policy | 0.31 / 0.30 | 0.39 / 0.35 | 0.12 / 0.25 | 0.15 / 0.15 | 0.53 / 0.30 | 0.25 / 0.38 | 0.18 / 0.38 |

Most scenarios have 8 tasks, so per-category numbers are very noisy; the 23-scenario mean
is about ±0.06.

- **Our CPU models are near chance on the decisions TavernBench tests** (0.24-0.31 vs 0.24),
  though they beat the scripted bots (2.94). They learned broad habits (fill the board,
  spend the gold, level) and not card-specific tactics. Beating the bots is a weak proxy
  for good play; TavernBench is now the main yardstick, and training logs it
  (`bench_dir=...`, ~20 s for the tiny model on CPU).
- The value head barely separates options within a turn: typically 0.01-0.1 expected places
  between the best and worst option.
- 18 of the 23 scenarios ask which option wins the next combat. The combat simulator answers
  those perfectly. As a benchmark score that is circular, since it's how the labels were made.
  As a tool it's legitimate: the overlay can simulate the next fight against the opponent's
  last-seen board the way Bob's Buddy does. The other five (hero, trinket, discover for final
  placement, turn plans) depend on long-term judgment that only training provides.
- Hero (D1) and trinket (D2) labels are top human players' average placements. Even
  TavernAgent gets 0.25 / 0.12 there.

## 2026-11-01: next-combat auxiliary head (h1) - no help

h1 = f5's recipe (tiny model, full rules, 4 learners + 4 bots, filter + caps, lr 1e-3, seed 0,
150 iterations) plus a head predicting the combat right after each turn (lost it, damage taken),
`combat_coef=0.5`. Labels come from the seat's own Health + Armor across turns. The head
learns (its loss falls from 0.78 to 0.61), but:

| it | 10 | 30 | 50 | 70 | 90 | 110 | 130 | 150 |
|---|---|---|---|---|---|---|---|---|
| f5 vs bots | 7.09 | 5.91 | 4.84 | 4.84 | 4.34 | 4.08 | 3.91 | 4.08 |
| h1 vs bots | 7.06 | 6.13 | 5.20 | 4.49 | 4.20 | 4.13 | 3.73 | 4.13 |
| f5 TavernBench value / policy | 0.30 / 0.29 | 0.40 / 0.29 | 0.31 / 0.26 | 0.35 / 0.25 | 0.32 / 0.25 | 0.36 / 0.25 | 0.36 / 0.32 | 0.35 / 0.29 |
| h1 TavernBench value / policy | 0.32 / 0.19 | 0.36 / 0.24 | 0.29 / 0.26 | 0.27 / 0.22 | 0.27 / 0.18 | 0.29 / 0.17 | 0.32 / 0.22 | 0.35 / 0.25 |

Mean over it 80-150 (8 evaluations): TavernBench value 0.35 (f5) vs 0.31 (h1), policy 0.28 vs
0.20 (chance 0.24). Against the bots they're the same within noise. The head is not in the
3070 recipe (`combat_coef` stays 0). f5's TavernBench curve was scored afterwards on its
snapshots (`runs/f5-caps/bench_curve.jsonl`); it is flat from it 20 on while its placement vs
the bots goes from 6.27 to 4.08.

## 2026-11-01: small model (h2) vs tiny (f5) at equal iterations

h2 = f5's recipe with `model=small` (2.9M), lr 5e-4 (f5: tiny, lr 1e-3), micro_batch 512,
torch_threads 2. The first attempt was killed for memory at iteration 8 while h1 ran beside it
(~3.4 GB; the container has 8 GB). Resumed alone, it reached iteration 85 before the 2-hour
background limit stopped it (~80 s per iteration). Checkpoints to 80 are in runs/h2-small.

| it | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 |
|---|---|---|---|---|---|---|---|---|
| f5 vs bots (128 sampled) | 7.09 | 6.27 | 5.91 | 5.19 | 4.84 | 5.02 | 4.84 | 4.13 |
| h2 vs bots (128 sampled) | 7.45 | 6.15 | 5.52 | 5.16 | 4.80 | 4.41 | 4.86 | 4.41 |
| f5 TavernBench value / policy | 0.30 / 0.29 | 0.36 / 0.23 | 0.40 / 0.29 | 0.34 / 0.25 | 0.31 / 0.26 | 0.33 / 0.24 | 0.35 / 0.25 | 0.37 / 0.26 |
| h2 TavernBench value / policy | 0.28 / 0.27 | 0.32 / 0.33 | 0.30 / 0.32 | 0.36 / 0.32 | 0.37 / 0.32 | 0.32 / 0.33 | 0.36 / 0.36 | 0.39 / 0.34 |

Mean over it 40-80 (5 evaluations each):
- vs bots: 4.73 (h2) vs 4.80 (f5), each ±0.35: the same.
- TavernBench policy pick: 0.33 vs 0.25 (chance 0.24), h2 higher at all 5. The gain is in
  economy (0.28 vs 0.08), positioning (0.34 vs 0.18) and targets (0.33 vs 0.23).
- TavernBench value pick: 0.36 vs 0.34, mixed: positioning and selection up, economy
  (0.05 vs 0.20) and targets (0.28 vs 0.41) down.
- Positioning is scored by the value head in both modes, so it is one result, not two.

So the bigger model is not faster against the bots per iteration, but its policy picks are
above chance on TavernBench where the tiny model's are not. Confounded with the learning rate
(5e-4 vs 1e-3) and only 80 iterations. It supports the full model in the 3070 recipe; nothing
changes for full-01. full-01 (below; 8x more decisions per iteration, so iterations don't line
up) has policy picks at 0.28-0.35 through it 130: the same band as h2 (0.32-0.36 from it 20),
above f5 (0.23-0.29 through it 120). Every model size so far gets policy picks in a flat band from early on
instead of rising with training.

## 2026-11-01: native Windows on the 3070

The 3070 machine has no WSL, no Visual Studio C++ tools and no Windows SDK, and installing
any of them needs admin rights. So TavernGym is built natively with Rust's **GNU** toolchain
(user folder, no admin; its bundled linker needs no MSVC). One snag: pyo3 links Python with
a raw-dylib import, for which rustc calls `dlltool`. The toolchain's bundled GNU dlltool
fails ("CreateProcess": no assembler shipped), so we use LLVM's dlltool instead: Rust's
`llvm-tools` component, `llvm-ar.exe` copied as `dlltool.exe`. Build: 2 min. All 16 tests
pass on Windows, including the TavernBench oracle check, so the Windows build plays the
same game. `scripts/setup_windows.ps1`, `docs/SETUP_WINDOWS.md`.

Speed: random play 11,040 decisions/s (64 games, 8 threads). PyTorch 2.14.1+cu130 on the
3070, driver 596.49.

**GPU memory spills instead of failing.** The first full-size run (micro_batch 2048) used
7.4 GB of VRAM plus 5.8 GB of shared system RAM (`\GPU Process Memory(*)\Shared Usage`), and
iteration 2's update took 237 s. The NVIDIA driver on Windows falls back to system memory
rather than raising out-of-memory, so the memory check that "passed" at 2048 had been
spilling all along. Fix: `gpu_mem_fraction=0.8` caps PyTorch at 80% of VRAM so an oversized
batch raises out-of-memory instead. Measured under the cap, one iteration of 16k decisions:

| micro_batch | peak reserved | update |
|---|---|---|
| 1024 | 6.39 GB (of the 6.4 GB cap) | 6 s |
| 512 | 3.39 GB | 6 s |

512 is the 3070 preset now. Full-size iteration (65,536 decisions): 12 s rollout + 22 s
update = 34 s, 3.7 GB VRAM, ~70 MB shared. The aborted run is `runs/full-01-aborted-sysmem`.

**runs/full-01 started 2026-11-01 10:21** on the 3070: `--preset 3070` (full model 16.9M,
4 learners + 4 bots, full rules, lr 3e-4 -> 1e-4 over 4,000 iterations), TavernBench every 10.
Capped at 500 iterations for a first review (`lr_schedule_iters=4000` keeps the schedule on the
4,000 path, so extending later doesn't jump); stopped at the it-25 checkpoint and resumed.
First evaluations (bots: 256 sampled games; TavernBench value / policy): it 10 7.73 ±0.10,
0.30 / 0.30; it 20 5.95 ±0.24, 0.33 / 0.28. The tiny CPU model took 50+ iterations to reach 6.

**full-01 check-in at it 140** (~3 h of training, ~9M decisions):
- vs bots (256 sampled games): it 60 3.72, 80 3.24, 100 3.23, 110 3.12, 120 3.12, 130 3.49
  (±0.2 each). Passed a bot's 4.50 before it 60 (f4 needed 70 CPU iterations); flattening
  near 3.1-3.5 since it 80. Sampled, so not comparable to g1's 2.94 greedy.
- TavernBench value 0.30 -> 0.40 (it 130, its best), policy flat at 0.28-0.35 the whole way.
  Positioning is the one category clearly rising: 0.28 -> 0.45 (one result: arrangement
  is scored by the value head in both modes). Policy
  ordering fell 0.42-0.50 (it 30-50) -> 0.17-0.23 (it 100-130) while value ordering stayed
  0.45-0.65: the policy's card order is drifting away from what its own value head prefers.
  Economy and strategy stay near 0-0.25 (few tasks each, so noisy).
- Behaviour (training rollouts): 21 actions/turn, 22% of them moves (11% at it 30, ~4.7
  moves a turn of the 6 allowed); 6.8 minions at turn 8, but **tier 2.6 at turn 8**: lower
  than f4 (3.4) and g1 (2.94). Each run levels less the longer it trains: flooding the board
  at low tier is what beats these bots. Not a bug, but real lobbies punish it; watch whether
  self-play pulls it back up by it 500, and if not, try hybrid opponents.
- Health: kl 0.005-0.009, entropy 1.45-1.50, 0 early stops, 4.5 GB VRAM, 66 C, 34 s/iter.

**full-01 crashed at it 268 (13:16): one huge minion made every weight NaN.** The log shows
`ppo.py:198: RuntimeWarning: overflow encountered in cast`, then `kl nan ent nan` for it 268,
then a CUDA assert in `torch.multinomial` on the next rollout. TavernGym encodes stats
linearly (attack / 10, health / 10, no ceiling) and rollouts store observations as float16
(max 65504), so a minion past 655,040 attack or health was stored as inf. The network saw the
real value while acting (float32) but inf while training, the loss went NaN, and one optimizer
step made the whole model NaN. Scale of the tail: in 256 training-like games with the it-250
policy, the largest stat was ~8,400 (feature 837); this took ~17M decisions to hit once.
Fix: the model clamps its input to ±65504 and rollouts clip to the same before storing (so
acting and training see the same value, and nothing below 65504 changes); a second guard skips
any optimizer step whose gradient norm isn't finite, logs `skipped`, and status.py warns.
latest.pt (it 265) was clean (no non-finite tensors) and the run resumed from it at 15:4x;
it 266-268 were rerun (status.py keeps the newest row per iteration).

## 2026-11-01: where our recipe departs from the paper's

The repo releases TavernGym and TavernBench only ("The trained TavernAgent, its weights and
training code ... are not included"), so the paper (arXiv 2610.33845, HTML v1) is the only
source for the method. Checked against it:
- **Opponents: the paper never used self-play.** Section 6: "Rather than self-play, the learner
  faces seven library seats." Appendix C: libraries "were first built from statistics of top
  human players"; "each turn a library seat takes an entry that matches a target strength
  curve"; later "recalibrated to the opponents the agent actually met on the client". Ours
  (4 copies of itself + 4 scripted bots) is our own choice, and so is a league of snapshots.
  Library seats would also explain why TavernAgent levels and ours doesn't: a seat that has a
  top player's turn-8 board beats a tier-2 board flood.
- **TavernBench combat and short-term answers use the simulator.** Section 7: "it applies each
  option and simulates 256 combats after it (one-step lookahead)." So TavernAgent's 0.89 is
  not network-only on those scenarios. Our combat-preview check (512 combats per option) gets
  1.00 on the 18 next-combat scenarios; our network-only scores there are a stricter test than
  the paper's. The remaining decisions (hero, trinket, the played-out D4/N1/N2) are the
  network's job in both.
- Live play: "the policy decides every operation, and a human presses the shortcut key the AI
  names"; the paper doesn't say whether the lookahead was used live.
- Scale: later stages are 300 iterations of ~1.05M decisions (~310M decisions, 570k games), lr
  6e-6 -> 5e-6, entropy 0.002, aux placement 0.1, value 0.5, target KL 0.05, 18M parameters.
  full-01 at it 268 has ~17.6M decisions in total.

## 2026-11-01: Firestone's public stats, and how far our agent is from top players

Firestone's app loads its Battlegrounds meta stats from static files (from its open-source code,
`libs/battlegrounds/...`): `static.zerotoheroes.com/api/bgs/hero-stats/mmr-{100,50,25,10,1}/
{last-patch,past-three,past-seven,all-time}/overview-from-hourly.gz.json`, plus `card-stats`
(same MMR folders), `trinket-stats`, `quest-stats`, `comp-stats` and an MMR-percentile table.
Its developer, asked by Cranny: "So long as the data is not used commercially, all data
available on the application itself can be fully utilized by any other developer". Its ToS is
a personal, non-commercial license. One sample of each, last patch:
- hero-stats, top 1% MMR: 107 heroes, 22,840 player-games. Per hero: average position,
  placement distribution, pick rate, tribe impact, and per turn `warbandStats` (average warband
  total stats) and `combatWinrate`. The paper's "warband strength by turn" is right there.
- card-stats, top 1%: 708 cards, 2.2M plays, each with plays and average placement by turn.
  The Tavern upgrade buttons are listed but barely counted (1-2 plays), so the leveling curve
  is not here; HSReplay Tier7's "When to Tavern Up" has it.
- trinket-stats: 243 trinkets, pick rate and placement at each MMR percentile.
- Every hero and trinket, and 702 of 708 cards, are in TavernGym's catalog (36.6.3.253932); the
  6 missing are the Coin and Tavern buttons. The live pool matches the simulator.

**Warband total stats by turn, top-1% humans (Firestone, hero-weighted) vs full-01 it 475**
(128 sampled games vs bots, at each End Turn; scratchpad script `warband_curve.py`):

| turn | 2 | 4 | 6 | 7 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|---|---|---|
| top 1% | 5.6 | 31.5 | 91.1 | 167.2 | 328.2 | 1,792.6 | 5,951.3 | 8,276.9 |
| full-01 | 11.3 | 45.3 | 116.6 | 160.3 | 207.8 | 339.1 | 508.8 | 863.8 |
| full-01 tier | 1.01 | 1.01 | 1.91 | 2.03 | 2.13 | 2.88 | 3.97 | 4.91 |

Turn 1 agrees (3.9 vs 4.1), so the turns line up. Ours is ahead through turn 6 (it stays on
tier 1 until turn 5 and spends everything on minions), even at 7, then falls behind: 0.63x at
turn 8, 0.19x at 10, 0.09x at 12. That is the board flood, measured against people: it buys an
early lead that the bots can't punish, and never builds the late game top players have. The
bots presumably sit even lower (not measured yet). This curve, per MMR bracket, is the target
strength curve for library opponents.

Decision (Cranny): no training on human moves; human data calibrates opponents and evaluation
only, as in the paper.

## 2026-11-01: full-01 at 500 iterations (~33M decisions), the review

- vs bots, training eval (256 sampled): 2.55 ±0.17, top-1 0.25, top-4 0.91 (its best).
- vs bots, greedy, 1,000 games: **2.88 ±0.09**, top-1 0.20, top-4 0.86. Same as g1's 2.94
  (tiny model, CPU, it 200): 8x more decisions per iteration and a 40x bigger model bought
  nothing measurable against the bots. Sampled beats greedy here (2.55 vs 2.88).
- Head to head (greedy, 512 games each): it 500 among seven it-250s **3.78 ±0.19**; among seven
  it-100s **3.71 ±0.18**. It did get stronger (0.7-0.8 places better than 4.5 against its own
  past), but no more against it 100 than against it 250.
- TavernBench at it 500: value 0.37, policy 0.33; flat since it 30 (policy 0.28-0.35).
- Warband stats by turn at it 500 match it 475 (turn 8 206, 10 350, 12 538 vs top 1% 328,
  1,793, 5,951); it levels a little sooner late (tier 3.36 at turn 10, was 2.88).
- One crash (it 268, float16 overflow, fixed above); otherwise kl 0.003-0.009, entropy
  1.50 -> 1.08, no early stops.

**Verdict:** the bots are spent as teachers. More iterations still improve it against itself,
but none of it shows up on TavernBench or against the human strength curve, and the model size
made no difference against the bots. Next is the paper's opponent: library seats calibrated to
Firestone's warband curve. TavernGym has no library-seat hook: `Game.debug` can edit boards
(not on the RL hot path), and `BatchEnv.save`/`load` round-trip a game's JSON, so library
seats mean either rewriting seat boards in the saved state between turns or a small
`set_board` addition to the engine. full-01 stays stopped at 500 until that is built.

## 2026-11-01: library opponents (the paper's method, our gap-filling)

Firestone stats for every bracket are in data/firestone (`scripts/fetch_firestone.py`, 18
files, last patch). Warband total stats by turn, hero-weighted (`bgagent/human.py`):

| turn | 6 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|
| all players (bracket 100) | 73 | 230 | 977 | 3,335 | 6,553 |
| top 50% (MMR >= 6,283) | 78 | 254 | 1,156 | 4,034 | 7,592 |
| top 25% (>= 6,814) | 82 | 274 | 1,299 | 4,507 | 7,949 |
| top 10% (>= 7,504) | 85 | 291 | 1,465 | 4,989 | 8,294 |
| top 1% (>= 9,947) | 91 | 328 | 1,793 | 5,951 | 8,277 |
| full-01 it 500 vs bots | 115 | 206 | 350 | 538 | 929 |

Even the all-players curve has ~3x our board at turn 10 and ~6x at 12, so the ladder starts
below the human curves.

`bgagent/library.py` implements library seats. What the paper states, we follow; where it is
silent (it publishes no code or library details), these are our choices:
- A library seat is a controlled slot that only presses End Turn (random legal option when a
  choice is forced). At each of its turn starts, its board, hand, Tier and the buff counters
  cards read in combat (`COPY_STATS`) are replaced with an entry via BatchEnv save/load
  (~2 ms each, 30-130 KB states). Hero, Health, damage and elimination are normal. Injected
  minions get fresh eids and pool_copies = 0 (TavernGym's own convention for non-pool copies).
  Checked: a seat given 7x 500/500 never lost Health or armor and finished 1st.
- Entries are harvested from simulated games (`scripts/build_library.py`): every alive seat's
  board at each turn start, labelled with the turn it fought. Per turn the 400 strongest are
  always kept, the rest reservoir-sampled up to 2,000.
- Pick: of 8 random entries for the turn, the closest to the target in log strength, its
  stats scaled onto the target. Target = ladder curve at difficulty d times a log-normal
  spread (sigma 0.25).
- Ladder: d = 0 is the library's own median; 1-5 are Firestone's brackets 100/50/25/10/1%;
  geometric interpolation between rungs. After each iteration d += 0.05 while the learner's
  training placement < 4.0, d -= 0.05 while > 5.0.
- Exam: a second library from other seeds and other snapshots, never trained on, evaluated at
  fixed d (0, 1, 5) next to the bots and TavernBench.
- New runs can start from a policy's weights (`init_from`), as the paper's stages did.

Smoke test (tiny fresh model, 3 iterations): opponents got exactly their targets (t8 204/204,
t12 543/543); a fresh policy places 8.00 against the library, as expected.

Libraries: train = 35,058 entries from 1,000 games (seed 1; seats 0-3 full-01 it 500, 4-5 it
250, 6-7 bots), 374 s; exam = 27,700 entries from 300 games (seed 2; it 475, it 375, bots).
Their strongest boards pass the top-1% mean at most turns (train turn 10: 2,681 vs 1,793;
turn 12: 7,239 vs 5,951), so even d = 5 needs little scaling. Medians track full-01 (turn 10:
385 train, 344 exam).

Speed (full model, 1 learner per game, 3070), decisions/s in the rollout:
- first version: 790-940 (rollout 70-86 s; mixed was 5,500-6,000). Profile: `prepare` 17.8 s
  of 38 s, 200k json decode calls (a per-minion deep copy through JSON), and games played on
  after the learner was out (library seats are controlled, so BatchEnv only restarts a game
  when they're done too).
- shallow minion copies, `end_finished` (marks the library seats done once the learner is
  out: the game restarts on the next step), orjson: 1,230-1,650 (rollout 41-53 s).
- What's left per 32k decisions: prepare ~13 s (state save/parse/load ~6 s, entry parse
  ~2.4 s), observe ~9 s (8 slots per learner instead of 1; only TavernGym can skip unready
  slots), GPU ~9 s. Python-side floor is ~60 s per iteration; a `set_board` / observe-ready
  change in TavernGym would be the next step if library training proves itself.

**runs/lib-01 started 2026-11-01 ~19:40**: init_from full-01 it 500, opponents=library,
learner_seats=(0,), n_games 256, lr 2e-4 -> 1e-4 over 4,000, 300 iterations, exam at d 0/1/5
(128 games each) next to bots and TavernBench every 10. First iterations: placement 5.17 at
d = 0 (library boards at their own median beat the learner: they never have an off turn),
~60 s per iteration (39-53 s rollout + 22 s update).

**lib-01 check-in at it 78** (21:10): no crashes, no skipped steps, 1,600-1,930 dec/s.
Training placement at d = 0, averaged per 10 iterations (256 games each, so ~±0.05):

| its | 1-10 | 11-20 | 21-30 | 31-40 | 41-50 | 51-60 | 61-70 | 71-79 |
|---|---|---|---|---|---|---|---|---|
| placement | 5.07 | 5.06 | 4.96 | 4.87 | 4.93 | 4.80 | 4.57 | 4.61 |
| top 4 | 0.38 | 0.38 | 0.40 | 0.41 | 0.39 | 0.42 | 0.47 | 0.45 |
| tier at t8 | 2.15 | 2.12 | 2.13 | 2.08 | 2.05 | 2.07 | 2.05 | 2.03 |

- It learns against its own median boards (5.07 -> 4.6), but d never left 0: the best single
  iteration was 4.22 and the climb needs < 4.0. By construction d = 0 is the learner's former
  self, so the up threshold asks it to beat that by half a place.
- Nothing outside moved: bots 2.48-2.75 (full-01 ended 2.55 on the same sampled eval),
  TavernBench value 0.31-0.35 / policy 0.28-0.32 (full-01: 0.37 / 0.33), exam d1 6.53-6.92,
  d5 7.34-7.50. Exam d0 4.34-4.94 at ±0.47 (128 games) is too noisy to see a 0.4 change.
- Tier at turn 8 keeps falling. At d = 0 the opponents are our own shape: stronger than
  humans early (turn 6: 123 vs 73), equal at turn 8 (225 vs 230), far weaker late (turn 10:
  396 vs 977). Flooding a tier-2 board is still the right answer to them; only climbing toward
  the human curves makes the late game punish it.

**Climb threshold 4.0 -> 4.5 at it 90** (Cranny's go-ahead): stopped after the it-90
checkpoint, resumed with `difficulty_up=4.5` (down stays 5.0), so d rises after any iteration
better than even. Expected to settle where the learner places ~4.75. Its 81-90 had averaged
4.58 at d = 0; it 90 eval: bots 2.34 ±0.16, exam d0 4.18 / d1 6.65 / d5 7.45, TavernBench
0.35 / 0.33. First climbs: it 93 (3.97) and it 94 (4.34) took d to 0.10. Turn-8 opponents barely
change with d (d0 225 vs d1 230 at turn 8); the ladder bites from turn 10 on.

## 2026-11-02: lib-01 at 300 iterations (~20M decisions), the review

Training, per 30 iterations (placement against the library, at the d of the time):

| its | 1-30 | 31-60 | 61-90 | 91-120 | 121-150 | 151-180 | 181-210 | 211-240 | 241-270 | 271-300 |
|---|---|---|---|---|---|---|---|---|---|---|
| placement | 5.03 | 4.87 | 4.57 | 4.72 | 4.68 | 4.69 | 4.75 | 4.76 | 4.74 | 4.77 |
| d at end | 0 | 0 | 0 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.20 | 0.20 |
| tier at t8 | 2.13 | 2.07 | 2.04 | 2.04 | 2.04 | 2.03 | 2.02 | 2.02 | 2.01 | 2.01 |
| entropy | 1.09 | 1.07 | 1.08 | 1.05 | 1.05 | 1.06 | 1.01 | 0.98 | 0.94 | 0.88 |

- The ladder settled where predicted (placement ~4.75) but stopped climbing at d ~0.2 from
  it ~190: after it 90, 26 iterations went up and 22 down. At d = 0.2 the turn-10 opponent is
  488 (all players: 977), turn 12 885 (3,335).
- vs bots, greedy, 1,000 games: **2.20 ±0.08**, top-1 0.35, top-4 0.95 (full-01 it 500:
  2.88 ±0.09). Training evals fell 2.54 -> ~2.2 by it 230, flat after.
- Head to head (greedy, 512 games): lib-01 it 300 among seven full-01 it-500s **3.41 ±0.19**.
- Exam (never trained on): d0 4.56 -> best 3.54-3.61 (its 180-230) -> 3.76 at 300; d1 flat the
  whole run (6.45-6.92); d5 flat (7.33-7.50).
- TavernBench flat: value 0.27-0.36, policy 0.26-0.33 (0.34 / 0.30 at it 300).
- Warband stats by turn (128 sampled games vs bots, scratchpad `warband_curve.py`):

| turn | 6 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|
| lib-01 it 300 | 122 | 238 | 479 | 743 | 1,191 |
| lib-01 tier | 1.83 | 2.02 | 2.02 | 2.10 | 3.08 |
| full-01 it 500 | 115 | 206 | 350 | 538 | 929 |
| full-01 tier | 1.94 | 2.18 | 3.36 | 4.74 | 5.73 |
| all players | 73 | 230 | 977 | 3,335 | 6,553 |

  Boards ~37% stronger at turns 10-12 than full-01, but by levelling *less*: it now sits on
  tier 2 from turn 7 to turn 12 and levels at 13 (the upgrade gets 1 Gold cheaper each turn it
  waits). Against people it is 0.49x at turn 10 and 0.22x at turn 12.

**Verdict:** library opponents beat the bots as teachers (2.88 -> 2.20 vs bots, 3.41 against
its own start), but the learner found a cheaper local optimum, not the human one: at d ~0.2
the late game never punishes a tier-2 board, and the ladder can't climb past where the learner
holds even. The exam at d1 never moved.

## 2026-11-02: Firestone's Perfect Games (real boards; missed until now)

The library above was built from our own agent's simulated boards scaled up to Firestone's
curves, so every opponent was a stretched copy of a tier-2 flood. Real boards were the first
thing on the list for Firestone and its permission covers them; only the stats files had been
pulled. Cranny caught it.

Firestone's app has a "Perfect Games" tab (`bgs-perfect-games.service.ts`):
`static.zerotoheroes.com/api/bgs/bgs-perfect-games.json` lists the newest 1,000 first-place
games, each with a `replayKey`; the replay viewer loads `xml.firestoneapp.com/<replayKey>`, a
zipped HSReplay XML of the whole game (~1 MB zipped, ~19 MB unzipped): every turn of the
winner's board, hand, Gold and Tier, every opponent's board at each combat, and the winner's
choices. The list's own `postMatchStats` holds only the final board.

The list on 2026-11-02: 1,000 games from 10-28 to 11-02, build 255790 (TavernGym's cards are
253932), player rating quartiles 5,522 / 6,230 / 6,766 (max 18,778; Firestone's top 50% starts
at 6,283). Every minion on the 1,000 final boards (6,816, 462 distinct) is in TavernGym's
catalog; 997 of 1,000 heroes. `scripts/fetch_perfect_games.py` merges the list into
data/firestone/perfect/games.jsonl (it keeps moving, so repeated runs grow it) and downloads each
replay, one request per 2 s.

Use, per the paper and Cranny's rule: opponent boards (library entries) and evaluation only,
never the winners' moves as training targets.

**Replay reader** (`bgagent/replay.py`): Battlegrounds plays turn k as game TURN 2k-1 (Tavern)
and 2k (combat). At each even TURN: the winner's minions in play (its board entering combat),
its Tier and hero; the opponent's board is the minions the game creates on the opponent side
(controller 9, zone PLAY) in the first top-level block of that TURN that creates any. A
minion is created with its card's base stats and the same block then sets the real ones (seen:
BG31_801_G created 2/2, set to 15/13), so the board is read when that block ends; first try,
reading at creation, gave turn-10 opponents 34 total stats. Later blocks are summons in the
fight. The opponent's Tier is on a hidden copy of its hero the game gives the winner (zone
SETASIDE, PLAYER_TECH_LEVEL). Player names are blanked before parsing: one replay had an
unescaped `name=""Lich King" Deryl"` that broke the XML. ~0.1 s per replay (8 processes).

Check against Firestone, first 300 replays (mean warband stats; Firestone's is per player-game):

| turn | 6 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|
| opponents in the replays | 63 | 175 | 712 | 3,604 | 11,696 |
| Firestone all players | 73 | 230 | 977 | 3,335 | 6,553 |
| winners in the replays | 86 | 282 | 1,806 | 11,595 | 32,089 |
| opponents' Tier (known 88-98%) | 3.62 | 4.70 | 5.24 | 5.55 | 5.74 |
| winners' Tier | 3.63 | 4.93 | 5.58 | 5.77 | 5.86 |

The opponents sit near the all-players mean, the winners well above it. Real players are on
Tier 4.7 at turn 8 and 5.2 at turn 10; lib-01 is on 2.0 and 2.0.

`scripts/build_real_library.py` on the first 536 replays: 0 failed, 42 s; 11,159 train entries
(425 games) and 2,865 exam (111 games, split by a hash of the reviewId), winners' and opponents'
boards; every minion is a TavernGym card. Median / mean by turn (train): t8 186 / 228, t10 799 /
1,256, t12 3,366 / 6,778.

All 1,000 replays (662 MB, 0 failed): **real-train 20,929 entries (798 games), real-exam 5,244
(202 games)**. Exam median by turn: t8 192, t10 876, t12 3,283.

## 2026-11-02: TavernBench, our AI vs the paper's LLMs

The release has the paper's answers for TavernAgent, ten LLMs and Jev (results/eval). Scored
with the benchmark's own code on the 23 decision scenarios our AI answers (`summarize`, run with
`python -X utf8`: the answer files aren't cp1252):

| system | decision | | system | decision |
|---|---|---|---|---|
| TavernAgent | 0.89 | | GLM 5.3 | 0.35 |
| GPT-6 Astra | 0.58 | | **ours (lib-01, value / policy)** | **0.34 / 0.30** |
| Claude Sonnet 5.5 | 0.49 | | Intern S2 preview | 0.31 |
| Claude Opus 5.5 | 0.46 | | Jev | 0.30 |
| GPT-6.1 Sol | 0.43 | | Kimi K2.6 | 0.26 |
| Gemini 3.8 Flash | 0.40 | | chance | 0.24 |
| | | | DeepSeek V4.1 Flash / Grok 4.7 | 0.23 / 0.17 |

Over all 30 play scenarios (`report`): TavernAgent 0.91, GPT-6 Astra 0.63, Claude Sonnet 0.53,
Opus 0.52. TavernAgent used 256-combat lookahead on the next-combat scenarios; ours is
network-only, and our combat-preview check got 1.00 on those 18.

**Scored the paper's way** (`lookahead` mode in bgagent/tavernbench.py: 256 simulated combats per
option on the 18 next-combat scenarios, the policy on the other five), lib-01 it 300:

| system | decision (23) | long-term (5) | select | econ | position | order | targets | strategy |
|---|---|---|---|---|---|---|---|---|
| chance | 0.24 | 0.27 | 0.30 | 0.12 | 0.20 | 0.21 | 0.25 | 0.25 |
| TavernAgent | 0.89 | 0.54 | 0.64 | 1.00 | 0.97 | 1.00 | 0.97 | 0.75 |
| GPT-6 Astra | 0.58 | 0.44 | 0.50 | 0.75 | 0.55 | 0.72 | 0.55 | 0.54 |
| Claude Opus 5.5 | 0.46 | 0.31 | 0.41 | 0.50 | 0.33 | 0.60 | 0.53 | 0.38 |
| ours, policy | 0.30 | 0.35 | 0.36 | 0.00 | 0.30 | 0.25 | 0.40 | 0.21 |
| **ours, lookahead** | **0.85** | 0.35 | 0.66 | 1.00 | 0.97 | 1.00 | 0.97 | 0.21 |

The five long-term scenarios, which no next-combat simulation answers (tasks: D1 8, D2 8, D4 16,
N1 29, N2 27):

| | D1 hero | D2 trinket | D4 discover | N1 turn plan | N2 turn plan |
|---|---|---|---|---|---|
| chance | 0.25 | 0.25 | 0.33 | 0.25 | 0.25 |
| TavernAgent | 0.25 | 0.12 | 0.81 | 0.83 | 0.67 |
| ours, policy | 0.50 | 0.38 | 0.44 | 0.24 | 0.19 |

With the paper's lookahead the tool is at 0.85 against TavernAgent's 0.89. The whole remaining gap
is long-term judgment: turn plans (N1/N2, 56 tasks, ours at chance) and the final-placement
discover. That is what training has to teach. Training logs `lookahead` and `long-term` from now
on (runs started before this change log value / policy only).

## 2026-11-02: the paper's training recipe vs ours (research, from the arXiv HTML)

Table 7 (final stage, TA-23) against our config:

| | paper | ours (lib-02) |
|---|---|---|
| minibatch / epochs | 8,192 / 1 | same |
| clip / target KL | 0.2 / 0.05 | same |
| gamma / lambda | 1.0 / 0.995 | same |
| grad clip | 1.0 | same |
| entropy / value / aux coef | 0.002 / 0.5 / 0.1 | same |
| reward | terminal placement utility | same (no shaping) |
| model | entity transformer, d 384, 8 layers, 18M | entity transformer, 16.9M |
| decisions per iteration | ~1.05M (512 games x 256 steps) | ~66k |
| per stage | 300 iterations, ~310M decisions, ~570k games | 300 iterations, ~20M |
| lr | 6e-6 -> 5e-6 (final stage) | 2e-4 -> ~1.9e-4 |
| KL anchor to the stage's start | 0.1 for the first 100 iterations | none |

- Stages: from scratch on simplified rules, then full content, the player's view, a first-place
  goal against opponents fitted to human statistics, then library opponents; later versions are
  chains of 300-iteration stages, each starting from the previous one under the KL anchor.
  Graded libraries run "from aligned libraries to harder ones with venomous minions, hands and
  ceiling builds". No search or simulation in training, no imitation, no pretraining.
- Compute (Appendix F): eight H200s and 128 CPU cores; ~1,900 GPU-hours for the 23 delivered
  stages, ~16,700 for everything else, ~1,300 for evaluation; 20 days in all.
  **Corrected the same day:** this note first scaled the GPU-hours by an H200's ~10x hardware
  edge and said "~20,000 hours of this PC". Wrong: count practice, not hardware. Same-size model
  (18M vs 16.9M). lib-02 makes 65,755 decisions per iteration in 57.8 s (medians over 179
  iterations; 35.7 s playing games, 22.1 s learning), ~4.1M decisions an hour. One 310M-decision
  stage is ~76 hours of this PC (~3 days); 23 stages of that size (~7.1B decisions) ~1,740 hours
  (~73 days nonstop). The paper's own pace, ~7.1B / 1,900 GPU-hours, is ~3.7M decisions per
  GPU-hour: each of their GPUs produced about what this PC does. Assumes every delivered stage
  was the final one's size; the paper's Table 7 covers the final stage only. The other ~16,700
  GPU-hours were experiments whose answers the paper gives us. lib-01 and lib-02 each had ~20M
  decisions: ~6% of one paper stage.
- What mattered most: recalibrating the libraries to the opponents met in live games, "the
  largest single gain" (Table 3, same checkpoint and budget: first place 20.9% -> 27.2% on the
  main exam, 10.3% -> 13.5% on the harder one).
- Their pitfalls: a search over copied games seemed to add 23.6 points of first-place rate but
  read the copies' future random numbers (reseeding: -0.65 ± 1.14); picking the best of many
  snapshots overstated gains (winner's curse); one version doubled its exam first-place rate and
  then lost ~1,300 rating live, from an early/mid-game weakness the exam hid.

- The first-place goal (checked 2026-11-02): the paper gives no numbers for it. Its only mentions
  are Appendix C ("then to a first-place goal against opponents fitted to human statistics, and
  then to library opponents") and Table 6 (TA-3: "opponents fitted to human statistics;
  first-place goal"). It doesn't say how the goal was weighted or whether it replaced the
  placement utility. TavernGym ships only the linear `placement_reward` (env.rs: 1st = 1 ... 8th
  = -1). The paper's model weights are not released either (AutoTavern README: "The trained
  TavernAgent, its weights and training code ... are not included"). Departure: our chain
  (full-01 -> lib-01 -> lib-02) never had a first-place stage; the paper's library stages came
  after one.
- Full pass over AutoTavern at fb61201 (2026-11-02, Cranny's prompt: TAVERNGYM.md). Read the
  README, TAVERNGYM.md, ARCHITECTURE.md and TAVERNBENCH.md, and searched the source (RulesConfig,
  the Python bindings, batch.rs, env.rs), the v1.0.0 release, the branches and the issues.
  Upstream has no newer commits, no other branch, no wiki, and only two unrelated issues. The
  only reward is the linear placement reward (TAVERNGYM.md "Batched environment"; env.rs
  `placement_reward`); RulesConfig has no reward fields. The release notes repeat: "The trained
  agent, its weights and training code are not included." Useful for us:
  - `step` returns each seat's final placement (`info["placement"]`), and ppo.py already adds
    the end reward itself, so a first-place goal needs no simulator change.
  - ARCHITECTURE.md "Determinism": "a rollout from [a copy] sees the original's future Taverns
    and Discovers unless the copy gets a fresh Rng". This is the paper's copied-game leak. The
    live tool's lookahead must reseed shops and Discovers, not just combats.
  - TAVERNBENCH.md: the N1/N2 turn plans were "cut from natural games, each played out 128
    times", and the hero and trinket picks are scored against top human players.
- Our setup against the docs (same day, Cranny: "anything else we may be handling differently
  by mistake?"):
  - Checked and fine. `rules=full` differs from TavernGym's defaults only in the 50-action
    turn cap. lib-02 has no shaping (`move_cost` 0.0), as in the paper. Turn numbers line up:
    the simulator's turn 1 is the first Recruit turn with 3 Gold; in the replays (300 games) the
    winners are on Tier 1 at turn 1 (297) and Tier 2 at turn 2 (268), so there's no off-by-one.
  - **A gap I hadn't recorded:** the real-board entries (`EntryBuilder`, replay.py) have
    `hand: []`, `stats: {}`, `counters: []`, and the minions carry only card, stats, golden and
    keywords: no magnetized cards, no added Deathrattles. The paper's entries have "its hand and
    hidden counters". A winner's replay never shows an opponent's hand or counters, so this
    can't be fully fixed. At equal total stats, our real-board opponents fight weaker than the
    real ones (tokens summoned in combat miss the counter buffs, nothing starts combat from
    hand, magnetic Deathrattles are missing). Scaling their stats to the target only partly
    offsets it. Not measured yet. Measurable on the simulated library by stripping its hands and
    counters and comparing combat win rates.
  - **Measured (same day):** simulated-library entries (data/library/train, which have hands,
    counters, attached cards and gained effects) rebuilt exactly as real entries are
    (EntryBuilder: card, stats, golden, keywords). Pairs matched within 10% of total stats (as
    library seats are matched to a target), 200 pairs per turn, 64 reseeded combats each.
    Combat win rate (+ half ties) of A against B, as it is vs rebuilt:

    | turn | 6 | 8 | 10 | 12 | 14 |
    |---|---|---|---|---|---|
    | as it is | 0.494 | 0.538 | 0.532 | 0.516 | 0.495 |
    | rebuilt | 0.479 | 0.533 | 0.523 | 0.510 | 0.471 |
    | difference | +0.015 ±0.012 | +0.005 ±0.007 | +0.009 ±0.013 | +0.006 ±0.010 | +0.024 ±0.015 |

    Rebuilding costs 0.5-2.4 points of combat win rate: too small to explain lib-02's Tier 1
    play. Caveat: these boards come from our policies and the bots, which may lean on counters
    less than real players do (e.g. Undead builds). First try paired random boards; they were
    so lopsided that every combat went one way (difference 0.000 everywhere), so pairs are now
    matched. Script: scratchpad measure_strip.py.

**lib-02 stopped at it 275 (Cranny: "End lib 2 and let's move forward"), 09:23, right after
the it-275 checkpoint; policy_00275.pt written from latest.pt.** Review (scratchpad
review_lib02.txt), against lib-01 it 300:

| | lib-01 it 300 | lib-02 it 275 |
|---|---|---|
| greedy vs bots, 1000 games | 2.20 ±0.08 | 2.19 ±0.07 (top-1 0.35) |
| among seven full-01 it500s, 512 games | 3.41 ±0.19 | **3.13 ±0.17** (top-1 0.24) |
| among seven lib-01 it300s, 512 games | (4.5 = even) | **4.20 ±0.20** (top-1 0.15) |
| real exam d0, same 512 games | 5.54 ±0.13 (top-4 0.26) | **5.17 ±0.13** (0.36) |
| real exam d1 | 5.79 ±0.13 (0.19) | **5.47 ±0.14** (0.26) |
| real exam d5 | 6.71 ±0.10 (0.03) | **6.42 ±0.11** (0.08) |
| first place on the real exam d0 / d1 | 0.01 / 0.02 | 0.02 / 0.03 |
| tier vs bots, turns 8 / 10 / 12 | 2.02 / 2.02 / 2.10 | 1.12 / 1.94 / 2.00 |
| warband stats vs bots, turns 10 / 12 | 479 / 743 | 472 / 696 |
| TavernBench lookahead / policy | 0.85 / 0.30 | 0.85 / 0.29 |
| long-term (D1 D2 D4 N1 N2) | 0.35 (0.50 0.38 0.44 0.24 0.19) | 0.34 (0.50 0.25 0.44 0.31 0.22) |

Better than lib-01 on every head-to-head and real-exam measure, while staying on Tier 1
longer and almost never winning (2-3% first place on the exam). Its ladder never moved off d =
0 (it needs placement < 4.5; it sat at ~5.2): it trained only against median real boards (t8
~191, t10 ~851; the Perfect Games winners have 282 / 1,806).

TavernBench `data/full` (Cranny asked, 2026-11-02): all 960 tasks; `data/eval` is the paper's
368-task subset (8 per scenario, plus every D4/N1/N2 task). The released answers of the other
systems cover eval only, so comparisons stay on eval. For our 23 scenarios full has 552 tasks
against eval's 232 (24 per scenario, so hero and trinket go from 8 to 24; D4/N1/N2 have no
more). lib-02 it 275 on full (`--set .../data/full --compare ""`): decision 0.28 policy / 0.85
lookahead, long-term 0.36; D1 hero 0.58 (eval: 0.50), D2 0.25, D4 0.44, N1 0.31, N2 0.22. Not for
training: hero and trinket labels are top human players' picks (imitation), the rest would
train on the benchmark, and the paper trained on neither.

**Options added (2026-11-02), off by default:** `placement_utility` (the end reward for
places 1..8; () = linear) and `anchor_coef` / `anchor_iters` (the paper's KL anchor to
`init_from`). Tests: `test_placement_utility`, `test_anchor_kl`. Smoke run with both on: the
anchor KL is logged (0.00018, 0.00174), then 0 after `anchor_iters`.

For us: our hyperparameters are the paper's except volume, lr and the KL anchor. Volume is a
gap of days to months on this PC, not years (one paper stage ~3 days, see the correction above).
Its biggest lever, live-game recalibration, needs the live tool and Cranny's games. Our combat previews reseed every combat (`combat_preview`), the guard
against their copied-game leak; the live tool must keep that.

## 2026-11-02: lib-02, real-board opponents

Start point, lib-01 it 300 on real-exam (256 sampled games each; d0 = the exam's own median,
d1 all players, d5 top 1%): **d0 5.31 ±0.18** (top-1 0.01, top-4 0.30), **d1 5.79 ±0.17** (0.01,
0.18), **d5 6.74 ±0.14** (0.00, 0.04).

**runs/lib-02 started 2026-11-02 03:16**: init_from lib-01 it 300, library real-train, exam
real-exam at d 0/1/5 with 256 games (lib-01: 128), learner_seats=(0,), n_games 256, lr 2e-4,
difficulty_up 4.5, 300 iterations. One variable against lib-01: the opponents' boards are real.
First iterations: placement 5.56 / 5.63 / 5.42 at d = 0 (opponent turn 8 ~189), tier at turn 8
2.00, ~60 s per iteration. The question: does the tier at turns 8-10 move off 2.0?

Check-in at it 179 (07:12), means over 20 iterations:

| iterations | tier t8 | board t8 | entropy | training placement |
|---|---|---|---|---|
| 1-20 | 2.01 | 6.27 | 0.75 | 5.50 |
| 61-80 | 2.00 | 6.16 | 0.63 | 5.40 |
| 121-140 | 1.99 | 6.07 | 0.57 | 5.38 |
| 161-179 | 1.92 | 6.08 | 0.54 | 5.30 |

Exam it 130-170: d0 5.18-5.36, d1 5.62-5.80, d5 6.46-6.74 (start 5.31 / 5.79 / 6.74). Bots
2.17-2.32. TavernBench value 0.31-0.35. Placement improves a little while the tier at turn 8
drifts *down* and entropy falls steadily: the run is polishing the stay-low plan, not finding
levelling. Untested guess: small batches at a high lr commit early. The real pace including
evals is 1.31 min per iteration (the ~60 s above leaves the evals out), so the run ends about
09:50, not the 08:20 I told Cranny.

Check-in at it 249 (08:44): tier by turn (means over the iteration ranges):

| iterations | t4 | t6 | t8 | t10 | training placement | top-1 | top-4 |
|---|---|---|---|---|---|---|---|
| 1-20 | 1.00 | 1.88 | 2.01 | 2.03 | 5.50 | 0.007 | 0.25 |
| 161-180 | 1.00 | 1.39 | 1.92 | 2.00 | 5.31 | 0.006 | 0.29 |
| 201-220 | 1.00 | 1.04 | 1.32 | 2.00 | 5.28 | 0.007 | 0.31 |
| 230-249 | 1.00 | 1.02 | 1.16 | 1.98 | 5.21 | 0.009 | 0.31 |

Exam (it 10 -> 240): d0 5.55 -> 5.41 (top-4 0.22 -> 0.28), d1 5.86 -> 5.57 (0.16 -> 0.25), d5
6.74 -> 6.48 (0.04 -> 0.09). It now stays on Tier 1 until about turn 9 (real winners: 4.7 at turn 8),
spends all its gold on minions, and places better against real boards and the exam, but first
place stays at ~1%. That is survival: under the linear placement reward, 6th -> 5th is worth the
same as 2nd -> 1st. The paper ran a stage with "a first-place goal"; to research before any
change.

## 2026-11-02: lib-03, a first-place stage (the paper's TA-3)

Why: lib-02 survives better than lib-01 but wins 2-3% on the real exam and stays on Tier 1 to
turn 8. Under the linear reward a place is a place, and at d = 0 survival was enough. The paper
ran "a first-place goal against opponents fitted to human statistics" (Table 6, TA-3) before
its library stages; our chain never had one. Cranny approved ("Alright start l3").

**runs/lib-03 started 2026-11-02 ~10:08** (first iteration logged 10:09:51): init_from lib-02
it 275, library real-train, exam real-exam (d 0/1/5, 256 games). Three settings change against
lib-02:
- `difficulty=1.0`, `difficulty_down=8.0` (never easier), `difficulty_up=4.5`: Firestone's
  all-players curve is our reading of "opponents fitted to human statistics".
- `placement_utility=(3, 0.714, 0.429, 0.143, -0.143, -0.429, -0.714, -1)`: 1st is worth 3
  instead of 1, the rest linear as before, so 2nd -> 1st (+2.29) is worth more than 8th -> 2nd
  (+1.71). The paper gives no numbers; 3 is ours.
- `anchor_coef=0.1`, `anchor_iters=100` (the paper's Table 7).
Unchanged: lr 2e-4, ~66k decisions per iteration, 256 games, 300 iterations. First
iterations: placement 5.66 / 5.58 at d = 1 (opponent turn 8 ~236), tier at turn 8 1.06 / 1.07,
anchor KL 0.003-0.006, ~70 s per iteration (update 29 s with the anchor, was 22-23 s).
Success: the tier at turn 8 above 2 and first place on the real exam above 3%, with nothing
else falling. If the tier hasn't moved by it 100, the reward alone isn't the lever; next is
the paper-scale run (~1M decisions per iteration, lower lr, ~3 days per stage).

## 2026-11-02: why it never levels (Cranny: "winning so many games but never tiering up")

**It never tries.** lib-02 it 275 against real-exam at d = 1, 256 sampled games per variant,
same lobbies (scratchpad forced_level.py). Its own P(Upgrade) at the first legal chance of the
turn: turn 2 mean 1.3e-8 (max 7.6e-8, n = 251), turns 3 and 4 ~1e-18. Over lib-03's ~77k games
that is ~0.001 expected early level-ups: the first-place reward can't reach turn-2 levelling.

**Forcing it to level makes it worse** (it never learned to play a levelled game):

| variant | placement | top-1 | top-4 | tier t4 / t8 / t10 |
|---|---|---|---|---|
| as it plays | 5.38 ±0.19 | 0.020 | 0.27 | 1.00 / 1.06 / 1.94 |
| Upgrade first on turn 2 | 6.02 ±0.17 | 0.008 | 0.15 | 1.98 / 1.98 / 2.91 |
| Upgrade first on turns 2, 4, 6 | 6.32 ±0.18 | 0.008 | 0.13 | 2.98 / 3.68 / 3.64 |

(A 16-game first try gave 6.25 / 6.50 / 7.38: noise, kept only as the smoke test.)

**Where the habit came from: the scripted bots.** full-01 (from scratch, 4 learner seats + 4
bots) levelled early while it was near-random and unlearned it while it learned to beat the
bots (means over 10 iterations):

| full-01 it | tier t4 | t6 | t8 | t10 | placement | entropy |
|---|---|---|---|---|---|---|
| 1 | 1.63 | 2.20 | 2.90 | 3.55 | 6.44 | 1.90 |
| 50 | 1.29 | 2.08 | 2.89 | 3.59 | 4.13 | 1.61 |
| 150 | 1.12 | 2.01 | 2.59 | 2.99 | 3.55 | 1.46 |
| 300 | 1.02 | 1.94 | 2.10 | 2.20 | 3.45 | 1.26 |
| 480 | 1.00 | 1.96 | 2.15 | 3.10 | 3.27 | 1.09 |

lib-01 and lib-02 then pushed the turn-8 tier from 2.0 to 1.1.

**Research:**
- **Corrected the same day:** I first wrote, and told Cranny, "the paper never trained against
  scripted bots". It doesn't say. A second read finds no opponent named for its first three
  stages (from scratch / full content / player's view); TA-3 has "opponents fitted to human
  statistics"; library seats come after. Scripted bots appear only as a baseline (0.0% first
  places against its reality-calibrated opponents). "Learning started from scratch on a simplified rule
  set, moved to the full content, then to the player's view", then library seats ("Rather than
  self-play, the learner faces seven library seats"), graded "from aligned libraries to harder
  ones". It says nothing about exploration or entropy, and nothing about how its agent learned
  tier timing.
- Departure we made: `opponents=mixed` (bots + learners) for the foundation. CLAUDE.md records
  why: bots alone gave no signal, and self-play plateaued at 6.9.
- Literature: "primacy bias", where an agent overfits its earliest experience and that harms
  everything after (Nikishin et al., ICML 2022, arXiv 2205.07802; remedy: re-initialise the
  last layers). PPO also loses plasticity under non-stationary training (Moalla et al.,
  NeurIPS 2024, arXiv 2405.00662). Entropy bonuses can't revive an action at P ~ 1e-8: the
  softmax gradient scales with p.

## 2026-11-02: a fresh start without bots (Cranny: "Yes")

lib-03 was stopped at it 40 (latest.pt; tier t4/t6/t8/t10 1.00/1.01/1.07/1.94, training
placement 5.52 at d = 1). Nothing moved, as P(Upgrade) ~1e-8 predicted.

**Easier opponents** (`Difficulty.target`, `difficulty_min`): below d = 0 the library's median
boards are scaled down by 10**d. Where does a beginner place? 128 games each, real-train
library, one tested seat against seven library seats:

| d | turn-8 target | uniform random play | untrained full network |
|---|---|---|---|
| 0 | 185 | 8.00 | 7.99 |
| -0.5 | 59 | 7.88 | 7.99 |
| -1 | 18 | 7.31 | 7.85 |
| -1.5 | 6 | 6.96 | 7.74 |
| -2 | 2 | 6.81 (top-4 0.10) | 7.77 (top-4 0.01) |

Even boards scaled to 1-2 total stats on turn 8 beat a beginner. The scaled boards keep their
minion count (each minion at least 1/1) and their real Tavern Tier, which sets the damage. One
learner against seven library seats gives almost no reward variation: the "bots alone gave no
signal" problem again.

So the fresh run keeps four learner seats, as full-01 did, and replaces the four bots with
library seats. The learners compete with each other, so placement varies from the first game,
while the library seats bring real boards that get stronger as the learners improve.
Departure: the paper's library stages use one learner and seven library seats, and it names no
opponents for its first stages.

## Open questions
- Learning rate and entropy for the full model: the paper's 6e-6 is for late fine-tuning
  stages; early stages need far higher. Tune on the small preset first.
- Whether self-play alone reaches 4.5 vs bots, or needs bot/library opponents mixed in.
  (Answered: mixed reaches 2.94 greedy; pure self-play plateaued near 6.9.)
- Where the tiny model's curve flattens, and how much the full model on the 3070 adds.
- Reward shaping: none so far (pure placement reward), as in TavernGym.
