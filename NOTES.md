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

## Open questions
- Learning rate and entropy for the full model: the paper's 6e-6 is for late fine-tuning
  stages; early stages need far higher. Tune on the small preset first.
- Whether self-play alone reaches 4.5 vs bots, or needs bot/library opponents mixed in.
  (Answered: mixed reaches 2.94 greedy; pure self-play plateaued near 6.9.)
- Where the tiny model's curve flattens, and how much the full model on the 3070 adds.
- Reward shaping: none so far (pure placement reward), as in TavernGym.
