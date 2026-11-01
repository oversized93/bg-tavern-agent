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

## Open questions
- Learning rate and entropy for the full model: the paper's 6e-6 is for late fine-tuning
  stages; early stages need far higher. Tune on the small preset first.
- Whether self-play alone reaches 4.5 vs bots, or needs bot/library opponents mixed in.
- Reward shaping: none so far (pure placement reward), as in TavernGym.
