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

## Open questions
- Learning rate and entropy for the full model: the paper's 6e-6 is for late fine-tuning
  stages; early stages need far higher. Tune on the small preset first.
- Whether self-play alone reaches 4.5 vs bots, or needs bot/library opponents mixed in.
- Reward shaping: none so far (pure placement reward), as in TavernGym.
