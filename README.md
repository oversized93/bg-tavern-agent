# bg-tavern-agent

A Hearthstone Battlegrounds **recommendation** agent. A transformer policy learns to play by
reinforcement learning inside [TavernGym](https://github.com/MCG-NJU/AutoTavern), a faithful
Battlegrounds simulator released with the paper *Build, Don't Play* (Ma & Wang, 2026). Once
trained, it reads your live game state and suggests moves. **It never plays for you**: no
input automation, no memory writing, read-only on the game's logs.

## What's here
| Path | What it does |
|---|---|
| `bgagent/layout.py` | Offsets into TavernGym's 3,227-float / 124-card observation, checked against the engine |
| `bgagent/model.py` | Entity transformer: one token per minion, hand card, shop offer, choice and opponent; pointer head over all 245 actions; value and placement heads |
| `bgagent/ppo.py` | PPO over TavernGym's batched environment, self-play / league opponents, checkpoints and resume |
| `bgagent/policy.py` | `Advisor`: ranked recommendations for any saved game state |
| `bgagent/tavernbench.py` | Answers TavernBench's 23 decision scenarios with a policy, scored by the benchmark's own code |
| `scripts/train.py` | Train (`--preset smoke / small / 3070`); rerun the same command to resume |
| `scripts/evaluate.py` | Average placement vs the seven scripted bots (or `--vs` another policy), with a 95% interval |
| `scripts/tavernbench.py` | TavernBench decision accuracy, next to the paper's agent and the LLMs |
| `scripts/advise.py` | Recommendations for a saved state, or a simulated demo game |
| `scripts/status.py` | Phone-sized progress summary of a run |
| `scripts/diagnose.py` | Behaviour stats (warband size, tier, unspent gold by turn) that show learning before placement moves |

## Quick start
Setup on Windows with an NVIDIA GPU: [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md) (native, no admin; how the 3070 runs) or [docs/SETUP_WSL2.md](docs/SETUP_WSL2.md) (WSL2).

```bash
pytest -q                                                        # sanity checks
python scripts/train.py --preset smoke --run runs/smoke          # 1-minute plumbing test
python scripts/train.py --preset small --run runs/stage1-small   # stage 1: hours
python scripts/status.py runs/stage1-small
python scripts/advise.py runs/stage1-small/latest.pt --demo --turn 6
```

## Model sizes
| preset | width x layers | parameters |
|---|---|---|
| tiny | 64 x 2 | 0.4M (tests only) |
| small | 192 x 4 | 2.9M |
| medium | 256 x 6 | 6.3M |
| full | 384 x 8 | 16.9M (the paper's final size is 18M) |

## Plan and log
- [docs/ROADMAP.md](docs/ROADMAP.md): training stages, TavernBench scoring, the Power.log
  bridge and how it gets verified, opponents from real games, patch drift, the overlay.
- [NOTES.md](NOTES.md): every experiment and decision, with numbers.

## Credits
TavernGym and TavernBench are MIT-licensed work by Changlian Ma and Limin Wang (Nanjing
University, Shanghai AI Laboratory). This project is independent and unaffiliated with them
or with Blizzard Entertainment. Hearthstone is a Blizzard trademark.
