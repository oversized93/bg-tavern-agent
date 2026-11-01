"""PPO for TavernGym's BatchEnv.

BatchEnv slots act only when `ready`, several per step, and a seat's reward arrives once, at
the end of its game. So rollouts are gathered as whole episodes: each slot keeps an open
trajectory, the final placement reward is added to its last decision when the slot is done,
and only finished trajectories go into an update (unfinished ones carry over to the next
iteration with the log-probs they were sampled with). With gamma = 1 every return is exact.
"""

import copy
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import taverngym
import torch
import torch.nn.functional as Fn

from . import layout as L
from .evaluation import evaluate
from .filters import ActionFilter
from .model import ModelConfig, TavernNet, param_count
from .rules import RULES, rules_config  # noqa: F401  (RULES re-exported for scripts)


@dataclass
class TrainConfig:
    model: str = "full"
    rules: str = "full"
    # "league": all eight seats are policy-controlled. The learner trains on `learner_seats`;
    # the other seats play a frozen past snapshot (the current policy until the first one).
    # With learner_seats = all eight it is pure self-play.
    # "mixed": the learner plays `learner_seats`, TavernGym's scripted bots play the rest.
    # The learners still compete with each other, so the reward varies even while the bots
    # beat them all, and the learner sees competent play from the start.
    # "bots": one learner seat against seven scripted bots. A fresh policy places last almost
    # every game there, so the reward barely varies and nothing is learned; start elsewhere.
    # "hybrid": the learner plays `learner_seats`, a frozen past snapshot plays
    # `snapshot_seats`, and the scripted bots play the rest. For when the bots get too easy.
    opponents: str = "league"
    learner_seats: tuple = (0, 1, 2, 3)
    snapshot_seats: tuple = (4, 5)  # "hybrid" only
    league_size: int = 8  # snapshots kept as opponents
    snapshot_every: int = 25  # iterations between policy snapshots (league + eval history)
    eval_every: int = 10  # iterations between evaluations against the scripted bots
    eval_games: int = 256
    # TavernBench decision accuracy at every evaluation (bgagent/tavernbench.py): the path to
    # AutoTavern's tavernbench/ folder, or "" to skip. About 20 s for the tiny model on CPU.
    bench_dir: str = ""
    # Reward shaping: subtracted from the reward of every minion Move. Moves are free in the
    # game, so an untrained policy shuffles minions dozens of times a turn; 0 disables it.
    move_cost: float = 0.0
    torch_threads: int = 0  # CPU threads for torch; 0 = torch's default
    factored: bool = True  # factored action head (see ModelConfig.factored)
    move_cap: int = 6  # action filter (see filters.py): minion moves allowed per turn, 0 = off
    no_refreeze: bool = True  # action filter: no Freeze while the shop is already frozen
    cancel_cap: int = 2  # action filter: Cancels allowed per turn (Play/Cancel loops), 0 = off
    n_games: int = 128
    threads: int = 8
    decisions_per_iter: int = 65536
    iterations: int = 300
    lr: float = 1e-4
    lr_final: float = 3e-5
    gamma: float = 1.0
    lam: float = 0.995
    clip: float = 0.2
    target_kl: float = 0.05
    epochs: int = 1
    minibatch: int = 8192
    micro_batch: int = 2048  # gradient accumulation chunk; lower it if the GPU runs out of memory
    ent_coef: float = 0.002
    vf_coef: float = 0.5
    aux_coef: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 0
    checkpoint_every: int = 5
    max_traj_len: int = 4000  # drop a trajectory that runs this long (a stuck game)
    device: str = "auto"

    @staticmethod
    def preset(name: str) -> "TrainConfig":
        if name == "smoke":  # CPU, a few minutes: checks the plumbing, learns nothing
            return TrainConfig(model="tiny", n_games=4, threads=2, decisions_per_iter=2048,
                               iterations=3, minibatch=1024, micro_batch=512, lr=3e-4,
                               lr_final=3e-4, checkpoint_every=1, eval_every=3, eval_games=16)
        # Defaults from the CPU tests (NOTES.md, 2026-10-31): the factored head plus "mixed"
        # opponents (4 learner seats, 4 scripted bots) beat every other setup by a wide
        # margin, and training on the full rules beat the simplified-rules curriculum
        # (3.78 vs 6.90 against the bots after 100 iterations). Learning rates for the
        # bigger models are extrapolated, not yet tested.
        if name == "small":  # fast iteration (hours)
            return TrainConfig(model="small", rules="full", opponents="mixed", n_games=64,
                               learner_seats=(0, 1, 2, 3), decisions_per_iter=32768,
                               minibatch=4096, micro_batch=2048, lr=5e-4, lr_final=2e-4)
        if name == "3070":  # the multi-day run on an 8 GB card
            return TrainConfig(model="full", opponents="mixed", n_games=64,
                               learner_seats=(0, 1, 2, 3), decisions_per_iter=65536,
                               minibatch=8192, micro_batch=2048, lr=3e-4, lr_final=1e-4)
        raise KeyError(name)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.inference_mode()
def act(model, feats, cards, mask, device, amp):
    f = torch.from_numpy(feats).to(device, non_blocking=True)
    c = torch.from_numpy(cards).to(device, non_blocking=True)
    m = torch.from_numpy(mask).to(device, non_blocking=True)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        logits, value, _ = model(f, c, m)
    logp_all = torch.log_softmax(logits, -1)
    a = torch.multinomial(logp_all.exp(), 1).squeeze(-1)
    logp = logp_all.gather(1, a[:, None]).squeeze(-1)
    return a.cpu().numpy(), logp.cpu().numpy(), value.float().cpu().numpy()


class Collector:
    """Runs BatchEnv and returns finished learner trajectories."""

    def __init__(self, cfg: TrainConfig, device):
        self.cfg, self.device = cfg, device
        self.amp = device.type == "cuda"
        league = cfg.opponents in ("league", "hybrid")
        controlled = {"league": tuple(range(8)), "mixed": tuple(cfg.learner_seats),
                      "hybrid": tuple(sorted(set(cfg.learner_seats) | set(cfg.snapshot_seats))),
                      "bots": (0,)}[cfg.opponents]
        self.env = taverngym.BatchEnv(cfg.n_games, seed=cfg.seed, controlled=controlled,
                                      config=rules_config(cfg.rules), threads=cfg.threads)
        n = self.env.n_slots
        k = len(controlled)
        seats = np.array([controlled[i % k] for i in range(n)])
        self.learner = np.isin(seats, cfg.learner_seats) if league else np.ones(n, bool)
        self.open = [[] for _ in range(n)]
        self.obs = self.env.observe()
        self.opponent = None  # frozen model for league seats
        self.filter = None  # built from the model's filter settings on first collect
        self.stats_reset()

    def stats_reset(self):
        self.placements, self.ep_lens, self.rejected, self.dropped = [], [], 0, 0
        # behaviour at each learner End Turn, by turn: (warband size, Tavern Tier, gold left)
        self.end_turn = {t: [] for t in (4, 6, 8, 10)}
        self.n_actions, self.n_moves, self.n_end_turns = 0, 0, 0

    def behaviour(self):
        out = {}
        for t, rows in self.end_turn.items():
            if rows:
                b, tier, gold = np.mean(rows, 0)
                out.update({f"board_t{t}": round(float(b), 2), f"tier_t{t}": round(float(tier), 2),
                            f"gold_left_t{t}": round(float(gold), 2)})
        if self.n_end_turns:
            out["actions_per_turn"] = round(self.n_actions / self.n_end_turns, 1)
        if self.n_actions:
            out["move_frac"] = round(self.n_moves / self.n_actions, 3)
        return out

    def collect(self, model):
        cfg, env = self.cfg, self.env
        done_trajs, n_dec = [], 0
        t0 = time.time()
        while n_dec < cfg.decisions_per_iter:
            o = self.obs
            ready = o["ready"]
            actions = np.zeros(env.n_slots, dtype=np.int64)
            if self.filter is None:
                self.filter = ActionFilter.for_model(env.n_slots, model)
            lr_idx = np.nonzero(ready & self.learner)[0]
            if len(lr_idx):
                lm = self.filter.apply(o["features"][lr_idx], o["action_mask"][lr_idx], lr_idx)
                a, lp, v = act(model, o["features"][lr_idx], o["cards"][lr_idx], lm,
                               self.device, self.amp)
                self.filter.record(lr_idx, a)
                actions[lr_idx] = a
                for j, i in enumerate(lr_idx):
                    f, aj = o["features"][i], int(a[j])
                    is_move = aj >= L.A_MOVE
                    self.open[i].append([
                        f.astype(np.float16), o["cards"][i].astype(np.int16),
                        np.packbits(lm[j]), aj, float(lp[j]),
                        float(v[j]), -cfg.move_cost if is_move else 0.0,
                    ])
                    self.n_actions += 1
                    self.n_moves += is_move
                    if aj == L.A_END and f[1] == 0:  # End Turn outside hero select
                        self.n_end_turns += 1
                        t = int(round(f[0] * 20))
                        if t in self.end_turn:  # global floats: turn/20, tier/6, gold/10, board/7
                            self.end_turn[t].append((f[33] * 7, f[10] * 6, f[7] * 10))
            op_idx = np.nonzero(ready & ~self.learner)[0]
            if len(op_idx):
                om = self.filter.apply(o["features"][op_idx], o["action_mask"][op_idx], op_idx)
                a, _, _ = act(self.opponent or model, o["features"][op_idx],
                              o["cards"][op_idx], om, self.device, self.amp)
                self.filter.record(op_idx, a)
                actions[op_idx] = a
            rewards, dones, info = env.step(actions)
            self.rejected += int((info["rejected"] & self.learner).sum())
            hit = np.nonzero(((rewards != 0) | dones) & self.learner)[0]
            for i in hit:
                traj = self.open[i]
                if traj:
                    traj[-1][6] += float(rewards[i])
                if dones[i]:
                    place = int(info["placement"][i])
                    if traj and place > 0:
                        done_trajs.append((traj, place))
                        n_dec += len(traj)
                        self.placements.append(place)
                        self.ep_lens.append(len(traj))
                    elif traj:
                        self.dropped += 1  # simulator guard stopped the game
                    self.open[i] = []
            for i in lr_idx:
                if len(self.open[i]) > cfg.max_traj_len:
                    self.open[i] = []
                    self.dropped += 1
            self.obs = env.observe()
        return done_trajs, time.time() - t0


def build_batch(trajs, gamma, lam):
    feats, cards, masks, acts, logps, vals, advs, rets, places = ([] for _ in range(9))
    for traj, place in trajs:
        v = np.array([t[5] for t in traj], np.float32)
        r = np.array([t[6] for t in traj], np.float32)
        nxt = np.append(v[1:], 0.0)  # terminal value is 0
        delta = r + gamma * nxt - v
        adv = np.zeros_like(v)
        acc = 0.0
        for t in range(len(traj) - 1, -1, -1):
            acc = delta[t] + gamma * lam * acc
            adv[t] = acc
        feats.append(np.stack([t[0] for t in traj]))
        cards.append(np.stack([t[1] for t in traj]))
        masks.append(np.stack([t[2] for t in traj]))
        acts.append(np.array([t[3] for t in traj], np.int64))
        logps.append(np.array([t[4] for t in traj], np.float32))
        vals.append(v)
        advs.append(adv)
        rets.append(adv + v)
        places.append(np.full(len(traj), place - 1, np.int64))
    cat = np.concatenate
    return {
        "feats": cat(feats), "cards": cat(cards),
        "mask": np.unpackbits(cat(masks), axis=1, count=L.N_ACTIONS).astype(bool),
        "act": cat(acts), "logp": cat(logps), "val": cat(vals), "adv": cat(advs),
        "ret": cat(rets), "place": cat(places),
    }


def ppo_update(model, opt, batch, cfg: TrainConfig, device):
    amp = device.type == "cuda"
    n = len(batch["act"])
    T = {k: torch.from_numpy(v) for k, v in batch.items()}
    stats = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "aux": 0.0, "kl": 0.0, "clipfrac": 0.0}
    steps, stop = 0, False
    for _ in range(cfg.epochs):
        perm = torch.randperm(n)
        for s in range(0, n, cfg.minibatch):
            mb = perm[s : s + cfg.minibatch]
            if len(mb) < cfg.minibatch // 4:
                continue
            adv_mb = T["adv"][mb]
            adv_mean, adv_std = adv_mb.mean(), adv_mb.std() + 1e-8
            opt.zero_grad(set_to_none=True)
            mb_stats = dict.fromkeys(stats, 0.0)
            for c0 in range(0, len(mb), cfg.micro_batch):
                ix = mb[c0 : c0 + cfg.micro_batch]
                w = len(ix) / len(mb)
                g = lambda k, dt=None: T[k][ix].to(device, dt)  # noqa: E731
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    logits, value, place = model(g("feats", torch.float32), g("cards"),
                                                 g("mask"))
                logp_all = torch.log_softmax(logits, -1)
                logp = logp_all.gather(1, g("act")[:, None]).squeeze(-1)
                ratio = torch.exp(logp - g("logp"))
                adv = (g("adv") - adv_mean.to(device)) / adv_std.to(device)
                pg = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()
                vf = 0.5 * (value - g("ret")).pow(2).mean()
                p = logp_all.exp()
                ent = -(p * logp_all.masked_fill(~g("mask"), 0.0)).sum(-1).mean()
                aux = Fn.cross_entropy(place, g("place"))
                loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent + cfg.aux_coef * aux
                (loss * w).backward()
                with torch.no_grad():
                    kl = ((ratio - 1) - torch.log(ratio)).mean()  # k3 estimator
                    mb_stats["kl"] += kl.item() * w
                    mb_stats["clipfrac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item() * w
                for k, val in (("pg", pg), ("vf", vf), ("ent", ent), ("aux", aux)):
                    mb_stats[k] += val.item() * w
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            steps += 1
            for k in stats:
                stats[k] += mb_stats[k]
            if mb_stats["kl"] > cfg.target_kl:
                stop = True
                break
        if stop:
            break
    out = {k: v / max(steps, 1) for k, v in stats.items()}
    out.update(updates=steps, early_stop=float(stop))
    return out


class Trainer:
    def __init__(self, cfg: TrainConfig, run_dir: Path):
        self.cfg, self.run_dir = cfg, Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.torch_threads:
            torch.set_num_threads(cfg.torch_threads)
        self.device = pick_device(cfg.device)
        mcfg = ModelConfig.preset(cfg.model)
        mcfg.factored = cfg.factored
        mcfg.move_cap = cfg.move_cap
        mcfg.no_refreeze = cfg.no_refreeze
        mcfg.cancel_cap = cfg.cancel_cap
        latest = self.run_dir / "latest.pt"
        if latest.exists():  # resume with the architecture the run was started with
            from .policy import model_config_from

            mcfg = model_config_from(torch.load(latest, map_location="cpu", weights_only=False))
            # the action filter is not architecture: the run's config decides it
            mcfg.move_cap, mcfg.no_refreeze = cfg.move_cap, cfg.no_refreeze
            mcfg.cancel_cap = cfg.cancel_cap
        self.model = TavernNet(mcfg).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=0.0,
                                     eps=1e-5)
        self.it = 0
        self.league = []  # list of (iteration, cpu state dict)
        self.load_latest()
        self.collector = Collector(cfg, self.device)
        self.metrics = open(self.run_dir / "metrics.jsonl", "a")
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(self.run_dir / "tb")
        except Exception:  # tensorboard missing: metrics.jsonl still has everything
            self.tb = None

    # ---- checkpoints
    def save(self):
        ck = {
            "iteration": self.it,
            "config": asdict(self.cfg),
            "model_config": asdict(self.model.cfg),
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "league_iters": [i for i, _ in self.league],
            "taverngym": {"version": taverngym.version(), "game": taverngym.game_version()},
        }
        tmp = self.run_dir / "latest.pt.tmp"
        torch.save(ck, tmp)
        tmp.replace(self.run_dir / "latest.pt")  # atomic, so a crash never leaves a half file

    def save_policy(self):
        """A weights-only snapshot: league opponents and the evaluation history read these."""
        torch.save({"model": self.model.state_dict(), "model_config": asdict(self.model.cfg),
                    "iteration": self.it}, self.run_dir / f"policy_{self.it:05d}.pt")

    def load_latest(self):
        p = self.run_dir / "latest.pt"
        if not p.exists():
            return
        ck = torch.load(p, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.it = ck["iteration"]
        iters = ck.get("league_iters", [])
        if not iters and self.cfg.opponents in ("league", "hybrid"):
            # a run that switches to league / hybrid opponents: start from its own snapshots
            snaps = sorted(int(f.stem.split("_")[1]) for f in self.run_dir.glob("policy_*.pt"))
            iters = [i for i in snaps if i <= self.it][-self.cfg.league_size :]
        for i in iters:
            f = self.run_dir / f"policy_{i:05d}.pt"
            if f.exists():
                self.league.append((i, torch.load(f, map_location="cpu")["model"]))
        print(f"resumed from {p} at iteration {self.it}"
              + (f", league {[i for i, _ in self.league]}" if self.league else ""))

    # ---- league
    def maybe_snapshot(self):
        cfg = self.cfg
        if self.it % cfg.snapshot_every:
            return
        self.save_policy()
        if cfg.opponents not in ("league", "hybrid"):
            return
        sd = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self.league.append((self.it, sd))
        self.league = self.league[-cfg.league_size :]

    def pick_opponent(self):
        if self.cfg.opponents not in ("league", "hybrid"):
            return None
        if not self.league:  # until the first snapshot, opponents are the current policy
            return None
        _, sd = random.choice(self.league)
        opp = copy.deepcopy(self.model)
        opp.load_state_dict(sd)
        return opp.eval()

    # ---- main loop
    def lr_at(self, it):
        cfg = self.cfg
        frac = min(it / max(cfg.iterations - 1, 1), 1.0)
        return cfg.lr + (cfg.lr_final - cfg.lr) * frac

    def run(self):
        cfg = self.cfg
        print(f"device={self.device} model={cfg.model} params={param_count(self.model)/1e6:.1f}M "
              f"games={cfg.n_games} opponents={cfg.opponents} rules={cfg.rules}")
        while self.it < cfg.iterations:
            for g in self.opt.param_groups:
                g["lr"] = self.lr_at(self.it)
            self.collector.opponent = self.pick_opponent()
            self.collector.stats_reset()
            self.model.eval()
            trajs, t_roll = self.collector.collect(self.model)
            batch = build_batch(trajs, cfg.gamma, cfg.lam)
            self.model.train()
            t1 = time.time()
            up = ppo_update(self.model, self.opt, batch, cfg, self.device)
            t_upd = time.time() - t1
            self.it += 1
            c = self.collector
            pl = np.array(c.placements)
            m = {
                "iteration": self.it,
                "decisions": int(len(batch["act"])),
                "episodes": int(len(pl)),
                "avg_placement": float(pl.mean()) if len(pl) else None,
                "top1": float((pl == 1).mean()) if len(pl) else None,
                "top4": float((pl <= 4).mean()) if len(pl) else None,
                "ep_len": float(np.mean(c.ep_lens)) if c.ep_lens else None,
                "rejected": c.rejected,
                "dropped": c.dropped,
                "rollout_s": round(t_roll, 1),
                "update_s": round(t_upd, 1),
                "decisions_per_s": round(len(batch["act"]) / max(t_roll, 1e-6), 1),
                "lr": self.lr_at(self.it - 1),
                **{k: round(v, 5) for k, v in up.items()},
                **c.behaviour(),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self.metrics.write(json.dumps(m) + "\n")
            self.metrics.flush()
            if self.tb:
                for k, v in m.items():
                    if isinstance(v, (int, float)) and k != "iteration":
                        self.tb.add_scalar(k, v, self.it)
            fmt = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
            print(f"it {self.it:4d} | place {fmt(m['avg_placement'])} top4 {fmt(m['top4'])} "
                  f"| t8 board {fmt(m.get('board_t8'))} tier {fmt(m.get('tier_t8'))} "
                  f"| acts/turn {m.get('actions_per_turn', '-')} moves {m.get('move_frac', '-')} "
                  f"| kl {m['kl']:.4f} ent {m['ent']:.3f} | {m['decisions_per_s']:.0f} dec/s "
                  f"roll {t_roll:.0f}s upd {t_upd:.0f}s", flush=True)
            if cfg.eval_every and self.it % cfg.eval_every == 0:
                self.model.eval()
                # Always on the full rules (the game the advice is for), whatever the training
                # rules. Sampled, not greedy: an early greedy policy can repeat a no-op such as
                # a minion move until the per-turn action cap, which makes evaluation crawl.
                # scripts/evaluate.py measures the greedy policy on snapshots.
                r = evaluate(self.model, cfg.eval_games, n_parallel=min(64, cfg.eval_games),
                             threads=cfg.threads, sample=True, device=self.device, rules="full")
                ev = {"iteration": self.it, **{f"eval_{k}": v for k, v in r.items()}}
                self.metrics.write(json.dumps(ev) + "\n")
                self.metrics.flush()
                if self.tb:
                    for k in ("avg_placement", "top1", "top4"):
                        self.tb.add_scalar(f"eval_vs_bots/{k}", r[k], self.it)
                print(f"     eval vs bots: place {r['avg_placement']:.2f} ±{r['ci95']:.2f} "
                      f"top1 {r['top1']:.2f} top4 {r['top4']:.2f} ({r['games']} games)", flush=True)
                if cfg.bench_dir:
                    from .tavernbench import bench_eval

                    b = bench_eval(self.model, cfg.bench_dir, self.run_dir / "tavernbench",
                                   self.device)
                    self.metrics.write(json.dumps({"iteration": self.it, **b}) + "\n")
                    self.metrics.flush()
                    if self.tb:
                        for k in ("bench_value", "bench_policy"):
                            self.tb.add_scalar(f"tavernbench/{k}", b[k], self.it)
                    print(f"     TavernBench decisions: value {b['bench_value']:.2f}  "
                          f"policy {b['bench_policy']:.2f}  (chance 0.24, TavernAgent 0.89)",
                          flush=True)
            self.maybe_snapshot()
            if self.it % cfg.checkpoint_every == 0 or self.it == cfg.iterations:
                self.save()
        self.save()
        self.save_policy()
