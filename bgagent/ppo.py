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
from .model import FEATURE_MAX, ModelConfig, TavernNet, param_count
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
    # "library": the paper's opponents (bgagent/library.py). The learner plays `learner_seats`;
    # every other seat is a library seat that fights with a board from `library_path`, picked
    # to match a target strength curve. Difficulty d climbs a ladder: 0 = the library's own
    # median board, 1-5 = Firestone's MMR brackets (all players, top 50/25/10/1%; fetch with
    # scripts/fetch_firestone.py). After each iteration d moves by `difficulty_step`: up while
    # the learner's average placement is below `difficulty_up`, down while above
    # `difficulty_down`.
    opponents: str = "league"
    learner_seats: tuple = (0, 1, 2, 3)
    snapshot_seats: tuple = (4, 5)  # "hybrid" only
    library_path: str = ""
    exam_path: str = ""  # held-out library for evaluation, never trained on ("" = skip)
    exam_difficulties: tuple = (0.0, 1.0, 5.0)  # library median, all players, top 1%
    exam_games: int = 128
    difficulty: float = 0.0  # starting d (a resumed run keeps its own)
    difficulty_step: float = 0.05
    difficulty_up: float = 4.0
    difficulty_down: float = 5.0
    # Lowest d. Below 0 the library's median boards are scaled down by 10**d (d = -1: a tenth
    # of the median), so a policy trained from scratch has opponents it can beat: the paper
    # trains against library seats only, "from aligned libraries to harder ones".
    difficulty_min: float = 0.0
    library_sigma: float = 0.25  # spread of each turn's target strength (log-normal sigma)
    # A new run starts from these policy weights (and their architecture) instead of random
    # ones: the paper's later stages start from the previous stage. Ignored on resume.
    init_from: str = ""
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
    # The end-of-game reward for places 1..8; () = TavernGym's linear one (1st = 1 ... 8th =
    # -1). The paper names a "first-place goal" stage (Table 6, TA-3) but gives no numbers
    # (NOTES.md, 2026-11-02).
    placement_utility: tuple = ()
    # KL anchor (the paper's Table 7: 0.1 for the first 100 iterations of each stage): adds
    # anchor_coef * KL(policy || the stage's starting policy) to the loss for the first
    # anchor_iters iterations. The starting policy is `init_from`, so it needs one.
    anchor_coef: float = 0.0
    anchor_iters: int = 100
    # Levelling bootcamp (bgagent/bootcamp.py; NOTES.md, 2026-11-02): (turn, tier) pairs; while
    # a learner seat is below them and can afford Upgrade, it upgrades instead of acting, in a
    # fraction `level_force` of its episodes. `level_force_decay` = (start, end) iterations:
    # the fraction falls linearly to 0 between them (the hand-back); () = constant. Forced
    # steps train the value head, not the policy. Evaluations use the same forcing.
    # bootcamp.WINNERS = real winners' averages (Cranny's OK, 2026-11-02). () = off.
    level_curve: tuple = ()
    level_force: float = 1.0
    level_force_decay: tuple = ()
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
    # Iterations over which lr falls linearly to lr_final; 0 = `iterations`. Set it to the
    # planned length to stop early and extend later (raise `iterations`) without the schedule
    # jumping.
    lr_schedule_iters: int = 0
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
    # Weight of the next-combat auxiliary loss (ModelConfig.combat_head); 0 = no head. Only
    # for new runs: a checkpoint without the head can't grow one on resume.
    combat_coef: float = 0.0
    max_grad_norm: float = 1.0
    seed: int = 0
    checkpoint_every: int = 5
    max_traj_len: int = 4000  # drop a trajectory that runs this long (a stuck game)
    device: str = "auto"
    # Cap on PyTorch's share of GPU memory (0 = no cap). On Windows the NVIDIA driver doesn't
    # fail when VRAM runs out: it spills into system RAM and the update gets ~5x slower
    # (NOTES.md, 2026-11-01). The cap makes an oversized micro_batch raise out-of-memory
    # instead, which the setup scripts' memory check catches.
    gpu_mem_fraction: float = 0.8

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
                               minibatch=8192, micro_batch=512, lr=3e-4, lr_final=1e-4)
        raise KeyError(name)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def store_features(feats):
    """Observation floats as stored in a rollout: float16, clipped to the range the model
    clamps its input to, so a huge stat can't overflow to inf."""
    return np.clip(feats, -FEATURE_MAX, FEATURE_MAX).astype(np.float16)


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
        league = cfg.opponents in ("league", "hybrid", "library")
        controlled = {"league": tuple(range(8)), "library": tuple(range(8)),
                      "mixed": tuple(cfg.learner_seats),
                      "hybrid": tuple(sorted(set(cfg.learner_seats) | set(cfg.snapshot_seats))),
                      "bots": (0,)}[cfg.opponents]
        self.env = taverngym.BatchEnv(cfg.n_games, seed=cfg.seed, controlled=controlled,
                                      config=rules_config(cfg.rules), threads=cfg.threads)
        n = self.env.n_slots
        k = len(controlled)
        seats = np.array([controlled[i % k] for i in range(n)])
        self.learner = np.isin(seats, cfg.learner_seats) if league else np.ones(n, bool)
        self.library_seats = None
        if cfg.opponents == "library":
            from .library import Library, LibrarySeats, ladder

            if not cfg.library_path:
                raise ValueError("opponents=library needs library_path (scripts/build_library.py)")
            lib = Library.load(cfg.library_path)
            others = tuple(s for s in range(8) if s not in set(cfg.learner_seats))
            self.library_seats = LibrarySeats(self.env, others, lib, ladder(lib),
                                              d=cfg.difficulty, sigma=cfg.library_sigma,
                                              seed=cfg.seed)
        self.level = None
        if cfg.level_curve:
            from .bootcamp import LevelCurve

            self.level = LevelCurve(self.env, cfg.level_curve, cfg.level_force, seed=cfg.seed)
        self.open = [[] for _ in range(n)]
        self.obs = self.env.observe()
        self.opponent = None  # frozen model for league seats
        self.filter = None  # built from the model's filter settings on first collect
        self.stats_reset()

    def stats_reset(self):
        self.placements, self.ep_lens, self.rejected, self.dropped = [], [], 0, 0
        # behaviour at each learner End Turn, by turn: (warband size, Tavern Tier, gold left)
        # (turns 2 and 3: real players are on Tier 2 by then, the earliest sign of levelling)
        self.end_turn = {t: [] for t in (2, 3, 4, 6, 8, 10)}
        self.n_actions, self.n_moves, self.n_end_turns, self.n_forced = 0, 0, 0, 0
        if getattr(self, "library_seats", None) is not None:
            self.library_seats.stats_reset()

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
        if self.level is not None:
            out["level_force"] = round(self.level.p, 3)
            out["forced_upgrades_per_game"] = round(self.n_forced / max(len(self.placements), 1), 2)
        if self.library_seats is not None:  # what the opponents fought with, vs the target
            for t in (4, 6, 8, 10, 12):
                rows = self.library_seats.given.get(t)
                if rows:
                    tg, gv = np.mean(rows, 0)
                    out[f"opp_target_t{t}"] = round(float(tg), 1)
                    out[f"opp_given_t{t}"] = round(float(gv), 1)
            out["difficulty"] = round(self.library_seats.d, 3)
        return out

    def collect(self, model):
        cfg, env = self.cfg, self.env
        done_trajs, n_dec = [], 0
        t0 = time.time()
        while n_dec < cfg.decisions_per_iter:
            o = self.obs
            if self.library_seats is not None and self.library_seats.prepare(o):
                o = self.obs = env.observe()  # library seats got this turn's boards
            ready = o["ready"]
            actions = np.zeros(env.n_slots, dtype=np.int64)
            if self.filter is None:
                self.filter = ActionFilter.for_model(env.n_slots, model)
            lr_idx = np.nonzero(ready & self.learner)[0]
            if len(lr_idx):
                lm = self.filter.apply(o["features"][lr_idx], o["action_mask"][lr_idx], lr_idx)
                a, lp, v = act(model, o["features"][lr_idx], o["cards"][lr_idx], lm,
                               self.device, self.amp)
                forced = np.zeros(len(a), bool)
                if self.level is not None:  # the bootcamp: Upgrade on the curve
                    forced = self.level.apply(o["features"][lr_idx], lm, a, lr_idx)
                    self.n_forced += int(forced.sum())
                self.filter.record(lr_idx, a)
                actions[lr_idx] = a
                stored = store_features(o["features"][lr_idx])
                for j, i in enumerate(lr_idx):
                    f, aj = o["features"][i], int(a[j])
                    is_move = aj >= L.A_MOVE
                    self.open[i].append([
                        stored[j].copy(), o["cards"][i].astype(np.int16),  # copy: no view
                        np.packbits(lm[j]), aj, float(lp[j]),
                        float(v[j]), -cfg.move_cost if is_move else 0.0,
                        bool(forced[j]),  # not the policy's choice: no policy gradient
                    ])
                    self.n_actions += 1
                    self.n_moves += is_move
                    if aj == L.A_END and f[1] == 0:  # End Turn outside hero select
                        self.n_end_turns += 1
                        t = int(round(f[0] * 20))
                        if t in self.end_turn:  # global floats: turn/20, tier/6, gold/10, board/7
                            self.end_turn[t].append((f[33] * 7, f[10] * 6, f[7] * 10))
            op_idx = np.nonzero(ready & ~self.learner)[0]
            if len(op_idx) and self.library_seats is not None:
                actions[op_idx] = self.library_seats.actions(o, op_idx)
            elif len(op_idx):
                om = self.filter.apply(o["features"][op_idx], o["action_mask"][op_idx], op_idx)
                a, _, _ = act(self.opponent or model, o["features"][op_idx],
                              o["cards"][op_idx], om, self.device, self.amp)
                self.filter.record(op_idx, a)
                actions[op_idx] = a
            rewards, dones, info = env.step(actions)
            if self.library_seats is not None:
                self.library_seats.end_finished(dones)
            self.rejected += int((info["rejected"] & self.learner).sum())
            hit = np.nonzero(((rewards != 0) | dones) & self.learner)[0]
            for i in hit:
                traj = self.open[i]
                if traj:
                    traj[-1][6] += float(rewards[i])
                if dones[i]:
                    place = int(info["placement"][i])
                    if traj and place > 0:
                        traj[-1][6] += utility_adjustment(place, cfg.placement_utility)
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


F_HEALTH, F_ARMOR, F_HERO_SELECT = 5, 6, 1  # global floats: health / 40, armor / 20


def linear_reward(place, n=8):
    """TavernGym's placement reward (env.rs `placement_reward`): 1st = 1, last = -1."""
    return 1.0 - 2.0 * (place - 1) / (n - 1)


def utility_adjustment(place, utility):
    """What to add to TavernGym's end reward so that it becomes `utility[place - 1]`."""
    if not utility:
        return 0.0
    return float(utility[place - 1]) - linear_reward(place, len(utility))


def combat_labels(f, place):
    """Per decision, the combat right after its turn: (lost it, damage taken, weight).

    Read from the seat's own observations: Health + Armor at its last decision of a turn
    against its first decision of the next. In the seat's last turn, 1st place won that
    combat and every other place died in it. Hero select has no combat (weight 0)."""
    f = f.astype(np.float32)
    turn = np.round(f[:, 0] * 20).astype(np.int64)
    hp = f[:, F_HEALTH] * 40 + f[:, F_ARMOR] * 20
    dmg = np.zeros(len(f), np.float32)
    starts = np.flatnonzero(np.r_[True, turn[1:] != turn[:-1]])
    ends = np.r_[starts[1:], len(f)]
    for k, (s, e) in enumerate(zip(starts, ends)):
        if k + 1 < len(starts):
            d = max(0.0, hp[e - 1] - hp[starts[k + 1]])
        else:
            d = 0.0 if place == 1 else max(hp[e - 1], 1.0)
        dmg[s:e] = d
    w = (f[:, F_HERO_SELECT] < 0.5).astype(np.float32)
    return (dmg > 0).astype(np.float32), dmg, w


def build_batch(trajs, gamma, lam):
    feats, cards, masks, acts, logps, vals, advs, rets, places = ([] for _ in range(9))
    lost, dmg, cw, train_pi = [], [], [], []
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
        # 0 for actions the bootcamp forced (bootcamp.py): they train the value head only
        train_pi.append(np.array([0.0 if len(t) > 7 and t[7] else 1.0 for t in traj], np.float32))
        lo, dm, w = combat_labels(feats[-1], place)
        lost.append(lo)
        dmg.append(dm)
        cw.append(w)
    cat = np.concatenate
    return {
        "feats": cat(feats), "cards": cat(cards),
        "mask": np.unpackbits(cat(masks), axis=1, count=L.N_ACTIONS).astype(bool),
        "act": cat(acts), "logp": cat(logps), "val": cat(vals), "adv": cat(advs),
        "ret": cat(rets), "place": cat(places),
        "lost": cat(lost), "dmg": cat(dmg), "cw": cat(cw), "train_pi": cat(train_pi),
    }


def ppo_update(model, opt, batch, cfg: TrainConfig, device, anchor=None, anchor_coef=0.0):
    """One PPO update. With `anchor` (a frozen model) and anchor_coef > 0, the loss also has
    anchor_coef * KL(policy || anchor) over the legal actions."""
    amp = device.type == "cuda"
    n = len(batch["act"])
    if "train_pi" not in batch:  # every action was the policy's own
        batch = {**batch, "train_pi": np.ones(n, np.float32)}
    T = {k: torch.from_numpy(v) for k, v in batch.items()}
    stats = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "aux": 0.0, "combat": 0.0, "kl": 0.0,
             "clipfrac": 0.0, "anchor_kl": 0.0}
    use_anchor = anchor is not None and anchor_coef > 0
    use_combat = cfg.combat_coef > 0
    steps, skipped, stop = 0, 0, False
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
                    out = model(g("feats", torch.float32), g("cards"), g("mask"),
                                combat=use_combat)
                logits, value, place = out[:3]
                logp_all = torch.log_softmax(logits, -1)
                logp = logp_all.gather(1, g("act")[:, None]).squeeze(-1)
                ratio = torch.exp(logp - g("logp"))
                adv = (g("adv") - adv_mean.to(device)) / adv_std.to(device)
                # Actions the bootcamp forced (train_pi = 0) were not sampled from the policy:
                # no policy gradient, and they don't count toward the KL / clip statistics.
                tp = g("train_pi")
                n_pi = tp.sum().clamp(min=1.0)
                ratio = torch.where(tp > 0, ratio, torch.ones_like(ratio))  # no 0 * inf = nan
                pg = -(torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv)
                       * tp).sum() / n_pi
                vf = 0.5 * (value - g("ret")).pow(2).mean()
                p = logp_all.exp()
                ent = -(p * logp_all.masked_fill(~g("mask"), 0.0)).sum(-1).mean()
                aux = Fn.cross_entropy(place, g("place"))
                loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent + cfg.aux_coef * aux
                combat = torch.zeros((), device=device)
                if use_combat:
                    c, cw = out[3], g("cw")
                    lost_l = Fn.binary_cross_entropy_with_logits(c[:, 0], g("lost"),
                                                                 reduction="none")
                    dmg_l = Fn.smooth_l1_loss(c[:, 1], g("dmg") / 10, reduction="none")
                    combat = ((lost_l + dmg_l) * cw).sum() / cw.sum().clamp(min=1)
                    loss = loss + cfg.combat_coef * combat
                anchor_kl = torch.zeros((), device=device)
                if use_anchor:
                    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16,
                                                         enabled=amp):
                        a_logits = anchor(g("feats", torch.float32), g("cards"), g("mask"))[0]
                    a_logp = torch.log_softmax(a_logits.float(), -1)
                    # illegal actions: p = 0 and both log-probs -inf; mask them, not 0 * nan
                    diff = (logp_all - a_logp).masked_fill(~g("mask"), 0.0)
                    anchor_kl = (p * diff).sum(-1).mean()
                    loss = loss + anchor_coef * anchor_kl
                (loss * w).backward()
                with torch.no_grad():
                    kl = (((ratio - 1) - torch.log(ratio)) * tp).sum() / n_pi  # k3 estimator
                    mb_stats["kl"] += kl.item() * w
                    mb_stats["clipfrac"] += (((ratio - 1).abs() > cfg.clip).float() * tp).sum().item() \
                        / n_pi.item() * w
                for k, val in (("pg", pg), ("vf", vf), ("ent", ent), ("aux", aux),
                               ("combat", combat), ("anchor_kl", anchor_kl)):
                    mb_stats[k] += val.item() * w
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            if not torch.isfinite(norm):  # one bad minibatch must not make every weight NaN
                opt.zero_grad(set_to_none=True)
                skipped += 1
                continue
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
    out.update(updates=steps, early_stop=float(stop), skipped=skipped)
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
        if self.device.type == "cuda" and cfg.gpu_mem_fraction:
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            torch.cuda.set_per_process_memory_fraction(cfg.gpu_mem_fraction, index)
        mcfg = ModelConfig.preset(cfg.model)
        mcfg.factored = cfg.factored
        mcfg.move_cap = cfg.move_cap
        mcfg.no_refreeze = cfg.no_refreeze
        mcfg.cancel_cap = cfg.cancel_cap
        mcfg.combat_head = cfg.combat_coef > 0
        latest = self.run_dir / "latest.pt"
        init = None
        if not latest.exists() and cfg.init_from:
            from .policy import model_config_from

            init = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
            mcfg = model_config_from(init)
            mcfg.move_cap, mcfg.no_refreeze = cfg.move_cap, cfg.no_refreeze
            mcfg.cancel_cap = cfg.cancel_cap
        if latest.exists():  # resume with the architecture the run was started with
            from .policy import model_config_from

            mcfg = model_config_from(torch.load(latest, map_location="cpu", weights_only=False))
            # the action filter is not architecture: the run's config decides it
            mcfg.move_cap, mcfg.no_refreeze = cfg.move_cap, cfg.no_refreeze
            mcfg.cancel_cap = cfg.cancel_cap
            if cfg.combat_coef > 0 and not mcfg.combat_head:
                raise ValueError("combat_coef > 0, but this run's model has no combat head; "
                                 "start a new run")
        self.model = TavernNet(mcfg).to(self.device)
        if init is not None:
            self.model.load_state_dict(init["model"])
            print(f"initialized from {cfg.init_from} (its iteration {init.get('iteration')})")
        if cfg.level_curve:
            from .bootcamp import check_curve

            check_curve(cfg.level_curve)
            if cfg.level_force_decay and (len(cfg.level_force_decay) != 2
                                          or cfg.level_force_decay[1] <= cfg.level_force_decay[0]):
                raise ValueError(f"level_force_decay needs (start, end) iterations, end > start; "
                                 f"got {cfg.level_force_decay}")
        if cfg.placement_utility and len(cfg.placement_utility) != 8:
            raise ValueError(f"placement_utility needs 8 values (places 1..8), got "
                             f"{cfg.placement_utility}")
        self.anchor = None
        if cfg.anchor_coef > 0:
            if not cfg.init_from:
                raise ValueError("anchor_coef > 0 needs init_from: the anchor is the stage's "
                                 "starting policy")
            from .policy import model_config_from

            a = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
            acfg = model_config_from(a)
            acfg.move_cap, acfg.no_refreeze, acfg.cancel_cap = mcfg.move_cap, mcfg.no_refreeze, \
                mcfg.cancel_cap
            self.anchor = TavernNet(acfg).to(self.device).eval()
            self.anchor.load_state_dict(a["model"])
            self.anchor.requires_grad_(False)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=0.0,
                                     eps=1e-5)
        self.it = 0
        self.league = []  # list of (iteration, cpu state dict)
        self.difficulty = cfg.difficulty  # library opponents' ladder position
        self.load_latest()
        self.collector = Collector(cfg, self.device)
        if self.collector.library_seats is not None:
            self.collector.library_seats.d = self.difficulty
        self.exam = None
        if cfg.exam_path:
            from .library import Library, ladder

            exam = Library.load(cfg.exam_path)
            self.exam = (exam, ladder(exam))
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
            "difficulty": self.difficulty,
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
        if "difficulty" in ck and self.cfg.opponents == "library" and ck["config"].get(
                "opponents") == "library":
            self.difficulty = ck["difficulty"]
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
    def adapt_difficulty(self, placement):
        """Library opponents: climb the ladder while the learner wins, step back while it
        loses (the thresholds are TrainConfig.difficulty_up / difficulty_down)."""
        seats = self.collector.library_seats
        if seats is None or placement is None:
            return
        cfg = self.cfg
        if placement < cfg.difficulty_up:
            self.difficulty += cfg.difficulty_step
        elif placement > cfg.difficulty_down:
            self.difficulty -= cfg.difficulty_step
        self.difficulty = float(np.clip(self.difficulty, cfg.difficulty_min, seats.difficulty.max))
        seats.d = self.difficulty

    def exam_eval(self):
        """The paper's exam: the policy against a library it never trained on, at fixed
        difficulties (TrainConfig.exam_difficulties)."""
        cfg = self.cfg
        lib, lad = self.exam
        out = {"iteration": self.it}
        for dd in cfg.exam_difficulties:
            r = evaluate(self.model, cfg.exam_games, n_parallel=min(64, cfg.exam_games),
                         threads=cfg.threads, sample=True, device=self.device, rules="full",
                         library=lib, difficulty=lad, d=dd, sigma=cfg.library_sigma,
                         level=self.level())
            out[f"exam_d{dd:g}"] = r["avg_placement"]
            out[f"exam_d{dd:g}_ci95"] = r["ci95"]
            out[f"exam_d{dd:g}_top4"] = r["top4"]
        return out

    def level_force_at(self, it):
        """The bootcamp's forcing fraction at iteration `it` (falls to 0 over the decay)."""
        cfg = self.cfg
        if not cfg.level_force_decay:
            return cfg.level_force
        a, b = cfg.level_force_decay
        return cfg.level_force * float(np.clip((b - it) / (b - a), 0.0, 1.0))

    def level(self):
        """The bootcamp as evaluations use it: (curve, fraction), or None."""
        if not self.cfg.level_curve:
            return None
        return self.cfg.level_curve, self.level_force_at(self.it)

    def lr_at(self, it):
        cfg = self.cfg
        span = cfg.lr_schedule_iters or cfg.iterations
        frac = min(it / max(span - 1, 1), 1.0)
        return cfg.lr + (cfg.lr_final - cfg.lr) * frac

    def run(self):
        cfg = self.cfg
        print(f"device={self.device} model={cfg.model} params={param_count(self.model)/1e6:.1f}M "
              f"games={cfg.n_games} opponents={cfg.opponents} rules={cfg.rules}")
        while self.it < cfg.iterations:
            for g in self.opt.param_groups:
                g["lr"] = self.lr_at(self.it)
            self.collector.opponent = self.pick_opponent()
            if self.collector.level is not None:
                self.collector.level.p = self.level_force_at(self.it)
            self.collector.stats_reset()
            self.model.eval()
            trajs, t_roll = self.collector.collect(self.model)
            batch = build_batch(trajs, cfg.gamma, cfg.lam)
            self.model.train()
            t1 = time.time()
            coef = cfg.anchor_coef if self.it < cfg.anchor_iters else 0.0
            up = ppo_update(self.model, self.opt, batch, cfg, self.device, self.anchor, coef)
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
            self.adapt_difficulty(m["avg_placement"])
            fmt = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
            print(f"it {self.it:4d} | place {fmt(m['avg_placement'])} top4 {fmt(m['top4'])} "
                  f"| t8 board {fmt(m.get('board_t8'))} tier {fmt(m.get('tier_t8'))} "
                  f"| acts/turn {m.get('actions_per_turn', '-')} moves {m.get('move_frac', '-')} "
                  f"| kl {m['kl']:.4f} ent {m['ent']:.3f} | {m['decisions_per_s']:.0f} dec/s "
                  f"roll {t_roll:.0f}s upd {t_upd:.0f}s"
                  + (f" | d {m['difficulty']:.2f} opp t8 {m.get('opp_given_t8', '-')}"
                     if "difficulty" in m else "")
                  + (f" | skipped {up['skipped']} non-finite steps" if up["skipped"] else ""),
                  flush=True)
            if cfg.eval_every and self.it % cfg.eval_every == 0:
                self.model.eval()
                # Always on the full rules (the game the advice is for), whatever the training
                # rules. Sampled, not greedy: an early greedy policy can repeat a no-op such as
                # a minion move until the per-turn action cap, which makes evaluation crawl.
                # scripts/evaluate.py measures the greedy policy on snapshots.
                r = evaluate(self.model, cfg.eval_games, n_parallel=min(64, cfg.eval_games),
                             threads=cfg.threads, sample=True, device=self.device, rules="full",
                             level=self.level())
                ev = {"iteration": self.it, **{f"eval_{k}": v for k, v in r.items()}}
                self.metrics.write(json.dumps(ev) + "\n")
                self.metrics.flush()
                if self.tb:
                    for k in ("avg_placement", "top1", "top4"):
                        self.tb.add_scalar(f"eval_vs_bots/{k}", r[k], self.it)
                print(f"     eval vs bots: place {r['avg_placement']:.2f} ±{r['ci95']:.2f} "
                      f"top1 {r['top1']:.2f} top4 {r['top4']:.2f} ({r['games']} games)", flush=True)
                if self.exam is not None:
                    ex = self.exam_eval()
                    self.metrics.write(json.dumps(ex) + "\n")
                    self.metrics.flush()
                    if self.tb:
                        for k, v in ex.items():
                            if k.startswith("exam_d") and not k.endswith(("_ci95", "_top4")):
                                self.tb.add_scalar(f"exam/{k}", v, self.it)
                    print("     exam library (never trained on): " + "  ".join(
                        f"d{dd:g} {ex[f'exam_d{dd:g}']:.2f} ±{ex[f'exam_d{dd:g}_ci95']:.2f}"
                        for dd in cfg.exam_difficulties), flush=True)
                if cfg.bench_dir:
                    from .tavernbench import bench_eval

                    b = bench_eval(self.model, cfg.bench_dir, self.run_dir / "tavernbench",
                                   self.device)
                    self.metrics.write(json.dumps({"iteration": self.it, **b}) + "\n")
                    self.metrics.flush()
                    if self.tb:
                        for k in ("bench_value", "bench_policy", "bench_lookahead",
                                  "bench_policy_long_term"):
                            self.tb.add_scalar(f"tavernbench/{k}", b[k], self.it)
                    print(f"     TavernBench decisions: value {b['bench_value']:.2f}  "
                          f"policy {b['bench_policy']:.2f}  lookahead {b['bench_lookahead']:.2f}  "
                          f"long-term {b['bench_policy_long_term']:.2f}  (chance 0.24 / 0.27, "
                          f"TavernAgent 0.89 / 0.54)", flush=True)
            self.maybe_snapshot()
            if self.it % cfg.checkpoint_every == 0 or self.it == cfg.iterations:
                self.save()
        self.save()
        self.save_policy()
