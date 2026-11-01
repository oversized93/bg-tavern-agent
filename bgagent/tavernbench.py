"""Answer TavernBench decision tasks with a trained policy (no simulation, no search).

TavernBench ships with AutoTavern (tavernbench/ in a checkout of the commit pinned in
pyproject.toml). 288 of its 368 evaluation tasks come with the saved game they were cut from
and, for each answer option, the concrete actions behind it. 23 of its 30 "play" scenarios
are decisions our policy can make:

  selection   D1 hero, D2 trinket, D3 discover, D4 discover (final placement), D5 which card
  economy     E3 which minions to buy
  positioning P1-P5 arrangement of the warband
  ordering    O1-O5 order of a turn's actions
  targets     T1-T5 target of a spell / Activate / Hero Power / magnetize / sell
  strategy    N1-N2 first action of a turn plan (final placement)

The other seven play scenarios (B1-B6, R2) ask for combat facts (who attacks first, win /
tie / loss odds, damage): there is no decision to make, and the honest way to answer them is
TavernGym's combat simulator, which is how the labels were made. The ten rule scenarios need
no state. Neither is answered here.

Two ways to pick an option, both from the network alone:
  value   apply the option's actions to a copy of the game and read the value head (expected
          final placement) in the position that results; take the best.
  policy  what the policy itself does with only the task's options open: at each step its
          most likely next action among those that keep an option alive (an option already
          complete continues with End Turn). This is what the overlay would recommend. Hero
          choice (D1) and warband arrangement (P1-P5) have no single action sequence per
          option, so they fall back to the value.

Answers are written in the benchmark's answers.jsonl format and scored with TavernBench's
own scoring code.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import torch

from .policy import load_policy, observe_state, placement_from_value

DECISION_SCENARIOS = (
    [f"D{i}" for i in range(1, 6)] + ["E3"] + [f"P{i}" for i in range(1, 6)]
    + [f"O{i}" for i in range(1, 6)] + [f"T{i}" for i in range(1, 6)] + ["N1", "N2"]
)


def import_bench(bench_dir):
    """TavernBench's engine and report modules from an AutoTavern checkout."""
    root = Path(bench_dir).expanduser().resolve()
    if not (root / "tavernbench" / "engine.py").exists():
        raise FileNotFoundError(f"{root} is not AutoTavern's tavernbench/ folder")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tavernbench import engine, report  # noqa: E402

    return engine, report


def load_set(set_dir):
    d = Path(set_dir)
    prompts = {r["id"]: r for r in map(json.loads, open(d / "prompts.jsonl"))}
    labels = {r["id"]: r for r in map(json.loads, open(d / "labels.jsonl"))}
    states = {}
    with gzip.open(d / "states.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            states[r["id"]] = r
    return prompts, labels, states


class Recorder:
    """A Game that remembers (state before, seat, action) for every step it takes."""

    def __init__(self, game):
        self.game, self.steps = game, []

    def step(self, p, a):
        self.steps.append((self.game.state_json(), p, int(a)))
        return self.game.step(p, a)

    def __getattr__(self, name):
        return getattr(self.game, name)


class Option:
    """One answer: the actions behind it and the states along the way (states[k] is the
    position before actions[k]; states[-1] the one after the last action)."""

    def __init__(self, answer, seat, actions, states, value=None):
        self.answer, self.seat, self.actions, self.states = answer, seat, actions, states
        self.value = value  # value head in the final position (filled by the Scorer)

    @classmethod
    def recorded(cls, answer, rec: Recorder, seat=0):
        return cls(answer, seat, [a for _, _, a in rec.steps],
                   [s for s, _, _ in rec.steps] + [rec.game.state_json()])


class Scorer:
    def __init__(self, policy, engine, device="cpu"):
        """`policy`: a policy file, or a TavernNet already in memory (the trainer's)."""
        if isinstance(policy, (str, Path)):
            self.model, self.iteration = load_policy(policy, device)
        else:
            self.model, self.iteration = policy, None
        self.device = torch.device(device)
        self.E = engine

    @torch.inference_mode()
    def net(self, state_json, seat):
        """(log-probs over the 245 actions or None when the seat has no choice, value)."""
        f, c, m, ready = observe_state(state_json, seat)
        has_choice = ready and m[0].any()
        if not has_choice:  # the value is still defined; give the masked softmax something legal
            m = m.copy()
            m[0, 0] = True
        logits, value, _ = self.model(torch.from_numpy(f).to(self.device),
                                      torch.from_numpy(c).to(self.device),
                                      torch.from_numpy(m).to(self.device))
        lp = torch.log_softmax(logits[0].float(), -1).cpu().numpy() if has_choice else None
        return lp, float(value[0])

    # ---- the options of each task shape
    def from_ops(self, state, options, seat=0):
        out = []
        for opt in options:
            g = Recorder(self.E.from_state(state))
            if self.E.apply_ops(g, opt["ops"]):
                out.append(Option.recorded(opt["answer"], g, seat))
        return out

    def from_perms(self, state, perms, answers):
        """The warband reordered in the saved state (some positions have a choice pending,
        which makes minion moves illegal, and many move sequences give one arrangement)."""
        out = []
        for perm, ans in zip(perms, answers):
            st = json.loads(state)
            board = st["players"][0]["board"]
            st["players"][0]["board"] = [board[i] for i in perm]
            out.append(Option(ans, 0, None, [json.dumps(st)]))
        return out

    def from_actions(self, state, mapping, seat):
        """`mapping`: action index (as a string) -> answer; the choice is made now."""
        out = []
        for a, ans in mapping.items():
            g = Recorder(self.E.from_state(state))
            g.step(seat, int(a))
            out.append(Option.recorded(ans, g, seat))
        return out

    def from_states(self, states, answers):
        return [Option(ans, 0, None, [s]) for s, ans in zip(states, answers)]

    def options(self, rec):
        sc = rec["id"].split("-")[0]
        seat = rec.get("seat", 0)
        if sc == "D1":
            opts = self.from_states(rec["states"], rec["answers"])
        elif sc.startswith("P"):
            opts = self.from_perms(rec["state"], rec["perms"], rec["answers"])
        elif "options" in rec:
            opts = self.from_ops(rec["state"], rec["options"], seat)
        elif "map" in rec:
            opts = self.from_actions(rec["state"], rec["map"], seat)
        else:
            raise ValueError(f"unknown task shape: {sorted(rec)}")
        for o in opts:
            o.value = self.net(o.states[-1], o.seat)[1]
        return opts

    # ---- choosing
    def by_value(self, opts):
        return max(opts, key=lambda o: o.value).answer if opts else None

    def by_policy(self, opts):
        """What the policy does when only the task's options are open to it: at each step, its
        most likely next action among those that keep some option alive, where an option that
        is already complete continues with End Turn ("take no other action"). Options without
        actions (hero choice, warband arrangement) fall back to the value."""
        cands = [o for o in opts if o.actions is not None]
        if not cands:
            return self.by_value(opts)
        k = 0
        while True:
            done = [o for o in cands if len(o.actions) == k]
            nxt = {o.actions[k] for o in cands if len(o.actions) > k}
            if not nxt:
                return done[0].answer
            lp, _ = self.net(cands[0].states[k], cands[0].seat)
            if lp is None:
                return self.by_value(cands)
            choice = {a: lp[a] for a in nxt}
            if done:
                choice[self.E.A_END] = lp[self.E.A_END]
            best = max(choice, key=choice.get)
            if done and best == self.E.A_END and best not in nxt:
                return done[0].answer
            cands = [o for o in cands if len(o.actions) > k and o.actions[k] == best]
            k += 1


def answer_set(policy, set_dir, bench_dir, device="cpu", scenarios=None, log=print):
    """Answers for every decision task: {mode: {task id: answer}}, plus per-task details.
    `policy`: a policy file or a TavernNet."""
    engine, _ = import_bench(bench_dir)
    prompts, _, states = load_set(set_dir)
    scorer = Scorer(policy, engine, device)
    wanted = set(scenarios or DECISION_SCENARIOS)
    answers = {"value": {}, "policy": {}}
    details = []
    ids = [i for i in prompts if prompts[i]["scenario"] in wanted and i in states]
    for n, tid in enumerate(ids, 1):
        opts = scorer.options(states[tid])
        picks = {"value": scorer.by_value(opts), "policy": scorer.by_policy(opts)}
        for mode, a in picks.items():
            if a is not None:
                answers[mode][tid] = a
        details.append({"id": tid, **picks, "options": [
            {"answer": o.answer, "steps": None if o.actions is None else len(o.actions),
             "expected_place": round(placement_from_value(o.value), 3)} for o in opts]})
        if n % 25 == 0 or n == len(ids):
            log(f"  {n}/{len(ids)} tasks")
    return answers, details


def write_answers(path, answers, model_name):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for tid, a in answers.items():
            f.write(json.dumps({"id": tid, "model": model_name, "response": json.dumps({"answer": a}),
                                "error": None}) + "\n")


def summarize(report, set_dir, answers_path, scenarios=None):
    """TavernBench's own scoring (a task without an answer counts as wrong), averaged the way
    the paper does: a category is the mean of its scenarios, a total the mean over scenarios.
    `decision` covers the scenarios this module answers; `play` all 30 play scenarios.
    `answers_path=None` gives chance (the expected accuracy of a uniform random answer)."""
    labels, prompts = report.load(set_dir)
    if answers_path is None:
        ch = {}
        for iid, p in prompts.items():
            ch.setdefault(p["scenario"], []).append(report.expected(labels[iid], p))
        acc = {s: sum(v) / len(v) for s, v in ch.items()}
    else:
        acc, _ = report.evaluate(labels, prompts, answers_path)
    wanted = list(scenarios or DECISION_SCENARIOS)
    cats = {}
    for s in wanted:
        cats.setdefault(report.CATS[s[0]], []).append(acc[s])
    full = report.summarize(acc)
    return {"decision": sum(acc[s] for s in wanted) / len(wanted), "play": full["play"],
            "by_category": {c: sum(v) / len(v) for c, v in cats.items()},
            "by_scenario": {s: acc[s] for s in wanted}}


def bench_eval(policy, bench_dir, out_dir, device="cpu"):
    """Decision accuracy (value and policy) on the evaluation set, for the training log."""
    bench_dir = Path(bench_dir).expanduser()
    set_dir = bench_dir / "data" / "eval"
    _, report = import_bench(bench_dir)
    answers, _ = answer_set(policy, set_dir, bench_dir, device, log=lambda *a: None)
    out = {}
    for mode, a in answers.items():
        path = Path(out_dir) / f"answers_{mode}.jsonl"
        write_answers(path, a, mode)
        s = summarize(report, set_dir, path)
        out[f"bench_{mode}"] = round(s["decision"], 4)
        out.update({f"bench_{mode}_{c}": round(v, 4) for c, v in s["by_category"].items()})
    return out
