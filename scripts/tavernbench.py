"""Score a policy on TavernBench's decision tasks and compare it with the paper's systems.

    python scripts/tavernbench.py runs/g1-control/policy_00200.pt --bench ~/AutoTavern/tavernbench

Needs a checkout of AutoTavern at the commit pinned in pyproject.toml (TavernBench's tasks,
scoring code and the paper's released answers live there, not in the pip package):

    git clone https://github.com/MCG-NJU/AutoTavern ~/AutoTavern
    git -C ~/AutoTavern checkout fb61201a557ddd36a411cf44aac51ce4d9b606ce

Answers the 23 decision scenarios two ways (bgagent/tavernbench.py): by the value head after
each option ("value") and by the policy's probability of each option ("policy"). Writes
answers_value.jsonl, answers_policy.jsonl and details.jsonl to --out, and prints the
benchmark's accuracy next to the released systems' on the same scenarios. "play" is the
paper's 30-scenario score: the seven combat-fact scenarios we don't answer count as wrong.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bgagent import tavernbench as TB  # noqa: E402

CAT_ORDER = ["selection", "economy", "positioning", "ordering", "targets", "strategy"]


def row(name, s):
    cats = " ".join(f"{s['by_category'].get(c, float('nan')):5.2f}" for c in CAT_ORDER)
    return f"{name:24s} {s['decision']:6.2f} {s['play']:5.2f}  {cats}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("--bench", default="~/AutoTavern/tavernbench",
                    help="AutoTavern's tavernbench/ folder")
    ap.add_argument("--set", help="task set folder (default: <bench>/data/eval)")
    ap.add_argument("--out", help="output folder (default: runs/tavernbench/<policy name>)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compare", default="tavernagent,claude-opus-5.5,claude-sonnet-5.5,jev",
                    help="released systems (folders in <bench>/results/eval) to print alongside")
    args = ap.parse_args()

    bench = Path(args.bench).expanduser()
    set_dir = Path(args.set) if args.set else bench / "data" / "eval"
    p = Path(args.policy)
    out = Path(args.out) if args.out else Path("runs/tavernbench") / f"{p.parent.name}-{p.stem}"
    _, report = TB.import_bench(bench)

    answers, details = TB.answer_set(p, set_dir, bench, args.device)
    for mode, a in answers.items():
        TB.write_answers(out / f"answers_{mode}.jsonl", a, f"{p.parent.name}/{p.stem}:{mode}")
    with open(out / "details.jsonl", "w") as f:
        for d in details:
            f.write(json.dumps(d) + "\n")

    results = {"chance": TB.summarize(report, set_dir, None)}
    for name in args.compare.split(","):
        if name:
            results[name] = TB.summarize(report, set_dir, bench / "results" / "eval" / name /
                                         "answers.jsonl")
    for mode in answers:
        results[f"ours ({mode})"] = TB.summarize(report, set_dir, out / f"answers_{mode}.jsonl")
    (out / "summary.json").write_text(json.dumps(results, indent=1))

    print(f"\n{'system':24s} decis.  play  " + " ".join(f"{c[:5]:>5s}" for c in CAT_ORDER))
    for name, s in results.items():
        print(row(name, s))
    print("\ndecis. = mean accuracy over the 23 decision scenarios; play = the paper's 30 "
          "play scenarios\n(combat facts we don't answer count as wrong). By scenario: "
          f"{out / 'summary.json'}")


if __name__ == "__main__":
    main()
