"""One-screen summary of a training run, sized for a phone.

    python scripts/status.py runs/full-01

Shows the latest iteration, recent training placement, every evaluation against the scripted
bots, speed and an ETA, read from metrics.jsonl (no TensorBoard needed).
"""

import json
import sys
from datetime import datetime
from pathlib import Path


def main():
    run = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    cfg = json.loads((run / "config.json").read_text())
    rows = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines() if x.strip()]
    train = [r for r in rows if "decisions" in r]
    evals = [r for r in rows if "eval_avg_placement" in r]
    bench = [r for r in rows if "bench_value" in r]
    if not train:
        print("no iterations logged yet")
        return
    # A resume from a checkpoint repeats the iterations after it: keep the newest of each.
    latest = lambda rs: list({r["iteration"]: r for r in rs}.values())  # noqa: E731
    train, evals, bench = latest(train), latest(evals), latest(bench)
    last = train[-1]
    it, total = last["iteration"], cfg["iterations"]
    recent = train[-10:]
    sec = sum(r["rollout_s"] + r["update_s"] for r in recent) / len(recent)
    print(f"{run.name}: iteration {it}/{total}  ({cfg['model']}, {cfg['opponents']}, "
          f"rules={cfg['rules']})")
    print(f"last update {last['time']}  ~{sec:.0f}s/iter  ETA {(total - it) * sec / 3600:.1f}h")
    pl = [r["avg_placement"] for r in recent if r["avg_placement"] is not None]
    if pl:
        print(f"training placement (last {len(pl)}): {sum(pl) / len(pl):.2f}  "
              f"kl {last['kl']:.4f}  entropy {last['ent']:.2f}  early-stops "
              f"{sum(r['early_stop'] for r in recent):.0f}/{len(recent)}")
    skipped = sum(r.get("skipped", 0) for r in train)
    if skipped:
        print(f"WARNING: {skipped} update steps skipped for non-finite gradients "
              f"(last at iteration {max(r['iteration'] for r in train if r.get('skipped'))})")
    tiers = [(t, last[f"tier_t{t}"]) for t in (2, 3, 4, 6, 8, 10) if f"tier_t{t}" in last]
    if tiers:  # real winners (Perfect Games): Tier 2 by turn 2-3, 2.45 at 4, 3.6 at 6, 4.9 at 8, 5.6 at 10
        print("tier by turn: " + "  ".join(f"t{t} {v:.2f}" for t, v in tiers)
              + "   (real winners: t2-3 2.0, t4 2.45, t6 3.6, t8 4.9, t10 5.6)")
    if "tier_t8" in last or "actions_per_turn" in last:
        print(f"behaviour at turn 8 End Turn: board {last.get('board_t8', '-')}  "
              f"tier {last.get('tier_t8', '-')}  gold left {last.get('gold_left_t8', '-')}  "
              f"| {last.get('actions_per_turn', '-')} actions/turn, "
              f"{last.get('move_frac', '-')} of them moves")
    if "difficulty" in last:
        pairs = "  ".join(f"t{t} {last[f'opp_given_t{t}']:.0f}/{last[f'opp_target_t{t}']:.0f}"
                          for t in (4, 6, 8, 10, 12) if f"opp_given_t{t}" in last)
        print(f"library opponents: difficulty {last['difficulty']:.2f} "
              f"(0 = library median, 1 = all players ... 5 = top 1%)  "
              f"board strength given/target: {pairs}")
    exams = latest([r for r in rows if any(k.startswith("exam_d") for k in r)])
    if exams:
        print("exam library, never trained on (placement, lower is better; 4.5 = even):")
        for r in exams[-5:]:
            cols = [k for k in r if k.startswith("exam_d") and not k.endswith(("_ci95", "_top4"))]
            print(f"  it {r['iteration']:4d}: " + "  ".join(
                f"{k[5:]} {r[k]:.2f} ±{r[k + '_ci95']:.2f}" for k in cols))
    if evals:
        print("vs scripted bots (lower is better; random = 7.95, bot = 4.50):")
        for r in evals[-8:]:
            print(f"  it {r['iteration']:4d}: {r['eval_avg_placement']:.2f} "
                  f"±{r['eval_ci95']:.2f}  top1 {r['eval_top1']:.2f}  top4 {r['eval_top4']:.2f}")
    if bench:
        print("TavernBench decisions (higher is better; chance 0.24, TavernAgent 0.89; long-term: "
              "hero, trinket, discover, turn plans, chance 0.27, TavernAgent 0.54):")
        for r in bench[-5:]:
            extra = ""
            if "bench_lookahead" in r:  # logged since 2026-11-02
                extra = (f"  lookahead {r['bench_lookahead']:.2f}  "
                         f"long-term {r['bench_policy_long_term']:.2f}")
            print(f"  it {r['iteration']:4d}: value {r['bench_value']:.2f}  "
                  f"policy {r['bench_policy']:.2f}{extra}")
    age = datetime.now() - datetime.strptime(last["time"], "%Y-%m-%d %H:%M:%S")
    if age.total_seconds() > max(3 * sec, 600):
        print(f"WARNING: no new iteration for {age}; the run may have stopped")


if __name__ == "__main__":
    main()
