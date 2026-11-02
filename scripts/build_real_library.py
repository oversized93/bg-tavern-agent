"""Opponent libraries from real games: Firestone's Perfect Games replays (bgagent/replay.py).

    python scripts/build_real_library.py                  # every replay in data/firestone/perfect
    python scripts/build_real_library.py --min-rank 6814  # top 25% games only

Step 1 reads every replay (in parallel) into boards.jsonl.gz next to them: per game its
reviewId, the winner's rating, build, and each combat's two boards (the winner's and the
opponent's), with no account names or ids. Replays already read are not read again.
Step 2 turns the boards into library entries (TavernGym's format) and splits them by game:
--exam-share of the games (chosen by a hash of their reviewId, so the split is stable as the
collection grows) go to the exam library, never trained on; the rest to the training one.

Which boards: by default the opponents' boards, the fair sample of players at that rating, and
the winners' boards too (--sides). These calibrate opponents only; no move is ever learned
from them (CLAUDE.md).
"""

import argparse
import gzip
import hashlib
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bgagent.library import Library  # noqa: E402
from bgagent.replay import Board, EntryBuilder, Minion, read_replay  # noqa: E402


def _read(path):
    try:
        g = read_replay(path)
        return Path(path).name.replace(".xml.zip", ""), g.to_json(), None
    except Exception as e:  # one bad file shouldn't stop the rest
        return Path(path).name.replace(".xml.zip", ""), None, f"{type(e).__name__}: {e}"[:200]


def _exam(review_id, share):
    h = int(hashlib.sha1(review_id.encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < share


def main():
    ap = argparse.ArgumentParser()
    perfect = ROOT / "data" / "firestone" / "perfect"
    ap.add_argument("--replays", default=str(perfect))
    ap.add_argument("--train", default=str(ROOT / "data" / "library" / "real-train.jsonl.gz"))
    ap.add_argument("--exam", default=str(ROOT / "data" / "library" / "real-exam.jsonl.gz"))
    ap.add_argument("--exam-share", type=float, default=0.2)
    ap.add_argument("--min-rank", type=int, default=0, help="only games whose winner had this rating")
    ap.add_argument("--sides", default="opponent,main", help="opponent, main, or both")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    src = Path(args.replays)

    ranks = {}
    for line in (src / "games.jsonl").read_text(encoding="utf-8").splitlines():
        g = json.loads(line)
        ranks[g["reviewId"]] = int(g["playerRank"]) if str(g.get("playerRank", "")).isdigit() else 0

    # ---- step 1: replays -> boards.jsonl.gz (cached per game)
    boards_path = src / "boards.jsonl.gz"
    games = {}
    if boards_path.exists():
        with gzip.open(boards_path, "rt", encoding="utf-8") as f:
            for line in f:
                g = json.loads(line)
                games[g["reviewId"]] = g
    todo = [p for p in sorted((src / "replays").glob("*.xml.zip"))
            if p.name.replace(".xml.zip", "") not in games]
    t0 = time.time()
    failed = []
    if todo:
        with ProcessPoolExecutor(args.workers) as pool:
            for i, (rid, g, err) in enumerate(pool.map(_read, todo, chunksize=4), 1):
                if g is None:
                    failed.append((rid, err))
                    continue
                g["reviewId"], g["rank"] = rid, ranks.get(rid, 0)
                games[rid] = g
                if i % 100 == 0:
                    print(f"  read {i}/{len(todo)}", flush=True)
        with gzip.open(boards_path, "wt", encoding="utf-8") as f:
            for g in games.values():
                f.write(json.dumps(g) + "\n")
    print(f"step 1: {len(todo)} replays read in {time.time() - t0:.0f}s ({len(failed)} failed), "
          f"{len(games)} games in {boards_path.name}")
    for rid, err in failed[:5]:
        print(f"  failed {rid}: {err}")

    # ---- step 2: boards -> library entries
    sides = {s.strip() for s in args.sides.split(",")}
    eb = EntryBuilder()
    train, exam = Library(), Library()
    n_games = {"train": 0, "exam": 0}
    t0 = time.time()
    for rid, g in games.items():
        if g["rank"] < args.min_rank:
            continue
        lib, name = (exam, "exam") if _exam(rid, args.exam_share) else (train, "train")
        n_games[name] += 1
        for b in g["boards"]:
            if b["side"] not in sides or not b["minions"]:
                continue
            board = Board(turn=b["turn"], side=b["side"], hero=b["hero"], tier=b["tier"],
                          minions=[Minion(**m) for m in b["minions"]])
            e = eb.entry(board)
            if e is not None:
                lib.add(e)
    train.save(args.train)
    exam.save(args.exam)
    print(f"step 2: {len(train)} train entries ({n_games['train']} games), {len(exam)} exam "
          f"({n_games['exam']} games) in {time.time() - t0:.0f}s; sides {sorted(sides)}, "
          f"min rank {args.min_rank}")
    if eb.unknown:
        print(f"  minions TavernGym doesn't know (skipped): {sum(eb.unknown.values())}, e.g. "
              f"{sorted(eb.unknown.items(), key=lambda kv: -kv[1])[:8]}")
    by_turn = defaultdict(list)
    for t in train.turns():
        by_turn[t] = [s for s, _ in train.by_turn[t]]
    print("train median / mean strength by turn:",
          {t: (round(float(np.median(v))), round(float(np.mean(v)))) for t, v in sorted(by_turn.items())
           if t <= 16 and t % 2 == 0})


if __name__ == "__main__":
    main()
