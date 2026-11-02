"""Download Firestone's Battlegrounds "Perfect Games" (its app's tab of recent first-place
games, each with its full replay) into data/firestone/perfect.

    python scripts/fetch_perfect_games.py            # the list, then every replay not yet here
    python scripts/fetch_perfect_games.py --limit 50

The list holds the newest 1,000 games and keeps moving, so each run merges it into
games.jsonl (one line per game, by reviewId) and the collection grows. Replays are the
HSReplay XML the app's replay viewer loads (xml.firestoneapp.com/<replayKey>, ~1 MB zipped),
kept zipped under replays/.

Same terms as fetch_firestone.py (Firestone's developer: "So long as the data is not used
commercially, all data available on the application itself can be fully utilized by any
other developer"): personal, non-commercial, never redistributed, under data/ (ignored by
git). One request every --delay seconds; any HTTP error stops the run rather than retrying.
These games calibrate opponents and evaluation only: the policy never trains on their moves.
Data: Firestone (https://www.firestoneapp.com).
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_firestone import UA, fetch  # noqa: E402

LIST_URL = "https://static.zerotoheroes.com/api/bgs/bgs-perfect-games.json"
REPLAY_URL = "https://xml.firestoneapp.com/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "data" / "firestone" / "perfect"))
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between requests")
    ap.add_argument("--limit", type=int, default=0, help="at most this many replays (0 = all)")
    ap.add_argument("--max-age", type=float, default=12.0, help="hours before re-fetching the list")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "replays").mkdir(parents=True, exist_ok=True)

    index = out / "games.jsonl"
    games = {}
    if index.exists():
        for line in index.read_text(encoding="utf-8").splitlines():
            g = json.loads(line)
            games[g["reviewId"]] = g
    stamp = out / "list.fetched"
    if not stamp.exists() or time.time() - stamp.stat().st_mtime > args.max_age * 3600:
        new = 0
        for g in fetch(LIST_URL):
            if g.get("reviewId") and g.get("replayKey") and g["reviewId"] not in games:
                games[g["reviewId"]] = g
                new += 1
        index.write_text("".join(json.dumps(g) + "\n" for g in games.values()), encoding="utf-8")
        stamp.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
        print(f"list: {new} new games, {len(games)} in total")
        time.sleep(args.delay)
    else:
        print(f"list: cached, {len(games)} games")

    todo = [g for g in games.values()
            if not (out / "replays" / f"{g['reviewId']}.xml.zip").exists()]
    if args.limit:
        todo = todo[: args.limit]
    print(f"replays: {len(games) - len(todo)} here, fetching {len(todo)}")
    for i, g in enumerate(todo, 1):
        key = g["replayKey"]
        if not key.endswith(".zip"):
            print(f"skip {g['reviewId']}: not a zip ({key})")
            continue
        req = urllib.request.Request(REPLAY_URL + key, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:  # an expired replay: note it and move on
                print(f"missing {g['reviewId']} (404)")
                time.sleep(args.delay)
                continue
            print(f"stopping: HTTP {e.code} on {g['reviewId']}", file=sys.stderr)
            return 1
        path = out / "replays" / f"{g['reviewId']}.xml.zip"
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(body)
        tmp.replace(path)
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}", flush=True)
        time.sleep(args.delay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
