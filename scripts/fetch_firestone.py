"""Download Firestone's Battlegrounds meta stats (the files its app loads) into data/firestone.

    python scripts/fetch_firestone.py                 # last patch, every MMR bracket
    python scripts/fetch_firestone.py --period past-seven

Firestone's developer: "So long as the data is not used commercially, all data available on
the application itself can be fully utilized by any other developer". Personal,
non-commercial use only; the files stay under data/ (ignored by git) and are never
redistributed. Requests are spaced a second apart, and a file is only fetched again when it
is older than --max-age hours. Data: Firestone (https://www.firestoneapp.com).
"""

import argparse
import gzip
import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://static.zerotoheroes.com/api/bgs"
MMR = (100, 50, 25, 10, 1)  # Firestone's brackets: top N% of players by MMR
UA = "bg-tavern-agent (personal, non-commercial research; github.com/oversized93/bg-tavern-agent)"


def files(period):
    out = {f"percentiles.json": f"{BASE}/hero-stats/{period}/mmr-percentiles.gz.json",
           "trinket-stats.json": f"{BASE}/trinket-stats/{period}/overview-from-hourly.gz.json",
           "comp-stats.json": f"{BASE}/comp-stats/{period}/overview-from-hourly.gz.json"}
    for m in MMR:
        out[f"hero-stats-mmr{m}.json"] = (
            f"{BASE}/hero-stats/mmr-{m}/{period}/overview-from-hourly.gz.json")
        out[f"card-stats-mmr{m}.json"] = (
            f"{BASE}/card-stats/mmr-{m}/{period}/overview-from-hourly.gz.json")
        out[f"quest-stats-mmr{m}.json"] = (
            f"{BASE}/quest-stats/mmr-{m}/{period}/overview-from-hourly.gz.json")
    return out


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read()
    if body[:2] == b"\x1f\x8b":
        body = gzip.decompress(body)
    return json.loads(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="last-patch",
                    choices=["last-patch", "past-three", "past-seven", "all-time"])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "firestone"))
    ap.add_argument("--max-age", type=float, default=12.0, help="hours before re-fetching")
    args = ap.parse_args()
    out = Path(args.out) / args.period
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, url in files(args.period).items():
        path = out / name
        if path.exists() and time.time() - path.stat().st_mtime < args.max_age * 3600:
            print(f"cached  {name}")
            continue
        try:
            data = fetch(url)
        except Exception as e:  # keep going: one missing file shouldn't lose the rest
            print(f"FAILED  {name}: {e}", file=sys.stderr)
            continue
        path.write_text(json.dumps(data))
        manifest[name] = {"url": url, "fetched": time.strftime("%Y-%m-%dT%H:%M:%S"),
                          "lastUpdateDate": data.get("lastUpdateDate") if isinstance(data, dict)
                          else None,
                          "dataPoints": data.get("dataPoints") if isinstance(data, dict) else None}
        print(f"fetched {name}  ({manifest[name]['dataPoints']} data points)")
        time.sleep(1.0)
    manifest_path.write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
