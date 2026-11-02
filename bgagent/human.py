"""Human reference curves from Firestone's meta stats (scripts/fetch_firestone.py).

Used only to calibrate opponents and evaluation, never to imitate human moves. The main one is
warband strength by turn (total Attack + Health of the warband, Firestone's `warbandStats`),
the paper's "target strength curve", for each MMR bracket (top 100/50/25/10/1% of players).
"""

import json
from pathlib import Path

MMR_BRACKETS = (100, 50, 25, 10, 1)
DATA = Path(__file__).resolve().parents[1] / "data" / "firestone"


def _load(name, period, data_dir=None):
    path = Path(data_dir or DATA) / period / name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run scripts/fetch_firestone.py")
    return json.loads(path.read_text())


def warband_curve(mmr=1, period="last-patch", data_dir=None, max_turn=20):
    """{turn: mean warband total stats} over all heroes, weighted by each hero's games."""
    stats = _load(f"hero-stats-mmr{mmr}.json", period, data_dir)["heroStats"]
    acc = {}
    for h in stats:
        for w in h.get("warbandStats") or []:
            if w.get("averageStats") is None or w["turn"] > max_turn:
                continue
            s, n = acc.get(w["turn"], (0.0, 0))
            acc[w["turn"]] = (s + w["averageStats"] * h["dataPoints"], n + h["dataPoints"])
    return {t: s / n for t, (s, n) in sorted(acc.items()) if n}


def mmr_thresholds(period="last-patch", data_dir=None):
    """{bracket percentile: minimum MMR} for the period."""
    return {r["percentile"]: r["mmr"] for r in _load("percentiles.json", period, data_dir)}
