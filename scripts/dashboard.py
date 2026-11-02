"""Live training dashboard in the browser (reads runs/*/metrics.jsonl, redraws every 30 s).

    python scripts/dashboard.py            # http://localhost:8050 (opens the browser)
    python scripts/dashboard.py --lan      # also from a phone on the same Wi-Fi
    scripts\\dashboard.cmd                  # the same, by double-click

Pick the run and a run to compare with at the top of the page; the choice is in the address
(e.g. http://localhost:8050/#boot-01:scratch-01), so a bookmark keeps it.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bgagent.dashboard import serve  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs", default=str(Path(__file__).resolve().parents[1] / "runs"))
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--lan", action="store_true", help="listen on the local network too (phone)")
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser window")
    a = ap.parse_args()
    serve(a.runs, a.port, a.lan, not a.no_browser)


if __name__ == "__main__":
    main()
