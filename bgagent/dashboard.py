"""A live training dashboard in the browser: `python scripts/dashboard.py`.

A small local web server (standard library only) that reads every run's metrics.jsonl and
serves one page (dashboard.html, next to this file) that redraws its charts every 30 seconds.
It only reads files under runs/; it never touches training or the game.

    python scripts/dashboard.py            # http://localhost:8050, opens the browser
    python scripts/dashboard.py --lan      # also reachable from a phone on the same Wi-Fi
"""

import json
import re
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

PAGE = Path(__file__).with_name("dashboard.html")
NAME = re.compile(r"^[A-Za-z0-9._-]+$")
# Only what the page draws (a metrics row has ~60 keys; the TavernBench rows ~25 more).
KEEP = re.compile(r"^(iteration|time|avg_placement|ent|kl|difficulty|tier_t\d+|board_t8|"
                  r"level_force|forced_upgrades_per_game|skipped|eval_avg_placement|eval_ci95|"
                  r"eval_top1|eval_top4|exam_d[\d.]+(_ci95|_top4)?|bench_(value|policy|lookahead)|"
                  r"bench_policy_long_term)$")
CONFIG = ("model", "opponents", "rules", "iterations", "level_curve", "level_force",
          "level_force_decay", "learner_seats", "exam_difficulties")


def load_rows(path):
    """metrics.jsonl merged into one row per iteration, oldest first.

    Training writes a training row for an iteration, then its eval, exam and TavernBench rows.
    A resume from a checkpoint repeats the iterations after it, so a training row starts that
    iteration afresh (the newest attempt wins, as in status.py). A half-written last line (the
    run is appending while we read) is skipped."""
    by = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if not isinstance(r, dict) or "iteration" not in r:
            continue
        it = int(r["iteration"])
        if "decisions" in r:
            by[it] = {}
        row = by.setdefault(it, {})
        row.update({k: v for k, v in r.items() if KEEP.match(k)})
        if "time" in r:
            try:
                row["ts"] = time.mktime(time.strptime(r["time"], "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                pass
    return [by[k] for k in sorted(by)]


class Runs:
    """The run folders under runs/, parsed once per change of metrics.jsonl."""

    def __init__(self, root):
        self.root = Path(root)
        self.cache = {}

    def names(self):
        found = [p for p in self.root.iterdir() if (p / "metrics.jsonl").is_file()]
        found.sort(key=lambda p: (p / "metrics.jsonl").stat().st_mtime, reverse=True)
        return [p.name for p in found]

    def summary(self):
        out = []
        for name in self.names():
            d = self.get(name)
            last = next((r for r in reversed(d["rows"]) if "ts" in r), {})
            out.append({"name": name, "iteration": last.get("iteration"),
                        "total": d["config"].get("iterations"), "updated": last.get("ts")})
        return out

    def get(self, name):
        if not NAME.match(name) or name not in self.names():
            raise KeyError(name)
        f = self.root / name / "metrics.jsonl"
        key = (f.stat().st_mtime_ns, f.stat().st_size)
        if name not in self.cache or self.cache[name][0] != key:
            cfg_file = self.root / name / "config.json"
            cfg = json.loads(cfg_file.read_text()) if cfg_file.is_file() else {}
            self.cache[name] = (key, {"name": name, "rows": load_rows(f),
                                      "config": {k: cfg[k] for k in CONFIG if k in cfg}})
        return self.cache[name][1]


def handler(runs):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = unquote(self.path.split("?")[0])
            try:
                if path in ("/", "/index.html"):
                    return self.send(PAGE.read_bytes(), "text/html; charset=utf-8")
                if path == "/api/runs":
                    return self.json({"now": time.time(), "runs": runs.summary()})
                if path.startswith("/api/run/"):
                    return self.json({"now": time.time(), **runs.get(path[len("/api/run/"):])})
            except KeyError:
                pass
            self.send_error(404)

        def json(self, obj):
            self.send(json.dumps(obj, separators=(",", ":")).encode(), "application/json")

        def send(self, body, kind):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # quiet: the page asks every 30 s
            pass

    return Handler


def lan_address():
    """This PC's address on the local network (no packet is sent: UDP connect only picks a route)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.168.0.1", 9))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def serve(root="runs", port=8050, lan=False, open_browser=True):
    server = ThreadingHTTPServer(("0.0.0.0" if lan else "127.0.0.1", port), handler(Runs(root)))
    url = f"http://localhost:{port}"
    print(f"Training dashboard: {url}")
    if lan:
        ip = lan_address()
        print(f"From a phone on the same Wi-Fi: http://{ip or '<this PC address>'}:{port}")
        print("(Windows may ask whether to allow Python on private networks: allow it for the phone.)")
    print("Leave this window open while you watch; close it (or Ctrl+C) to stop the page.")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
