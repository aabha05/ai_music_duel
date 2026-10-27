"""
app.py - the live demo. Starts a small web server on your laptop and opens the website.

    python app.py

Pick a mood (or click anywhere on the mood map), press "Compose", and your Python AI
composes a brand-new duel while the page shows its progress. Uses only Python's
built-in http.server, so there is nothing extra to install.

    GET  /                 the demo website (web/index.html)
    POST /api/compose      start a duel        {"energy":..,"darkness":..,"seed":..,...}
    GET  /api/job/<id>     progress + the finished duel
    POST /api/save/<id>    keep a live duel in web/data.js (so it is on the website for good)
"""
import json
import os
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

import engine
import web_export
from moods import MOODS, nearest, settings_for

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(ROOT, "web")
LIVE = os.path.join(ROOT, "output", "live")
HOST = os.environ.get("HOST", "127.0.0.1")      # a hosting service sets HOST=0.0.0.0
PORT = int(os.environ.get("PORT", "8000"))
MAX_JOBS = 50                                    # forget old duels so a public server never fills up

jobs = {}
run_lock = threading.Lock()          # one duel at a time (the chord progression is shared state)


def compose(job_id, cfg, label):
    job = jobs[job_id]

    def say(text):
        job["lines"].append(text)
        if text.startswith("Turn ") and " of " in text:
            job["turn"] = int(text.split()[1])
        print(text)

    try:
        with run_lock:
            job["state"] = "running"
            result = engine.run_duel(cfg, say=say)
            out = os.path.join(LIVE, job_id)
            tree = engine.save_outputs(result, out, audio=False)
            job["tree"] = tree
            job["run"] = web_export.build_run(result, label)
            job["run"]["treeUrl"] = f"/live/{job_id}/search_tree.png"
            job["state"] = "done"
    except Exception as e:                                             # report the error on the page
        traceback.print_exc()
        job["state"], job["error"] = "error", f"{type(e).__name__}: {e}"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=WEB, **kw)

    def log_message(self, fmt, *args):                                 # keep the terminal readable
        if "/api/job/" not in (str(args[0]) if args else ""):
            super().log_message(fmt, *args)

    def send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/ping":
            return self.send_json({"ok": True})
        if path.startswith("/api/job/"):
            job = jobs.get(path.rsplit("/", 1)[-1])
            if not job:
                return self.send_json({"error": "No duel with that id. Press Compose to start one."}, 404)
            return self.send_json({k: job.get(k) for k in ("state", "lines", "turn", "turns", "run", "error")})
        if path.startswith("/live/"):
            parts = path.split("/")
            if len(parts) == 4 and parts[2] in jobs and parts[3] == "search_tree.png":
                with open(os.path.join(LIVE, parts[2], "search_tree.png"), "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                return self.wfile.write(data)
            return self.send_error(404)
        return super().do_GET()

    def end_headers(self):
        if self.path.split("?")[0] in ("/", "/index.html", "/data.js"):
            self.send_header("Cache-Control", "no-store")          # always show the newest saved duels
        super().end_headers()

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/api/compose":
            body = self.read_json()
            mood_id = body.get("mood")
            e, d = float(body.get("energy", 0.5)), float(body.get("darkness", 0.5))
            base = settings_for(mood_id if mood_id in MOODS else nearest(e, d))   # chords + personalities
            base.update({k: v for k, v in body.items() if k in engine.DEFAULTS})
            cfg = engine.clean_config(base)
            if os.environ.get("LIGHT_MODE"):          # small free servers: keep each duel quick
                cfg["depth"] = min(cfg["depth"], 2)
                cfg["population"], cfg["generations"] = 30, 15
            if mood_id in MOODS:
                label = MOODS[mood_id]["name"]
            else:
                label = f"Custom (near {MOODS[nearest(cfg['energy'], cfg['darkness'])]['name'].lower()})"
            label += f", variation {cfg['seed']}"
            for old in sorted(jobs, key=lambda k: jobs[k]["created"])[:-MAX_JOBS]:
                jobs.pop(old, None)
            job_id = uuid.uuid4().hex[:10]
            jobs[job_id] = {"state": "queued", "lines": [], "turn": 0, "turns": cfg["turns"],
                            "created": time.time(), "label": label}
            threading.Thread(target=compose, args=(job_id, cfg, label), daemon=True).start()
            return self.send_json({"id": job_id, "label": label})
        if path.startswith("/api/save/"):
            job_id = path.rsplit("/", 1)[-1]
            job = jobs.get(job_id)
            if not job or job.get("state") != "done":
                return self.send_json({"error": "That duel is not finished yet."}, 400)
            run = dict(job["run"])
            run.pop("treeUrl", None)
            name = "".join(ch if ch.isalnum() else "_" for ch in run["label"].lower()).strip("_")
            web_export.save_run(name, run, job["tree"])
            return self.send_json({"saved": name})
        self.send_error(404)


def main():
    os.makedirs(LIVE, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://localhost:{PORT}/"
    print(f"AI Music Duel demo running at {url}  (press Ctrl+C to stop)")
    if HOST in ("127.0.0.1", "localhost"):                      # on your own laptop, open the browser
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()