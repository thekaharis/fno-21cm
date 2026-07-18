#!/usr/bin/env python3
"""Live training-metrics dashboard for fno-21cm runs.

Stdlib only. Scans checkpoints_*/metrics.jsonl and checkpoint-archive/*/metrics.jsonl
under the project root, plus any extra run directories added via --add or the UI.

Usage:
    python serve.py [--port 8080] [--host 127.0.0.1] [--root /path/to/fno-21cm]
                    [--add /path/to/some/checkpoint_dir ...]

View from your laptop through an SSH tunnel:
    ssh -L 8080:localhost:8080 <binac-login>
    then open http://localhost:8080
"""
import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DASH_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = DASH_DIR.parent
EXTRA_RUNS_FILE = DASH_DIR / "extra_runs.json"

# Fallback liveness window (seconds) when a run has no epoch_train_time yet.
LIVE_FALLBACK_S = 2 * 3600


def load_extra_runs():
    try:
        paths = json.loads(EXTRA_RUNS_FILE.read_text())
        return [str(p) for p in paths if isinstance(p, str)]
    except (OSError, json.JSONDecodeError):
        return []


def save_extra_runs(paths):
    EXTRA_RUNS_FILE.write_text(json.dumps(sorted(set(paths)), indent=2) + "\n")


def read_metrics(path):
    """Parse a metrics.jsonl, skipping blank/partial lines (job may be mid-write)."""
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        return []
    return rows


def read_label(run_dir):
    """Short config label from run_metadata.json, e.g. 'localfno m=[8,8,16] h=32'."""
    try:
        meta = json.loads((run_dir / "run_metadata.json").read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    mc = meta.get("model_config", {})
    if not isinstance(mc, dict):
        return ""
    parts = []
    if mc.get("kind"):
        parts.append(str(mc["kind"]))
    for key, tag in (
        ("localfno_modes", "m"),
        ("n_modes", "m"),
        ("hidden_channels", "h"),
        ("localfno_base_width", "w"),
    ):
        if key in mc:
            v = mc[key]
            v = json.dumps(v, separators=(",", ",")) if isinstance(v, list) else v
            parts.append(f"{tag}={v}")
    return " ".join(parts)


# e.g. "    [train 1300/1320] 1.18 samples/s (0.29 batches/s, bs=4)  elapsed 4422.9s  ETA   1.1 min"
PROGRESS_RE = re.compile(
    r"\[(\w+)\s+(\d+)/(\d+)\]\s+([\d.]+)\s*samples/s.*?elapsed\s+([\d.]+)s\s+ETA\s+([\d.]+)")
CKPT_DIR_RE = re.compile(r"CHECKPOINT_DIR:\s*(\S+)")
# rough share of an epoch spent in each phase (train dominates at ~4500s vs ~2x125s)
PHASE_SPAN = {"train": (0.0, 0.95), "val": (0.95, 0.025), "test": (0.975, 0.025)}


def read_task(run_dir, name):
    """Training-target tag for page grouping: metadata 'task', else name."""
    try:
        meta = json.loads((run_dir / "run_metadata.json").read_text())
        task = meta.get("task")
        if isinstance(task, str) and task:
            return task
    except (OSError, json.JSONDecodeError):
        pass
    return "zre" if "zre" in name else "3d"


def read_total_epochs(run_dir):
    try:
        meta = json.loads((run_dir / "run_metadata.json").read_text())
        v = meta.get("training", {}).get("epochs")
        return v if isinstance(v, int) and v > 0 else None
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def find_progress(run_dir, root, metrics_mtime):
    """Within-epoch progress for a live run, parsed from its slurm log.

    Finds the freshest <root>/logs/*.out whose header names this run's checkpoint
    dir, then parses the last '[phase i/N] ...' line from its tail.
    """
    log_dir = root / "logs"
    try:
        candidates = sorted(log_dir.glob("*.out"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return None
    log = None
    for lg in candidates:
        try:
            st = lg.stat()
            if time.time() - st.st_mtime > 6 * 3600:
                break  # sorted by recency; the rest are older still
            with open(lg, errors="replace") as f:
                head = f.read(8192)
        except OSError:
            continue
        m = CKPT_DIR_RE.search(head)
        named = m and m.group(1).rstrip("/").rsplit("/", 1)[-1] == run_dir.name
        if named or f"{run_dir.name}/metrics.jsonl" in head:
            log = lg
            break
    if log is None:
        return None
    try:
        st = log.stat()
        with open(log, "rb") as f:
            f.seek(max(0, st.st_size - 16384))
            tail = f.read().decode(errors="replace")
    except OSError:
        return None
    matches = PROGRESS_RE.findall(tail)
    if not matches:
        return None
    phase, done, total, sps, elapsed, eta_min = matches[-1]
    done, total = int(done), int(total)
    off, span = PHASE_SPAN.get(phase, (0.0, 1.0))
    epoch_frac = off + span * (done / total if total else 0.0)
    # if metrics.jsonl was written after the last progress line, that epoch is
    # already counted as complete and the next one hasn't started printing yet
    if st.st_mtime <= metrics_mtime:
        epoch_frac = 0.0
    return {
        "phase": phase,
        "done": done,
        "total": total,
        "samples_per_s": float(sps),
        "elapsed_s": float(elapsed),
        "eta_min": float(eta_min),
        "epoch_frac": epoch_frac,
        "log": log.name,
        "log_age_s": round(time.time() - st.st_mtime, 1),
    }


def is_live(mtime, rows):
    """A run counts as live if metrics.jsonl was written within ~2 epochs."""
    window = LIVE_FALLBACK_S
    for row in reversed(rows):
        t = row.get("epoch_train_time")
        if isinstance(t, (int, float)) and t > 0:
            window = max(2.0 * t, 600.0)
            break
    return (time.time() - mtime) < window


def columnize(rows):
    """Turn row dicts into {key: [values]} with keys in first-appearance order."""
    keys = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    series = {}
    for k in keys:
        col = []
        for row in rows:
            v = row.get(k)
            col.append(v if isinstance(v, (int, float)) and not isinstance(v, bool) else None)
        series[k] = col
    return keys, series


def discover_run_dirs(root, extras):
    """Yield (name, dir) pairs for every directory holding a metrics.jsonl."""
    found = {}
    for d in sorted(root.glob("checkpoints_*")):
        if (d / "metrics.jsonl").is_file():
            found[d.name] = d
    for d in sorted((root / "checkpoints").glob("*")):
        if (d / "metrics.jsonl").is_file():
            found[d.name] = d
    for d in sorted((root / "checkpoint-archive").glob("*")):
        if (d / "metrics.jsonl").is_file():
            found[f"archive/{d.name}"] = d
    for p in extras:
        d = Path(p)
        if not (d / "metrics.jsonl").is_file():
            continue
        name = d.name
        if name in found and found[name] != d:
            name = f"{d.parent.name}/{d.name}"
        found[name] = d
    return found


def build_payload(root, extras):
    runs = []
    for name, d in discover_run_dirs(root, extras).items():
        mfile = d / "metrics.jsonl"
        rows = read_metrics(mfile)
        if not rows:
            continue
        try:
            mtime = mfile.stat().st_mtime
        except OSError:
            continue
        keys, series = columnize(rows)
        live = is_live(mtime, rows)
        runs.append({
            "name": name,
            "path": str(d),
            "mtime": mtime,
            "live": live,
            "epochs": len(rows),
            "total_epochs": read_total_epochs(d),
            "progress": find_progress(d, root, mtime) if live else None,
            "label": read_label(d),
            "keys": keys,
            "series": series,
            "task": read_task(d, name),
        })
    return {"generated": time.time(), "root": str(root), "extras": extras, "runs": runs}


class Handler(BaseHTTPRequestHandler):
    root = DEFAULT_ROOT

    def parse_request(self):
        # VS Code's port forwarder sometimes prefixes the first request on a
        # reused connection with NUL bytes, which strict parsing rejects as
        # "Unsupported method ('\x00...GET')".
        self.raw_requestline = self.raw_requestline.lstrip(b"\x00")
        return super().parse_request()

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            try:
                html = (DASH_DIR / "index.html").read_bytes()
            except OSError:
                self._send(500, {"error": "index.html not found next to serve.py"})
                return
            self._send(200, html, "text/html; charset=utf-8")
        elif path == "/api/runs":
            self._send(200, build_payload(self.root, load_extra_runs()))
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            target = str(body.get("path", "")).strip()
        except (ValueError, json.JSONDecodeError):
            self._send(400, {"error": "bad request body"})
            return
        if not target:
            self._send(400, {"error": "missing 'path'"})
            return
        d = Path(target).expanduser()
        if d.name == "metrics.jsonl":
            d = d.parent
        d = d.resolve()
        extras = load_extra_runs()
        if path == "/api/add":
            if not (d / "metrics.jsonl").is_file():
                self._send(400, {"error": f"no metrics.jsonl in {d}"})
                return
            if str(d) not in extras:
                extras.append(str(d))
                save_extra_runs(extras)
            self._send(200, {"ok": True, "extras": extras})
        elif path == "/api/remove":
            extras = [p for p in extras if p != str(d)]
            save_extra_runs(extras)
            self._send(200, {"ok": True, "extras": extras})
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        pass  # keep the terminal quiet; errors still surface as HTTP responses


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (keep 127.0.0.1 on shared login nodes; use SSH tunnel)")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="project root scanned for checkpoints/, checkpoints_*/ and checkpoint-archive/")
    ap.add_argument("--add", action="append", default=[], metavar="DIR",
                    help="extra run directory containing a metrics.jsonl (repeatable)")
    args = ap.parse_args()

    extras = load_extra_runs()
    for p in args.add:
        d = Path(p).expanduser().resolve()
        if (d / "metrics.jsonl").is_file():
            if str(d) not in extras:
                extras.append(str(d))
        else:
            print(f"warning: ignoring --add {d} (no metrics.jsonl)")
    save_extra_runs(extras)

    Handler.root = args.root.resolve()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"dashboard: http://{args.host}:{args.port}  (root: {Handler.root})")
    print(f"tunnel:    ssh -L {args.port}:localhost:{args.port} <binac-login>")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
