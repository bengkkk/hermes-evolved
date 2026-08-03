#!/usr/bin/env python3
"""Safe DOWN-bridge drill simulator (goal_20260803075438_0).

Validates the bridge recovery path (probe -> restart -> re-probe) against a
SIMULATED bridge so the live host bridge (127.0.0.1:8791) is never disrupted.

Why this exists: reliability engineering needs recovery validation, but the
real bridge is infrastructure. Killing a healthy bridge to "test" recovery
is exactly what goal 22 forbids ("never by arbitrarily killing a healthy
bridge"). This script is the safe simulation mechanism: a throwaway fake
bridge on an ephemeral port that behaves like the real one for the purposes
of scripts/bridge_healthcheck.py, plus start/stop subcommands so a drill can
script the full lifecycle:

  DOWN (fake stopped) -> restart command starts fake -> re-probe UP

Usage:
  scripts/bridge_drill_sim.py --serve --port N     # foreground fake bridge
  scripts/bridge_drill_sim.py --start-bg --port N  # detached fake bridge (usable
                                                   # as a healthcheck --restart-cmd)
  scripts/bridge_drill_sim.py --stop --port N      # stop detached fake bridge

The fake bridge serves GET /bridge/v1/health -> 200 {"ok": true, "pid": N},
mirroring the real bridge's health contract (see evolve_bridge.py).

Stdlib only. Never touches the real bridge, world model, or repo state.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PID_FILE_TMPL = "/tmp/bridge_drill_sim_{port}.pid"
LOG_FILE_TMPL = "/tmp/bridge_drill_sim_{port}.log"


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        if self.path.rstrip("/") == "/bridge/v1/health":
            body = json.dumps({"ok": True, "pid": os.getpid()}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def log_message(self, *args) -> None:  # silence per-request stderr noise
        pass


def _pid_file(port: int) -> str:
    return PID_FILE_TMPL.format(port=port)


def _log_file(port: int) -> str:
    return LOG_FILE_TMPL.format(port=port)


def _pid_alive(pid: int) -> bool:
    return os.path.isdir(f"/proc/{pid}")


def _read_pid(port: int) -> int | None:
    path = _pid_file(port)
    if not os.path.exists(path):
        return None
    try:
        return int(open(path).read().strip())
    except Exception:
        return None


def cmd_serve(port: int) -> int:
    """Run the fake bridge in the foreground (child of --start-bg)."""
    server = ThreadingHTTPServer(("127.0.0.1", port), _HealthHandler)
    server.serve_forever()
    return 0


def cmd_start_bg(port: int) -> int:
    """Detach a fake bridge on `port`; write its PID file. Exit 0."""
    existing = _read_pid(port)
    if existing and _pid_alive(existing):
        print(f"fake bridge already running on port {port} (pid {existing})")
        return 0
    log = open(_log_file(port), "a")
    proc = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--serve", "--port", str(port)],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    with open(_pid_file(port), "w") as f:
        f.write(str(proc.pid) + "\n")
    # Bounded wait for the health endpoint so the healthcheck's first
    # re-probe finds it UP (mirrors launcher's 2s liveness check).
    import urllib.request

    url = f"http://127.0.0.1:{port}/bridge/v1/health"
    for _ in range(20):
        try:
            with urllib.request.urlopen(url, timeout=0.5) as resp:
                if resp.status == 200:
                    print(f"fake bridge UP on port {port} (pid {proc.pid})")
                    return 0
        except Exception:
            pass
        time.sleep(0.1)
    print(f"WARNING: fake bridge on port {port} did not answer in time", file=sys.stderr)
    return 1


def cmd_stop(port: int) -> int:
    """Stop a detached fake bridge; remove its PID file."""
    pid = _read_pid(port)
    if pid is None:
        print(f"no fake bridge PID file for port {port}")
        return 1
    if _pid_alive(pid):
        import signal

        os.kill(pid, signal.SIGTERM)
        for _ in range(10):
            if not _pid_alive(pid):
                break
            time.sleep(0.1)
        if _pid_alive(pid):
            os.kill(pid, signal.SIGKILL)
        print(f"fake bridge on port {port} stopped (pid {pid})")
    else:
        print(f"fake bridge on port {port} already dead (pid {pid})")
    os.remove(_pid_file(port))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serve", action="store_true", help="run fake bridge in foreground")
    ap.add_argument("--start-bg", action="store_true", help="start detached fake bridge")
    ap.add_argument("--stop", action="store_true", help="stop detached fake bridge")
    ap.add_argument("--port", type=int, default=59998, help="port (default 59998)")
    args = ap.parse_args(argv)

    if args.serve:
        return cmd_serve(args.port)
    if args.start_bg:
        return cmd_start_bg(args.port)
    if args.stop:
        return cmd_stop(args.port)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
