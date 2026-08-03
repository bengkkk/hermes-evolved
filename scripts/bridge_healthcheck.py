#!/usr/bin/env python3
"""Standalone host-bridge health check + restart (Gap 10 resilience goal).

Goal: "Automate host bridge health check and restart" — when bridge
liveness is DOWN, this single shell action locates and restarts it,
restoring api_call capability within one cycle.

Why a standalone script when the daemon already self-heals in-process?
  - The daemon's ``_bridge_ensure_running`` only runs while the daemon is
    alive; the bridge can die while the daemon is mid-cycle or while a
    cron session is the only actor.
  - The goal's verification criterion is "one shell action locates and
    restarts it" — this script IS that action: probe -> restart ->
    re-probe, with a deterministic exit code a caller can gate on.
  - It reuses the launcher's own restart path (``evolve_daemon.sh bridge
    restart``), so token persistence, PID file handling, and logging all
    stay consistent with the existing bridge lifecycle.

Behavior:
  1. Probe GET {BRIDGE_URL}/bridge/v1/health (default 127.0.0.1:8791).
  2. UP    -> print status, exit 0 (nothing to do).
  3. DOWN  -> if --check-only: print status, exit 1 (no side effects).
              otherwise run the restart command, wait for the bridge to
              come back (bounded retries), then report final status.
  4. Exit 0 iff the bridge is UP at the end; 1 iff it is DOWN.

Usage:
  scripts/bridge_healthcheck.py [--check-only] [--bridge-url URL]
      [--restart-cmd CMD] [--timeout SEC] [--retries N] [--wait SEC]

Stdlib only (urllib + subprocess). Never raises; always exits 0/1.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from urllib import request as _request
from urllib.error import URLError as _URLError

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BRIDGE_URL = "http://127.0.0.1:8791"
DEFAULT_RESTART_CMD = os.path.join(REPO_ROOT, "evolve_daemon.sh") + " bridge restart"


def probe_health(bridge_url: str, timeout: float) -> tuple[bool, str]:
    """Probe /bridge/v1/health; return (up, status_string). Never raises."""
    url = bridge_url.rstrip("/") + "/bridge/v1/health"
    try:
        with _request.urlopen(url, timeout=timeout) as resp:
            body = resp.read(512).decode("utf-8", "replace")
            status = getattr(resp, "status", 200)
            try:
                parsed = json.loads(body)
                ok = bool(parsed.get("ok")) if isinstance(parsed, dict) else False
                pid = parsed.get("pid", "?") if isinstance(parsed, dict) else "?"
            except Exception:
                ok, pid = False, "?"
            if status == 200 and ok:
                return True, f"UP (HTTP {status}, pid {pid})"
            return False, f"UP? (HTTP {status}, body mismatch)"
    except _URLError as e:
        return False, "DOWN (" + str(getattr(e, "reason", e))[:60] + ")"
    except Exception as e:  # never raise into the caller
        return False, "DOWN (" + str(e)[:60] + ")"


def run_restart(restart_cmd: str, timeout: float) -> str:
    """Run the restart command (shlex-split); return a short result string."""
    import shlex

    parts = shlex.split(restart_cmd)
    try:
        r = subprocess.run(
            parts,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        detail = (r.stdout.strip()[:160] + " " + r.stderr.strip()[:80]).strip()
        return f"restart exit={r.returncode}: {detail[:200]}"
    except Exception as e:
        return f"restart failed: {str(e)[:100]}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Host-bridge health check + restart")
    ap.add_argument("--check-only", action="store_true",
                    help="probe only; never restart (exit 1 if DOWN)")
    ap.add_argument("--bridge-url", default=os.environ.get(
        "HERMES_EVOLVED_BRIDGE_URL", DEFAULT_BRIDGE_URL))
    ap.add_argument("--restart-cmd", default=os.environ.get(
        "HERMES_EVOLVED_BRIDGE_RESTART_CMD", DEFAULT_RESTART_CMD))
    ap.add_argument("--timeout", type=float, default=3.0)
    ap.add_argument("--retries", type=int, default=10)
    ap.add_argument("--wait", type=float, default=2.0)
    args = ap.parse_args(argv)

    up, status = probe_health(args.bridge_url, args.timeout)
    print(f"bridge health: {status}")
    if up:
        return 0

    if args.check_only:
        print("bridge DOWN (check-only; no restart attempted)")
        return 1

    print("bridge DOWN — restarting via: " + args.restart_cmd)
    print(run_restart(args.restart_cmd, timeout=max(args.timeout * 2, 10.0)))

    # Bounded re-probe loop: give the launcher time to start the bridge.
    for attempt in range(1, args.retries + 1):
        time.sleep(args.wait)
        up, status = probe_health(args.bridge_url, args.timeout)
        print(f"re-probe {attempt}/{args.retries}: {status}")
        if up:
            print("bridge RECOVERED after restart")
            return 0

    print("bridge STILL DOWN after restart")
    return 1


if __name__ == "__main__":
    sys.exit(main())
