"""Tests for scripts/bridge_drill_sim.py (goal_20260803075438_0).

Validates the safe DOWN-bridge drill simulator: the fake bridge serves the
same /bridge/v1/health contract as the real bridge, start-bg/stop manage a
detached instance, and the full recovery loop (healthcheck DOWN -> restart
cmd starts fake bridge -> re-probe UP) works end-to-end against the
simulated bridge — no live bridge or external network required.
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM = REPO_ROOT / "scripts" / "bridge_drill_sim.py"
HEALTHCHECK = REPO_ROOT / "scripts" / "bridge_healthcheck.py"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _probe(port: int, timeout: float = 1.0) -> tuple[bool, str]:
    url = f"http://127.0.0.1:{port}/bridge/v1/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = json.loads(resp.read(512).decode("utf-8", "replace"))
            return (resp.status == 200 and bool(body.get("ok")), str(body.get("pid", "?")))
    except Exception as e:
        return (False, str(e)[:40])


def _run(args: list[str], timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SIM), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_serve_serves_health_contract():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(SIM), "--serve", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        # The server needs a moment to bind — bounded retry like cmd_start_bg.
        up, pid = False, ""
        for _ in range(20):
            up, pid = _probe(port)
            if up:
                break
            time.sleep(0.1)
        assert up, f"fake bridge should answer /bridge/v1/health (port {port})"
        assert pid.isdigit()
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_start_bg_then_stop_lifecycle():
    port = _free_port()
    r = _run(["--start-bg", "--port", str(port)])
    assert r.returncode == 0, r.stdout + r.stderr
    try:
        up, _ = _probe(port)
        assert up, "detached fake bridge should come UP"
    finally:
        r2 = _run(["--stop", "--port", str(port)])
        assert r2.returncode == 0, r2.stdout + r2.stderr
    up, _ = _probe(port)
    assert not up, "fake bridge should be DOWN after --stop"


def test_healthcheck_recovery_loop_against_simulated_down():
    """The core drill: healthcheck detects DOWN, restart cmd starts the fake
    bridge, re-probe reports RECOVERED (exit 0). No real bridge involved."""
    port = _free_port()
    restart_cmd = f"{sys.executable} {SIM} --start-bg --port {port}"
    r = subprocess.run(
        [sys.executable, str(HEALTHCHECK),
         "--bridge-url", f"http://127.0.0.1:{port}",
         "--restart-cmd", restart_cmd,
         "--retries", "8", "--wait", "1"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    try:
        assert "bridge DOWN" in r.stdout, r.stdout + r.stderr
        assert "bridge RECOVERED after restart" in r.stdout, r.stdout + r.stderr
        assert r.returncode == 0, r.stdout + r.stderr
        # fake bridge left running by the restart cmd — verify then clean up
        up, _ = _probe(port)
        assert up
    finally:
        _run(["--stop", "--port", str(port)])
    up, _ = _probe(port)
    assert not up, "cleanup should leave the simulated bridge DOWN"
