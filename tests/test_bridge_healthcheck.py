"""Tests for scripts/bridge_healthcheck.py (Gap 10 resilience goal).

Covers the standalone health-check/restart script: probe semantics,
check-only behavior (probe, never restart), and the restart-then-reprobe
recovery path — all against a local in-process HTTP server so no live
bridge or external network is required.
"""

import http.server
import json
import socketserver
import threading
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bridge_healthcheck.py"


class _OkHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/bridge/v1/health":
            body = json.dumps({"ok": True, "version": 1, "pid": 424242}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):  # silence
        pass


class _BadHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        self.send_response(500)
        self.end_headers()

    def log_message(self, *args):  # silence
        pass


@pytest.fixture
def ok_server():
    with socketserver.TCPServer(("127.0.0.1", 0), _OkHandler) as srv:
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        yield f"http://127.0.0.1:{srv.server_address[1]}"
        srv.shutdown()
        t.join(timeout=5)


@pytest.fixture
def bad_server():
    with socketserver.TCPServer(("127.0.0.1", 0), _BadHandler) as srv:
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        yield f"http://127.0.0.1:{srv.server_address[1]}"
        srv.shutdown()
        t.join(timeout=5)


def _run(*args, timeout=60):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ── probe semantics ─────────────────────────────────────────────────────

def test_up_bridge_exits_zero(ok_server):
    r = _run("--check-only", "--bridge-url", ok_server)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "UP" in r.stdout
    assert "pid 424242" in r.stdout


def test_down_bridge_check_only_exits_one():
    # Point at a port with nothing listening.
    r = _run("--check-only", "--bridge-url", "http://127.0.0.1:1")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "DOWN" in r.stdout
    assert "no restart attempted" in r.stdout


def test_bad_health_body_is_down(bad_server):
    r = _run("--check-only", "--bridge-url", bad_server)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "DOWN" in r.stdout


# ── restart path ────────────────────────────────────────────────────────

def test_restart_still_down_after_restart(tmp_path):
    """Restart command runs but the bridge stays down -> exit 1."""
    restart_script = tmp_path / "fake_restart.sh"
    restart_script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    restart_script.chmod(0o755)
    r = _run(
        "--bridge-url", "http://127.0.0.1:1",
        "--restart-cmd", str(restart_script),
        "--retries", "2", "--wait", "0.1",
    )
    assert r.returncode == 1, r.stdout + r.stderr
    assert "restarting via" in r.stdout
    assert "STILL DOWN after restart" in r.stdout


def _free_port() -> int:
    """Return a currently-free localhost port (race-tolerant for tests)."""
    with socketserver.TCPServer(("127.0.0.1", 0), _OkHandler) as srv:
        return srv.server_address[1]


def test_restart_recovery_end_to_end(tmp_path):
    """Probe DOWN -> restart actually starts a server -> re-probe UP -> exit 0."""
    port = _free_port()
    url = f"http://127.0.0.1:{port}"

    restart_script = tmp_path / "fake_restart.sh"
    restart_script.write_text(
        "#!/bin/sh\n"
        f"# start the healthy server on port {port}\n"
        f"nohup {sys.executable} -c '\n"
        "import http.server, json, socketserver\n"
        "class H(http.server.BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        b=json.dumps({\"ok\": True, \"pid\": 1}).encode()\n"
        "        self.send_response(200); self.send_header(\"Content-Length\", str(len(b))); self.end_headers()\n"
        "        self.wfile.write(b)\n"
        "    def log_message(self, *a): pass\n"
        f"with socketserver.TCPServer((\"127.0.0.1\", {port}), H) as s:\n"
        "    s.serve_forever()\n"
        "' >/dev/null 2>&1 &\n"
        "exit 0\n",
        encoding="utf-8",
    )
    restart_script.chmod(0o755)

    r = _run(
        "--bridge-url", url,
        "--restart-cmd", str(restart_script),
        "--retries", "20", "--wait", "0.3",
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RECOVERED after restart" in r.stdout
