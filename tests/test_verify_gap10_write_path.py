"""Tests for verify_gap10_write_path.py — the state-aware live write-path probe.

Pins the script's decision logic (exit-code mapping, assertion messages, and
the no-real-PUT-when-granted safety) with mocked probes. Never touches the
network or the live bridge.

Probe call order inside main() when github.write is DENIED:
    P1 GET  api.github.com/                  -> (200, exit 0, '"status": 200')
    P3 PUT  contents/docs/other.md (sibling) -> (403, "endpoint not in allowlist")
    P2 PUT  contents/docs/gap10-level2-plan.md -> (403, "permission denied: no write grant")
When GRANTED: P1 and P3 only — P2 must NOT fire (no real PUT from a probe).
"""

import sys

import pytest

import verify_gap10_write_path as v

READ_OK = (200, {"exit": 0, "output": '{"status": 200, "bytes": 2262, "body": "{}"}'})
SIBLING_DENIED = (403, {"exit": 1, "output": "BLOCKED: endpoint not in allowlist: PUT ..."})
GATE_DENIED = (403, {"exit": 1, "output": "BLOCKED: permission denied: no write grant for resource 'github'"})


@pytest.fixture
def denied(monkeypatch):
    monkeypatch.setattr(v, "_github_write_granted", lambda: False)


@pytest.fixture
def granted(monkeypatch):
    monkeypatch.setattr(v, "_github_write_granted", lambda: True)


def run_main(monkeypatch, capsys, probe_results):
    """Run main() with a mocked _probe; return (exit_code, stdout).

    ``probe_results`` is either a callable (used as-is) or a sequence of
    (status, payload) tuples returned one per call, in call order.
    """
    monkeypatch.setattr(sys, "argv", ["verify_gap10_write_path.py", "--port", "8791"])
    if callable(probe_results):
        monkeypatch.setattr(v, "_probe", probe_results)
    else:
        seq = iter(probe_results)
        monkeypatch.setattr(v, "_probe", lambda port, method, endpoint, body=None: next(seq))
    code = v.main()
    return code, capsys.readouterr().out


def test_denied_all_probes_ok_exit0(denied, monkeypatch, capsys):
    code, out = run_main(monkeypatch, capsys, [READ_OK, SIBLING_DENIED, GATE_DENIED])
    assert code == 0
    assert "VERIFY PASS" in out
    assert "P1 read=200/ok" in out
    assert "P3 exact-path=403/ok" in out
    assert "P2 gate=denied/ok" in out
    assert "UNREACHABLE" not in out


def test_denied_read_failure_exit1(denied, monkeypatch, capsys):
    code, out = run_main(monkeypatch, capsys, [(500, {"exit": 1, "output": "boom"}), SIBLING_DENIED, GATE_DENIED])
    assert code == 1
    assert "VERIFY FAIL" in out


def test_denied_exact_path_not_gated_exit1(denied, monkeypatch, capsys):
    """A write that bypasses the permission gate (e.g. 200) must trip FAIL."""
    code, _ = run_main(monkeypatch, capsys, [READ_OK, SIBLING_DENIED, (200, {"exit": 0, "output": "{}"})])
    assert code == 1


def test_denied_exact_path_wrong_reason_exit1(denied, monkeypatch, capsys):
    """403 with the wrong reason (allowlist vs permission) is a drift signal."""
    wrong = (403, {"exit": 1, "output": "BLOCKED: endpoint not in allowlist: PUT ..."})
    code, _ = run_main(monkeypatch, capsys, [READ_OK, SIBLING_DENIED, wrong])
    assert code == 1


def test_denied_sibling_allowlist_break_exit1(denied, monkeypatch, capsys):
    """Sibling PUT escaping the exact-path rule must trip FAIL."""
    code, _ = run_main(monkeypatch, capsys, [READ_OK, (200, {"exit": 0, "output": "{}"}), GATE_DENIED])
    assert code == 1


def test_granted_exit2_and_never_probes_p2(granted, monkeypatch, capsys):
    """When github.write is granted the script must NOT fire a real PUT."""
    calls = []
    def probes(port, method, endpoint, body=None):
        calls.append((method, endpoint))
        return READ_OK if method == "GET" else SIBLING_DENIED

    monkeypatch.setattr(sys, "argv", ["verify_gap10_write_path.py", "--port", "8791"])
    monkeypatch.setattr(v, "_probe", probes)
    code = v.main()
    out = capsys.readouterr().out

    assert code == 2
    assert "GRANT ACTIVE" in out
    assert "fire the first write triple deliberately" in out
    # Exactly P1 (GET) + P3 (sibling PUT); the exact allowlisted plan.md PUT must never be probed.
    assert len(calls) == 2
    assert calls[0][0] == "GET"
    assert calls[1][1].endswith("contents/docs/other.md")


def test_unreachable_bridge_exit1(denied, monkeypatch, capsys):
    def boom(port, method, endpoint, body=None):
        raise RuntimeError("connection refused")

    code, out = run_main(monkeypatch, capsys, boom)
    assert code == 1
    assert "UNREACHABLE" in out


def test_token_resolution_env_wins(monkeypatch):
    monkeypatch.setenv("HERMES_EVOLVED_BRIDGE_TOKEN", "env-token")
    monkeypatch.setattr(v, "get_evolve_dir", lambda: None)
    assert v._bridge_token() == "env-token"
