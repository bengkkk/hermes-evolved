"""Tests for evolve_bridge.py (Gap 10 step 3 — host bridge for api_call).

Covers the three deny layers (auth / allowlist / permission registry),
the deny-never-executes security property, credential parsing, audit
trail, and the allowlist-drift guard against think_daemon.py.
"""

import json

import pytest
from fastapi.testclient import TestClient

import evolve_bridge
from evolve_bridge import app


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolate evolve dir + token for every test."""
    monkeypatch.setenv("HERMES_EVOLVE_DIR", str(tmp_path))
    monkeypatch.setenv("HERMES_EVOLVED_BRIDGE_TOKEN", "test-token")
    return tmp_path


@pytest.fixture
def client(env):
    return TestClient(app)


def _auth(token="test-token"):
    return {"Authorization": f"Bearer {token}"}


def _write_self_model(env, github_read=False):
    d = env / "evolve"
    d.mkdir(exist_ok=True)
    (d / "self_model.json").write_text(
        json.dumps(
            {
                "identity": {"name": "test"},
                "permissions": {"github": {"read": github_read, "write": False,
                                           "act": False, "cap": None}},
            }
        ),
        encoding="utf-8",
    )


# ── health ───────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/bridge/v1/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["version"] == 1


# ── auth layer ───────────────────────────────────────────────────────────

def test_missing_token_denied(client, monkeypatch):
    calls = []
    monkeypatch.setattr(evolve_bridge, "_github_get", lambda ep: calls.append(ep))
    r = client.post("/bridge/v1/exec", json={"endpoint": "https://api.github.com/x"})
    assert r.status_code == 401
    assert r.json()["exit"] == 1
    assert calls == []  # deny must never execute


def test_wrong_token_denied(client, monkeypatch):
    calls = []
    monkeypatch.setattr(evolve_bridge, "_github_get", lambda ep: calls.append(ep))
    r = client.post(
        "/bridge/v1/exec",
        json={"endpoint": "https://api.github.com/x"},
        headers=_auth("wrong-token"),
    )
    assert r.status_code == 401
    assert calls == []


# ── allowlist layer ──────────────────────────────────────────────────────

def test_non_allowlisted_host_denied(client, env, monkeypatch):
    _write_self_model(env, github_read=True)  # permission granted — still denied
    calls = []
    monkeypatch.setattr(evolve_bridge, "_github_get", lambda ep: calls.append(ep))
    r = client.post(
        "/bridge/v1/exec",
        json={"endpoint": "https://evil.example.com/steal", "method": "GET"},
        headers=_auth(),
    )
    assert r.status_code == 403
    assert "BLOCKED" in r.json()["output"]
    assert calls == []


def test_wrong_method_denied(client, env, monkeypatch):
    _write_self_model(env, github_read=True)
    calls = []
    monkeypatch.setattr(evolve_bridge, "_github_get", lambda ep: calls.append(ep))
    r = client.post(
        "/bridge/v1/exec",
        json={"endpoint": "https://api.github.com/repos/x", "method": "POST",
              "body": {"x": 1}},
        headers=_auth(),
    )
    assert r.status_code == 403
    assert "BLOCKED" in r.json()["output"]
    assert calls == []


# ── permission layer ─────────────────────────────────────────────────────

def test_permission_deny_default(client, env, monkeypatch):
    """No self_model file → deny-by-default github entry → blocked."""
    calls = []
    monkeypatch.setattr(evolve_bridge, "_github_get", lambda ep: calls.append(ep))
    r = client.post(
        "/bridge/v1/exec",
        json={"endpoint": "https://api.github.com/repos/NousResearch/hermes-agent"},
        headers=_auth(),
    )
    assert r.status_code == 403
    assert "permission denied" in r.json()["output"]
    assert calls == []


def test_permission_explicit_false_denied(client, env, monkeypatch):
    _write_self_model(env, github_read=False)
    calls = []
    monkeypatch.setattr(evolve_bridge, "_github_get", lambda ep: calls.append(ep))
    r = client.post(
        "/bridge/v1/exec",
        json={"endpoint": "https://api.github.com/repos/NousResearch/hermes-agent"},
        headers=_auth(),
    )
    assert r.status_code == 403
    assert "permission denied" in r.json()["output"]
    assert calls == []


# ── happy path ───────────────────────────────────────────────────────────

def test_allowlisted_and_granted_executes(client, env, monkeypatch):
    _write_self_model(env, github_read=True)
    seen = {}

    def fake_get(endpoint):
        seen["endpoint"] = endpoint
        return {"exit": 0, "output": json.dumps({"status": 200, "body": "ok"})}

    monkeypatch.setattr(evolve_bridge, "_github_get", fake_get)
    ep = "https://api.github.com/repos/NousResearch/hermes-agent"
    r = client.post(
        "/bridge/v1/exec",
        json={"endpoint": ep, "method": "GET",
              "expected_outcome": "repo metadata returned"},
        headers=_auth(),
    )
    assert r.status_code == 200
    assert r.json()["exit"] == 0
    assert seen["endpoint"] == ep


def test_audit_trail_written(client, env, monkeypatch):
    _write_self_model(env, github_read=True)
    monkeypatch.setattr(
        evolve_bridge,
        "_github_get",
        lambda ep: {"exit": 0, "output": "{}"},
    )
    client.post(
        "/bridge/v1/exec",
        json={"endpoint": "https://api.github.com/repos/NousResearch/hermes-agent"},
        headers=_auth(),
    )
    # and one blocked attempt
    client.post(
        "/bridge/v1/exec",
        json={"endpoint": "https://evil.example.com/"},
        headers=_auth(),
    )
    audit = env / "evolve" / "bridge_audit.log"
    assert audit.exists()
    lines = audit.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "exec"
    assert json.loads(lines[0])["exit"] == 0
    assert json.loads(lines[1])["event"] == "deny_allowlist"
    assert "ts" in json.loads(lines[0])


# ── credential parsing ───────────────────────────────────────────────────

def test_git_credentials_token_parsing(tmp_path):
    cred = tmp_path / ".git-credentials"
    cred.write_text(
        "https://user1:tok1@github.com/\n"
        "https://user2:tok%40en@gitlab.com/\n",
        encoding="utf-8",
    )
    assert evolve_bridge._read_git_credentials_token("github.com", cred) == "tok1"
    assert evolve_bridge._read_git_credentials_token("gitlab.com", cred) == "tok@en"
    assert evolve_bridge._read_git_credentials_token("bitbucket.org", cred) == ""
    # never crash on a missing file
    assert evolve_bridge._read_git_credentials_token("github.com", tmp_path / "nope") == ""


# ── drift guard: bridge allowlist must stay identical to the daemon's ────

def test_allowlist_matches_daemon():
    from think_daemon import _API_CALL_ALLOWLIST

    assert evolve_bridge.BRIDGE_ALLOWLIST == _API_CALL_ALLOWLIST


def test_main_refuses_to_start_without_token(monkeypatch, capsys, env):
    monkeypatch.delenv("HERMES_EVOLVED_BRIDGE_TOKEN", raising=False)
    # also remove the token-file fallback path by pointing evolve dir at an
    # empty tmp dir (fixture already did) — no bridge_token file there
    assert evolve_bridge.main(["--port", "9999"]) == 2
    out = capsys.readouterr().err
    assert "refusing to start without a token" in out
