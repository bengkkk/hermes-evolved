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


def _write_self_model(env, github_read=False, github_write=False):
    d = env / "evolve"
    d.mkdir(exist_ok=True)
    (d / "self_model.json").write_text(
        json.dumps(
            {
                "identity": {"name": "test"},
                "permissions": {"github": {"read": github_read, "write": github_write,
                                           "act": False, "cap": None}},
            }
        ),
        encoding="utf-8",
    )


# Level 2 write target (first planned Contents API PUT — must stay in sync
# with BRIDGE_WRITE_ALLOWLIST / _API_WRITE_ALLOWLIST).
_WRITE_PATH = (
    "https://api.github.com/repos/bengkkk/hermes-evolved/"
    "contents/docs/gap10-level2-plan.md"
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


# ── Level 2 write path (armed, deny-until-granted) ─────────────────────

def test_write_denied_without_write_grant(client, env, monkeypatch):
    """The Level 2 gate: allowlisted PUT + read grant is still BLOCKED."""
    _write_self_model(env, github_read=True, github_write=False)
    calls = []
    monkeypatch.setattr(
        evolve_bridge, "_github_put", lambda ep, body: calls.append((ep, body))
    )
    r = client.post(
        "/bridge/v1/exec",
        json={
            "endpoint": _WRITE_PATH,
            "method": "PUT",
            "body": {"message": "test", "content": "eA=="},
        },
        headers=_auth(),
    )
    assert r.status_code == 403
    assert "no write grant" in r.json()["output"]
    assert calls == []  # deny must never execute


def test_write_denied_non_allowlisted_paths(client, env, monkeypatch):
    """A write grant does NOT widen the endpoint allowlist (exact-path match)."""
    _write_self_model(env, github_write=True)
    calls = []
    monkeypatch.setattr(
        evolve_bridge, "_github_put", lambda ep, body: calls.append((ep, body))
    )
    for ep in (
        "https://api.github.com/repos/bengkkk/hermes-evolved/contents/other.md",
        _WRITE_PATH + "-evil",  # prefix widening must stay blocked
        "https://api.github.com/repos/other/repo/contents/docs/gap10-level2-plan.md",
    ):
        r = client.post(
            "/bridge/v1/exec",
            json={
                "endpoint": ep,
                "method": "PUT",
                "body": {"message": "test", "content": "eA=="},
            },
            headers=_auth(),
        )
        assert r.status_code == 403, ep
        assert "BLOCKED" in r.json()["output"]
    assert calls == []


def test_write_invalid_body_rejected(client, env, monkeypatch):
    """Malformed write bodies are rejected before anything reaches the wire."""
    _write_self_model(env, github_write=True)
    calls = []
    monkeypatch.setattr(
        evolve_bridge, "_github_put", lambda ep, body: calls.append((ep, body))
    )
    for body in (
        {"content": "eA=="},                            # missing message
        {"message": "test"},                            # missing content
        {"message": "", "content": "eA=="},             # empty message
        {"message": "test", "content": "not-base64!"},  # invalid base64
        {"message": "test", "content": "eA==", "sha": 123},  # non-str sha
    ):
        r = client.post(
            "/bridge/v1/exec",
            json={"endpoint": _WRITE_PATH, "method": "PUT", "body": body},
            headers=_auth(),
        )
        assert r.status_code == 400, body
        assert "invalid body" in r.json()["output"]
    # Non-dict body: rejected by FastAPI's schema layer (422) — still a deny.
    r = client.post(
        "/bridge/v1/exec",
        json={"endpoint": _WRITE_PATH, "method": "PUT", "body": []},
        headers=_auth(),
    )
    assert r.status_code == 422
    assert calls == []


def test_write_executes_with_grant(client, env, monkeypatch):
    """Allowlisted PUT + write grant + valid body executes ``_github_put``."""
    _write_self_model(env, github_write=True)
    seen = {}

    def fake_put(endpoint, body):
        seen["endpoint"] = endpoint
        seen["body"] = body
        return {"exit": 0, "output": json.dumps({"status": 201, "body": "created"})}

    monkeypatch.setattr(evolve_bridge, "_github_put", fake_put)
    body = {"message": "test", "content": "eA==", "sha": "abc"}
    r = client.post(
        "/bridge/v1/exec",
        json={"endpoint": _WRITE_PATH, "method": "PUT", "body": body},
        headers=_auth(),
    )
    assert r.status_code == 200
    assert r.json()["exit"] == 0
    assert seen["endpoint"] == _WRITE_PATH
    assert seen["body"] == body


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
    from think_daemon import _API_CALL_ALLOWLIST, _API_WRITE_ALLOWLIST

    assert evolve_bridge.BRIDGE_ALLOWLIST == _API_CALL_ALLOWLIST
    assert evolve_bridge.BRIDGE_WRITE_ALLOWLIST == _API_WRITE_ALLOWLIST


def test_policy_reference_matches_enforced_allowlists():
    """gap10_level2_policy.py is the readable policy contract: its lists must
    equal the daemon's enforced lists so the reference can never drift wider
    than what actually executes (the exact-path write invariant included)."""
    import gap10_level2_policy
    from think_daemon import (
        _API_CALL_ALLOWLIST,
        _API_METHOD_ACTION,
        _API_WRITE_ALLOWLIST,
    )

    assert gap10_level2_policy.READ_ALLOWLIST == _API_CALL_ALLOWLIST
    assert gap10_level2_policy.WRITE_ALLOWLIST == _API_WRITE_ALLOWLIST
    assert gap10_level2_policy.METHOD_ACTION == evolve_bridge._METHOD_ACTION
    assert gap10_level2_policy.METHOD_ACTION == _API_METHOD_ACTION


def test_policy_reference_decisions_match_bridge_gate():
    """Decision-matrix equivalence: the reference pre-flight must agree with
    the enforced bridge gate on the exact Level 2 endpoints. The over-
    permissive ``contents/.+`` regex draft would fail the sibling-path cases
    here — this test is what prevents that widening from coming back."""
    import gap10_level2_policy

    read_only = {"github": {"read": True, "write": False}}
    write = {"github": {"read": True, "write": True}}
    empty = {}

    # (endpoint, method, permissions) → expected ok/deny
    cases = [
        # read semantics
        ("https://api.github.com/repos/x", "GET", read_only, True),
        ("https://api.github.com/repos/x", "GET", empty, False),
        ("https://api.github.com/repos/x", "GET", {"github": {"read": False, "write": True}}, False),
        ("https://evil.example.com/", "GET", read_only, False),
        # write semantics — exact-path allowlist + explicit write grant
        (_WRITE_PATH, "PUT", write, True),
        (_WRITE_PATH, "PUT", read_only, False),
        (_WRITE_PATH, "PUT", empty, False),
        # a write grant never widens the allowlist (the security invariant)
        ("https://api.github.com/repos/bengkkk/hermes-evolved/contents/other.md",
         "PUT", write, False),
        (_WRITE_PATH + "-evil", "PUT", write, False),
        ("https://api.github.com/repos/other/repo/contents/docs/gap10-level2-plan.md",
         "PUT", write, False),
        # methods/endpoints not in ANY allowlist stay denied even with a grant
        (_WRITE_PATH, "DELETE", write, False),
        ("https://api.github.com/repos/x", "POST", write, False),
    ]
    for endpoint, method, perms, expected_ok in cases:
        ok, reason = gap10_level2_policy.method_aware_preflight(endpoint, method, perms)
        assert ok is expected_ok, (endpoint, method, perms, ok, reason)


def test_contents_body_validator_matches_bridge_gate():
    """think_daemon._validate_contents_body must stay behaviorally identical
    to evolve_bridge._validate_contents_body — both docstrings claim the
    invariant ("the bridge is authoritative, so a mismatch there can only
    tighten, never widen"), but nothing pinned it until this test. The
    bridge is the final gate, so a looser daemon copy is still safe; a
    tighter one silently diverges the two pre-flight layers. Asserting a
    decision matrix (not source text) keeps this a behavior contract."""
    import think_daemon

    bodies = [
        None,
        [],
        "not-a-dict",
        {},
        {"message": "", "content": "eA=="},
        {"message": "test", "content": ""},
        {"message": "test", "content": "eA=="},
        {"message": 42, "content": "eA=="},
        {"message": "test", "content": "not base64!!!"},
        {"message": "test", "content": "eA==", "sha": "abc123"},
        {"message": "test", "content": "eA==", "sha": 123},
        {"message": "test", "content": "eA==", "sha": None},
    ]
    for body in bodies:
        bridge_ok, bridge_err = evolve_bridge._validate_contents_body(body)
        daemon_ok, daemon_err = think_daemon._validate_contents_body(body)
        assert daemon_ok == bridge_ok, (body, daemon_ok, bridge_ok)
        assert daemon_err == bridge_err, (body, daemon_err, bridge_err)


def test_main_refuses_to_start_without_token(monkeypatch, capsys, env):
    monkeypatch.delenv("HERMES_EVOLVED_BRIDGE_TOKEN", raising=False)
    # also remove the token-file fallback path by pointing evolve dir at an
    # empty tmp dir (fixture already did) — no bridge_token file there
    assert evolve_bridge.main(["--port", "9999"]) == 2
    out = capsys.readouterr().err
    assert "refusing to start without a token" in out


# ── compact GET/PUT summary (sha surfaced for read-then-write) ─────────────

def test_compact_get_summary_extracts_contents_sha():
    """A Contents API object's blob sha + size must survive the truncation
    (they appear AFTER the base64 content field, beyond MAX_OUTPUT_BYTES), so
    a read-then-write flow can learn the current sha for an update PUT."""
    body = json.dumps(
        {
            "type": "file",
            "encoding": "base64",
            "size": 12173,
            "name": "gap10-level2-plan.md",
            "path": "docs/gap10-level2-plan.md",
            "content": "x" * 6000,  # beyond MAX_OUTPUT_BYTES — sha comes after
            "sha": "deadbeef" * 5,
            "url": "https://api.github.com/...",
        }
    )
    s = evolve_bridge._compact_get_summary(200, body)
    assert s["status"] == 200
    assert s["sha"] == "deadbeef" * 5
    assert s["size"] == 12173
    assert len(s["body"]) <= evolve_bridge.MAX_OUTPUT_BYTES
    assert "error" not in s


def test_compact_get_summary_unwraps_put_content_wrapper():
    """A Contents API PUT response wraps the file object under ``content``;
    sha/size must be surfaced from there (create AND update responses)."""
    body = json.dumps(
        {
            "content": {
                "name": "gap10-level2-plan.md",
                "path": "docs/gap10-level2-plan.md",
                "sha": "273306db" + "0" * 32,
                "size": 12173,
                "url": "https://api.github.com/...",
            },
            "commit": {"sha": "c0ffee" * 5},
        }
    )
    s = evolve_bridge._compact_get_summary(201, body)
    assert s["status"] == 201
    assert s["sha"] == "273306db" + "0" * 32
    assert s["size"] == 12173
    assert "error" not in s


def test_compact_get_summary_shapes():
    """Non-contents shapes degrade safely: error object → message, list →
    count, non-JSON → status/bytes/body only."""
    err = evolve_bridge._compact_get_summary(404, json.dumps({"message": "Not Found"}))
    assert err["status"] == 404
    assert err["error"] == "Not Found"

    listing = evolve_bridge._compact_get_summary(200, json.dumps([{"name": "a"}, {"name": "b"}]))
    assert listing["count"] == 2
    assert "sha" not in listing

    raw = evolve_bridge._compact_get_summary(200, "not json at all")
    assert raw["status"] == 200
    assert "sha" not in raw
    assert "error" not in raw
    assert raw["bytes"] == len("not json at all")
