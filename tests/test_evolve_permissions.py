"""Tests for evolve_permissions.py (Gap 10 Level 2 on-ramp — grant CLI).

Covers the explicit-user-grant channel: grant persists through the
canonical SelfModel file path, revoke flips flags back, check reflects
the registry, invalid usage is rejected, and the daemon/bridge-facing
SelfModel sees exactly what the CLI wrote (deny-by-default preserved).
"""

import json

import pytest

import evolve_permissions as ep
from data_layer import SelfModel


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolate the evolve dir so no test touches live daemon state."""
    monkeypatch.setenv("HERMES_EVOLVE_DIR", str(tmp_path))
    return tmp_path


def _registry(env):
    p = env / "evolve" / "self_model.json"
    return json.loads(p.read_text(encoding="utf-8"))["permissions"] if p.exists() else {}


# ── grant ───────────────────────────────────────────────────────────────

def test_grant_writes_registry_and_persists(env):
    rc = ep.main(["grant", "github", "write", "--yes"])
    assert rc == 0
    reg = _registry(env)
    assert reg["github"]["write"] is True
    assert reg["github"]["read"] is False      # untouched, deny-by-default
    # The daemon/bridge path sees the same grant:
    assert SelfModel.load().check_permission("github", "write") is True
    assert SelfModel.load().check_permission("github", "read") is False


def test_grant_is_validate_clean_and_merges(env):
    ep.main(["grant", "github", "read", "--yes"])
    ep.main(["grant", "github", "write", "--yes"])
    sm = SelfModel.load()
    assert sm.validate_permissions() == []
    assert sm.permissions["github"] == {
        "read": True, "write": True, "act": False, "cap": None,
    }


def test_grant_with_cap(env):
    rc = ep.main(["grant", "github", "act", "5.0", "--yes"])
    assert rc == 0
    assert SelfModel.load().permissions["github"]["cap"] == 5.0


def test_grant_unknown_action_rejected(env, capsys):
    with pytest.raises(SystemExit) as exc:
        ep.main(["grant", "github", "delete", "--yes"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
    assert _registry(env) == {}  # nothing written


def test_grant_negative_cap_rejected(env):
    rc = ep.main(["grant", "github", "act", "-1", "--yes"])
    assert rc == 2
    assert _registry(env) == {}


def test_grant_requires_confirmation(env, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: (_ for _ in ()).throw(EOFError()))
    rc = ep.main(["grant", "github", "write"])
    assert rc == 2  # aborted: no tty / no --yes
    assert _registry(env) == {}  # nothing written


def test_grant_declined_by_user(env, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    rc = ep.main(["grant", "github", "write"])
    assert rc == 1
    assert _registry(env) == {}


# ── revoke ───────────────────────────────────────────────────────────────

def test_revoke_flips_flag(env):
    ep.main(["grant", "github", "write", "--yes"])
    assert ep.main(["revoke", "github", "write"]) == 0
    assert SelfModel.load().check_permission("github", "write") is False
    assert _registry(env)["github"]["write"] is False


def test_revoke_already_unset(env):
    assert ep.main(["revoke", "github", "write"]) == 1


# ── check ────────────────────────────────────────────────────────────────

def test_check_exit_codes(env, capsys):
    ep.main(["grant", "github", "write", "--yes"])
    assert ep.main(["check", "github", "write"]) == 0
    assert "GRANTED" in capsys.readouterr().out
    assert ep.main(["check", "github", "act"]) == 1
    assert "denied" in capsys.readouterr().out
    assert ep.main(["check", "notion", "read"]) == 1  # undeclared resource denies


# ── show ─────────────────────────────────────────────────────────────────

def test_show_empty_and_single(env, capsys):
    # The default self-model always declares a deny-by-default github entry,
    # so an empty file still shows it with no grants.
    assert ep.main(["show"]) == 0
    out = capsys.readouterr().out
    assert "github" in out and "(none)" in out
    ep.main(["grant", "github", "read", "--yes"])
    assert ep.main(["show", "github"]) == 0
    assert "read" in capsys.readouterr().out
    assert ep.main(["show", "notion"]) == 1  # undeclared → deny-by-default


# ── no args / unknown command ───────────────────────────────────────────

def test_no_command_errors(env):
    with pytest.raises(SystemExit) as exc:
        ep.main([])
    assert exc.value.code == 2
