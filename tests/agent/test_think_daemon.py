"""Tests for think_daemon.py — persistent cognition daemon (Gap 1 + Gap 6).

Tests the testable units in the daemon:

  - JSON extraction from LLM responses (``_try_parse_json``)
  - PID lock acquire/release (``_acquire_daemon_lock`` / ``_release_daemon_lock``)
  - Daemon state persistence (``load_daemon_state`` / ``save_daemon_state``)
  - Timeline load/save round-trip (``load_timeline`` / ``save_timeline``)
  - Self-model load/save round-trip (``load_self_model`` / ``save_self_model``)
  - Prompt building from state (``_build_thinking_prompt``)
  - Reliability printout (``_print_cycle_stats``)

Each test class runs in isolation with a temp evolve directory.
"""

from __future__ import annotations

import asyncio
import importlib
import io
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Generator

import pytest

# Disable logging noise during tests — think_daemon has its own logger
logging.disable(logging.CRITICAL)


# ═══════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def evolve_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[Dict[str, Path], None, None]:
    """Set up an isolated evolve directory and import think_daemon fresh.

    Since ``data_layer`` caches ``_EVOLVE_DIR`` at module level, we must
    force a full re-import of BOTH ``data_layer`` and ``think_daemon``
    to point them at our temp directory. Tests run in isolated subprocesses
    (``scripts/run_tests_parallel.py``), so this is per-file-repeatable.
    """
    evolve_dir = tmp_path / "evolve"
    evolve_dir.mkdir(parents=True, exist_ok=True)

    # Point the evolve data layer at our temp directory — must be set
    # BEFORE module imports run their module-level code.
    monkeypatch.setenv("HERMES_EVOLVE_DIR", str(tmp_path))

    # ── Force fresh import of BOTH data_layer and think_daemon ──
    # data_layer._EVOLVE_DIR is set once at module level; we must
    # evict it from sys.modules so it recomputes on re-import.
    for mod in ("data_layer", "think_daemon"):
        if mod in sys.modules:
            del sys.modules[mod]

    import data_layer  # noqa: F811  — recomputes _EVOLVE_DIR from HERMES_EVOLVE_DIR
    import think_daemon as td  # noqa: F811 — uses the fresh data_layer

    # Double-reload to catch any inter-module dependency chains
    td = importlib.reload(td)

    paths = {
        "evolve_dir": evolve_dir,
        "timeline_file": evolve_dir / "timeline.json",
        "self_model_file": evolve_dir / "self_model.json",
        "orientation_file": evolve_dir / "orientation.json",
        "daemon_state_file": evolve_dir / "daemon_state.json",
        "daemon_lock_file": evolve_dir / "daemon.lock",
    }
    yield {"module": td, "paths": paths}


# ═══════════════════════════════════════════════════════════════════
#  JSON parsing from LLM responses
# ═══════════════════════════════════════════════════════════════════


class TestTryParseJson:
    """_try_parse_json extracts JSON objects from LLM text responses."""

    def test_plain_json(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        result = td._try_parse_json('{"insight": "test", "action": null}')
        assert result is not None
        assert result["insight"] == "test"

    def test_plain_json_array(self, evolve_env: Dict) -> None:
        """_try_parse_json only handles JSON objects, not arrays."""
        td = evolve_env["module"]
        # An array won't work — the function looks for { }
        result = td._try_parse_json('[1, 2, 3]')
        assert result is None

    def test_markdown_fenced_json(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        text = """Here is my thinking:

```json
{"insight": "I should focus on testing", "action": null}
```"""
        result = td._try_parse_json(text)
        assert result is not None
        assert result["insight"] == "I should focus on testing"

    def test_markdown_code_block_no_lang(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        text = """```
{"insight": "no lang fence", "action": null}
```"""
        result = td._try_parse_json(text)
        assert result is not None
        assert result["insight"] == "no lang fence"

    def test_json_with_leading_text_inline(self, evolve_env: Dict) -> None:
        """Leading text before a JSON object is handled by scanning for { }."""
        td = evolve_env["module"]
        text = 'Some preliminary text.\n\n{"focus_next": "test parsing", "action": null}'
        result = td._try_parse_json(text)
        assert result is not None
        assert result["focus_next"] == "test parsing"

    def test_json_with_trailing_text_inline(self, evolve_env: Dict) -> None:
        """Trailing text after a JSON object is handled by scanning for { }."""
        td = evolve_env["module"]
        text = '{"insight": "trailing"} but then the model kept talking'
        result = td._try_parse_json(text)
        assert result is not None
        assert result["insight"] == "trailing"

    def test_multiple_json_objects_not_fenced(self, evolve_env: Dict) -> None:
        """Multiple JSON objects outside a fence — not parseable as one object."""
        td = evolve_env["module"]
        text = '{"first": true} some text {"second": true}'
        result = td._try_parse_json(text)
        # The first { to last } spans both objects with non-JSON in between,
        # so the whole thing is invalid. The function correctly returns None.
        assert result is None

    def test_invalid_json_returns_none(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        result = td._try_parse_json("This is not JSON {{broken")
        assert result is None

    def test_empty_string(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        assert td._try_parse_json("") is None

    def test_fenced_with_extra_newlines(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        text = "\n\n```\n\n{\"insight\": \"fenced with space\"}\n\n```\n"
        result = td._try_parse_json(text)
        assert result is not None
        assert result["insight"] == "fenced with space"

    def test_fenced_complex_nested_json(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        text = """```json
{
  "insight": "complex test",
  "action": {"type": "shell", "command": "ls", "description": "list"},
  "prediction": {"text": "will work", "confidence": 0.8}
}
```"""
        result = td._try_parse_json(text)
        assert result is not None
        assert result["insight"] == "complex test"
        assert result["action"]["type"] == "shell"
        assert result["prediction"]["confidence"] == 0.8

    def test_fenced_with_leading_text(self, evolve_env: Dict) -> None:
        """Leading text before a fenced code block is fine."""
        td = evolve_env["module"]
        text = "Let me think about this...\n\n```\n{\"insight\": \"thoughts\"}\n```"
        result = td._try_parse_json(text)
        assert result is not None
        assert result["insight"] == "thoughts"


# ═══════════════════════════════════════════════════════════════════
#  PID lock
# ═══════════════════════════════════════════════════════════════════


class TestDaemonLock:
    """PID-based lock prevents concurrent daemon runs."""

    def test_acquire_lock_creates_file(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        lock_path = evolve_env["paths"]["daemon_lock_file"]

        assert td._acquire_daemon_lock() is True
        assert lock_path.exists()
        content = lock_path.read_text().strip()
        assert content == str(os.getpid())

    def test_lock_held_by_self_is_ok(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        # Acquire twice — same PID should be fine
        assert td._acquire_daemon_lock() is True
        assert td._acquire_daemon_lock() is True

    def test_release_lock_removes_file(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        lock_path = evolve_env["paths"]["daemon_lock_file"]

        td._acquire_daemon_lock()
        assert lock_path.exists()
        td._release_daemon_lock()
        assert not lock_path.exists()

    def test_release_when_not_held(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        # Should not crash
        td._release_daemon_lock()

    def test_stale_lock_reacquired(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        lock_path = evolve_env["paths"]["daemon_lock_file"]

        # Write a PID that doesn't exist
        lock_path.write_text("99999999")  # Very unlikely to be a real PID

        # Should reacquire (stale lock takeover)
        assert td._acquire_daemon_lock() is True
        assert lock_path.read_text().strip() == str(os.getpid())

    def test_corrupted_lock_file(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        lock_path = evolve_env["paths"]["daemon_lock_file"]

        lock_path.write_text("not_a_number")

        # Should overwrite
        assert td._acquire_daemon_lock() is True
        assert lock_path.read_text().strip() == str(os.getpid())

    def test_lock_refused_when_held_by_live_foreign_pid(self, evolve_env: Dict) -> None:
        """A live lock owner must block a second daemon from starting.

        Regression test for the duplicate-daemon failure mode: a daemon
        that started before the lock file existed never re-checked lock
        ownership, so a second instance would start while the first kept
        running — interleaving cycles, doubling LLM load, and racing
        state writes.  The lock MUST be refused (and left untouched)
        while another live process owns it.
        """
        td = evolve_env["module"]
        lock_path = evolve_env["paths"]["daemon_lock_file"]

        # Our parent process is guaranteed alive during the test run.
        foreign_pid = os.getppid()
        assert foreign_pid != os.getpid()
        lock_path.write_text(str(foreign_pid))

        # Acquisition must fail...
        assert td._acquire_daemon_lock() is False
        # ...and the live owner's lock must NOT be clobbered.
        assert lock_path.read_text().strip() == str(foreign_pid)

    def test_lock_takeover_only_when_owner_dead(self, evolve_env: Dict) -> None:
        """Stale-PID takeover must never clobber a live owner's lock.

        Guards the ownership re-check added to the daemon loop: the
        per-cycle check calls _acquire_daemon_lock() and exits when it
        returns False.  If a live owner's lock were wrongly taken over,
        BOTH daemons would keep running (the exact duplicate-instance
        bug the re-check exists to prevent).
        """
        td = evolve_env["module"]
        lock_path = evolve_env["paths"]["daemon_lock_file"]

        # Live foreign owner → refused, lock preserved.
        foreign_pid = os.getppid()
        lock_path.write_text(str(foreign_pid))
        assert td._acquire_daemon_lock() is False
        assert lock_path.read_text().strip() == str(foreign_pid)

        # Dead owner → takeover allowed.
        lock_path.write_text("99999999")
        assert td._acquire_daemon_lock() is True
        assert lock_path.read_text().strip() == str(os.getpid())

    def test_release_only_own_lock(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        lock_path = evolve_env["paths"]["daemon_lock_file"]

        # Another PID's lock
        lock_path.write_text("77777")
        td._release_daemon_lock()
        # Should NOT remove — it's not our PID
        assert lock_path.exists()
        assert lock_path.read_text().strip() == "77777"

    def test_acquire_release_cycle(self, evolve_env: Dict) -> None:
        """Full acquire → release → re-acquire cycle works."""
        td = evolve_env["module"]
        assert td._acquire_daemon_lock() is True
        td._release_daemon_lock()
        assert not evolve_env["paths"]["daemon_lock_file"].exists()
        assert td._acquire_daemon_lock() is True
        assert evolve_env["paths"]["daemon_lock_file"].read_text().strip() == str(os.getpid())
        td._release_daemon_lock()


# ═══════════════════════════════════════════════════════════════════
#  Daemon state persistence
# ═══════════════════════════════════════════════════════════════════


class TestDaemonState:
    """load_daemon_state / save_daemon_state round-trip."""

    def test_default_state(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        state = td.load_daemon_state()
        assert state["status"] == "initialized"
        assert state["version"] == 2, f"Expected version 2, got {state.get('version')}"
        assert state["tick_count"] == 0
        assert state.get("first_tick") is None
        assert state.get("interval_seconds") == 600
        assert state.get("cycle_history") == [], "Expected empty cycle_history"

    def test_save_and_load_round_trip(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        state = td.load_daemon_state()
        state["status"] = "running"
        state["tick_count"] = 5
        state["interval_seconds"] = 300
        td.save_daemon_state(state)

        loaded = td.load_daemon_state()
        assert loaded["status"] == "running"
        assert loaded["tick_count"] == 5
        assert loaded["interval_seconds"] == 300

    def test_state_persists_custom_keys(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        state = td.load_daemon_state()
        state["custom_key"] = "custom_value"
        state["last_output"] = {"insight": "test", "confidence": 0.9}
        td.save_daemon_state(state)

        loaded = td.load_daemon_state()
        assert loaded["custom_key"] == "custom_value"
        assert loaded["last_output"]["insight"] == "test"
        assert loaded["last_output"]["confidence"] == 0.9

    def test_file_location(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        state_path = evolve_env["paths"]["daemon_state_file"]
        assert not state_path.exists()

        td.save_daemon_state(td.load_daemon_state())
        assert state_path.exists()
        data = json.loads(state_path.read_text())
        assert data["status"] == "initialized"

    def test_corrupted_state_returns_default(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        state_path = evolve_env["paths"]["daemon_state_file"]
        state_path.write_text("{{invalid json}}")

        state = td.load_daemon_state()
        assert state["status"] == "initialized"
        assert state["tick_count"] == 0

    def test_cycle_stats_initialization(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        state = td.load_daemon_state()
        assert "cycle_stats" not in state  # Not created until first cycle

    def test_cycle_stats_saved(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        state = td.load_daemon_state()
        state["cycle_stats"] = {
            "total": 10, "ok": 8, "error": 1, "parse_error": 1,
            "avg_duration": 5.2, "max_duration": 12.0,
        }
        td.save_daemon_state(state)

        loaded = td.load_daemon_state()
        cs = loaded["cycle_stats"]
        assert cs["total"] == 10
        assert cs["ok"] == 8
        assert cs["avg_duration"] == 5.2


# ═══════════════════════════════════════════════════════════════════
#  Code-drift detection
# ═══════════════════════════════════════════════════════════════════


class TestCodeDriftDetection:
    """_git_head / _check_code_drift flag stale daemon processes."""

    def test_git_head_returns_short_sha(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        head = td._git_head()
        # The workspace is a git repo, so a 7+ char short SHA is expected.
        # In an exotic non-git checkout this may be None — that is allowed.
        if head is not None:
            assert len(head) >= 7
            assert all(c in "0123456789abcdef" for c in head)

    def test_no_drift_when_startup_head_matches(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        head = td._git_head()
        if head is None:
            import pytest as _pt
            _pt.skip("workspace is not a git checkout")
        ds = {"startup_head": head}
        assert td._check_code_drift(ds) is False
        assert "code_drift" not in ds

    def test_drift_detected_when_startup_head_stale(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        head = td._git_head()
        if head is None:
            import pytest as _pt
            _pt.skip("workspace is not a git checkout")
        # Simulate a daemon that started on an older commit.
        stale = "0" * len(head)
        ds = {"startup_head": stale}
        assert td._check_code_drift(ds) is True
        drift = ds.get("code_drift")
        assert drift is not None
        assert drift["startup_head"] == stale
        assert drift["current_head"] == head
        assert "detected_at" in drift

    def test_no_drift_without_startup_head(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        ds = {}  # legacy state, no startup_head recorded
        assert td._check_code_drift(ds) is False

    def test_mark_startup_clears_stale_drift(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        head = td._git_head()
        stale = "0" * (len(head) if head else 7)
        # State left behind by a previous daemon that was restarted:
        # startup_head is stale AND a code_drift block claims drift.
        ds = {
            "startup_head": stale,
            "code_drift": {
                "startup_head": stale,
                "current_head": "fffffff",
                "detected_at": "2026-07-31T18:04:20+00:00",
            },
        }
        td._mark_startup(ds, 900, head)
        assert ds["startup_head"] == head
        assert ds["interval_seconds"] == 900
        assert ds["status"] == "running"
        # A fresh start runs the code at `head` — the stale drift marker
        # must not linger and keep status/state claiming the daemon is
        # behind the repo.
        assert "code_drift" not in ds

    def test_mark_startup_preserves_current_head_without_drift(
        self, evolve_env: Dict
    ) -> None:
        td = evolve_env["module"]
        head = td._git_head()
        ds = {"startup_head": head, "interval_seconds": 600}
        td._mark_startup(ds, 900, head)
        assert ds["startup_head"] == head
        assert ds["interval_seconds"] == 900
        assert "code_drift" not in ds

    def test_mark_startup_resets_inherited_outage_counter(
        self, evolve_env: Dict
    ) -> None:
        """A fresh daemon must not inherit the previous process's LLM-outage
        counter.

        ``consecutive_fallback_cycles`` is evidence gathered by the
        (now-dead) process that last held the lock.  If a fresh process
        inherited a skip-tier value (>= 3) it would skip LLM probes for
        up to 2 more cycles even after the provider recovered — a
        needless blind window after every drift/manual restart.  The
        reset targets the probe tier (4 → one bounded 90 s attempt), so
        a still-down endpoint costs exactly one bounded probe before the
        skip tier re-engages.
        """
        td = evolve_env["module"]
        head = td._git_head()
        # Extended-outage state left behind by a dead daemon (observed
        # 2026-08-01: drift-restart inherited counter=10, then ticks
        # 304-305 skipped probes while a direct _call_llm probe returned
        # PONG in 2.9 s).
        ds = {"consecutive_fallback_cycles": 10}
        td._mark_startup(ds, 900, head)
        assert ds["consecutive_fallback_cycles"] == 4  # probe tier

    def test_mark_startup_keeps_healthy_counter_untouched(
        self, evolve_env: Dict
    ) -> None:
        """A healthy/warm counter (0-2) is first-hand evidence for the
        fresh process only when it is small; values below the extended-
        outage skip tier (>= 3) are left alone."""
        td = evolve_env["module"]
        head = td._git_head()
        for healthy in (0, 1, 2):
            ds = {"consecutive_fallback_cycles": healthy}
            td._mark_startup(ds, 900, head)
            assert ds["consecutive_fallback_cycles"] == healthy


class TestDriftAutoRestart:
    """_schedule_drift_restart hands a stale daemon over to fresh code."""

    def test_restart_scheduled_when_drifted(self, evolve_env: Dict) -> None:
        from unittest.mock import patch as _patch

        td = evolve_env["module"]
        # A daemon that started long ago, so no throttle applies.
        old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        ds = {"startup_head": "0000000", "last_auto_restart": old}
        with _patch("subprocess.Popen") as popen:
            ok = td._schedule_drift_restart(ds)
        assert ok is True
        # Throttle timestamp recorded for the next check.
        assert "last_auto_restart" in ds
        # A detached helper was spawned to run evolve_daemon.sh restart.
        assert popen.call_count == 1
        argv = popen.call_args.args[0]
        assert "restart" in argv[2]  # bash -c "<...restart...>"
        assert argv[4] == str(td._WORKSPACE_ROOT / "evolve_daemon.sh")
        kwargs = popen.call_args.kwargs
        assert kwargs.get("start_new_session") is True  # setsid: detached
        assert kwargs.get("stdin") is not None

    def test_restart_throttled_when_recent(self, evolve_env: Dict) -> None:
        from unittest.mock import patch as _patch

        td = evolve_env["module"]
        recent = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
        ds = {"startup_head": "0000000", "last_auto_restart": recent}
        with _patch("subprocess.Popen") as popen:
            ok = td._schedule_drift_restart(ds)
        assert ok is False
        popen.assert_not_called()
        # Throttle must not rewrite the recorded timestamp.
        assert ds["last_auto_restart"] == recent

    def test_restart_refused_without_launcher(self, evolve_env: Dict, tmp_path: Path) -> None:
        from unittest.mock import patch as _patch

        td = evolve_env["module"]
        # Point the workspace root somewhere without evolve_daemon.sh.
        bare_dir = tmp_path / "bare"
        bare_dir.mkdir()
        old_root = td._WORKSPACE_ROOT
        td._WORKSPACE_ROOT = bare_dir
        try:
            ds = {"startup_head": "0000000"}
            with _patch("subprocess.Popen") as popen:
                ok = td._schedule_drift_restart(ds)
            assert ok is False
            popen.assert_not_called()
            assert "last_auto_restart" not in ds
        finally:
            td._WORKSPACE_ROOT = old_root


# ═══════════════════════════════════════════════════════════════════
#  Timeline persistence
# ═══════════════════════════════════════════════════════════════════


class TestTimelinePersistence:
    """load_timeline / save_timeline round-trip."""

    def test_default_timeline(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        tl = td.load_timeline()
        assert tl["version"] == 1
        assert "past" in tl
        assert "present" in tl
        assert "future" in tl
        assert tl["past"]["events"] == []

    def test_save_and_load_round_trip(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        tl = td.load_timeline()
        tl["past"]["events"].append({
            "id": "test_event",
            "type": "milestone",
            "summary": "Testing persistence",
        })
        tl["present"]["active_project"] = "test_project"
        tl["future"]["goals"].append("Test goal")
        td.save_timeline(tl)

        loaded = td.load_timeline()
        assert len(loaded["past"]["events"]) == 1
        assert loaded["past"]["events"][0]["id"] == "test_event"
        assert loaded["present"]["active_project"] == "test_project"
        assert "Test goal" in loaded["future"]["goals"]

    def test_file_created_on_save(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        tl_path = evolve_env["paths"]["timeline_file"]
        assert not tl_path.exists()
        td.save_timeline(td.load_timeline())
        assert tl_path.exists()

    def test_corrupted_file_returns_default(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        tl_path = evolve_env["paths"]["timeline_file"]
        tl_path.write_text("garbage")
        tl = td.load_timeline()
        assert tl["past"]["events"] == []


# ═══════════════════════════════════════════════════════════════════
#  Self-model persistence
# ═══════════════════════════════════════════════════════════════════


class TestSelfModelPersistence:
    """load_self_model / save_self_model round-trip."""

    def test_default_self_model(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        sm = td.load_self_model()
        assert sm["version"] == 1
        assert sm["identity"]["name"] == "Hermes (evolved)"

    def test_save_and_load_round_trip(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        sm = td.load_self_model()
        sm["identity"]["name"] = "Custom Name"
        sm["state"]["current_gap_focus"] = "Gap 6"
        sm["capabilities"]["known_skills"] = ["testing", "coding"]
        td.save_self_model(sm)

        loaded = td.load_self_model()
        assert loaded["identity"]["name"] == "Custom Name"
        assert loaded["state"]["current_gap_focus"] == "Gap 6"
        assert "testing" in loaded["capabilities"]["known_skills"]

    def test_file_created_on_save(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        sm_path = evolve_env["paths"]["self_model_file"]
        assert not sm_path.exists()
        td.save_self_model(td.load_self_model())
        assert sm_path.exists()

    def test_corrupted_file_returns_default(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        sm_path = evolve_env["paths"]["self_model_file"]
        sm_path.write_text("{{{corrupted")
        sm = td.load_self_model()
        assert sm["identity"]["name"] == "Hermes (evolved)"


# ═══════════════════════════════════════════════════════════════════
#  Prompt builder
# ═══════════════════════════════════════════════════════════════════


class TestBuildThinkingPrompt:
    """_build_thinking_prompt formats a comprehensive state summary."""

    def _make_daemon_state(self, **overrides: Any) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "status": "running",
            "tick_count": 42,
            "interval_seconds": 600,
            "last_tick": "2026-07-29T00:00:00",
            "first_tick": "2026-07-01T00:00:00",
            "last_action_output": "exit=0: testing complete",
            "cycle_stats": {"total": 42, "ok": 40, "error": 1, "parse_error": 1},
        }
        state.update(overrides)
        return state

    def _make_state(self, **daemon_overrides: Any) -> Dict[str, Any]:
        return {
            "daemon_state": self._make_daemon_state(**daemon_overrides),
            "timeline": {
                "version": 1,
                "past": {"events": [], "completed_sessions": []},
                "present": {},
                "future": {"goals": []},
            },
            "self_model": {
                "version": 1,
                "identity": {"name": "Hermes", "role": "Self-evolving AI"},
                "state": {"current_gap_focus": "Gap 6", "evolution_version": 5},
                "capabilities": {"strengths": ["shell", "write_file"], "weaknesses": [], "unknown_areas": []},
                "commitments": {},
            },
            "orientation": None,
            "world_model": None,
        }

    def test_prompt_structure(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        prompt = td._build_thinking_prompt(self._make_state())

        # Core sections
        assert "CURRENT STATE" in prompt
        assert "YOUR TASK" in prompt
        assert "Identity:" in prompt
        assert "tick_count" in prompt or "thinking cycles" in prompt

        # World model section (Gap 6)
        assert "World Model" in prompt or "world model" in prompt

    def test_prompt_includes_tick_count(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        prompt = td._build_thinking_prompt(self._make_state())
        assert "42" in prompt  # tick_count from daemon state

    def test_prompt_includes_identity(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        prompt = td._build_thinking_prompt(self._make_state())
        assert "Hermes" in prompt
        assert "Self-evolving AI" in prompt

    def test_prompt_includes_gap_focus(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        prompt = td._build_thinking_prompt(self._make_state())
        assert "Gap 6" in prompt

    def test_prompt_includes_live_root_note(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        prompt = td._build_thinking_prompt(self._make_state())
        assert "Live code root" in prompt
        assert td._WORKSPACE_ROOT_STR in prompt

    def test_live_root_note_flags_stale_copy(
        self, evolve_env: Dict, tmp_path: Any
    ) -> None:
        td = evolve_env["module"]
        stale = tmp_path / "stale-deploy"
        stale.mkdir()
        note = td._live_root_prompt_note(
            td._WORKSPACE_ROOT, stale_candidates=(str(stale),)
        )
        assert "Stale copies exist" in note
        assert str(stale) in note
        # Missing candidate dir → no stale warning.
        note2 = td._live_root_prompt_note(
            td._WORKSPACE_ROOT,
            stale_candidates=(str(tmp_path / "does-not-exist"),),
        )
        assert "Stale copies exist" not in note2
        # Workspace root itself is never flagged as stale.
        note3 = td._live_root_prompt_note(
            td._WORKSPACE_ROOT, stale_candidates=(td._WORKSPACE_ROOT_STR,)
        )
        assert "Stale copies exist" not in note3

    def test_prompt_with_events(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        state = self._make_state()
        state["timeline"]["past"]["events"] = [
            {"type": "milestone", "summary": "Completed testing"},
            {"type": "decision", "summary": "Focus on quality"},
        ]
        prompt = td._build_thinking_prompt(state)
        assert "Completed testing" in prompt
        assert "Focus on quality" in prompt

    def test_prompt_with_tasks(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        state = self._make_state()
        state["timeline"]["present"]["active_tasks"] = ["Write tests", "Fix bugs"]
        prompt = td._build_thinking_prompt(state)
        assert "Write tests" in prompt
        assert "Fix bugs" in prompt

    def test_prompt_with_goals(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        state = self._make_state()
        # Goals are now loaded from the Goals store (evolve/goals.json),
        # not from state[\"timeline\"][\"future\"][\"goals\"] — see commit 22cf22fde.
        # We must write goals to the Goals store so _build_thinking_prompt sees them.
        from data_layer import Goals
        g = Goals()
        g.propose(
            "Achieve 90% test coverage",
            description="Improve test coverage across all modules",
            rationale="Quality assurance",
            priority=1,
        )
        g.propose(
            "Implement Gap 8",
            description="Self-directed evolution — autonomous goal pursuit",
            rationale="Phase 2 completion",
            priority=2,
        )
        g.save()
        prompt = td._build_thinking_prompt(state)
        assert "90% test coverage" in prompt or "Achieve" in prompt
        assert "Implement Gap 8" in prompt

    def test_prompt_shows_goal_verification_criteria(self, evolve_env: Dict) -> None:
        """Active goals surface verification_criteria + evidence in the prompt."""
        td = evolve_env["module"]
        state = self._make_state()
        from data_layer import Goals
        g = Goals()
        g.propose(
            "Gather more shell action samples for reliable calibration",
            description="Need more shell samples for per-type calibration",
            rationale="Calibration",
            priority=1,
            verification_criteria="Prediction error for shell drops below 0.3",
        )
        g.save()
        prompt = td._build_thinking_prompt(state)
        assert "Gather more shell action samples" in prompt
        assert "Prediction error for shell drops below 0.3" in prompt

    def test_goal_evidence_line(self, evolve_env: Dict) -> None:
        """_format_goal_evidence mirrors the reconciler's title patterns."""
        td = evolve_env["module"]
        per_type = {"shell": {"count": 54, "avg_error": 0.21}}
        acc = {"avg_triple_error": 0.20}
        ev = td._format_goal_evidence(
            {"title": "Gather more shell action samples"}, per_type, acc
        )
        assert "shell has 54" in ev
        ev2 = td._format_goal_evidence(
            {"title": "Investigate shell prediction failures"}, per_type, acc
        )
        assert "0.21" in ev2
        ev3 = td._format_goal_evidence(
            {"title": "Fix overconfidence at high confidence"}, {}, acc
        )
        assert "0.20" in ev3
        assert td._format_goal_evidence({"title": "Unrelated goal"}, {}, {}) == ""
        # Missing stats must not crash — degrades to a count of 0
        ev4 = td._format_goal_evidence(
            {"title": "Gather more write_file action samples"}, {}, {}
        )
        assert "write_file has 0" in ev4

    def test_prompt_shows_no_active_plan(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        state = self._make_state()
        prompt = td._build_thinking_prompt(state)
        assert "NO ACTIVE PLAN" in prompt

    def test_prompt_shows_last_action_output(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        prompt = td._build_thinking_prompt(self._make_state())
        assert "testing complete" in prompt

    def test_prompt_with_empty_state(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        empty_state = {
            "daemon_state": {"status": "initialized", "tick_count": 0},
            "timeline": {"version": 1, "past": {}, "present": {}, "future": {}},
            "self_model": {
                "version": 1, "identity": {}, "state": {},
                "capabilities": {}, "commitments": {},
            },
            "orientation": None,
            "world_model": None,
        }
        prompt = td._build_thinking_prompt(empty_state)
        assert "CURRENT STATE" in prompt
        assert "world model" in prompt.lower()

    def test_prompt_api_call_capabilities_closed_gate(self, evolve_env: Dict) -> None:
        """No read grants -> api_call capability line keeps the discourage text."""
        td = evolve_env["module"]
        prompt = td._build_thinking_prompt(self._make_state())
        assert "Currently NO read grants are open" in prompt
        assert "do not spam" in prompt

    def test_prompt_api_call_capabilities_open_gate(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """github.read granted + allowlisted endpoint -> prompt advertises api_call."""
        td = evolve_env["module"]
        # Stub the live probe: the bridge may or may not be up in the test env.
        monkeypatch.setattr(td, "_bridge_liveness", lambda: "UP (probed)")
        state = self._make_state()
        state["self_model"]["permissions"] = {
            "github": {"read": True, "write": False, "act": False, "cap": None},
        }
        prompt = td._build_thinking_prompt(state)
        assert ">>> GATE OPEN <<<" in prompt
        assert "https://api.github.com/" in prompt
        # The old static discouragement must NOT appear when the gate is open.
        assert "do not spam" not in prompt

    def test_api_call_capability_text_unit(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Helper directly: empty/malformed permissions never crash."""
        td = evolve_env["module"]
        # Stub the live probe so the unit test is deterministic (no network).
        monkeypatch.setattr(td, "_bridge_liveness", lambda: "UP (probed)")
        assert "NO read grants" in td._api_call_capability_text(None)
        assert "NO read grants" in td._api_call_capability_text({})
        assert "NO read grants" in td._api_call_capability_text(
            {"github": {"read": False}}
        )
        out = td._api_call_capability_text({"github": {"read": True}})
        assert "GATE OPEN" in out
        assert "https://api.github.com/" in out

    def test_api_call_capability_text_reports_liveness(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Open-gate text must include the live bridge liveness signal.

        Gap 10 step 5: without it the LLM plans to "start the bridge
        integration" even when the bridge is already up and verified.
        """
        td = evolve_env["module"]
        monkeypatch.setattr(td, "_bridge_liveness", lambda: "UP (probed)")
        out = td._api_call_capability_text({"github": {"read": True}})
        assert "Host bridge liveness: UP (probed)" in out
        assert "fire api_call now" in out
        # Closed gate: no liveness signal, still discourages doomed calls.
        monkeypatch.setattr(td, "_bridge_liveness", lambda: "DOWN (refused)")
        closed = td._api_call_capability_text({"github": {"read": False}})
        assert "Host bridge liveness" not in closed

    def test_bridge_liveness_probe(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_bridge_liveness maps health responses to compact status strings."""
        td = evolve_env["module"]

        class _FakeResp:
            def __init__(self, status: int = 200, body: bytes = b'{"ok":true}'):
                self.status = status
                self._body = body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, n: int = -1) -> bytes:
                return self._body

        import urllib.request as _ur
        import urllib.error as _uer

        monkeypatch.setattr(
            _ur, "urlopen",
            lambda url, timeout: _FakeResp(200, b'{"ok":true,"pid":1}'),
        )
        assert td._bridge_liveness() == "UP (probed)"

        monkeypatch.setattr(
            _ur, "urlopen",
            lambda url, timeout: _FakeResp(500, b"boom"),
        )
        assert "UP?" in td._bridge_liveness()

        def _refused(url, timeout):
            raise _uer.URLError("Connection refused")

        monkeypatch.setattr(_ur, "urlopen", _refused)
        assert td._bridge_liveness().startswith("DOWN (")

        def _boom(url, timeout):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(_ur, "urlopen", _boom)
        assert td._bridge_liveness().startswith("DOWN (")

    def test_prompt_includes_commitments(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        state = self._make_state()
        state["timeline"]["present"]["commitments"] = [
            {"what": "Write daemon tests", "status": "active", "deadline": "2026-08-01"},
        ]
        prompt = td._build_thinking_prompt(state)
        assert "Write daemon tests" in prompt

    def test_prompt_prefers_newest_active_commitments(self, evolve_env: Dict) -> None:
        """The prompt must surface the MOST RECENT active commitments.

        add_commitment() appends (oldest -> newest), so taking the head of
        the active list would show stale pre-grant read-gate commitments
        and shadow the current guidance (e.g. an old 'awaiting github.write
        grant' row after that grant already landed). Regression test for the
        active_commits[-3:] selection fix.
        """
        td = evolve_env["module"]
        state = self._make_state()
        state["timeline"]["present"]["commitments"] = [
            {"what": "STALE: fire GET until github.write grant", "status": "active", "deadline": None},
            {"what": "STALE: old read-gate steady state", "status": "active", "deadline": None},
            {"what": "STALE: pre-grant calibration", "status": "active", "deadline": None},
            {"what": "CURRENT: one GET per cycle while awaiting Level 3 grants", "status": "active", "deadline": None},
        ]
        prompt = td._build_thinking_prompt(state)
        assert "CURRENT: one GET per cycle while awaiting Level 3 grants" in prompt
        # The oldest stale row must not be the one surfaced at the front.
        stale_idx = prompt.find("STALE: fire GET until github.write grant")
        current_idx = prompt.find("CURRENT: one GET per cycle")
        assert stale_idx == -1 or current_idx < stale_idx, (
            "newest active commitment must be surfaced over stale older rows"
        )

    def test_world_model_unavailable_graceful(self, evolve_env: Dict) -> None:
        """When world model fails to load, prompt should be graceful."""
        td = evolve_env["module"]
        # Simulate world model load failure by not providing it
        state = self._make_state()
        state["world_model"] = None
        prompt = td._build_thinking_prompt(state)
        # Should not crash, and should mention world model is unavailable
        assert "world model" in prompt.lower()

    def test_prompt_format_is_valid_string(self, evolve_env: Dict) -> None:
        """Prompt must be a valid non-empty string."""
        td = evolve_env["module"]
        prompt = td._build_thinking_prompt(self._make_state())
        assert isinstance(prompt, str)
        assert len(prompt) > 200

    def test_prompt_includes_all_required_sections(self, evolve_env: Dict) -> None:
        """Check all major sections from the thinking prompt template."""
        td = evolve_env["module"]
        prompt = td._build_thinking_prompt(self._make_state())

        sections = [
            "CURRENT STATE",
            "YOUR TASK",
            "Identity:",
            "Evolution version:",
            "Total thinking cycles",
            "Strengths:",
            "Weaknesses:",
            "Unknown areas:",
            "Commitments:",
            "Permissions (Gap 10 registry",
            "Recent timeline events:",
            "Active project:",
            "Active tasks:",
            "Active plan:",
            "World Model",
            "Orientation",
            "Future goals:",
            "Last action result:",
        ]
        for section in sections:
            assert section in prompt, f"Missing section: '{section}'"


class TestExternalBlockGuidance:
    """Gap 4: bounded multi-gap fallback guidance during external blocks.

    The guidance must fire exactly when a write allowlist entry is armed
    (the write path is ready) but the permission registry carries no
    truthy ``write`` grant — a real permission/user dependency. Normal
    cycles (grant present, or nothing armed) must be byte-identical.
    """

    def _blocked_state(self, write: bool = False) -> Dict[str, Any]:
        return {
            "daemon_state": {
                "status": "running",
                "tick_count": 42,
                "interval_seconds": 600,
                "last_tick": "2026-07-29T00:00:00",
                "first_tick": "2026-07-01T00:00:00",
                "last_action_output": "exit=0: read-gate GET ok",
                "cycle_stats": {"total": 42, "ok": 40, "error": 1, "parse_error": 1},
            },
            "timeline": {
                "version": 1,
                "past": {"events": [], "completed_sessions": []},
                "present": {},
                "future": {"goals": []},
            },
            "self_model": {
                "version": 1,
                "identity": {"name": "Hermes", "role": "Self-evolving AI"},
                "state": {"current_gap_focus": "Gap 10", "evolution_version": 5},
                "capabilities": {
                    "strengths": ["shell"],
                    "weaknesses": [],
                    "unknown_areas": [],
                },
                "commitments": {},
                "permissions": {"github": {"read": True, "write": write}},
            },
            "orientation": None,
            "world_model": None,
        }

    def test_blocked_when_write_allowlist_armed_but_no_grant(
        self, evolve_env: Dict
    ) -> None:
        td = evolve_env["module"]
        sm = {"permissions": {"github": {"read": True, "write": False}}}
        guidance = td._external_block_guidance(sm)
        assert "EXTERNAL BLOCK ACTIVE" in guidance
        assert "github" in guidance
        assert "steady-state" in guidance
        assert "ONE secondary" in guidance

    def test_empty_when_write_granted(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        sm = {"permissions": {"github": {"read": True, "write": True}}}
        assert td._external_block_guidance(sm) == ""

    def test_empty_when_no_permissions_declared(self, evolve_env: Dict) -> None:
        """A self-model with NO permissions section at all (corrupt/absent
        registry) must not change normal-cycle behavior."""
        td = evolve_env["module"]
        assert td._external_block_guidance({}) == ""
        assert td._external_block_guidance(None) == ""

    def test_blocked_when_resource_missing_from_registry(self, evolve_env: Dict) -> None:
        """Deny-by-default: a missing resource entry is still an external block
        while the write allowlist is armed for it."""
        td = evolve_env["module"]
        guidance = td._external_block_guidance({"permissions": {}})
        assert "EXTERNAL BLOCK ACTIVE" in guidance
        assert "github" in guidance

    def test_prompt_injects_guidance_when_blocked(self, evolve_env: Dict) -> None:
        """_build_thinking_prompt appends the Gap 4 guidance when a write
        grant is pending and goals exist (goals section must be populated)."""
        td = evolve_env["module"]
        from data_layer import Goals

        g = Goals()
        g.propose(
            "Resilient multi-gap progression during external blocks",
            description="Advance secondary gaps while the active gap is blocked",
            rationale="Phase 2 autonomy",
            priority=2,
        )
        g.save()
        prompt = td._build_thinking_prompt(self._blocked_state(write=False))
        assert "EXTERNAL BLOCK ACTIVE" in prompt
        assert "github" in prompt

    def test_prompt_unchanged_when_write_granted(self, evolve_env: Dict) -> None:
        """No EXTERNAL BLOCK text when the write grant exists — normal cycles
        stay byte-identical."""
        td = evolve_env["module"]
        prompt = td._build_thinking_prompt(self._blocked_state(write=True))
        assert "EXTERNAL BLOCK ACTIVE" not in prompt


# ═══════════════════════════════════════════════════════════════════
#  Reliability cycle-stat printout
# ═══════════════════════════════════════════════════════════════════


class TestPrintCycleStats:
    """_print_cycle_stats shows reliability metrics."""

    def test_prints_nothing_for_first_cycle(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        # No cycle → no output
        captured = io.StringIO()
        out = sys.stdout
        sys.stdout = captured
        try:
            td._print_cycle_stats()
        finally:
            sys.stdout = out
        assert captured.getvalue() == ""

    def test_shows_reliability_after_multiple_cycles(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        ds = td.load_daemon_state()
        ds["cycle_stats"] = {"total": 10, "ok": 8, "error": 1, "parse_error": 1,
                              "avg_duration": 5.2, "max_duration": 12.0}
        td.save_daemon_state(ds)

        captured = io.StringIO()
        out = sys.stdout
        sys.stdout = captured
        try:
            td._print_cycle_stats()
        finally:
            sys.stdout = out

        output = captured.getvalue()
        assert "reliability:" in output
        assert "80%" in output  # 8/10 = 80%

    def test_shows_stable_health(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        ds = td.load_daemon_state()
        ds["cycle_stats"] = {"total": 5, "ok": 5, "error": 0, "parse_error": 0,
                              "avg_duration": 3.0, "max_duration": 4.5}
        td.save_daemon_state(ds)

        captured = io.StringIO()
        out = sys.stdout
        sys.stdout = captured
        try:
            td._print_cycle_stats()
        finally:
            sys.stdout = out

        output = captured.getvalue()
        assert "stable" in output.lower()

    def test_shows_degraded_health(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        ds = td.load_daemon_state()
        ds["cycle_stats"] = {"total": 10, "ok": 5, "error": 3, "parse_error": 2,
                              "avg_duration": 8.0, "max_duration": 20.0}
        td.save_daemon_state(ds)

        captured = io.StringIO()
        out = sys.stdout
        sys.stdout = captured
        try:
            td._print_cycle_stats()
        finally:
            sys.stdout = out

        output = captured.getvalue()
        assert "degraded" in output.lower()


class TestDaemonStateMigration:
    """Version migration tests for daemon_state.json."""

    def test_v1_to_v2_migration_adds_cycle_history(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        ds_path = evolve_env["paths"]["daemon_state_file"]
        # Write a version 1 state without cycle_history
        import json as _json
        ds_path.parent.mkdir(parents=True, exist_ok=True)
        ds_path.write_text(_json.dumps({"version": 1, "status": "initialized", "tick_count": 5}))
        loaded = td.load_daemon_state()
        assert loaded["version"] == 2, f"Expected version 2, got {loaded['version']}"
        assert "cycle_history" in loaded, "cycle_history should be added during migration"
        assert loaded["cycle_history"] == [], "cycle_history should be empty list after migration"
        assert loaded["tick_count"] == 5, "Existing data should be preserved"

    def test_v2_state_unchanged(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        ds_path = evolve_env["paths"]["daemon_state_file"]
        import json as _json
        ds_path.parent.mkdir(parents=True, exist_ok=True)
        ds_path.write_text(_json.dumps({
            "version": 2, "status": "running", "tick_count": 10, "cycle_history": [
                {"status": "ok", "error": None, "duration": 5.0}
            ]
        }))
        loaded = td.load_daemon_state()
        assert loaded["version"] == 2
        assert loaded["tick_count"] == 10
        assert len(loaded["cycle_history"]) == 1


class TestCycleHistory:
    """cycle_history tracks per-cycle outcomes for diagnostics."""

    def test_cycle_history_appended_on_ok(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        ds = td.load_daemon_state()
        ds.setdefault("cycle_history", [])
        td.save_daemon_state(ds)

        # Simulate a successful cycle recording
        entry = {
            "timestamp": "2026-07-29T10:00:00+00:00",
            "status": "ok",
            "error": None,
            "duration": 12.5,
            "tick_count": 1,
        }
        ds["cycle_history"].append(entry)
        ds["cycle_history"] = ds["cycle_history"][-20:]
        td.save_daemon_state(ds)

        loaded = td.load_daemon_state()
        assert len(loaded["cycle_history"]) == 1
        assert loaded["cycle_history"][0]["status"] == "ok"
        assert loaded["cycle_history"][0]["error"] is None
        assert loaded["cycle_history"][0]["duration"] == 12.5

    def test_cycle_history_tracks_errors(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        ds = td.load_daemon_state()

        # Add error entries
        for i, (status, err) in enumerate([
            ("ok", None),
            ("error", "LLM returned no response"),
            ("timeout", "Cycle exceeded 120s hard limit"),
            ("parse_error", "Could not parse JSON from: xy"),
        ]):
            ds.setdefault("cycle_history", []).append({
                "timestamp": f"2026-07-29T10:0{i}:00+00:00",
                "status": status,
                "error": err,
                "duration": 10.0 + i * 5,
                "tick_count": i,
            })
        ds["cycle_history"] = ds["cycle_history"][-20:]
        td.save_daemon_state(ds)

        loaded = td.load_daemon_state()
        assert len(loaded["cycle_history"]) == 4
        error_entries = [c for c in loaded["cycle_history"] if c["status"] != "ok"]
        assert len(error_entries) == 3
        statuses = [c["status"] for c in error_entries]
        assert "timeout" in statuses
        assert "parse_error" in statuses
        assert "error" in statuses

    def test_cycle_history_persists_large_list(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        ds = td.load_daemon_state()
        ds["cycle_history"] = [{"status": "ok", "error": None, "duration": 1.0,
                                 "tick_count": i, "timestamp": "2026-07-29T10:00:00"}
                               for i in range(25)]
        td.save_daemon_state(ds)

        loaded = td.load_daemon_state()
        assert len(loaded["cycle_history"]) == 25, "save_daemon_state preserves full list"

    def test_print_cycle_stats_shows_recent_failures(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        ds = td.load_daemon_state()
        ds["cycle_stats"] = {"total": 6, "ok": 3, "error": 2, "parse_error": 1,
                              "avg_duration": 8.0, "max_duration": 25.0}
        ds["cycle_history"] = [
            {"status": "ok", "error": None, "duration": 5.0, "tick_count": 0,
             "timestamp": "2026-07-29T10:00:00"},
            {"status": "timeout", "error": "Cycle exceeded 120s hard limit",
             "duration": 120.0, "tick_count": 0,
             "timestamp": "2026-07-29T10:05:00"},
            {"status": "ok", "error": None, "duration": 6.0, "tick_count": 1,
             "timestamp": "2026-07-29T10:10:00"},
            {"status": "error", "error": "LLM returned no response",
             "duration": 45.0, "tick_count": 1,
             "timestamp": "2026-07-29T10:15:00"},
            {"status": "ok", "error": None, "duration": 4.5, "tick_count": 2,
             "timestamp": "2026-07-29T10:20:00"},
        ]
        td.save_daemon_state(ds)

        captured = io.StringIO()
        out = sys.stdout
        sys.stdout = captured
        try:
            td._print_cycle_stats()
        finally:
            sys.stdout = out

        output = captured.getvalue()
        assert "recent failures" in output.lower()
        assert "timeout" in output
        assert "LLM returned" in output


# ═══════════════════════════════════════════════════════════════════
#  Combined state persistence (integration)
# ═══════════════════════════════════════════════════════════════════


class TestStatePersistenceIntegration:
    """All three state files (timeline, self-model, daemon) coexist."""

    def test_all_state_files_independent(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]

        # Write all three with distinct data
        tl = td.load_timeline()
        tl["present"]["active_project"] = "integration_test"
        td.save_timeline(tl)

        sm = td.load_self_model()
        sm["state"]["evolution_version"] = 99
        td.save_self_model(sm)

        ds = td.load_daemon_state()
        ds["tick_count"] = 100
        td.save_daemon_state(ds)

        # Reload and verify each independently
        tl2 = td.load_timeline()
        assert tl2["present"]["active_project"] == "integration_test"

        sm2 = td.load_self_model()
        assert sm2["state"]["evolution_version"] == 99

        ds2 = td.load_daemon_state()
        assert ds2["tick_count"] == 100

    def test_files_created_in_evolve_dir(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        paths = evolve_env["paths"]

        # Save each
        td.save_timeline(td.load_timeline())
        td.save_self_model(td.load_self_model())
        td.save_daemon_state(td.load_daemon_state())

        assert paths["timeline_file"].exists()
        assert paths["self_model_file"].exists()
        assert paths["daemon_state_file"].exists()

    def test_lock_file_separate_from_state(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        lock_path = evolve_env["paths"]["daemon_lock_file"]

        td._acquire_daemon_lock()
        assert lock_path.exists()

        # Lock file is separate from state file
        state_path = evolve_env["paths"]["daemon_state_file"]
        assert not state_path.exists()

        td._release_daemon_lock()


# ═══════════════════════════════════════════════════════════════════
#  Orientation load/save (legacy support)
# ═══════════════════════════════════════════════════════════════════


class TestOrientationPersistence:
    """load_orientation / save_orientation, primarily used for legacy support."""

    def test_no_orientation_file_returns_none(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        orient = td.load_orientation()
        assert orient is None

    def test_save_and_load_round_trip(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        data = {"focus": "Gap 6", "insights": ["test"], "next_steps": ["step1"]}
        td.save_orientation(data)

        loaded = td.load_orientation()
        assert loaded is not None
        assert loaded["focus"] == "Gap 6"
        assert loaded["insights"] == ["test"]

    def test_file_created_on_save(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        orient_path = evolve_env["paths"]["orientation_file"]
        assert not orient_path.exists()

        td.save_orientation({"focus": "test"})
        assert orient_path.exists()


# ═══════════════════════════════════════════════════════════════════
#  Provider auto-detection from env vars
# ═══════════════════════════════════════════════════════════════════


class TestAutoDetectProviderFromEnv:
    """_auto_detect_provider_from_env maps env vars to provider/model pairs."""

    def test_detects_opencode_from_env(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        result = td._auto_detect_provider_from_env()
        # OPENCODE_API_KEY is set in CI/test environment
        if result:
            assert result[0] == "opencode-go"
            assert result[1] == "glm-5"

    def test_detects_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib
        import sys
        # Fresh import with a clean env
        monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test123")
        for mod in ("data_layer", "think_daemon"):
            if mod in sys.modules:
                del sys.modules[mod]
        import think_daemon as td  # type: ignore
        result = td._auto_detect_provider_from_env()
        assert result is not None
        assert result[0] == "anthropic"
        assert result[1] == "claude-sonnet-4-20250514"

    def test_returns_none_with_no_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib
        import sys
        monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        for mod in ("data_layer", "think_daemon"):
            if mod in sys.modules:
                del sys.modules[mod]
        import think_daemon as td  # type: ignore
        result = td._auto_detect_provider_from_env()
        assert result is None

    def test_ensure_runtime_main_uses_env_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When config.yaml doesn't exist but OPENCODE_API_KEY is set, runtime should init."""
        import importlib
        import sys
        # Clear any cached runtime state
        monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
        monkeypatch.setenv("OPENCODE_API_KEY", "sk-test-fallback")
        for mod in ("data_layer", "think_daemon", "agent.auxiliary_client"):
            sys.modules.pop(mod, None)
        import think_daemon as td  # type: ignore
        import data_layer  # noqa: F401
        # Reset global flag
        td._RUNTIME_INITIALIZED = False
        td._ensure_runtime_main()
        assert td._RUNTIME_INITIALIZED
        assert td._RUNTIME_PROVIDER == "opencode-go"
        assert td._RUNTIME_MODEL == "glm-5"


# ═══════════════════════════════════════════════════════════════════
#  Self-model pruning (_prune_self_model)
# ═══════════════════════════════════════════════════════════════════


class TestPruneSelfModel:
    """_prune_self_model removes stale/duplicate entries from self-model."""

    def test_empty_sm_no_changes(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        sm = {"capabilities": {}, "commitments": {}}
        removed = td._prune_self_model(sm)
        assert removed == 0

    def test_no_duplicates_keeps_all_within_cap(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        sm = {
            "capabilities": {
                "weaknesses": [
                    "widget prediction has high error (avg 0.85)",
                    "shell actions timeout on large outputs",
                    "git push fails on auth",
                ],
                "unknown_areas": [
                    "How to deploy to production",
                ],
            },
            "commitments": {
                "promised_features": [
                    "Fix widget prediction",
                    "Add retry for shell timeout",
                ],
            },
        }
        removed = td._prune_self_model(sm)
        assert removed == 0
        assert len(sm["capabilities"]["weaknesses"]) == 3
        assert len(sm["capabilities"]["unknown_areas"]) == 1
        assert len(sm["commitments"]["promised_features"]) == 2

    def test_deduplicates_near_duplicate_weaknesses(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        sm = {
            "capabilities": {
                "weaknesses": [
                    "Prediction calibration needs more shell samples",
                    "Prediction calibration needs more samples for shell",
                ],
            },
        }
        removed = td._prune_self_model(sm)
        assert removed == 1, "Expected 1 weakness removed as duplicate"
        assert len(sm["capabilities"]["weaknesses"]) == 1

    def test_prune_collapses_summarize_instead_of_execute_class(self, evolve_env: Dict) -> None:
        """Pattern 10: recurring self-flagellation variants must not fill the
        weakness budget — all phrasings of the same class collapse to zero
        (the LLM regenerates one fresh instance if the behavior is still real),
        while a genuinely distinct weakness survives."""
        td = evolve_env["module"]
        sm = {
            "capabilities": {
                "weaknesses": [
                    "Still at risk of summarizing instead of executing; must treat the read as an immediate command.",
                    "Can still slip into orientation-summary output while deferring code reads/edits.",
                    "Still prone to one or more summary-only cycles before executing; must treat the next cycle as a patch-writing cycle.",
                    "Still at risk of deferring the patch to 'next cycle' after seeing grep output.",
                    "Still prone to announcing next steps instead of executing them; must treat the action field as the immediate command.",
                    "Still at risk of turning a completed inspection into a summary instead of a patch.",
                    "I keep deferring implementation after orientation notes; this cycle I bind the plan to a concrete patch step.",
                    "Still at risk of writing orientation notes instead of executing concrete code reads/edits.",
                    "My predictions for database migrations are systematically overconfident (0.8 avg error)",
                ],
            },
        }
        removed = td._prune_self_model(sm, daemon_state={"tick_count": 340, "last_output": {"fallback": False}})
        remaining = sm["capabilities"]["weaknesses"]
        # All 8 self-flagellation variants removed; the distinct weakness survives.
        assert removed == 8, f"Expected 8 template weaknesses removed, got {removed}"
        assert len(remaining) == 1, f"Expected 1 remaining weakness, got {remaining!r}"
        assert "database migrations" in remaining[0]

    def test_prune_pattern10_gated_on_tick_count(self, evolve_env: Dict) -> None:
        """Pattern 10 must not fire on early cycles (tick < 10) so the
        self-model is not over-pruned before the daemon accumulates context."""
        td = evolve_env["module"]
        sm = {
            "capabilities": {
                "weaknesses": [
                    "Still at risk of summarizing instead of executing; must patch now.",
                ],
            },
        }
        removed = td._prune_self_model(sm, daemon_state={"tick_count": 3, "last_output": {"fallback": False}})
        assert removed == 0, f"Pattern 10 fired too early: removed {removed}"
        assert len(sm["capabilities"]["weaknesses"]) == 1

    def test_llm_weakness_survives_prune_when_current_cycle_llm_failed(
        self, evolve_env: Dict
    ) -> None:
        """Regression (2026-08-03, cycle 501): the post-cycle prune must not
        remove the 'LLM API unreliable' weakness in the very cycle where the
        LLM call just failed.

        The prune's fallback heuristic reads ``last_output.fallback`` which
        reflects the PREVIOUS cycle. Cycle 500 was llm-backed
        (fallback=False), so the heuristic concluded "LLM is working" and
        removed the weakness — even though cycle 501's LLM call timed out
        twice and fell back to local analysis. The fix: callers that know
        the CURRENT cycle's outcome pass ``llm_working`` explicitly, and the
        heuristic is bypassed.
        """
        td = evolve_env["module"]
        sm = {
            "capabilities": {
                "weaknesses": [
                    "LLM API unreliable: timeouts during thinking cycles",
                ],
            },
        }
        # Previous cycle was llm-backed → the stale heuristic alone would prune.
        daemon_state = {"tick_count": 501, "last_output": {"fallback": False}}
        # Current cycle LLM FAILED → llm_working=False → weakness must survive.
        removed = td._prune_self_model(
            sm, daemon_state=daemon_state, llm_working=False
        )
        assert removed == 0, (
            f"LLM weakness pruned despite current-cycle LLM failure: {removed}"
        )
        assert len(sm["capabilities"]["weaknesses"]) == 1, (
            f"Weakness wrongly removed: {sm['capabilities']['weaknesses']!r}"
        )

    def test_llm_weakness_pruned_when_current_cycle_llm_working(
        self, evolve_env: Dict
    ) -> None:
        """Sanity companion: when the CURRENT cycle is llm-backed
        (llm_working=True), the stale LLM-reliability weakness is pruned as
        before — the override must not disable legitimate pruning."""
        td = evolve_env["module"]
        sm = {
            "capabilities": {
                "weaknesses": [
                    "LLM API unreliable: timeouts during thinking cycles",
                ],
            },
        }
        daemon_state = {"tick_count": 501, "last_output": {"fallback": False}}
        removed = td._prune_self_model(
            sm, daemon_state=daemon_state, llm_working=True
        )
        assert removed == 1, f"Expected LLM weakness pruned, removed={removed}"
        assert sm["capabilities"]["weaknesses"] == []

    def test_llm_weakness_heuristic_unchanged_without_override(
        self, evolve_env: Dict
    ) -> None:
        """Backward compatibility: callers that don't pass ``llm_working``
        keep the previous-cycle heuristic (pre-cycle prune path and all
        existing test call sites)."""
        td = evolve_env["module"]
        sm = {
            "capabilities": {
                "weaknesses": [
                    "LLM API unreliable: timeouts during thinking cycles",
                ],
            },
        }
        # Previous cycle llm-backed (fallback=False) → heuristic prunes.
        removed = td._prune_self_model(
            sm, daemon_state={"tick_count": 501, "last_output": {"fallback": False}}
        )
        assert removed == 1
        # Previous cycle fallback → heuristic keeps the weakness.
        sm2 = {
            "capabilities": {
                "weaknesses": [
                    "LLM API unreliable: timeouts during thinking cycles",
                ],
            },
        }
        removed2 = td._prune_self_model(
            sm2, daemon_state={"tick_count": 501, "last_output": {"fallback": True}}
        )
        assert removed2 == 0
        assert len(sm2["capabilities"]["weaknesses"]) == 1


    def test_caps_commitments_to_eight(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        sm = {
            "capabilities": {"weaknesses": []},
            "commitments": {
                "promised_features": [f"Commitment {i}" for i in range(15)],
            },
        }
        removed = td._prune_self_model(sm)
        assert removed == 7, "Expected 7 commitments removed (15→8)"
        assert len(sm["commitments"]["promised_features"]) == 8
        assert sm["commitments"]["promised_features"] == [
            "Commitment 7", "Commitment 8", "Commitment 9",
            "Commitment 10", "Commitment 11", "Commitment 12",
            "Commitment 13", "Commitment 14",
        ]

    def test_deduplicates_unknown_areas(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        sm = {
            "capabilities": {
                "weaknesses": [],
                "unknown_areas": [
                    "How to inject orientation into system prompt here",
                    "inject orientation into the system prompt here",
                    "A completely different unknown area",
                ],
            },
        }
        removed = td._prune_self_model(sm)
        # First and second share >50% word overlap → 1 removed
        assert removed == 1
        assert len(sm["capabilities"]["unknown_areas"]) == 2

    def test_exact_match_deduplication(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        sm = {
            "capabilities": {
                "weaknesses": [
                    "Same exact text multiple",
                    "Same exact text multiple",
                    "A third unique weakness",
                ],
            },
        }
        removed = td._prune_self_model(sm)
        assert removed == 1
        assert len(sm["capabilities"]["weaknesses"]) == 2
        assert sm["capabilities"]["weaknesses"][0] == "Same exact text multiple"

    def test_substring_deduplication(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        sm = {
            "capabilities": {
                "weaknesses": [
                    "Insufficient install_package samples (n=2) for reliable calibration",
                    "Insufficient install_package samples",
                ],
            },
        }
        removed = td._prune_self_model(sm)
        assert removed == 1
        assert len(sm["capabilities"]["weaknesses"]) == 1

    # ── Health-aware (daemon_state) pruning ──────────────────────────
    # The stale patterns are gated on daemon health data (tick_count,
    # last_output.fallback). These tests exercise that branch, which the
    # earlier tests above never hit.

    def test_health_aware_removes_test_plan_weaknesses(
        self, evolve_env: Dict
    ) -> None:
        """Test-plan weakness class is pruned once the plan is complete."""
        td = evolve_env["module"]
        sm = {
            "capabilities": {
                "weaknesses": [
                    "Still at risk of re-planning instead of executing; mitigations must force at least one inspect/run action per cycle until Tests pass.",
                    "Deliberation-to-action latency on the Test plan persists until execution completes",
                    "Still prone to producing orientation summaries without executing pending steps; must treat shell/write actions as first-class outputs every cycle.",
                    "Repeated planning without execution on the pending Test plan; must maintain action-first discipline until both Test steps are complete.",
                    "Risk of pausing to deliberate immediately after locating the file; next move must be immediate test execution.",
                    "Deliberation-to-action latency still recurring; mitigation: forced shell inspection plus test run this cycle",
                    "Still prone to re-planning; must keep concrete inspect/run actions first in every cycle until Test steps pass.",
                ],
            },
        }
        removed = td._prune_self_model(sm, daemon_state={"tick_count": 263})
        assert removed == 6, "Expected all 6 Test-plan weaknesses removed"
        remaining = sm["capabilities"]["weaknesses"]
        assert len(remaining) == 1
        assert "orientation summaries" in remaining[0]

    def test_health_aware_removes_think_daemon_location_unknowns(
        self, evolve_env: Dict
    ) -> None:
        """Unknowns about think_daemon.py location / test harness are stale."""
        td = evolve_env["module"]
        sm = {
            "capabilities": {
                "weaknesses": [],
                "unknown_areas": [
                    "Location of think_daemon.py and the exact test harness for world-model predictions.",
                    "Location of think_daemon.py (being resolved this cycle).",
                    "Exact test harness for world-model predictions still unknown until script inspection completes.",
                    "Exact test harness invocation",
                    "A genuinely unresolved unknown area",
                ],
            },
        }
        removed = td._prune_self_model(sm, daemon_state={"tick_count": 263})
        assert removed == 4
        assert sm["capabilities"]["unknown_areas"] == [
            "A genuinely unresolved unknown area"
        ]

    def test_health_aware_removes_stale_test_plan_commitments(
        self, evolve_env: Dict
    ) -> None:
        """Commitments promising to run the completed Test plan are pruned."""
        td = evolve_env["module"]
        sm = {
            "capabilities": {"weaknesses": [], "unknown_areas": []},
            "commitments": {
                "promised_features": [
                    "Locate think_daemon.py and run the two Test steps this cycle, using /proc and filesystem inspection instead of ps.",
                    "Complete Step A inspection and Step B execution of the Test plan before any further planning.",
                    "Run both pending Test steps as soon as think_daemon.py is located, without adding another planning-only cycle.",
                    "As soon as the shell result reveals the script path, run the two pending Test steps in the same working session without re-planning.",
                    "Ship the widget refactor by Friday",
                ],
            },
        }
        removed = td._prune_self_model(sm, daemon_state={"tick_count": 263})
        assert removed == 4
        assert sm["commitments"]["promised_features"] == [
            "Ship the widget refactor by Friday"
        ]

    def test_health_aware_gate_below_tick_threshold(
        self, evolve_env: Dict
    ) -> None:
        """Below the tick threshold, Test-plan entries are NOT pruned."""
        td = evolve_env["module"]
        sm = {
            "capabilities": {
                "weaknesses": [
                    "Deliberation-to-action latency on the Test plan persists until execution completes",
                ],
                "unknown_areas": [
                    "Location of think_daemon.py (being resolved this cycle).",
                ],
            },
            "commitments": {
                "promised_features": [
                    "Run the two Test steps this cycle",
                ],
            },
        }
        removed = td._prune_self_model(sm, daemon_state={"tick_count": 3})
        assert removed == 0
        assert len(sm["capabilities"]["weaknesses"]) == 1
        assert len(sm["capabilities"]["unknown_areas"]) == 1
        assert len(sm["commitments"]["promised_features"]) == 1

    # ── Evidence-based resolution (documented subjects) ─────────────

    def test_evidence_based_resolves_new_phrasings(self, evolve_env: Dict) -> None:
        """Entries restating established facts are pruned regardless of phrasing.

        The canonical case: the think_daemon.py path/structure/read was
        executed (cycle 293) and documented in docs/think_daemon_loop.md,
        yet new LLM phrasings of "unknown" about it evaded every hardcoded
        pattern.  Evidence-based resolution clears any phrasing.
        """
        td = evolve_env["module"]
        sm = {
            "capabilities": {
                "weaknesses": [
                    "I keep noting LLM unavailability instead of forcing the pending think_daemon.py read to the top of every cycle.",
                    "Still prone to producing orientation summaries without executing pending steps; must treat shell/write actions as first-class outputs every cycle.",
                ],
                "unknown_areas": [
                    "Exact path, length, and core-loop structure of think_daemon.py.",
                    "think_daemon.py path, line count, and core-loop structure still unresolved until this cycle's action runs.",
                    "A genuinely unresolved unknown area",
                ],
            },
            "commitments": {
                "promised_features": [
                    "Execute the think_daemon.py read in this immediate cycle and document the core loop within the next 2 cycles.",
                    "Ship the widget refactor by Friday",
                ],
            },
        }
        removed = td._prune_self_model(sm, daemon_state={"tick_count": 263})
        # 1 weakness + 2 unknowns + 1 commitment resolved by evidence
        assert removed == 4
        assert sm["capabilities"]["weaknesses"] == [
            "Still prone to producing orientation summaries without executing pending steps; must treat shell/write actions as first-class outputs every cycle.",
        ]
        assert sm["capabilities"]["unknown_areas"] == [
            "A genuinely unresolved unknown area",
        ]
        assert sm["commitments"]["promised_features"] == [
            "Ship the widget refactor by Friday",
        ]

    def test_evidence_based_gate_below_tick_threshold(self, evolve_env: Dict) -> None:
        """Below the tick threshold, evidence-based resolution is skipped."""
        td = evolve_env["module"]
        sm = {
            "capabilities": {
                "weaknesses": [],
                "unknown_areas": [
                    "Exact path, length, and core-loop structure of think_daemon.py.",
                ],
            },
            "commitments": {"promised_features": []},
        }
        removed = td._prune_self_model(sm, daemon_state={"tick_count": 3})
        assert removed == 0
        assert len(sm["capabilities"]["unknown_areas"]) == 1

    def test_evidence_based_keeps_design_questions(self, evolve_env: Dict) -> None:
        """Design questions that merely mention a file are NOT resolved."""
        td = evolve_env["module"]
        sm = {
            "capabilities": {
                "weaknesses": [],
                "unknown_areas": [
                    "What new capability should the daemon add next for self-directed evolution?",
                    "How can the world model reduce prediction error for shell actions?",
                ],
            },
            "commitments": {"promised_features": []},
        }
        removed = td._prune_self_model(sm, daemon_state={"tick_count": 263})
        assert removed == 0
        assert len(sm["capabilities"]["unknown_areas"]) == 2

    def test_subject_is_resolved_predicate(self, evolve_env: Dict) -> None:
        """The predicate keys on evidence + knowledge keywords, not wording."""
        td = evolve_env["module"]
        assert td._subject_is_resolved(
            "Exact path, length, and core-loop structure of think_daemon.py."
        )
        assert td._subject_is_resolved(
            "think_daemon.py path, line count, and core-loop structure still unresolved until this cycle's action runs."
        )
        assert td._subject_is_resolved(
            "exact structure of world_model.py data and its prediction API"
        )
        # Design questions / unrelated text are NOT resolved
        assert not td._subject_is_resolved(
            "What new capability should the daemon add next for self-directed evolution?"
        )
        assert not td._subject_is_resolved(
            "Ship the widget refactor by Friday"
        )
        assert not td._subject_is_resolved(
            "A genuinely unresolved unknown area"
        )

    def test_subject_is_resolved_covers_gap8_wiring_and_type_guards(
        self, evolve_env: Dict
    ) -> None:
        """The Gap 8 retry/budget-wiring and type-guard subjects are resolved.

        Regression for the 2026-08-01 fixation loop: the daemon re-asked
        "where is the retry/budget policy wired" for many cycles even
        though the policy was committed and documented.  The actual live
        entries (self_model.json at the time of writing) must now resolve
        so _prune_self_model removes them and _apply_insights blocks
        re-adding them.
        """
        td = evolve_env["module"]
        stale_unknowns = [
            "Whether the live daemon's LLM call is sync or async and where "
            "the retry/budget policy is currently referenced.",
            "Exact LLM call-site lines and current integration status in "
            "think_daemon.py.",
            "Whether the existing retry/budget policy is currently invoked "
            "anywhere else in the daemon or only defined at lines 133-138.",
            "Exact mechanics of the live LLM call-site (sync/async) and how "
            "the retain/budget policy at lines 133-138 is intended to wrap "
            "the auxiliary_client call.",
            "Whether wiring the retry/budget policy requires changes inside "
            "auxiliary_client or only at the invocation point; also whether "
            "the daemon has a test harness to run after the patch.",
            "Exact LLM call-site lines in think_daemon.py and how the "
            "retry/budget helper should be invoked around the "
            "auxiliary_client call.",
            "Exact LLM call-site line numbers and sync/async nature of the "
            "call; whether retry/budget helpers are referenced anywhere "
            "besides lines 133-138.",
        ]
        stale_commitments = [
            "Next cycle, move from outlining to modifying/integrating the "
            "daemon's retry/budget functions into a concrete evolution "
            "change toward Gap 8.",
            "Integrate retry/budget logic into think_daemon.py and verify "
            "it compiles/tests within the next two cycles.",
            "After seeing the call-site grep output, write the integration "
            "patch for retry/budget logic in think_daemon.py within the "
            "next cycle.",
            "After seeing the extracted retry/budget section, emit the "
            "targeted edit to think_daemon.py and verify it compiles — by "
            "the end of next cycle.",
            "After the grep output returns, apply the retry/budget "
            "integration patch before writing any further orientation "
            "summary.",
            "After this inspection, apply the minimal patch to "
            "think_daemon.py connecting the retry/budget policy to the LLM "
            "call and verify with py_compile.",
            "Wire the retry/budget helper into the auxiliary_client LLM "
            "call site and verify python3 -m py_compile passes within the "
            "next 2 cycles.",
            "Write and py_compile-verify the retry/budget wiring patch in "
            "think_daemon.py during the next cycle.",
        ]
        stale_weakness = (
            "Crash-prone when action/outcome fields contain non-string types "
            "(e.g., TypeError: 'int' object has no attribute 'strip'); needs "
            "type-guards in cycle processing."
        )
        for entry in stale_unknowns + stale_commitments:
            assert td._subject_is_resolved(entry), entry[:80]
        assert td._subject_is_resolved(stale_weakness)

        # _prune_self_model actually removes them (evidence section needs
        # daemon_state with tick_count >= 10).
        sm = {
            "capabilities": {
                "weaknesses": [stale_weakness],
                "unknown_areas": list(stale_unknowns),
            },
            "commitments": {"promised_features": list(stale_commitments)},
        }
        removed = td._prune_self_model(
            sm, daemon_state={"tick_count": 340, "last_output": {"fallback": False}}
        )
        assert removed == 1 + len(stale_unknowns) + len(stale_commitments)
        assert sm["capabilities"]["weaknesses"] == []
        assert sm["capabilities"]["unknown_areas"] == []
        assert sm["commitments"]["promised_features"] == []

    def test_subject_is_resolved_covers_gap10_l2_ramp(
        self, evolve_env: Dict
    ) -> None:
        """The Gap 10 Level 2 write-ramp evidence gap is resolved.

        Regression for the 2026-08-02 stale-commitment accumulation: after
        the github.write grant landed and both allowlisted docs were
        published (plan sha 273306d..., evidence sha 7efe838...;
        bridge_audit.log shows exec PUT entries for both), the live
        self_model.json still carried 6 promised_features, 5 of which
        restated the completed ramp (fire the evidence write, verify the
        audit trail, sustain the read-gate steady state, never re-verify
        the committed L2 path).  Those must now resolve so _prune_self_model
        removes them and _apply_insights blocks re-adding them, while the
        durable bridge-recovery policy and genuinely new Level 3 design
        questions survive.
        """
        td = evolve_env["module"]
        stale_commitments = [
            "Supersede any stale bounded-verify scheduler.py/think_daemon.py "
            "preflight commitment: that wiring is fulfilled and committed "
            "(0de506ab7, facbda171); do not re-verify it while the "
            "github.write grant is pending.",
            "Reserve the single action slot for the primary steady-state "
            "action each cycle (GET api_call if bridge UP; targeted recovery "
            "probe if bridge DOWN); encode secondary advancing slices as "
            "plan/goal bookkeeping or timeline decisions, and never "
            "re-verify the committed L2 path.",
            "Fire the second allowlisted write (PUT evidence/gap10-level1.md) "
            "next cycle via scripts/fire_gap10_write.py evidence1; then the "
            "Level 2 ramp is complete and Gap 10's remaining work is "
            "post-ramp verification.",
            "Publish evidence/gap10-level1.md via the allowlisted bridge and "
            "verify bridge_audit.log shows exec PUT entries for both Level 2 "
            "docs.",
            "Keep the Level 2 write ramp moving to completion: fire the "
            "evidence write, verify audit entries, then close the ramp "
            "without re-verifying committed wiring.",
        ]
        keep_commitments = [
            # Durable operational policy — no gap10/L2 vocabulary.
            "On any cycle where the bridge is DOWN, spend the first action "
            "on a targeted bridge recovery probe (locate launcher/health "
            "script, attempt restart), not on passive ls or a doomed "
            "api_call.",
            # Genuinely new gap10 work — lacks completed-ramp vocabulary.
            "Request a Level 3 github.write grant for docs/ and draft the "
            "proposal document.",
        ]
        for entry in stale_commitments:
            assert td._subject_is_resolved(entry), entry[:80]
        for entry in keep_commitments:
            assert not td._subject_is_resolved(entry), entry[:80]

        sm = {
            "capabilities": {"weaknesses": [], "unknown_areas": []},
            "commitments": {
                "promised_features": list(stale_commitments) + keep_commitments
            },
        }
        removed = td._prune_self_model(
            sm, daemon_state={"tick_count": 340, "last_output": {"fallback": False}}
        )
        assert removed == len(stale_commitments)
        assert sm["commitments"]["promised_features"] == keep_commitments

    def test_subject_is_resolved_covers_gap10_bridge_healthcheck(
        self, evolve_env: Dict
    ) -> None:
        """The Gap 10 bridge health-check/restart validation gap is resolved.

        Regression for the 2026-08-03 stale-weakness accumulation: commit
        08dabcb2b completed the bridge health-check/restart goal with live
        evidence (bridge stopped -> scripts/bridge_healthcheck.py detected
        DOWN -> restarted -> re-probed UP -> following allowlisted GET HTTP
        200; goal_20260802151450_3 completed), yet self_model.json still
        carried a weakness claiming the restart path was "not yet validated",
        a DOWN-bridge simulation unknown, and two promised_features about
        validating/recording the recovery procedure by 2026-08-10.  Those
        must now resolve so _prune_self_model removes them, while the durable
        DOWN-recovery policy and genuinely new health-check design questions
        survive.
        """
        td = evolve_env["module"]
        stale_entries = [
            "The bridge health-check script is committed but its restart "
            "path is not yet validated against a real/simulated bridge DOWN; "
            "until validated, the automation is only partially proven.",
            "Risk of over-planning the DOWN-bridge validation and "
            "disrupting a healthy live bridge; must first inspect for a "
            "safe simulation path.",
            "Whether a controlled DOWN-bridge simulation is safe and what "
            "its invocation looks like.",
            "Complete validation of the bridge health-check/restart path by "
            "2026-08-10 while maintaining one allowlisted GET per cycle.",
            "Inspect for a safe DOWN-bridge simulation path before any live "
            "shutdown; validate and record the recovery procedure by "
            "2026-08-10.",
        ]
        keep_entries = [
            # Durable operational policy — bridge DOWN handling, no
            # validation/restart-path vocabulary.
            "On any cycle where the bridge is DOWN, spend the first action "
            "on a targeted bridge recovery probe (locate launcher/health "
            "script, attempt restart), not on passive ls or a doomed "
            "api_call.",
            # Genuinely new design question — health-check feature work
            # without resolved-validation vocabulary.
            "Should the health-check script alert a human operator when "
            "the bridge fails to come back up?",
        ]
        for entry in stale_entries:
            assert td._subject_is_resolved(entry), entry[:80]
        for entry in keep_entries:
            assert not td._subject_is_resolved(entry), entry[:80]

        sm = {
            "capabilities": {
                "weaknesses": stale_entries[:2],
                "unknown_areas": stale_entries[2:3],
            },
            "commitments": {
                "promised_features": stale_entries[3:] + keep_entries
            },
        }
        removed = td._prune_self_model(
            sm, daemon_state={"tick_count": 340, "last_output": {"fallback": False}}
        )
        assert removed == len(stale_entries)
        assert sm["commitments"]["promised_features"] == keep_entries
        assert sm["capabilities"]["weaknesses"] == []
        assert sm["capabilities"]["unknown_areas"] == []

    def test_subject_is_resolved_covers_p3_retry_telemetry(
        self, evolve_env: Dict
    ) -> None:
        """The P3 retry-path telemetry evidence gap is resolved.

        Regression for the 2026-08-01 fixation loop: after the per-attempt
        retry telemetry patch landed (commit 8ef5bb051, _call_llm logs
        {attempt, outcome, wait_s, reason} into _last_llm_call_stats
        ["attempts"] and the world-model llm_call triple), the daemon kept
        re-asking "exact retry-helper variable names and block boundaries
        before the patch can be written" for cycles 361-369.  The live
        self_model.json entries at the time of writing must now resolve so
        _prune_self_model removes them and _apply_insights blocks
        re-adding them in new phrasings.
        """
        td = evolve_env["module"]
        stale_weaknesses = [
            "I still depend on this cycle's grep output for exact "
            "retry-helper variable names and block boundaries, so the "
            "patch cannot be safely written from memory alone.",
            "Still reliant on exact inner-block source for the retry "
            "helper; need to see AST output.",
            "Still need exact inner-block source; risk of patching wrong "
            "variable names remains until lines 125-165 are inspected.",
        ]
        stale_unknowns = [
            "Whether the retry helper already exposes per-attempt values "
            "(attempt, wait, reason) that can be logged without changing "
            "control flow.",
            "Exact call-site variable names available for logging retry "
            "attempts/budget decisions.",
            "Internal structure of the adaptive retry/budget block beyond "
            "confirmed header lines 136/138/140 (variable names, loop "
            "shape, and available per-attempt values).",
            "Whether retry/budget is a standalone function or inline "
            "block; AST will resolve.",
            "Exact body of _llm_retry_policy (lines 239-288) and how "
            "skip/exhaustion decisions are represented at the call site.",
            "Exact variable names in the retry/budget helper lines 125-165",
            "Exact retry helper parameter names and call-site scope in "
            "think_daemon.py beyond confirmed boundary lines.",
        ]
        stale_commitments = [
            "Land retry-telemetry logging in think_daemon.py and verify "
            "via grep + py_compile before the next two cycles elapse.",
            "Do not write the retry-telemetry patch until the exact "
            "retry-helper line range and variable names are confirmed "
            "from compact grep/sed output.",
            "Extract exact helper bodies in small chunks, then land "
            "retry-telemetry with py_compile verification before "
            "declaring P3 done.",
            "Inspect lines 125-165 now and land the retry telemetry patch "
            "within this or the next cycle, verified by py_compile.",
            "Write the retry-telemetry patch only after this cycle's "
            "AST/grep evidence confirms variable names; then run "
            "py_compile and commit within the next two cycles.",
        ]
        for entry in stale_weaknesses + stale_unknowns + stale_commitments:
            assert td._subject_is_resolved(entry), entry[:80]

        # _prune_self_model actually removes them (evidence section needs
        # daemon_state with tick_count >= 10).
        sm = {
            "capabilities": {
                "weaknesses": list(stale_weaknesses),
                "unknown_areas": list(stale_unknowns),
            },
            "commitments": {"promised_features": list(stale_commitments)},
        }
        removed = td._prune_self_model(
            sm, daemon_state={"tick_count": 340, "last_output": {"fallback": False}}
        )
        assert removed == len(stale_weaknesses) + len(stale_unknowns) + len(
            stale_commitments
        )
        assert sm["capabilities"]["weaknesses"] == []
        assert sm["capabilities"]["unknown_areas"] == []
        assert sm["commitments"]["promised_features"] == []

        # Inspection-discipline lessons are NOT part of the retry-helper
        # subject and must survive.
        discipline = [
            "Repeated temptation to use full-line sed dumps despite known "
            "truncation; must always use grep/head/python slicing for "
            "source inspection.",
            "Use bounded line slices and compact greps for any further "
            "think_daemon.py inspection; never full-file dumps.",
        ]
        for entry in discipline:
            assert not td._subject_is_resolved(entry), entry[:80]

    def test_subject_is_resolved_keeps_new_design_questions(
        self, evolve_env: Dict
    ) -> None:
        """Genuine design questions about the same files must survive."""
        td = evolve_env["module"]
        new_design_questions = [
            "How should the daemon's retry policy add exponential backoff "
            "to reduce outage cost?",
            "Should the LLM call use a longer timeout for reasoning models?",
            "How should cycle processing guard against entirely new field "
            "types from the LLM?",
            "What new capability should the daemon add next for "
            "self-directed evolution?",
        ]
        for q in new_design_questions:
            assert not td._subject_is_resolved(q), q


# ═══════════════════════════════════════════════════════════════════
#  Action deduplication gate (break LLM fixation loops)
# ═══════════════════════════════════════════════════════════════════


class TestActionDedupGate:
    """_apply_insights' action deduplication gate overrides repetitive LLM actions."""

    def test_repetitive_shell_action_overridden(self, evolve_env: Dict) -> None:
        """When the LLM proposes the same type+description as recent triples, it's overridden."""
        td = evolve_env["module"]
        from world_model import WorldModel

        # Create a world model with 3 repetitive "read think_daemon" shell actions
        wm = WorldModel()
        # Use realistic descriptions from the actual world model fixation pattern
        # (source: /root/.hermes-evolved/evolve/world_model.json)
        realistic_descriptions = [
            "Read think_daemon source code to understand its loop structure and identify integration points.",
            "Read think_daemon source code to understand its loop and extension points",
            "Read the think_daemon source code to understand its loop and action selection.",
        ]
        for desc in realistic_descriptions:
            tid = wm.record_action("shell", desc, "should read file")
            wm.complete_action(tid, "exit=0: file contents shown")

        # Build a minimal daemon state with tick_count >= 10 so auto-defaults are active
        ds = {"tick_count": 15, "last_action_output": ""}

        # Build a minimal orientation to avoid KeyError in prompt building
        orient = {"vision": "Test", "phase": "test"}

        # Propose a new action that is also "read think_daemon" → should be dedup'd
        result = {
            "action": {
                "type": "shell",
                "command": "cat think_daemon.py",
                "description": "Read the think_daemon.py source code to understand its internal structure",
            },
            "fallback": False,
            "insight": "test",
            "focus_next": "continue",
            "confidence": 0.5,
            "reasoning": "test",
            "event_to_record": None,
            "outcome_to_record": None,
            "commitment": None,
            "prediction": None,
            "session_record": None,
            "plan_action": None,
            "new_plan": None,
            "new_goal": None,
            "goal_action": None,
            "search_query": None,
            "episodic_record": None,
            "self_model_update": {"weakness": None, "unknown": None, "new_commitment": None},
            "next_gap": None,
        }

        state = {
            "daemon_state": ds,
            "world_model": wm,
            "timeline": {"version": 1, "past": {"events": []}, "present": {}, "future": {}},
            "self_model": {
                "identity": {"name": "test", "role": "test"},
                "state": {},
                "capabilities": {"strengths": [], "weaknesses": [], "unknown_areas": []},
                "commitments": {},
            },
            "orientation": orient,
        }

        # mock out subprocess.run so _apply_insights doesn't actually execute shell commands.
        # Let real file I/O happen (evolve dir already exists from fixture).
        import subprocess
        original_run = subprocess.run
        try:
            def _mock_run(*a, **kw):
                return type("_R", (), {"returncode": 0, "stdout": "mocked\n", "stderr": ""})()

            subprocess.run = _mock_run

            updates = td._apply_insights(result, state)
        finally:
            subprocess.run = original_run

        # The dedup gate should have replaced the repetitive action with a rotating default,
        # which was then executed as a shell command (mocked subprocess.run).
        ds_after = updates.get("daemon_state", {})
        last_output = ds_after.get("last_action_output", "")
        assert isinstance(last_output, str), f"Expected string output, got: {type(last_output)}"
        # The output should contain our mocked 'exit=0: mocked' text
        assert "exit=0" in last_output or not last_output, f"Unexpected output: {last_output[:100]}"

    def test_failed_action_sets_prediction_feedback(self, evolve_env: Dict) -> None:
        """A shell action that fails must still close the prediction loop.

        The LLM needs expected-vs-actual feedback for FAILED actions too —
        otherwise failures silently skip the [PREDICTION] comparison and the
        next prompt never sees why its prediction was wrong (the exact
        moment calibration feedback matters most).
        """
        td = evolve_env["module"]
        from world_model import WorldModel

        wm = WorldModel()
        ds = {"tick_count": 5, "last_action_output": ""}

        result = {
            "action": {
                "type": "shell",
                "command": "python3 -c 'print(1)'",
                "description": "Run a harmless python one-liner",
                "expected_outcome": "exit=0: 1",
            },
            "fallback": False,
            "insight": "test",
            "focus_next": "continue",
            "confidence": 0.5,
            "reasoning": "test",
            "event_to_record": None,
            "outcome_to_record": None,
            "commitment": None,
            "prediction": None,
            "session_record": None,
            "plan_action": None,
            "new_plan": None,
            "new_goal": None,
            "goal_action": None,
            "search_query": None,
            "episodic_record": None,
            "self_model_update": {"weakness": None, "unknown": None, "new_commitment": None},
            "next_gap": None,
        }

        state = {
            "daemon_state": ds,
            "world_model": wm,
            "timeline": {"version": 1, "past": {"events": []}, "present": {}, "future": {}},
            "self_model": {
                "identity": {"name": "test", "role": "test"},
                "state": {},
                "capabilities": {"strengths": [], "weaknesses": [], "unknown_areas": []},
                "commitments": {},
            },
            "orientation": {"vision": "Test", "phase": "test"},
        }

        import subprocess
        original_run = subprocess.run
        try:
            def _boom(*a, **kw):
                raise RuntimeError("boom")

            subprocess.run = _boom
            td._apply_insights(result, state)
        finally:
            subprocess.run = original_run

        last_output = ds.get("last_action_output", "")
        assert "[PREDICTION" in last_output, (
            f"No prediction feedback written for failed action: {last_output!r}"
        )
        assert "FAILED: boom" in last_output, (
            f"Failure outcome missing from feedback: {last_output!r}"
        )
        assert "exit=0: 1" in last_output, (
            f"Expected-outcome text missing from feedback: {last_output!r}"
        )

        # The triple must be completed with the failure as the actual outcome
        completed = [t for t in wm.data.get("action_triples", []) if t.get("actual_outcome")]
        assert completed, "No completed action triple recorded"
        assert completed[-1]["actual_outcome"] == "FAILED: boom", completed[-1]

    def test_timed_out_action_sets_prediction_feedback(self, evolve_env: Dict) -> None:
        """A timed-out shell action must also produce prediction feedback."""
        td = evolve_env["module"]
        from world_model import WorldModel

        wm = WorldModel()
        ds = {"tick_count": 5, "last_action_output": ""}

        result = {
            "action": {
                "type": "shell",
                "command": "sleep 999",
                "description": "Sleep forever (will time out)",
                "expected_outcome": "exit=0: sleep completes",
            },
            "fallback": False,
            "insight": "test",
            "focus_next": "continue",
            "confidence": 0.5,
            "reasoning": "test",
            "event_to_record": None,
            "outcome_to_record": None,
            "commitment": None,
            "prediction": None,
            "session_record": None,
            "plan_action": None,
            "new_plan": None,
            "new_goal": None,
            "goal_action": None,
            "search_query": None,
            "episodic_record": None,
            "self_model_update": {"weakness": None, "unknown": None, "new_commitment": None},
            "next_gap": None,
        }

        state = {
            "daemon_state": ds,
            "world_model": wm,
            "timeline": {"version": 1, "past": {"events": []}, "present": {}, "future": {}},
            "self_model": {
                "identity": {"name": "test", "role": "test"},
                "state": {},
                "capabilities": {"strengths": [], "weaknesses": [], "unknown_areas": []},
                "commitments": {},
            },
            "orientation": {"vision": "Test", "phase": "test"},
        }

        import subprocess
        original_run = subprocess.run
        try:
            def _hang(*a, **kw):
                raise subprocess.TimeoutExpired(cmd="sleep 999", timeout=60)

            subprocess.run = _hang
            td._apply_insights(result, state)
        finally:
            subprocess.run = original_run

        last_output = ds.get("last_action_output", "")
        assert "[PREDICTION" in last_output, (
            f"No prediction feedback written for timed-out action: {last_output!r}"
        )
        assert "TIMEOUT: shell" in last_output, (
            f"Timeout outcome missing from feedback: {last_output!r}"
        )

    def test_unique_action_not_overridden(self, evolve_env: Dict) -> None:
        """New action types not matching recent triples should pass through."""
        td = evolve_env["module"]
        from world_model import WorldModel

        wm = WorldModel()
        # Populate with shell actions
        for i in range(2):
            tid = wm.record_action("shell", f"list workspace files {i}", "should list")
            wm.complete_action(tid, "exit=0: files")

        # Propose a git_commit action — different type, should NOT be dedup'd
        result = {
            "action": {
                "type": "git_commit",
                "message": "fix: test commit",
                "description": "Commit test changes to repository",
            },
            "fallback": False,
            "insight": "test",
            "focus_next": "continue",
            "confidence": 0.5,
            "reasoning": "test",
            "event_to_record": None,
            "outcome_to_record": None,
            "commitment": None,
            "prediction": None,
            "session_record": None,
            "plan_action": None,
            "new_plan": None,
            "new_goal": None,
            "goal_action": None,
            "search_query": None,
            "episodic_record": None,
            "self_model_update": {"weakness": None, "unknown": None, "new_commitment": None},
            "next_gap": None,
        }

        state = {
            "daemon_state": {"tick_count": 15, "last_action_output": ""},
            "world_model": wm,
            "timeline": {"version": 1, "past": {"events": []}, "present": {}, "future": {}},
            "self_model": {
                "identity": {"name": "test", "role": "test"},
                "state": {},
                "capabilities": {"strengths": [], "weaknesses": [], "unknown_areas": []},
                "commitments": {},
            },
            "orientation": {"vision": "Test", "phase": "test"},
        }

        import subprocess
        original_run = subprocess.run
        try:
            def _mock_run(*a, **kw):
                return type("_R", (), {"returncode": 0, "stdout": "mocked\n", "stderr": ""})()

            subprocess.run = _mock_run

            updates = td._apply_insights(result, state)
        finally:
            subprocess.run = original_run

        # git_commit executed (different type, not dedup'd by gate)
        ds_after = updates.get("daemon_state", {})
        last_output = ds_after.get("last_action_output", "")
        assert isinstance(last_output, str), f"Expected string output, got: {type(last_output)}"

    def test_resolved_subject_not_reappended(self, evolve_env: Dict) -> None:
        """_apply_insights refuses to re-add entries about documented subjects.

        The class-level half of evidence-based resolution: even when the
        LLM re-reports the think_daemon.py structure as an unknown /
        weakness / commitment, the append path must drop it.  The prune
        alone would only clean it up next cycle — but the self-model is
        fed back into the LLM prompt, so blocking at append time stops the
        fixation loop from re-seeding itself.
        """
        td = evolve_env["module"]
        from world_model import WorldModel

        wm = WorldModel()
        result = {
            "action": None,
            "fallback": False,
            "insight": "test",
            "focus_next": "continue",
            "confidence": 0.5,
            "reasoning": "test",
            "event_to_record": None,
            "outcome_to_record": None,
            "commitment": None,
            "prediction": None,
            "session_record": None,
            "plan_action": None,
            "new_plan": None,
            "new_goal": None,
            "goal_action": None,
            "search_query": None,
            "episodic_record": None,
            "self_model_update": {
                "weakness": "I keep noting LLM unavailability instead of forcing the pending think_daemon.py read to the top of every cycle.",
                "unknown": "Exact path, length, and core-loop structure of think_daemon.py.",
                "new_commitment": "Execute the think_daemon.py read in this immediate cycle and document the core loop within the next 2 cycles.",
            },
            "next_gap": None,
        }

        state = {
            "daemon_state": {"tick_count": 15, "last_action_output": ""},
            "world_model": wm,
            "timeline": {"version": 1, "past": {"events": []}, "present": {}, "future": {}},
            "self_model": {
                "identity": {"name": "test", "role": "test"},
                "state": {},
                "capabilities": {"strengths": [], "weaknesses": [], "unknown_areas": []},
                "commitments": {},
            },
            "orientation": {"vision": "Test", "phase": "test"},
        }

        import subprocess
        original_run = subprocess.run
        try:
            def _mock_run(*a, **kw):
                return type("_R", (), {"returncode": 0, "stdout": "mocked\n", "stderr": ""})()

            subprocess.run = _mock_run

            updates = td._apply_insights(result, state)
        finally:
            subprocess.run = original_run

        sm_out = updates["self_model"]
        assert sm_out["capabilities"]["weaknesses"] == []
        assert sm_out["capabilities"]["unknown_areas"] == []
        assert sm_out["commitments"].get("promised_features", []) == []

    def test_apply_insights_llm_weakness_survives_current_cycle_failure(
        self, evolve_env: Dict
    ) -> None:
        """Integration regression (2026-08-03, cycle 501): the main loop
        passes the CURRENT cycle's LLM outcome into _apply_insights, which
        forwards it to the post-cycle prune. A cycle whose LLM call failed
        must NOT have its 'LLM API unreliable' weakness removed — even when
        the daemon_state still carries the previous cycle's llm-backed
        (fallback=False) flag.

        This guards the wiring at the call site: if a refactor drops the
        ``llm_working`` argument (or the _apply_insights forwarding), this
        test fails by re-introducing the regression.
        """
        td = evolve_env["module"]
        from world_model import WorldModel

        wm = WorldModel()
        ds = {
            "tick_count": 501,
            # Previous cycle (500) was llm-backed — the stale heuristic alone
            # would conclude "LLM is working".
            "last_output": {"fallback": False, "insight": "c500",
                            "focus_next": "x", "confidence": 0.5,
                            "reasoning": "r"},
            "last_action_output": "",
        }
        result = {
            "action": None,
            "llm_fallback": True,  # CURRENT cycle LLM FAILED (2 timeouts)
            "insight": "LLM unavailable, local fallback",
            "focus_next": "continue",
            "confidence": 0.5,
            "reasoning": "local fallback",
            "event_to_record": None,
            "outcome_to_record": None,
            "commitment": None,
            "prediction": None,
            "session_record": None,
            "plan_action": None,
            "new_plan": None,
            "new_goal": None,
            "goal_action": None,
            "search_query": None,
            "episodic_record": None,
            "self_model_update": {"weakness": None, "unknown": None,
                                  "new_commitment": None},
            "next_gap": None,
        }
        state = {
            "daemon_state": ds,
            "world_model": wm,
            "timeline": {"version": 1, "past": {"events": []},
                         "present": {}, "future": {}},
            "self_model": {
                "identity": {"name": "test", "role": "test"},
                "state": {},
                "capabilities": {
                    "strengths": [],
                    "weaknesses": [
                        "LLM API unreliable: timeouts during thinking cycles",
                    ],
                    "unknown_areas": [],
                },
                "commitments": {},
            },
            "orientation": {"vision": "Test", "phase": "test"},
        }

        import subprocess
        original_run = subprocess.run
        try:
            def _mock_run(*a, **kw):
                return type("_R", (), {"returncode": 0, "stdout": "mocked\n",
                                       "stderr": ""})()

            subprocess.run = _mock_run
            # Mirror the main loop's call exactly (step 5).
            updates = td._apply_insights(
                result, state,
                llm_working=not bool(result.get("llm_fallback", False)),
            )
        finally:
            subprocess.run = original_run

        weaks = updates["self_model"]["capabilities"]["weaknesses"]
        assert any("LLM API unreliable" in w for w in weaks), (
            f"LLM weakness pruned despite current-cycle LLM failure: {weaks!r}"
        )

        # Companion: when the current cycle IS llm-backed, pruning still works.
        state2 = {
            "daemon_state": dict(ds),
            "world_model": wm,
            "timeline": {"version": 1, "past": {"events": []},
                         "present": {}, "future": {}},
            "self_model": {
                "identity": {"name": "test", "role": "test"},
                "state": {},
                "capabilities": {
                    "strengths": [],
                    "weaknesses": [
                        "LLM API unreliable: timeouts during thinking cycles",
                    ],
                    "unknown_areas": [],
                },
                "commitments": {},
            },
            "orientation": {"vision": "Test", "phase": "test"},
        }
        result2 = dict(result)
        result2["llm_fallback"] = False
        subprocess.run = lambda *a, **kw: type("_R", (), {
            "returncode": 0, "stdout": "mocked\n", "stderr": ""})()
        try:
            updates2 = td._apply_insights(
                result2, state2,
                llm_working=not bool(result2.get("llm_fallback", False)),
            )
        finally:
            subprocess.run = original_run
        weaks2 = updates2["self_model"]["capabilities"]["weaknesses"]
        assert not any("LLM API unreliable" in w for w in weaks2), (
            f"LLM weakness NOT pruned when current cycle LLM worked: {weaks2!r}"
        )


# ═══════════════════════════════════════════════════════════════════
#  Gap 8 slice 2: stale-guidance gate
# ═══════════════════════════════════════════════════════════════════


class TestStaleGuidanceGate:
    """_stale_guidance_skip_reason + the _apply_insights gate skip stale directives.

    The stale-guidance policy (stale_guidance_policy.py) is conservative:
    a directive is skipped ONLY on positive evidence (commit SHA in
    history, artifact verified, file clean). These tests pin the daemon
    wiring: the gate returns a reason for stale directives, None for
    everything else, and _apply_insights converts a stale action into a
    [STALE-SKIP] last_action_output instead of executing it.
    """

    def _make_result(self, action: Optional[dict]) -> dict:
        return {
            "action": action,
            "fallback": False,
            "insight": "test",
            "focus_next": "continue",
            "confidence": 0.5,
            "reasoning": "test",
            "event_to_record": None,
            "outcome_to_record": None,
            "commitment": None,
            "prediction": None,
            "session_record": None,
            "plan_action": None,
            "new_plan": None,
            "new_goal": None,
            "goal_action": None,
            "search_query": None,
            "episodic_record": None,
            "self_model_update": {
                "weakness": None,
                "unknown": None,
                "new_commitment": None,
            },
            "next_gap": None,
        }

    def _make_state(self) -> dict:
        from world_model import WorldModel

        wm = WorldModel()
        return {
            "daemon_state": {"tick_count": 15, "last_action_output": ""},
            "world_model": wm,
            "timeline": {"version": 1, "past": {"events": []}, "present": {}, "future": {}},
            "self_model": {
                "identity": {"name": "test", "role": "test"},
                "state": {},
                "capabilities": {"strengths": [], "weaknesses": [], "unknown_areas": []},
                "commitments": {},
            },
            "orientation": {"vision": "Test", "phase": "test"},
        }

    def test_no_references_returns_none(self, evolve_env: Dict) -> None:
        """A directive with no SHA/path/verify references can never be stale."""
        td = evolve_env["module"]
        assert (
            td._stale_guidance_skip_reason(
                {"type": "shell", "command": "ls /tmp", "description": "List temp files"}
            )
            is None
        )

    def test_unknown_sha_returns_none(self, evolve_env: Dict) -> None:
        """A SHA not in git history is unknown → execute (conservative)."""
        td = evolve_env["module"]
        import subprocess

        original_run = subprocess.run
        try:
            def _mock_run(*a, **kw):
                return type("_R", (), {
                    "returncode": 0,
                    "stdout": "d5ae20b8ac6e41c9a1a2867a7676c6bb8390f70b d5ae20b\n",
                    "stderr": "",
                })()
            subprocess.run = _mock_run
            reason = td._stale_guidance_skip_reason(
                {"type": "shell",
                 "description": "verify that 9999999999999999999999999999999999999999 landed"}
            )
        finally:
            subprocess.run = original_run
        assert reason is None

    def test_known_sha_returns_skip(self, evolve_env: Dict) -> None:
        """A SHA present in git history is positive stale evidence → skip."""
        td = evolve_env["module"]
        import subprocess

        original_run = subprocess.run
        try:
            def _mock_run(*a, **kw):
                return type("_R", (), {
                    "returncode": 0,
                    "stdout": "d5ae20b8ac6e41c9a1a2867a7676c6bb8390f70b d5ae20b\n",
                    "stderr": "",
                })()
            subprocess.run = _mock_run
            reason = td._stale_guidance_skip_reason(
                {"type": "shell",
                 "description": "verify that d5ae20b8ac6e41c9a1a2867a7676c6bb8390f70b landed"}
            )
        finally:
            subprocess.run = original_run
        assert reason and "already in history" in reason

    def test_gate_skips_stale_action(self, evolve_env: Dict) -> None:
        """_apply_insights converts a stale action into a [STALE-SKIP] output."""
        td = evolve_env["module"]
        import subprocess

        state = self._make_state()
        result = self._make_result({
            "type": "shell",
            "command": "git log --oneline",
            "description": "Verify that d5ae20b8ac6e41c9a1a2867a7676c6bb8390f70b is committed",
        })

        original_run = subprocess.run
        try:
            def _mock_run(*a, **kw):
                return type("_R", (), {
                    "returncode": 0,
                    "stdout": "d5ae20b8ac6e41c9a1a2867a7676c6bb8390f70b d5ae20b\n",
                    "stderr": "",
                })()
            subprocess.run = _mock_run
            updates = td._apply_insights(result, state)
        finally:
            subprocess.run = original_run

        last_output = state["daemon_state"].get("last_action_output", "")
        assert last_output.startswith("[STALE-SKIP]"), last_output

    def test_gate_executes_non_stale_action(self, evolve_env: Dict) -> None:
        """A non-stale action still executes (mocked subprocess, exit=0)."""
        td = evolve_env["module"]
        import subprocess

        state = self._make_state()
        result = self._make_result({
            "type": "shell",
            "command": "python3 -c 'print(1)'",
            "description": "Run a harmless python one-liner",
            "expected_outcome": "exit=0: 1",
        })

        original_run = subprocess.run
        try:
            def _mock_run(*a, **kw):
                return type("_R", (), {"returncode": 0, "stdout": "mocked\n", "stderr": ""})()
            subprocess.run = _mock_run
            updates = td._apply_insights(result, state)
        finally:
            subprocess.run = original_run

        last_output = state["daemon_state"].get("last_action_output", "")
        assert "exit=0" in last_output, last_output

    def test_gate_skips_commit_of_clean_file(self, evolve_env: Dict) -> None:
        """A commit directive naming a checked-clean file is a no-op → skip."""
        td = evolve_env["module"]
        import subprocess

        state = self._make_state()
        result = self._make_result({
            "type": "git_commit",
            "message": "commit docs/gap10-level2-plan.md",
            "description": "Commit the already-committed plan doc",
        })

        original_run = subprocess.run
        try:
            def _mock_run(*a, **kw):
                args = kw.get("args") or (a[0] if a else [])
                stdout = ""
                if args and args[:2] == ["git", "-C"] and "log" in args:
                    stdout = "d5ae20b8ac6e41c9a1a2867a7676c6bb8390f70b d5ae20b\n"
                # git status --porcelain → empty stdout = clean file
                return type("_R", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()
            subprocess.run = _mock_run
            updates = td._apply_insights(result, state)
        finally:
            subprocess.run = original_run

        last_output = state["daemon_state"].get("last_action_output", "")
        assert last_output.startswith("[STALE-SKIP]"), last_output


class TestCycleBodyEmptyLlmResponse:
    """An empty/whitespace-only LLM response is an outage, not a parse error.

    ``_call_llm`` can return ``""`` when the provider returns empty content
    (e.g. dict-style responses default to ``content=""``).  Previously an
    empty string fell through to ``_try_parse_json("")`` → ``parse_error``,
    which returned early and skipped every state update — the cycle was
    wasted even though the LLM was merely unavailable.  The cycle body must
    treat empty/whitespace-only responses exactly like ``raw is None``:
    local-analysis fallback, status ``ok``, consecutive-fallback tracking.
    """

    async def _run_cycle(self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch, raw: str) -> tuple:
        td = evolve_env["module"]

        async def _fake_llm(messages: list, task: str = "thinking") -> str:
            return raw

        monkeypatch.setattr(td, "_call_llm", _fake_llm)

        result = {"status": "ok", "tick_duration": 0, "insight": None, "error": None}
        ds = td.load_daemon_state()
        ds.setdefault(
            "cycle_stats",
            {"total": 0, "ok": 0, "error": 0, "parse_error": 0,
             "avg_duration": 0.0, "max_duration": 0.0},
        )

        # Local-analysis fallback still executes a state-checking action via
        # subprocess; mock it (same pattern as the action-dedup tests above)
        # so the test is hermetic and deterministic.
        import subprocess

        original_run = subprocess.run

        def _mock_run(*a, **kw):
            return type("_R", (), {"returncode": 0, "stdout": "mocked\n", "stderr": ""})()

        try:
            subprocess.run = _mock_run
            result = await td._run_cycle_body(result, ds)
        finally:
            subprocess.run = original_run

        return result, ds

    def test_empty_response_falls_back_to_local_analysis(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result, ds = asyncio.run(self._run_cycle(evolve_env, monkeypatch, ""))

        assert result["status"] == "ok", (
            f"Expected ok (fallback), got {result['status']}: {result.get('error')}"
        )
        assert result["llm_fallback"] is True
        assert ds["consecutive_fallback_cycles"] == 1, (
            "Empty response must count as a consecutive fallback cycle"
        )
        assert ds["tick_count"] == 1, "Cycle must advance tick_count (not return early)"
        assert ds["status"] == "ok"
        assert result.get("insight"), "Local-analysis insight should be produced"

    def test_whitespace_only_response_falls_back(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result, ds = asyncio.run(self._run_cycle(evolve_env, monkeypatch, "   \n\t  "))

        assert result["status"] == "ok", (
            f"Expected ok (fallback), got {result['status']}: {result.get('error')}"
        )
        assert result["llm_fallback"] is True
        assert ds["consecutive_fallback_cycles"] == 1
        assert ds["tick_count"] == 1

    def test_non_json_text_still_parse_error(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-empty, non-JSON text is a genuine parse error — not a fallback."""
        result, ds = asyncio.run(
            self._run_cycle(evolve_env, monkeypatch, "I am sorry, I cannot do that.")
        )

        assert result["status"] == "parse_error"
        assert result.get("llm_fallback") is not True
        assert "Could not parse JSON" in (result.get("error") or "")


class TestLlmRetryPolicy:
    """Adaptive LLM retry budget (outage-aware retry policy).

    ``_llm_retry_policy`` shrinks the retry budget as
    ``consecutive_fallback_cycles`` grows, so outage cycles stop burning
    ~60s per dead-end retry and keep their time for local analysis +
    action execution. The healthy tier must stay identical to the
    pre-adaptive behavior (2 × 90s).
    """

    def test_healthy_keeps_full_budget(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        assert td._llm_retry_policy(0) == (2, 90.0)
        # Negative fallback count (corrupt state) must not escalate the budget
        assert td._llm_retry_policy(-1) == (2, 90.0)

    def test_warm_outage_single_retry(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        assert td._llm_retry_policy(1) == (1, 60.0)

    def test_deep_outage_single_probe(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        # Depth 2: still probe once with a tight 45s cap
        assert td._llm_retry_policy(2) == (1, 45.0)

    def test_extended_outage_skips_probe(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        # Depth >= 3 on an odd (skip) cycle: 0 attempts — that cycle's
        # budget goes to local analysis + action execution instead of a
        # doomed 45s probe. Odd counters alternate with even (probe)
        # counters so recovery is detected within 2 cycles.
        assert td._llm_retry_policy(3) == (0, 0.0)
        assert td._llm_retry_policy(5) == (0, 0.0)
        assert td._llm_retry_policy(7) == (0, 0.0)
        assert td._llm_retry_policy(9) == (0, 0.0)
        assert td._llm_retry_policy(11) == (0, 0.0)

    def test_extended_outage_probe_cycle(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        # Every 2nd cycle (depth % 2 == 0): a bounded 90s probe so
        # recovery is still detected within 2 cycles of the provider
        # coming back. 90s (not 30s) so the auxiliary client's internal
        # transport timeout + one retry fit inside the cap; healthy
        # opencode-go latencies have been observed up to 31s. Every-2nd
        # (not every-4th) since 2026-08-01: the endpoint was observed
        # intermittent on a ~15 min period, so the 4-cycle cadence left
        # the daemon blind for up to 60 min during up-windows.
        assert td._llm_retry_policy(4) == (1, 90.0)
        assert td._llm_retry_policy(6) == (1, 90.0)
        assert td._llm_retry_policy(8) == (1, 90.0)
        assert td._llm_retry_policy(10) == (1, 90.0)
        assert td._llm_retry_policy(12) == (1, 90.0)

    def test_setter_applies_policy_to_globals(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        # Fresh import must start at the full budget
        assert td._llm_max_retries == 2
        assert td._llm_attempt_timeout == 90.0

        applied = td._set_llm_retry_policy(3)
        assert applied == (0, 0.0)
        assert td._llm_max_retries == 0
        assert td._llm_attempt_timeout == 0.0

        # Returning to health restores the full budget
        applied = td._set_llm_retry_policy(0)
        assert applied == (2, 90.0)
        assert td._llm_max_retries == 2
        assert td._llm_attempt_timeout == 90.0

    def test_tier_names_match_policy_budgets(self, evolve_env: Dict) -> None:
        """Invariant: tier name and applied budget can never drift.

        ``_llm_retry_policy`` derives its budget from ``_LLM_RETRY_BUDGETS``
        via ``_llm_retry_tier``, so for every outage depth the tier name
        must map back to exactly the budget the policy applies. This pins
        the single-source-of-truth relationship (P3 telemetry), not a
        value snapshot.
        """
        td = evolve_env["module"]
        for n in range(-1, 14):
            tier = td._llm_retry_tier(n)
            assert tier in td._LLM_RETRY_BUDGETS, f"unknown tier {tier!r} for n={n}"
            assert td._LLM_RETRY_BUDGETS[tier] == td._llm_retry_policy(n), (
                f"tier {tier} budget mismatch for n={n}"
            )

    def test_setter_records_policy_tier(self, evolve_env: Dict) -> None:
        """_set_llm_retry_policy records the chosen tier for telemetry."""
        td = evolve_env["module"]
        assert td._llm_policy_tier == "healthy"  # fresh import default
        td._set_llm_retry_policy(1)
        assert td._llm_policy_tier == "warm"
        td._set_llm_retry_policy(2)
        assert td._llm_policy_tier == "deep"
        td._set_llm_retry_policy(3)
        assert td._llm_policy_tier == "extended_skip"
        td._set_llm_retry_policy(4)
        assert td._llm_policy_tier == "extended_probe"
        td._set_llm_retry_policy(0)
        assert td._llm_policy_tier == "healthy"

    def test_call_llm_records_tier_and_backoff_in_stats(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Retry-path telemetry: tier, attempt count, and backoff time are
        recorded into ``_last_llm_call_stats`` (which the cycle body persists
        as the world-model ``llm_call`` triple's parameters)."""
        td = evolve_env["module"]
        td._set_llm_retry_policy(0)  # healthy: 2 retries × 90s
        assert td._llm_policy_tier == "healthy"

        monkeypatch.setattr(td, "_ensure_runtime_main", lambda: None)

        import agent.auxiliary_client as _aux

        calls = {"n": 0}

        async def _flakey(**kwargs: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                raise asyncio.TimeoutError("simulated first-attempt timeout")
            return type("_R", (), {"choices": [type("_C", (), {
                "message": type("_M", (), {"content": '{"insight": "ok"}'}),
            })()]})()
        monkeypatch.setattr(_aux, "async_call_llm", _flakey)

        # Neutralize the real 2s backoff sleep — record it instead.
        slept = {"total": 0.0}

        async def _fast_sleep(seconds: float) -> None:
            slept["total"] += seconds
        monkeypatch.setattr(asyncio, "sleep", _fast_sleep)

        async def _run() -> Optional[str]:
            return await td._call_llm([{"role": "user", "content": "x"}])

        result = asyncio.run(_run())

        assert result is not None
        stats = dict(td._last_llm_call_stats)
        assert stats.get("policy_tier") == "healthy"
        assert stats.get("success") is True
        assert stats.get("attempts_used") == 2
        assert stats.get("max_retries") == 2
        assert stats.get("backoff_slept_s") == 2.0  # 2**1 between attempts
        assert slept["total"] == 2.0
        assert stats.get("_fresh") is True

    def test_call_llm_records_per_attempt_telemetry(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """P3: per-attempt retry values (attempt, wait, reason) are recorded
        into ``_last_llm_call_stats[\"attempts\"]`` so the world-model
        ``llm_call`` triple carries observable attempt-by-attempt policy
        behavior, not just aggregates."""
        td = evolve_env["module"]
        td._set_llm_retry_policy(0)  # healthy: 2 retries × 90s
        monkeypatch.setattr(td, "_ensure_runtime_main", lambda: None)

        import agent.auxiliary_client as _aux

        calls = {"n": 0}

        async def _flakey(**kwargs: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                raise asyncio.TimeoutError("simulated first-attempt timeout")
            return type("_R", (), {"choices": [type("_C", (), {
                "message": type("_M", (), {"content": '{"insight": "ok"}'}),
            })()]})()
        monkeypatch.setattr(_aux, "async_call_llm", _flakey)

        async def _noop_sleep(seconds: float) -> None:
            return None
        monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

        async def _run() -> Optional[str]:
            return await td._call_llm([{"role": "user", "content": "x"}])

        result = asyncio.run(_run())
        assert result is not None
        attempts = list(td._last_llm_call_stats.get("attempts", []))
        assert len(attempts) == 2, f"expected 2 per-attempt entries, got {attempts}"
        # Attempt 1: failed with a timeout, waited nothing before it.
        assert attempts[0]["attempt"] == 1
        assert attempts[0]["outcome"] == "timeout"
        assert attempts[0]["wait_s"] == 0.0
        assert "timeout" in attempts[0]["reason"]
        # Attempt 2: succeeded after the 2s backoff (2**1).
        assert attempts[1]["attempt"] == 2
        assert attempts[1]["outcome"] == "success"
        assert attempts[1]["wait_s"] == 2.0
        assert attempts[1]["reason"] is None
        # Aggregates remain consistent with the per-attempt log.
        assert td._last_llm_call_stats.get("attempts_used") == 2
        assert td._last_llm_call_stats.get("backoff_slept_s") == 2.0

    def test_call_llm_records_per_attempt_log_on_total_failure(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """P3: every failed attempt is present in the per-attempt log even
        when the call exhausts its retry budget (the terminal path)."""
        td = evolve_env["module"]
        td._set_llm_retry_policy(0)  # healthy: 2 retries × 90s
        monkeypatch.setattr(td, "_ensure_runtime_main", lambda: None)

        import agent.auxiliary_client as _aux

        async def _always_fail(**kwargs: Any) -> Any:
            raise RuntimeError("endpoint down")
        monkeypatch.setattr(_aux, "async_call_llm", _always_fail)

        async def _noop_sleep(seconds: float) -> None:
            return None
        monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

        async def _run() -> Optional[str]:
            return await td._call_llm([{"role": "user", "content": "x"}])

        result = asyncio.run(_run())
        assert result is None
        attempts = list(td._last_llm_call_stats.get("attempts", []))
        assert len(attempts) == 2
        assert [a["outcome"] for a in attempts] == ["error", "error"]
        assert [a["wait_s"] for a in attempts] == [0.0, 2.0]
        assert all("endpoint down" in a["reason"] for a in attempts)
        assert td._last_llm_call_stats.get("last_error") == "endpoint down"

    def test_call_llm_skip_records_tier_and_zero_backoff(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A skip-cycle _call_llm records tier + zero backoff in stats."""
        td = evolve_env["module"]
        td._set_llm_retry_policy(3)  # extended_skip: 0 attempts
        assert td._llm_policy_tier == "extended_skip"
        monkeypatch.setattr(td, "_ensure_runtime_main", lambda: None)

        import agent.auxiliary_client as _aux

        async def _boom(*a: Any, **k: Any) -> None:
            raise AssertionError("async_call_llm must not be called on a skip cycle")
        monkeypatch.setattr(_aux, "async_call_llm", _boom)

        async def _run() -> Optional[str]:
            return await td._call_llm([{"role": "user", "content": "x"}])

        result = asyncio.run(_run())
        assert result is None
        stats = dict(td._last_llm_call_stats)
        assert stats.get("skipped") is True
        assert stats.get("policy_tier") == "extended_skip"
        assert stats.get("backoff_slept_s") == 0.0
        assert stats.get("attempts_used") == 0

    def test_call_llm_skips_probe_when_policy_is_zero(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With a 0-attempt policy, _call_llm returns None without ever
        invoking the auxiliary client."""
        td = evolve_env["module"]
        td._set_llm_retry_policy(3)
        assert td._llm_max_retries == 0

        monkeypatch.setattr(td, "_ensure_runtime_main", lambda: None)

        called = {"n": 0}
        try:
            import agent.auxiliary_client as _aux
        except ImportError:
            _aux = None  # import path unavailable — guard returns None anyway
        if _aux is not None:
            async def _boom(*a: Any, **k: Any) -> None:
                called["n"] += 1
                raise AssertionError(
                    "async_call_llm must not be called with a 0-attempt policy"
                )
            monkeypatch.setattr(_aux, "async_call_llm", _boom)

        async def _run() -> Optional[str]:
            return await td._call_llm([{"role": "user", "content": "x"}])

        result = asyncio.run(_run())
        assert result is None
        assert called["n"] == 0

    def test_call_llm_passes_timeout_into_auxiliary_client(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``_call_llm`` must forward its per-attempt timeout INTO
        ``async_call_llm``, not just wrap it in an outer ``wait_for``.

        Regression guard for the 2026-08-01 outage-blindness fix: the
        auxiliary client applies its internal ``_DEFAULT_AUX_TIMEOUT`` (30 s)
        to the HTTP request when no ``timeout=`` is passed, cutting off
        healthy-but-slow completions (opencode-go observed up to 31 s)
        before the daemon's outer 90 s cap can help. The daemon then
        records a fallback even though the endpoint was merely slow.
        """
        td = evolve_env["module"]
        td._set_llm_retry_policy(4)  # extended-outage probe: (1, 90.0)
        assert td._llm_max_retries == 1
        assert td._llm_attempt_timeout == 90.0

        monkeypatch.setattr(td, "_ensure_runtime_main", lambda: None)

        captured = {}

        import agent.auxiliary_client as _aux

        async def _fake_async_call_llm(**kwargs: Any) -> Any:
            captured["timeout"] = kwargs.get("timeout")
            captured["task"] = kwargs.get("task")
            captured["provider"] = kwargs.get("provider")
            captured["model"] = kwargs.get("model")
            return type("_R", (), {"choices": [type("_C", (), {
                "message": type("_M", (), {"content": '{"insight": "ok"}'}),
            })()]})()

        monkeypatch.setattr(_aux, "async_call_llm", _fake_async_call_llm)

        async def _run() -> Optional[str]:
            return await td._call_llm([{"role": "user", "content": "x"}])

        result = asyncio.run(_run())

        assert captured.get("timeout") == 90.0, (
            "async_call_llm must receive the daemon's per-attempt timeout; "
            f"got {captured.get('timeout')!r}"
        )
        assert captured.get("task") == "thinking"
        assert result is not None

    def test_call_llm_records_unexpected_response_shape_in_stats(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An anomalous auxiliary-client response must be recorded accurately.

        Regression guard (2026-08-01): the unexpected-response-shape path used
        to ``return None`` without touching ``_last_llm_call_stats``, so the
        cycle body persisted a world-model ``llm_call`` triple claiming
        ``failed: unknown`` with ``attempts_used: 0`` — even though the call
        really consumed attempt(s) and hit a distinct failure class. That
        mislabeled calibration data for the adaptive retry policy's
        predict→observe loop.
        """
        td = evolve_env["module"]
        td._set_llm_retry_policy(0)  # full budget: 2 retries × 90s
        assert td._llm_max_retries == 2

        monkeypatch.setattr(td, "_ensure_runtime_main", lambda: None)

        import agent.auxiliary_client as _aux

        async def _weird_response(**kwargs: Any) -> Any:
            return {"no": "choices", "here": True}  # dict WITHOUT choices

        monkeypatch.setattr(_aux, "async_call_llm", _weird_response)

        async def _run() -> Optional[str]:
            return await td._call_llm([{"role": "user", "content": "x"}])

        result = asyncio.run(_run())

        assert result is None
        stats = dict(td._last_llm_call_stats)
        assert stats.get("_fresh") is True, "real call must set the freshness marker"
        assert stats.get("success") is False
        assert stats.get("attempts_used") == 1, (
            "the anomalous attempt must be counted, got "
            f"{stats.get('attempts_used')}"
        )
        assert stats.get("last_error") == "unexpected_response_shape", (
            "the distinct failure class must be recorded, got "
            f"{stats.get('last_error')!r}"
        )
        assert stats.get("duration_s", 0.0) >= 0.0


class TestBudgetMetrics:
    """``_budget_metrics``: budget-consumed telemetry for the llm_call triple.

    P3 completion: the retry/budget telemetry already records wall time
    (``duration_s``) but never expressed it relative to the cycle budget in
    force (``_cycle_budget_seconds``, set for ``--once`` cron runs) — the
    exact cap ``_clamp_retry_budget`` optimizes against. These tests pin the
    pure mapping: fraction = duration / budget, capped at 1.0, None when no
    budget is set or the duration is unavailable.
    """

    def test_budget_metrics_no_budget(self, evolve_env: Dict) -> None:
        """No cycle budget (persistent daemon mode) → fraction None."""
        td = evolve_env["module"]
        td._cycle_budget_seconds = None
        assert td._budget_metrics(40.0) == {
            "cycle_budget_s": None,
            "budget_fraction": None,
        }

    def test_budget_metrics_with_budget(self, evolve_env: Dict) -> None:
        """Budget set → fraction is duration / budget (rounded to 4dp)."""
        td = evolve_env["module"]
        td._cycle_budget_seconds = 200.0
        assert td._budget_metrics(40.0) == {
            "cycle_budget_s": 200.0,
            "budget_fraction": 0.2,
        }
        assert td._budget_metrics(1.5)["budget_fraction"] == round(1.5 / 200.0, 4)

    def test_budget_metrics_caps_at_one(self, evolve_env: Dict) -> None:
        """Over-budget LLM phase still records fraction 1.0, not >1."""
        td = evolve_env["module"]
        td._cycle_budget_seconds = 200.0
        assert td._budget_metrics(500.0)["budget_fraction"] == 1.0

    def test_budget_metrics_nonpositive_duration(self, evolve_env: Dict) -> None:
        """Zero/unknown duration (e.g. skip path) → fraction None."""
        td = evolve_env["module"]
        td._cycle_budget_seconds = 200.0
        assert td._budget_metrics(0.0)["budget_fraction"] is None
        assert td._budget_metrics(-1.0)["budget_fraction"] is None


class TestCoerceLlmResponseFields:
    """Type-guard for LLM response fields (``_coerce_llm_response_fields``).

    The LLM emits any response field as a non-string/non-numeric type
    (same bug class as the 2026-08-01 ``'int' object has no attribute
    'strip'`` crash on gap_reference). Without the coercion, an
    int/dict ``insight`` crashes the ``--once`` display path at
    ``r.get("insight","")[:80]`` and a non-numeric
    ``prediction.confidence`` raises TypeError inside
    ``adjust_confidence``'s range comparison — both AFTER the expensive
    LLM call, wasting the whole cycle.
    """

    def test_non_string_insight_is_coerced(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        parsed: Dict[str, Any] = {"insight": 42, "focus_next": {"nested": True}}
        td._coerce_llm_response_fields(parsed)
        assert parsed["insight"] == "42"
        assert isinstance(parsed["focus_next"], str)
        # The coerced value must survive the --once display slice
        assert parsed["insight"][:80] == "42"

    def test_non_numeric_prediction_confidence_is_coerced(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        parsed: Dict[str, Any] = {
            "prediction": {"text": ["a", "list"], "confidence": "high"},
        }
        td._coerce_llm_response_fields(parsed)
        assert parsed["prediction"]["text"] == "['a', 'list']"
        assert parsed["prediction"]["confidence"] == 0.5

    def test_numeric_string_confidence_parses(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        parsed: Dict[str, Any] = {"prediction": {"confidence": "0.8"}}
        td._coerce_llm_response_fields(parsed)
        assert parsed["prediction"]["confidence"] == 0.8

    def test_top_level_confidence_defaults_to_zero(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        parsed: Dict[str, Any] = {"confidence": "n/a"}
        td._coerce_llm_response_fields(parsed)
        assert parsed["confidence"] == 0

    def test_none_and_valid_values_are_preserved(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        parsed: Dict[str, Any] = {
            "insight": "real insight",
            "next_gap": None,
            "reasoning": None,
            "confidence": 0.7,
            "prediction": {"text": "valid", "confidence": 0.3},
        }
        td._coerce_llm_response_fields(parsed)
        assert parsed["insight"] == "real insight"
        assert parsed["next_gap"] is None
        assert parsed["reasoning"] is None
        assert parsed["confidence"] == 0.7
        assert parsed["prediction"]["confidence"] == 0.3

    def test_event_record_inner_fields_coerced(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        parsed: Dict[str, Any] = {
            "event_to_record": {"summary": 42, "type": {"nested": True}, "impact": None},
        }
        td._coerce_llm_response_fields(parsed)
        ev = parsed["event_to_record"]
        assert ev["summary"] == "42"
        assert ev["type"] == "{'nested': True}"
        assert ev["impact"] is None  # None preserved

    def test_outcome_record_inner_fields_coerced(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        parsed: Dict[str, Any] = {
            "outcome_to_record": {"summary": ["done", "well"], "impact": 7, "event_id": {"x": 1}},
        }
        td._coerce_llm_response_fields(parsed)
        oc = parsed["outcome_to_record"]
        assert oc["summary"] == "['done', 'well']"
        assert oc["impact"] == "7"
        assert isinstance(oc["event_id"], str)

    def test_commitment_inner_fields_coerced(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        parsed: Dict[str, Any] = {
            "commitment": {"what": 42, "deadline": {"d": "2026-08-01"}},
        }
        td._coerce_llm_response_fields(parsed)
        cm = parsed["commitment"]
        assert cm["what"] == "42"
        assert cm["deadline"] == "{'d': '2026-08-01'}"

    def test_session_record_focus_and_outcomes_list_coerced(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        parsed: Dict[str, Any] = {
            "session_record": {"focus": {"f": 1}, "outcomes_list": "single outcome"},
        }
        td._coerce_llm_response_fields(parsed)
        ss = parsed["session_record"]
        assert ss["focus"] == "{'f': 1}"
        assert ss["outcomes_list"] == ["single outcome"]

    def test_session_record_empty_non_list_outcomes_degrade_to_list(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        parsed: Dict[str, Any] = {
            "session_record": {"focus": "ok", "outcomes_list": {}},
        }
        td._coerce_llm_response_fields(parsed)
        assert parsed["session_record"]["outcomes_list"] == []

    def test_apply_insights_persists_coerced_commitment(self, evolve_env: Dict) -> None:
        """The full pipeline: an int ``commitment.what`` from the LLM must land
        in the timeline as a string, so the next cycle's prompt builder
        (``"; ".join(c["what"] ...)``) cannot raise TypeError."""
        td = evolve_env["module"]
        parsed: Dict[str, Any] = {"commitment": {"what": 42}, "action": None}
        td._coerce_llm_response_fields(parsed)
        state = {
            "timeline": {"version": 1, "past": {}, "present": {}, "future": {}},
            "self_model": {"version": 1, "identity": {}, "state": {},
                           "capabilities": {}, "commitments": {}},
            "orientation": None,
            "daemon_state": {"tick_count": 1, "last_action_output": ""},
        }
        updates = td._apply_insights(parsed, state)
        commits = updates["timeline"]["present"]["commitments"]
        assert len(commits) == 1
        assert commits[0]["what"] == "42"
        assert isinstance(commits[0]["what"], str)

    def test_prompt_builder_skips_non_string_commitment_what(self, evolve_env: Dict) -> None:
        """Legacy non-string commitment rows (predating the record-block
        guard) must not crash the prompt builder — they are filtered out
        while valid string commitments still appear."""
        td = evolve_env["module"]
        state = {
            "daemon_state": {"status": "initialized", "tick_count": 0},
            "timeline": {
                "version": 1, "past": {}, "future": {},
                "present": {"commitments": [
                    {"what": 42, "status": "active"},
                    {"what": "real commitment", "status": "active"},
                ]},
            },
            "self_model": {"version": 1, "identity": {}, "state": {},
                           "capabilities": {}, "commitments": {}},
            "orientation": None,
            "world_model": None,
        }
        prompt = td._build_thinking_prompt(state)  # must not raise TypeError
        assert "real commitment" in prompt

    def test_new_goal_gap_reference_int_coerced(self, evolve_env: Dict) -> None:
        """Exact 2026-08-01 06:17 crash reproduction: an int
        ``new_goal.gap_reference`` must be coerced to a string so
        ``propose_goal`` → ``_find_similar_active_goal`` can never hit
        ``'int' object has no attribute 'strip'`` again."""
        td = evolve_env["module"]
        parsed: Dict[str, Any] = {
            "new_goal": {
                "title": "Finish Phase 2", "description": "Close all gaps",
                "gap_reference": 6,  # the crash value (emitted as int)
            },
        }
        td._coerce_llm_response_fields(parsed)
        ng = parsed["new_goal"]
        assert ng["gap_reference"] == "6"
        assert isinstance(ng["gap_reference"], str)

    def test_new_goal_all_string_fields_coerced(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        parsed: Dict[str, Any] = {
            "new_goal": {
                "title": 123,
                "description": {"nested": True},
                "rationale": ["list"],
                "gap_reference": 8,
                "verification_criteria": 0,
            },
        }
        td._coerce_llm_response_fields(parsed)
        ng = parsed["new_goal"]
        assert ng["title"] == "123"
        assert ng["description"] == "{'nested': True}"
        assert ng["rationale"] == "['list']"
        assert ng["gap_reference"] == "8"
        assert ng["verification_criteria"] == "0"

    def test_new_goal_priority_coerced(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        parsed: Dict[str, Any] = {
            "new_goal": {"title": "t", "description": "d", "priority": "2"},
        }
        td._coerce_llm_response_fields(parsed)
        assert parsed["new_goal"]["priority"] == 2
        parsed2: Dict[str, Any] = {
            "new_goal": {"title": "t", "description": "d", "priority": "high"},
        }
        td._coerce_llm_response_fields(parsed2)
        assert parsed2["new_goal"]["priority"] == 3  # schema default

    def test_goal_action_fields_coerced(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        parsed: Dict[str, Any] = {
            "goal_action": {
                "goal_id": 123, "new_status": ["active"], "note": {"n": 1},
            },
        }
        td._coerce_llm_response_fields(parsed)
        ga = parsed["goal_action"]
        assert ga["goal_id"] == "123"
        assert ga["new_status"] == "['active']"
        assert ga["note"] == "{'n': 1}"

    def test_plan_action_fields_coerced(self, evolve_env: Dict) -> None:
        """P4: ``plan_action`` is the same LLM-JSON surface as goal_action
        but was missing from the coercion list — an int ``step_id`` would
        silently never match a string step id in update_plan_step (dead
        update) and a dict ``note`` would persist garbage into the plan
        step. All string fields must coerce like goal_action."""
        td = evolve_env["module"]
        parsed: Dict[str, Any] = {
            "plan_action": {
                "step_id": 2, "new_status": ["complete"], "note": {"n": 1},
            },
        }
        td._coerce_llm_response_fields(parsed)
        pa = parsed["plan_action"]
        assert pa["step_id"] == "2"
        assert pa["new_status"] == "['complete']"
        assert pa["note"] == "{'n': 1}"

    def test_new_plan_fields_coerced(self, evolve_env: Dict) -> None:
        """P4: ``new_plan`` was unguarded — an int ``goal`` would persist a
        garbage plan title via create_plan and an int ``steps`` CRASHES the
        step comprehension in _apply_insights (``for s in np["steps"]``),
        which runs outside any try/except after the LLM call, killing the
        whole cycle. All string fields coerce; ``steps`` degrades to []."""
        td = evolve_env["module"]
        parsed: Dict[str, Any] = {
            "new_plan": {
                "goal": 42, "verification": {"v": 1}, "steps": 5,
            },
        }
        td._coerce_llm_response_fields(parsed)
        np_ = parsed["new_plan"]
        assert np_["goal"] == "42"
        assert np_["verification"] == "{'v': 1}"
        assert np_["steps"] == []

    def test_new_plan_valid_steps_list_preserved(self, evolve_env: Dict) -> None:
        """A legitimate list of step dicts must pass through untouched."""
        td = evolve_env["module"]
        steps = [{"description": "Step A", "verification": "V"}]
        parsed: Dict[str, Any] = {"new_plan": {"goal": "g", "steps": steps}}
        td._coerce_llm_response_fields(parsed)
        assert parsed["new_plan"]["steps"] is steps

    def test_action_dict_api_call_fields_coerced(self, evolve_env: Dict) -> None:
        """P4: the action-dict guard in _apply_insights must coerce api_call
        fields too — a dict ``endpoint`` would otherwise flow raw into
        _action_params_from_act → world-model parameters, polluting
        calibration features with unhashable garbage. After coercion the
        dict endpoint becomes a string that fails the http(s) allowlist
        check, so the action is cleanly BLOCKED with no network access."""
        td = evolve_env["module"]
        act: Dict[str, Any] = {
            "type": "api_call", "endpoint": {"url": "x"},
            "method": ["GET"], "description": 1,
            "expected_outcome": "exit=0: response",
        }
        parsed: Dict[str, Any] = {"action": act}
        state = {
            "timeline": {"version": 1, "past": {}, "present": {}, "future": {}},
            "self_model": {"version": 1, "identity": {}, "state": {},
                           "capabilities": {}, "commitments": {}},
            "orientation": None,
            "daemon_state": {"tick_count": 1, "last_action_output": ""},
            "world_model": None,
        }
        updates = td._apply_insights(parsed, state)  # must not raise
        assert updates is not None
        # The guard coerced the api_call fields in place on the action dict
        assert isinstance(act["endpoint"], str)
        assert isinstance(act["method"], str)
        assert isinstance(act["description"], str)
        # Coerced dict endpoint fails the http(s) pre-flight → clean BLOCKED,
        # no urllib call attempted (deny-by-default preserved)
        assert "BLOCKED" in state["daemon_state"].get("last_action_output", "")

    def test_episodic_record_fields_coerced(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        parsed: Dict[str, Any] = {
            "episodic_record": {
                "mtype": 42, "summary": {"s": 1}, "details": None,
                "salience": "high",
            },
        }
        td._coerce_llm_response_fields(parsed)
        er = parsed["episodic_record"]
        assert er["mtype"] == "42"
        assert er["summary"] == "{'s': 1}"
        assert er["details"] is None  # None preserved
        assert er["salience"] == 0.5  # unconvertible → schema default

    def test_semantic_record_fields_coerced(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        parsed: Dict[str, Any] = {
            "semantic_record": {
                "topic": 7, "fact": ["x"], "source": {"src": 1},
                "confidence": "n/a",
            },
        }
        td._coerce_llm_response_fields(parsed)
        sr = parsed["semantic_record"]
        assert sr["topic"] == "7"
        assert sr["fact"] == "['x']"
        assert sr["source"] == "{'src': 1}"
        assert sr["confidence"] == 0.7  # unconvertible → schema default

    def test_procedural_record_fields_coerced(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        parsed: Dict[str, Any] = {
            "procedural_record": {
                "pattern": 1, "trigger": {"t": 2}, "procedure": ["p"],
            },
        }
        td._coerce_llm_response_fields(parsed)
        pr = parsed["procedural_record"]
        assert pr["pattern"] == "1"
        assert pr["trigger"] == "{'t': 2}"
        assert pr["procedure"] == "['p']"

    def test_apply_insights_goal_and_memory_blocks_no_crash(self, evolve_env: Dict) -> None:
        """Full pipeline: an int ``new_goal.gap_reference`` AND an int
        ``episodic_record.summary`` from the LLM must land in the goal and
        memory stores as strings (the 06:17 crash class) instead of
        killing the cycle or polluting stores with non-string rows."""
        td = evolve_env["module"]
        parsed: Dict[str, Any] = {
            "new_goal": {
                "title": "Type-guard test goal", "description": "Verify coercion",
                "gap_reference": 6,
            },
            "episodic_record": {"summary": 42, "mtype": "observation"},
            "action": None,
        }
        td._coerce_llm_response_fields(parsed)
        state = {
            "timeline": {"version": 1, "past": {}, "present": {}, "future": {}},
            "self_model": {"version": 1, "identity": {}, "state": {},
                           "capabilities": {}, "commitments": {}},
            "orientation": None,
            "daemon_state": {"tick_count": 1, "last_action_output": ""},
            "world_model": None,
        }
        updates = td._apply_insights(parsed, state)  # must not raise
        # Episodic memory persisted with the coerced string summary
        import json as _json
        mem_path = evolve_env["paths"]["evolve_dir"].parent / "memory.json"
        if mem_path.exists():
            mem = _json.loads(mem_path.read_text(encoding="utf-8"))
            for e in mem.get("episodic", []):
                if e.get("summary") is not None:
                    assert isinstance(e["summary"], str), e
        # Timeline unchanged (no event/outcome/commitment in parsed)
        assert "timeline" in updates


class TestCoerceLlmResponseFieldsFuzz:
    """P4 verification criterion: the fuzz test.

    The P4 goal (``Crash-proof daemon outcome processing``) lists its
    verification criteria as: *"Run 50+ cycles or a fuzz test feeding
    int/None/dict values into outcome fields; zero crashes and each
    non-string value is coerced or safely rejected."* The targeted tests
    above prove individual fields with hand-picked values; this class
    closes the other half of the criterion by feeding every guarded
    field (top-level, prediction, event/outcome/commitment/session
    records, goal/memory/plan blocks) every poison class (int, dict,
    list, bool, float, None) and asserting the coercion invariants hold
    for all of them — zero exceptions, strings coerced, numerics
    coerced-or-defaulted, lists guaranteed, ``None`` preserved.
    """

    # Mirror of the guarded tables in _coerce_llm_response_fields:
    # (block, (field, ...)) where block == "" means top-level.
    _STRING_FIELDS = (
        ("", ("insight", "focus_next", "next_gap", "reasoning", "search_query")),
        ("prediction", ("text",)),
        ("event_to_record", ("summary", "type", "impact")),
        ("outcome_to_record", ("summary", "impact", "event_id")),
        ("commitment", ("what", "deadline")),
        ("session_record", ("focus",)),
        ("new_goal", ("title", "description", "rationale",
                      "gap_reference", "verification_criteria")),
        ("goal_action", ("goal_id", "new_status", "note")),
        ("episodic_record", ("mtype", "summary", "details")),
        ("semantic_record", ("topic", "fact", "source")),
        ("procedural_record", ("pattern", "trigger", "procedure")),
        ("plan_action", ("step_id", "new_status", "note")),
        ("new_plan", ("goal", "verification")),
    )
    # Numeric fields: (block, field, schema-default) — default is the
    # fallback when the value is present but unconvertible.
    _NUMERIC_FIELDS = (
        ("", "confidence", 0),
        ("prediction", "confidence", 0.5),
        ("new_goal", "priority", 3),
        ("new_goal", "salience", 0.5),
        ("new_goal", "confidence", 0.7),
        ("episodic_record", "salience", 0.5),
        ("semantic_record", "confidence", 0.7),
    )
    # Every non-string poison class the LLM has emitted or could emit.
    _POISONS = (42, {"nested": True}, ["a", "list"], True, 3.14, None)

    def test_string_fields_never_crash_and_always_coerce(
        self, evolve_env: Dict
    ) -> None:
        """Feed every poison into every string-guarded field. The choke
        point must never raise, and each non-None value must come out as
        ``str`` (``None`` is preserved per the null-allowed contract)."""
        td = evolve_env["module"]
        for block, fields in self._STRING_FIELDS:
            for fld in fields:
                for poison in self._POISONS:
                    parsed: Dict[str, Any] = (
                        {fld: poison} if not block else {block: {fld: poison}}
                    )
                    td._coerce_llm_response_fields(parsed)  # must not raise
                    got = parsed[fld] if not block else parsed[block][fld]
                    if poison is None:
                        assert got is None, (
                            f"{block or '<top>'}.{fld}: None must be preserved, "
                            f"got {got!r}"
                        )
                    else:
                        assert isinstance(got, str), (
                            f"{block or '<top>'}.{fld} with {poison!r} must "
                            f"coerce to str, got {got!r} ({type(got).__name__})"
                        )

    def test_numeric_fields_never_crash_and_coerce_or_default(
        self, evolve_env: Dict
    ) -> None:
        """Numeric-guarded fields must come out int/float for every
        poison — either converted or fallen back to the schema default.
        ``None`` stays ``None`` (null-allowed)."""
        td = evolve_env["module"]
        for block, fld, default in self._NUMERIC_FIELDS:
            for poison in self._POISONS:
                parsed: Dict[str, Any] = (
                    {fld: poison} if not block else {block: {fld: poison}}
                )
                td._coerce_llm_response_fields(parsed)  # must not raise
                got = parsed[fld] if not block else parsed[block][fld]
                if poison is None:
                    assert got is None, (
                        f"{block or '<top>'}.{fld}: None must be preserved, "
                        f"got {got!r}"
                    )
                else:
                    # bool is an int subclass and numeric-compatible, so
                    # the choke point's isinstance((int, float)) guard
                    # legitimately lets True/False pass through untouched.
                    assert isinstance(got, (int, float)), (
                        f"{block or '<top>'}.{fld} with {poison!r} must be a "
                        f"number (or default {default!r}), got {got!r} "
                        f"({type(got).__name__})"
                    )

    def test_list_guarded_fields_always_end_up_lists(
        self, evolve_env: Dict
    ) -> None:
        """``session_record.outcomes_list`` and ``new_plan.steps`` are
        list-shaped: a scalar degrades to a one-item list, an empty or
        non-list value degrades to ``[]``, and a real list passes through
        untouched — never an exception, never a non-list survivor."""
        td = evolve_env["module"]
        for poison in self._POISONS + ("single outcome",):
            parsed: Dict[str, Any] = {
                "session_record": {"focus": "f", "outcomes_list": poison},
                "new_plan": {"goal": "g", "steps": poison},
            }
            td._coerce_llm_response_fields(parsed)  # must not raise
            assert isinstance(parsed["session_record"]["outcomes_list"], list), (
                f"outcomes_list with {poison!r} must be a list, got "
                f"{parsed['session_record']['outcomes_list']!r}"
            )
            assert isinstance(parsed["new_plan"]["steps"], list), (
                f"steps with {poison!r} must be a list, got "
                f"{parsed['new_plan']['steps']!r}"
            )
        # A legitimate list of step dicts must survive untouched.
        steps = [{"description": "Step A", "verification": "V"}]
        parsed = {"new_plan": {"goal": "g", "steps": steps}}
        td._coerce_llm_response_fields(parsed)
        assert parsed["new_plan"]["steps"] is steps

    def test_fully_poisoned_document_survives_coerce_and_apply(
        self, evolve_env: Dict
    ) -> None:
        """Worst case: EVERY guarded block present at once, every field
        poisoned with a different non-string class. The full pipeline —
        ``_coerce_llm_response_fields`` then ``_apply_insights`` — must
        complete without raising, and everything persisted to the
        timeline must be string-typed (so next cycle's prompt builder
        ``\"; \".join(...)`` cannot TypeError)."""
        td = evolve_env["module"]
        parsed: Dict[str, Any] = {
            "insight": 42,
            "focus_next": {"nested": True},
            "next_gap": ["gap"],
            "reasoning": True,
            "search_query": 3.14,
            "confidence": "n/a",
            "prediction": {"text": 7, "confidence": "high"},
            "event_to_record": {"summary": 1, "type": ["t"], "impact": {"i": 2}},
            "outcome_to_record": {"summary": 3, "impact": {"x": 1}, "event_id": 4},
            "commitment": {"what": 5, "deadline": ["2026-08-01"]},
            "session_record": {"focus": 6, "outcomes_list": 7},
            "new_goal": {"title": 8, "description": 9, "rationale": 10,
                         "gap_reference": 11, "verification_criteria": 12,
                         "priority": "high"},
            "goal_action": {"goal_id": 13, "new_status": ["active"], "note": {"n": 1}},
            "episodic_record": {"mtype": 14, "summary": {"s": 1}, "details": 15,
                                "salience": "high"},
            "semantic_record": {"topic": 16, "fact": 17, "source": {"src": 1},
                                "confidence": "n/a"},
            "procedural_record": {"pattern": 18, "trigger": {"t": 1},
                                  "procedure": ["p"]},
            "plan_action": {"step_id": 19, "new_status": ["complete"], "note": 20},
            "new_plan": {"goal": 21, "verification": {"v": 1}, "steps": 22},
        }
        td._coerce_llm_response_fields(parsed)  # must not raise
        state = {
            "timeline": {"version": 1, "past": {}, "present": {}, "future": {}},
            "self_model": {"version": 1, "identity": {}, "state": {},
                           "capabilities": {}, "commitments": {}},
            "orientation": None,
            "daemon_state": {"tick_count": 1, "last_action_output": ""},
            "world_model": None,
        }
        updates = td._apply_insights(parsed, state)  # must not raise
        tl = updates.get("timeline", {})
        for ev in tl.get("past", {}).get("events", []):
            assert isinstance(ev["summary"], str), ev
            assert isinstance(ev["type"], str), ev
        for oc in tl.get("past", {}).get("outcomes", []):
            assert isinstance(oc["summary"], str), oc
            assert isinstance(oc["event_id"], str), oc
        for cm in tl.get("present", {}).get("commitments", []):
            assert isinstance(cm["what"], str), cm
        for pr in tl.get("future", {}).get("predictions", []):
            assert isinstance(pr["text"], str), pr
        # A poisoned plan must not have been created as the active plan.
        assert not any(
            p.get("status") == "active"
            for p in tl.get("future", {}).get("plans", [])
        )


class TestShellPreflightValidation:
    """Pre-flight shell command validation (``_validate_shell_command``).

    Catches the two observed generation-defect classes — unbalanced
    quotes and uncompilable embedded ``python3 -c`` code — that
    previously burned a wasted subprocess call and polluted world-model
    calibration with prediction errors that were actually command
    generation defects, not world-model misses.

    ``_execute_shell_action`` must block defective commands WITHOUT
    spawning a subprocess, returning an explicit ``exit=-1: PRE-FLIGHT
    BLOCKED: <reason>`` outcome, while still running well-formed
    commands normally.
    """

    def test_plain_command_passes(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        assert td._validate_shell_command("ls -la") is None
        assert td._validate_shell_command("echo hello world") is None
        assert td._validate_shell_command("git log --oneline -3") is None

    def test_balanced_quotes_pass(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        assert td._validate_shell_command("echo 'single quoted'") is None
        assert td._validate_shell_command('echo "double quoted"') is None
        assert td._validate_shell_command('echo "mixed \'nested\' quotes"') is None
        assert td._validate_shell_command("echo 'a' && echo \"b\"") is None

    def test_unbalanced_single_quote_detected(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        assert td._validate_shell_command("echo 'unterminated") == (
            "unbalanced single quote"
        )

    def test_unbalanced_double_quote_detected(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        assert td._validate_shell_command('echo "unterminated') == (
            "unbalanced double quote"
        )

    def test_escaped_quote_inside_double_quotes_is_balanced(self, evolve_env: Dict) -> None:
        # A backslash-escaped quote inside a double-quoted region must not
        # terminate the region (this is the exact failure mode of the
        # observed 0.85-error triple: the generator escaped the closing
        # quote, leaving the string unterminated).
        td = evolve_env["module"]
        # This one is genuinely balanced: the escaped quote is consumed
        assert td._validate_shell_command('echo "it\\"s fine"') is None

    def test_generated_python_c_quote_escape_detected(self, evolve_env: Dict) -> None:
        # Reproduction of the observed failure: python3 -c "..." whose
        # trailing quote was escaped by the generator — the double-quoted
        # region never closes.
        td = evolve_env["module"]
        cmd = 'python3 -c "import pathlib; print(\'x\')\\"'
        defect = td._validate_shell_command(cmd)
        assert defect is not None
        assert "unbalanced" in defect

    def test_embedded_python_compiles_passes(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        assert td._validate_shell_command("python3 -c \"print('hi')\"") is None
        assert (
            td._validate_shell_command(
                "python -c 'import sys; print(sys.version_info[0])'"
            )
            is None
        )

    def test_embedded_python_syntax_error_detected(self, evolve_env: Dict) -> None:
        # Reproduction of the observed second failure class: a `for` loop
        # after `;` in a one-liner is a Python SyntaxError.
        td = evolve_env["module"]
        cmd = 'python3 -c "import sys; for x in [1,2]: print(x)"'
        defect = td._validate_shell_command(cmd)
        assert defect is not None
        assert "does not compile" in defect

    def test_execute_blocks_defective_command_without_subprocess(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        td = evolve_env["module"]
        # Prove the subprocess is never spawned for a defective command
        import subprocess

        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("subprocess.run must not be called for a defective command")

        monkeypatch.setattr(subprocess, "run", _boom)
        out = td._execute_shell_action("echo 'unterminated")
        assert out.startswith("exit=-1:")
        assert "PRE-FLIGHT BLOCKED" in out

    def test_execute_runs_valid_command(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        out = td._execute_shell_action("echo preflight-ok")
        assert out.startswith("exit=0:")
        assert "preflight-ok" in out

    def test_execute_reports_real_failure_exit_code(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        out = td._execute_shell_action("exit 3")
        assert out.startswith("exit=3:")


class TestApiCallPreflightValidation:
    """Pre-flight validation for the ``api_call`` action type (Gap 10 step 2).

    Deny-by-default, two layers: the endpoint must match the allowlist
    (method + host + path prefix) AND the self-model permission registry
    must grant ``read`` on the entry's resource. Unauthorized endpoints
    must be rejected with a clear error and NO execution — the first
    verification criterion of the Gap 10 design doc.
    """

    def test_allowlisted_github_endpoint_matches(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        entry = td._api_call_allowlist_entry(
            "GET", "https://api.github.com/repos/NousResearch/hermes-agent"
        )
        assert entry is not None
        assert entry["resource"] == "github"

    def test_method_mismatch_rejected(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        assert (
            td._api_call_allowlist_entry(
                "POST", "https://api.github.com/repos/NousResearch/hermes-agent"
            )
            is None
        )

    def test_unknown_host_rejected(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        assert td._api_call_allowlist_entry("GET", "https://example.com/") is None

    def test_missing_endpoint_rejected(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        ok, err = td._validate_api_call({}, {})
        assert not ok
        assert "requires an 'endpoint'" in err

    def test_non_http_endpoint_rejected(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        ok, err = td._validate_api_call(
            {"endpoint": "file:///etc/passwd", "method": "GET"}, {}
        )
        assert not ok
        assert "absolute http(s)" in err

    def test_allowlisted_but_no_read_grant_is_denied(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        # github is allowlisted, but the registry declares it with NO grants
        ok, err = td._validate_api_call(
            {
                "endpoint": "https://api.github.com/repos/NousResearch/hermes-agent",
                "method": "GET",
            },
            {"github": {"read": False, "write": False, "act": False, "cap": None}},
        )
        assert not ok
        assert "permission denied" in err
        assert "github" in err

    def test_missing_resource_entry_is_denied(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        ok, err = td._validate_api_call(
            {
                "endpoint": "https://api.github.com/repos/NousResearch/hermes-agent",
                "method": "GET",
            },
            {},
        )
        assert not ok
        assert "permission denied" in err

    def test_read_grant_passes(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        ok, err = td._validate_api_call(
            {
                "endpoint": "https://api.github.com/repos/NousResearch/hermes-agent",
                "method": "GET",
            },
            {"github": {"read": True}},
        )
        assert ok
        assert err == ""

    # ── Level 2 write path (armed, deny-until-granted) ────────────────────
    # The write allowlist exists in the daemon now, but nothing passes until
    # the explicit ``write`` grant lands (docs/gap10-level2-plan.md §3).

    def test_write_allowlisted_put_with_write_grant_passes(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        ok, err = td._validate_api_call(
            {
                "endpoint": "https://api.github.com/repos/bengkkk/hermes-evolved/"
                "contents/docs/gap10-level2-plan.md",
                "method": "PUT",
                "body": {"message": "test", "content": "eA=="},
            },
            {"github": {"read": True, "write": True}},
        )
        assert ok, err
        assert err == ""

    def test_write_denied_without_write_grant(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        ok, err = td._validate_api_call(
            {
                "endpoint": "https://api.github.com/repos/bengkkk/hermes-evolved/"
                "contents/docs/gap10-level2-plan.md",
                "method": "PUT",
                "body": {"message": "test", "content": "eA=="},
            },
            {"github": {"read": True, "write": False}},
        )
        assert not ok
        assert "no write grant" in err

    def test_write_denied_non_allowlisted_path(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        ok, err = td._validate_api_call(
            {
                "endpoint": "https://api.github.com/repos/bengkkk/hermes-evolved/"
                "contents/other.md",
                "method": "PUT",
                "body": {"message": "test", "content": "eA=="},
            },
            {"github": {"write": True}},
        )
        assert not ok
        assert "write allowlist" in err

    def test_write_denied_prefix_widening(self, evolve_env: Dict) -> None:
        """Exact-path matching: an entry for plan.md never widens to siblings."""
        td = evolve_env["module"]
        ok, err = td._validate_api_call(
            {
                "endpoint": "https://api.github.com/repos/bengkkk/hermes-evolved/"
                "contents/docs/gap10-level2-plan.md-evil",
                "method": "PUT",
                "body": {"message": "test", "content": "eA=="},
            },
            {"github": {"write": True}},
        )
        assert not ok
        assert "write allowlist" in err

    def test_write_denied_invalid_body(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        ok, err = td._validate_api_call(
            {
                "endpoint": "https://api.github.com/repos/bengkkk/hermes-evolved/"
                "contents/docs/gap10-level2-plan.md",
                "method": "PUT",
                "body": {"message": "test"},  # missing content
            },
            {"github": {"write": True}},
        )
        assert not ok
        assert "invalid write body" in err

    def test_unsupported_method_rejected(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        ok, err = td._validate_api_call(
            {"endpoint": "https://api.github.com/x", "method": "TRACE"}, {}
        )
        assert not ok
        assert "unsupported method" in err

    def test_action_params_extractor_includes_api_call_fields(
        self, evolve_env: Dict
    ) -> None:
        td = evolve_env["module"]
        params = td._action_params_from_act(
            {
                "type": "api_call",
                "endpoint": "https://api.github.com/repos/NousResearch/hermes-agent",
                "method": "GET",
            },
            "api_call",
        )
        assert params["endpoint"].startswith("https://api.github.com")
        assert params["method"] == "GET"

    def test_execute_api_call_reports_bridge_unavailable(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        td = evolve_env["module"]
        # Port 1 refuses connections immediately — no bridge is listening
        monkeypatch.setattr(td, "_API_BRIDGE_URL", "http://127.0.0.1:1")
        out = td._execute_api_call(
            {
                "endpoint": "https://api.github.com/repos/NousResearch/hermes-agent",
                "method": "GET",
            }
        )
        assert out.startswith("exit=1:")
        assert "bridge unavailable" in out

    def test_execute_api_call_parses_structured_success(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        td = evolve_env["module"]
        import urllib.request

        captured: Dict[str, Any] = {}

        class _FakeResp:
            status = 200

            def __init__(self, body: bytes) -> None:
                self._body = body

            def read(self, n: int = -1) -> bytes:
                return self._body

            def __enter__(self) -> "_FakeResp":
                return self

            def __exit__(self, *a: Any) -> bool:
                return False

        def _fake_urlopen(req: Any, timeout: float = 30.0) -> _FakeResp:
            captured["url"] = req.full_url
            captured["method"] = req.get_method()
            captured["data"] = req.data
            return _FakeResp(b'{"exit": 0, "output": {"repo": "hermes-agent"}}')

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
        out = td._execute_api_call(
            {
                "endpoint": "https://api.github.com/repos/NousResearch/hermes-agent",
                "method": "GET",
            }
        )
        assert out.startswith("exit=0:")
        assert "hermes-agent" in out
        assert captured["url"].endswith("/bridge/v1/exec")
        assert captured["method"] == "POST"

    def test_apply_insights_blocks_api_call_without_execution(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unauthorized api_call must be BLOCKED and never executed."""
        td = evolve_env["module"]

        def _boom(*args: Any, **kwargs: Any) -> str:
            raise AssertionError(
                "_execute_api_call must not run for a denied action"
            )

        monkeypatch.setattr(td, "_execute_api_call", _boom)
        parsed: Dict[str, Any] = {
            "action": {
                "type": "api_call",
                "endpoint": "https://api.github.com/repos/NousResearch/hermes-agent",
                "method": "GET",
                "expected_outcome": "exit=0: repo info",
            },
            "event_to_record": None,
            "outcome_to_record": None,
            "commitment": None,
            "prediction": None,
            "session_record": None,
        }
        state = {
            "timeline": {"version": 1, "past": {}, "present": {}, "future": {}},
            "self_model": {
                "version": 1,
                "identity": {},
                "state": {},
                "capabilities": {},
                "commitments": {},
                "permissions": {
                    "github": {"read": False, "write": False, "act": False, "cap": None}
                },
            },
            "orientation": None,
            "daemon_state": {"tick_count": 1, "last_action_output": ""},
        }
        updates = td._apply_insights(parsed, state)
        # _apply_insights mutates the daemon_state dict in place (it is not
        # part of the returned dict), so read the outcome from `state`.
        out = state["daemon_state"].get("last_action_output", "")
        assert "BLOCKED:" in out
        assert "permission denied" in out


class TestLocalAnalysisStateCheckCommands:
    """The local-analysis fallback's rotating state-check commands must be
    valid shell commands (pre-flight validation passes) and execute
    successfully (``exit=0``).

    Regression: the goals state-check one-liner used a statement-level
    ``for`` loop after ``;`` — illegal Python in a single-line compound
    statement — so every LLM-outage cycle that rotated onto the goals
    command executed a command that could never work, recorded ``exit=1``
    with a SyntaxError, and polluted the world model with a 0.6-error
    triple that was a command-generation defect, not a world-model miss.
    Observed 5x on 2026-07-31 (06:09-11:11 UTC) before the one-liner was
    rewritten as a list comprehension.
    """

    def _action_for_tick(self, td: Any, tick_count: int) -> Dict[str, Any]:
        """Return the local-analysis action for a daemon tick.

        ``_local_analysis`` uses ``tick_count + 1`` for the rotation index,
        so ``tick_count=4`` yields tick 5 -> slot 0 (goals check).
        """
        result = td._local_analysis({"daemon_state": {"tick_count": tick_count}})
        return result["action"]

    def test_goals_state_check_command_valid_and_runs(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        action = self._action_for_tick(td, 4)  # tick 5 -> slot 0 (goals)
        assert action["type"] == "shell"
        cmd = action["command"]
        assert "Goals" in cmd
        assert td._validate_shell_command(cmd) is None
        out = td._execute_shell_action(cmd)
        assert out.startswith("exit=0:"), f"goals command failed: {out[:200]}"

    def test_self_model_state_check_command_valid_and_runs(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        action = self._action_for_tick(td, 5)  # tick 6 -> slot 1 (self-model)
        assert action["type"] == "shell"
        cmd = action["command"]
        assert td._validate_shell_command(cmd) is None
        out = td._execute_shell_action(cmd)
        assert out.startswith("exit=0:"), f"self-model command failed: {out[:200]}"

    def test_daemon_state_check_command_valid_and_runs(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        action = self._action_for_tick(td, 6)  # tick 7 -> slot 2 (daemon health)
        assert action["type"] == "shell"
        cmd = action["command"]
        assert td._validate_shell_command(cmd) is None
        out = td._execute_shell_action(cmd)
        assert out.startswith("exit=0:"), f"daemon command failed: {out[:200]}"


class TestFallbackSearchQuery:
    """Gap 5: the local-analysis fallback must keep self-directed info
    seeking alive during LLM outages by deriving a search_query from the
    self-model's unresolved ``unknown_areas``.

    Regression: the fallback hardcoded ``search_query=None``, so the only
    channel that can ask questions (the LLM) was the same channel that was
    down — the daemon's open questions were never searched during outages,
    and the search phase (4.5) had nothing to act on. The fallback now
    derives a bounded query from the top unknown area instead.
    """

    def test_helper_returns_top_unknown_area(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        sm = {"capabilities": {"unknown_areas": [
            "How to persist a plan across restarts",
            "Whether the bridge restart has a dry-run path",
        ]}}
        assert td._fallback_search_query(sm) == "How to persist a plan across restarts"

    def test_helper_returns_none_when_no_unknowns(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        assert td._fallback_search_query({"capabilities": {}}) is None
        assert td._fallback_search_query({}) is None
        assert td._fallback_search_query({"capabilities": {"unknown_areas": []}}) is None
        assert td._fallback_search_query({"capabilities": {"unknown_areas": "not-a-list"}}) is None

    def test_helper_trims_overlong_unknown(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        long_unknown = "x" * 500
        q = td._fallback_search_query({"capabilities": {"unknown_areas": [long_unknown]}})
        assert q is not None and len(q) == 200

    def test_local_analysis_emits_search_query_from_unknowns(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        result = td._local_analysis({
            "self_model": {"capabilities": {"unknown_areas": ["How to persist a plan"]}},
            "daemon_state": {"tick_count": 1, "last_action_output": ""},
        })
        assert result["search_query"] == "How to persist a plan"

    def test_local_analysis_search_query_none_without_unknowns(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        result = td._local_analysis({
            "self_model": {"capabilities": {"unknown_areas": []}},
            "daemon_state": {"tick_count": 1, "last_action_output": ""},
        })
        assert result["search_query"] is None


class TestRecordWebSearchTriple:
    """Gap 5: an executed web search must land in the world model as a
    completed ``web_search`` triple so info seeking is observable and its
    reliability learnable (low error when results are returned, higher
    error on a zero-hit search).
    """

    def _wm(self, evolve_env: Dict):
        td = evolve_env["module"]
        return td.WorldModel()

    def test_records_results_bearing_search_low_error(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        wm = self._wm(evolve_env)
        results = [{"title": "GitHub Docs", "href": "https://docs.github.com"}]
        tid = td._record_web_search_triple(wm, "github api", results)
        assert tid is not None
        triple = [t for t in wm.data["action_triples"] if t["id"] == tid][0]
        assert triple["action_type"] == "web_search"
        assert triple["completed"] is True
        assert triple["prediction_error"] <= 0.25, triple["prediction_error"]
        assert "exit=0: 1 result(s)" in triple["actual_outcome"]

    def test_records_zero_hit_search_higher_error(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        wm = self._wm(evolve_env)
        tid = td._record_web_search_triple(wm, "no such query xyz", [])
        assert tid is not None
        triple = [t for t in wm.data["action_triples"] if t["id"] == tid][0]
        assert triple["completed"] is True
        assert triple["prediction_error"] > 0.4, triple["prediction_error"]
        assert "exit=1: 0 results" in triple["actual_outcome"]

    def test_recording_failure_is_non_blocking(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        wm = self._wm(evolve_env)
        # A record_action that raises must not propagate — helper returns None
        def _boom(*a, **k):
            raise RuntimeError("simulated failure")
        wm.record_action = _boom  # type: ignore[method-assign]
        assert td._record_web_search_triple(wm, "query", [{"title": "t"}]) is None
        assert len(wm.data["action_triples"]) == 0


class TestStdlibSearchFallback:
    """Gap 5: the search phase must survive without third-party search
    packages. The production daemon env has neither ``duckduckgo_search``
    nor ``ddgs`` installed, so every search attempt logged
    "No search module available" and no web_search triple ever landed.
    ``_run_web_search`` now falls back to a stdlib-only Bing RSS fetch so
    self-directed info seeking keeps working on a bare Python env.
    """

    RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>
    <item><title>First Result</title><link>https://example.com/1</link>
    <description>First body text.</description></item>
    <item><title>Second Result</title><link>https://example.com/2</link>
    <description>Second body text.</description></item>
    <item><title>Third Result</title><link>https://example.com/3</link>
    <description>Third body text.</description></item>
    </channel></rss>"""

    def test_parses_rss_into_ddgs_shape(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        res = td._parse_search_rss(self.RSS, max_results=4)
        assert [r["title"] for r in res] == ["First Result", "Second Result", "Third Result"]
        assert res[0]["href"] == "https://example.com/1"
        assert res[0]["body"] == "First body text."

    def test_parses_rss_caps_at_max_results(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        res = td._parse_search_rss(self.RSS, max_results=2)
        assert len(res) == 2
        assert res[0]["title"] == "First Result"

    def test_parses_empty_feed_to_empty_list(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        assert td._parse_search_rss(
            b'<?xml version="1.0"?><rss version="2.0"><channel></channel></rss>'
        ) == []

    def test_parses_missing_fields_to_empty_strings(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        res = td._parse_search_rss(
            b'<?xml version="1.0"?><rss version="2.0"><channel><item></item></channel></rss>'
        )
        assert res == [{"title": "", "href": "", "body": ""}]

    def test_search_stdlib_builds_request_and_parses(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        td = evolve_env["module"]
        captured: Dict[str, Any] = {}

        class _FakeResp:
            def __enter__(self) -> "_FakeResp":
                return self

            def __exit__(self, *a: Any) -> bool:
                return False

            def read(self) -> bytes:
                return TestStdlibSearchFallback.RSS

        def _fake_urlopen(req: Any, timeout: int = 20) -> _FakeResp:
            captured["url"] = req.full_url
            captured["ua"] = req.get_header("User-agent")
            captured["timeout"] = timeout
            return _FakeResp()

        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
        res = td._search_stdlib("github api docs", max_results=4)
        assert "bing.com/search" in captured["url"]
        assert "q=github" in captured["url"]
        assert captured["ua"] == "hermes-evolved/1.0 (self-improving agent)"
        assert captured["timeout"] == 20
        assert res[0]["title"] == "First Result"

    def test_run_web_search_uses_ddgs_when_available(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins
        import types

        td = evolve_env["module"]
        fake_mod = types.ModuleType("duckduckgo_search")
        instances: list = []

        class _FakeDDGS:
            def __init__(self) -> None:
                instances.append(self)

            def __enter__(self) -> "_FakeDDGS":
                return self

            def __exit__(self, *a: Any) -> bool:
                return False

            def text(self, query: str, max_results: int = 4) -> list:
                self.called_with = (query, max_results)
                return [{"title": "ddg", "href": "https://ddg", "body": "b"}]

        fake_mod.DDGS = _FakeDDGS
        real_import = builtins.__import__

        def _fake_import(name: str, *a: Any, **kw: Any) -> Any:
            if name == "duckduckgo_search":
                return fake_mod
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        res = td._run_web_search("query", max_results=3)
        assert res[0]["title"] == "ddg"
        assert instances[0].called_with == ("query", 3)

    def test_run_web_search_falls_back_on_import_error(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        td = evolve_env["module"]
        real_import = builtins.__import__

        def _fake_import(name: str, *a: Any, **kw: Any) -> Any:
            if name in ("duckduckgo_search", "ddgs"):
                raise ImportError(f"No module named {name}")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        monkeypatch.setattr(
            td, "_search_stdlib",
            lambda q, max_results=4: [{"title": "fallback", "href": "https://fb", "body": "b"}],
        )
        res = td._run_web_search("query")
        assert res[0]["title"] == "fallback"

    def test_run_web_search_falls_back_on_ddgs_runtime_failure(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins
        import types

        td = evolve_env["module"]
        fake_mod = types.ModuleType("duckduckgo_search")

        class _BrokenDDGS:
            def __enter__(self) -> "_BrokenDDGS":
                return self

            def __exit__(self, *a: Any) -> bool:
                return False

            def text(self, query: str, max_results: int = 4) -> list:
                raise RuntimeError("rate limited")

        fake_mod.DDGS = _BrokenDDGS
        real_import = builtins.__import__

        def _fake_import(name: str, *a: Any, **kw: Any) -> Any:
            if name == "duckduckgo_search":
                return fake_mod
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        monkeypatch.setattr(
            td, "_search_stdlib",
            lambda q, max_results=4: [{"title": "fb2", "href": "https://fb2", "body": "b"}],
        )
        res = td._run_web_search("query")
        assert res[0]["title"] == "fb2"




class TestRotatingAutoCommands:
    """Every rotating auto-default command (module-level ``_ROTATING_AUTOS``)
    must pass pre-flight validation and execute successfully (exit=0).

    Regression: the sibling-dirs slot (index 5) used double-quoted
    ``python3 -c "..."`` with the closing quote escaped by the generator,
    leaving the shell string unterminated — recorded as a 0.85-error
    world-model triple on 2026-07-31 (05:19) that was a command-generation
    defect, not a world-model miss.  The pre-flight validator blocks it
    today, but the slot silently no-oped on every rotation.  This guard
    proves every rotation slot is healthy.
    """

    def test_all_rotating_auto_commands_valid_and_run(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        autos = td._ROTATING_AUTOS
        # Invariant: the rotation actually rotates across several slots
        assert len(autos) >= 3
        for idx, auto in enumerate(autos):
            assert auto["type"] == "shell"
            cmd = auto["command"]
            defect = td._validate_shell_command(cmd)
            assert defect is None, f"auto[{idx}] defective: {defect} | {cmd}"
            out = td._execute_shell_action(cmd)
            assert out.startswith("exit=0:"), f"auto[{idx}] failed: {out[:200]} | {cmd}"


# ═══════════════════════════════════════════════════════════════════
#  Placeholder-plan guard (prevents deliberation fixation loops)
# ═══════════════════════════════════════════════════════════════════


class TestPlaceholderPlanGuard:
    """The daemon must never persist a plan whose steps carry no
    actionable content (e.g. goal 'Test', steps 'Step A'/'Step B',
    verification 'V'). Such plans re-enter the prompt as the active
    plan every cycle and trap the daemon in a fixation loop.
    """

    def test_is_placeholder_step_detects_generic_labels(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        # Generic 'Step N' / 'S1' labels with bare 'V' verification
        assert td._is_placeholder_step({"description": "Step A", "verification": "V"})
        assert td._is_placeholder_step({"description": "Step 1", "verification": "Verify"})
        assert td._is_placeholder_step({"description": "S1", "verification": "V"})
        assert td._is_placeholder_step({"description": "step_2", "verification": "n/a"})
        # Empty fields
        assert td._is_placeholder_step({"description": "", "verification": ""})
        assert td._is_placeholder_step({"description": "  ", "verification": ""})
        assert td._is_placeholder_step({"description": "Run tests", "verification": ""})
        # Trivially short description
        assert td._is_placeholder_step({"description": "Do", "verification": "tests pass"})

    def test_is_placeholder_step_accepts_actionable_steps(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        assert not td._is_placeholder_step({
            "description": "Run the world-model test suite",
            "verification": "401 tests pass",
        })
        assert not td._is_placeholder_step({
            "description": "Implement predict-action-verify loop",
            "verification": "world_model.py imports and unit tests green",
        })

    def test_placeholder_new_plan_rejected_and_not_persisted(self, evolve_env: Dict) -> None:
        """A plan whose steps are all placeholders must be rejected with an
        observation event, and no active plan may exist afterwards.
        """
        td = evolve_env["module"]
        result = {
            "action": None,
            "fallback": True,
            "insight": "test",
            "focus_next": "continue",
            "confidence": 0.5,
            "reasoning": "test",
            "event_to_record": None,
            "outcome_to_record": None,
            "commitment": None,
            "prediction": None,
            "session_record": None,
            "plan_action": None,
            "new_plan": {
                "goal": "Test",
                "steps": [
                    {"description": "Step A", "verification": "V"},
                    {"description": "Step B", "verification": "V"},
                ],
            },
            "new_goal": None,
            "goal_action": None,
            "search_query": None,
            "episodic_record": None,
            "self_model_update": {"weakness": None, "unknown": None, "new_commitment": None},
            "next_gap": None,
        }
        state = {
            "daemon_state": {"tick_count": 15, "last_action_output": ""},
            "world_model": WorldModel() if "WorldModel" in globals() else None,
            "timeline": {"version": 1, "past": {"events": []}, "present": {}, "future": {}},
            "self_model": {
                "identity": {"name": "test", "role": "test"},
                "state": {},
                "capabilities": {"strengths": [], "weaknesses": [], "unknown_areas": []},
                "commitments": {},
            },
            "orientation": {"vision": "Test", "phase": "test"},
        }
        if state["world_model"] is None:
            from world_model import WorldModel
            state["world_model"] = WorldModel()

        td._apply_insights(result, state)

        # No active plan must have been created
        from data_layer import get_active_plan
        assert get_active_plan() is None
        # The rejection must be recorded as an observation event
        tl = td.load_timeline()
        events = tl.get("past", {}).get("events", [])
        assert any("Rejected placeholder plan" in (e.get("summary") or "") for e in events)

    def test_actionable_new_plan_still_created(self, evolve_env: Dict) -> None:
        """Plans with real, verifiable steps must still be created."""
        td = evolve_env["module"]
        from world_model import WorldModel
        result = {
            "action": None,
            "fallback": True,
            "insight": "test",
            "focus_next": "continue",
            "confidence": 0.5,
            "reasoning": "test",
            "event_to_record": None,
            "outcome_to_record": None,
            "commitment": None,
            "prediction": None,
            "session_record": None,
            "plan_action": None,
            "new_plan": {
                "goal": "Calibrate world model",
                "steps": [
                    {"description": "Run the world-model test suite", "verification": "all tests pass"},
                    {"description": "Collect five more shell action triples", "verification": "per-type count >= 5"},
                ],
            },
            "new_goal": None,
            "goal_action": None,
            "search_query": None,
            "episodic_record": None,
            "self_model_update": {"weakness": None, "unknown": None, "new_commitment": None},
            "next_gap": None,
        }
        state = {
            "daemon_state": {"tick_count": 15, "last_action_output": ""},
            "world_model": WorldModel(),
            "timeline": {"version": 1, "past": {"events": []}, "present": {}, "future": {}},
            "self_model": {
                "identity": {"name": "test", "role": "test"},
                "state": {},
                "capabilities": {"strengths": [], "weaknesses": [], "unknown_areas": []},
                "commitments": {},
            },
            "orientation": {"vision": "Test", "phase": "test"},
        }
        td._apply_insights(result, state)

        from data_layer import get_active_plan
        plan = get_active_plan()
        assert plan is not None
        assert plan["goal"] == "Calibrate world model"
        assert len(plan["steps"]) == 2
        assert plan["steps"][0]["description"] == "Run the world-model test suite"


class TestCycleBodyLlmCallWorldModelRecording:
    """Gap 8: the adaptive retry policy's outcome must reach the world model.

    ``_run_cycle_body`` records the LLM call as an ``llm_call`` action triple
    (expected = applied retry budget, actual = observed outcome) so endpoint
    reliability feeds per-type accuracy and the calibration curve instead of
    vanishing after the cycle. The recording is guarded by a single-use
    ``_fresh`` marker that only the real ``_call_llm`` sets — a monkeypatched
    ``_call_llm`` (as used across this test file) must never produce
    synthetic triples.
    """

    _VALID_JSON = json.dumps({
        "insight": "test insight",
        "self_model_update": [],
        "timeline_update": {"events": []},
        "goals_update": [],
        "search_query": "",
        "action": None,
        "prediction": None,
    })

    async def _run_cycle(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch,
        populate_stats: bool, skip: bool = False,
        budget_seconds: Optional[float] = None,
        fallback_cycles: int = 0,
    ) -> Dict[str, Any]:
        td = evolve_env["module"]
        if budget_seconds is not None:
            td._cycle_budget_seconds = budget_seconds

        async def _fake_llm(messages: list, task: str = "thinking") -> Optional[str]:
            if populate_stats:
                # Mimic what the real _call_llm now does: fresh outcome stats.
                td._last_llm_call_stats.clear()
                if skip:
                    # Extended-outage skip path: policy tier with 0 attempts.
                    td._last_llm_call_stats.update({
                        "_fresh": True,
                        "success": False,
                        "skipped": True,
                        "attempts_used": 0,
                        "last_error": None,
                        "duration_s": 0.0,
                        "max_retries": 0,
                        "per_attempt_timeout": 0.0,
                    })
                    return None
                td._last_llm_call_stats.update({
                    "_fresh": True,
                    "success": True,
                    "skipped": False,
                    "attempts_used": 1,
                    "last_error": None,
                    "duration_s": 1.5,
                    "max_retries": 2,
                    "per_attempt_timeout": 90.0,
                })
            return self._VALID_JSON

        monkeypatch.setattr(td, "_call_llm", _fake_llm)

        result = {"status": "ok", "tick_duration": 0, "insight": None, "error": None}
        ds = td.load_daemon_state()
        if fallback_cycles:
            ds["consecutive_fallback_cycles"] = fallback_cycles
        ds.setdefault(
            "cycle_stats",
            {"total": 0, "ok": 0, "error": 0, "parse_error": 0,
             "avg_duration": 0.0, "max_duration": 0.0},
        )
        return await td._run_cycle_body(result, ds)

    def test_success_outcome_recorded_as_llm_call_triple(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A real-style call records a completed triple with low error."""
        td = evolve_env["module"]
        asyncio.run(self._run_cycle(evolve_env, monkeypatch, populate_stats=True))

        wm = td.load_world_model()
        triples = [t for t in wm.data.get("action_triples", [])
                   if t["action_type"] == "llm_call"]
        assert len(triples) == 1, f"expected 1 llm_call triple, got {len(triples)}"
        t = triples[0]
        assert t["completed"] is True
        assert t["prediction_error"] is not None
        assert t["prediction_error"] <= 0.25, (
            f"success phrasing must score low, got {t['prediction_error']}"
        )
        assert "succeeded on attempt 1" in t["actual_outcome"]
        assert "2 attempt(s)" in t["expected_outcome"]
        assert t["action_parameters"]["max_retries"] == 2
        assert "_fresh" not in t["action_parameters"], (
            "freshness marker must not persist into stored params"
        )

    def test_monkeypatched_llm_without_stats_records_nothing(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-fresh (monkeypatched) call must not create a synthetic triple."""
        td = evolve_env["module"]
        asyncio.run(self._run_cycle(evolve_env, monkeypatch, populate_stats=False))

        wm = td.load_world_model()
        triples = [t for t in wm.data.get("action_triples", [])
                   if t["action_type"] == "llm_call"]
        assert triples == [], "monkeypatched _call_llm must not record llm_call triples"

    def test_skip_outcome_recorded_as_llm_call_triple(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deliberate extended-outage skip must record expected == actual.

        Regression guard: before the fix, the expected string was built
        unconditionally as "success: LLM responds within 0 attempt(s) × 0s
        budget" for a skipped call, scoring a phantom ~0.85 prediction error
        against the actual "skipped: no probe..." — polluting llm_call
        calibration with a fake failure signal on every outage cycle.
        """
        td = evolve_env["module"]
        asyncio.run(self._run_cycle(evolve_env, monkeypatch, populate_stats=True,
                                    skip=True))

        wm = td.load_world_model()
        triples = [t for t in wm.data.get("action_triples", [])
                   if t["action_type"] == "llm_call"]
        assert len(triples) == 1, f"expected 1 llm_call triple, got {len(triples)}"
        t = triples[0]
        assert t["completed"] is True
        assert t["prediction_error"] is not None
        assert t["prediction_error"] <= 0.25, (
            f"policy skip must score low, got {t['prediction_error']}"
        )
        assert "skipped" in t["expected_outcome"], (
            "skip expected must read as a skip, not as success-with-0-attempts"
        )
        assert "skipped" in t["actual_outcome"]
        assert t["expected_outcome"] == t["actual_outcome"], (
            "a deliberate policy skip is its own prediction: expected == actual"
        )
        assert t["action_parameters"]["max_retries"] == 0

    def test_llm_call_triple_records_budget_fraction(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With a cycle budget in force, the llm_call triple records how much
        of it the LLM phase consumed (budget-consumed telemetry, P3)."""
        td = evolve_env["module"]
        asyncio.run(self._run_cycle(evolve_env, monkeypatch, populate_stats=True,
                                    budget_seconds=200.0))

        wm = td.load_world_model()
        triples = [t for t in wm.data.get("action_triples", [])
                   if t["action_type"] == "llm_call"]
        assert len(triples) == 1, f"expected 1 llm_call triple, got {len(triples)}"
        params = triples[0]["action_parameters"]
        # Fake stats use duration_s = 1.5, budget 200.0 → fraction 0.0075.
        assert params.get("cycle_budget_s") == 200.0
        assert params.get("budget_fraction") == round(1.5 / 200.0, 4), params
        assert "_fresh" not in params

    def test_outage_tier_records_disjunctive_expected_outcome(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-healthy retry tier must hedge, not predict flat success.

        Regression guard for the llm_call calibration bias observed
        2026-08-01: every recorded failure (all under warm/deep/healthy
        tiers) was predicted as "success: ..." — a tautology that scored
        each real timeout as a 0.85 surprise.  Under an outage tier the
        expected outcome must read as an explicit hedge ("success or
        timeout: ...") naming the tier, and score the intermediate 0.4
        that _compute_prediction_error assigns to disjunctive
        predictions.
        """
        td = evolve_env["module"]
        asyncio.run(self._run_cycle(evolve_env, monkeypatch,
                                    populate_stats=True,
                                    fallback_cycles=2))

        wm = td.load_world_model()
        triples = [t for t in wm.data.get("action_triples", [])
                   if t["action_type"] == "llm_call"]
        assert len(triples) == 1, f"expected 1 llm_call triple, got {len(triples)}"
        t = triples[0]
        assert "success or timeout" in t["expected_outcome"], t["expected_outcome"]
        assert "outage tier 'deep'" in t["expected_outcome"], t["expected_outcome"]
        # Fake stats succeed on attempt 1; the hedge resolves to success
        # but must score the intermediate disjunctive error, not a
        # confident-correct 0.15 (nor the 0.85 a flat "success" would
        # have scored on the failure branch).
        assert t["prediction_error"] == 0.4, t["prediction_error"]

    def test_healthy_tier_with_recent_failures_hedges(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A healthy retry tier must still hedge when recent llm_call
        history shows failures.

        Regression guard for the 2026-08-02 03:16 observation: the
        consecutive-fallback counter reset to 0 after one successful
        cycle, so the next timeout under a "healthy" tier was predicted
        with flat "success: ..." and scored 0.85 — the single remaining
        llm_call discrepancy pattern (7 high-error triples).  Recent
        recorded outcomes are the second signal: with failures in the
        last _LLM_HEALTH_WINDOW calls, even a healthy tier must emit the
        disjunctive hedge and score the intermediate 0.4 (honest
        uncertainty), never a confident 0.15 or 0.85.
        """
        td = evolve_env["module"]
        wm = td.load_world_model()
        for _ in range(3):
            tid = wm.record_action(
                "llm_call",
                "LLM thinking call (adaptive retry policy)",
                "success: LLM responds within 2 attempt(s) × 90s budget",
                expected_source="daemon",
                parameters={"success": False, "last_error": "timeout"},
            )
            wm.complete_action(tid, "failed: timeout after 2 attempt(s)")
        wm.save()

        asyncio.run(self._run_cycle(evolve_env, monkeypatch,
                                    populate_stats=True,
                                    fallback_cycles=0))

        wm = td.load_world_model()
        triples = [t for t in wm.data.get("action_triples", [])
                   if t["action_type"] == "llm_call"]
        assert len(triples) == 4, f"expected 4 llm_call triples, got {len(triples)}"
        t = triples[-1]
        assert "success or timeout" in t["expected_outcome"], t["expected_outcome"]
        assert "outage tier 'healthy'" in t["expected_outcome"], t["expected_outcome"]
        assert "3/5 recent calls failed" in t["expected_outcome"], t["expected_outcome"]
        # The fake call succeeds on attempt 1, but the hedge must still
        # score the intermediate disjunctive error — the endpoint was
        # known-flaky from history, so 0.15 would be overconfident.
        assert t["prediction_error"] == 0.4, t["prediction_error"]

    def test_healthy_tier_clean_history_stays_confident(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No recent failures → a healthy tier keeps the confident
        prediction (no over-hedging).

        Boundary guard for the recent-history hedge: when the last
        _LLM_HEALTH_WINDOW llm_call triples all succeeded, a healthy
        tier must stay confident ("success: ...") and score the
        confident-correct 0.15 — flipping to "success or timeout" on
        clean history would degrade calibration with phantom
        uncertainty.
        """
        td = evolve_env["module"]
        wm = td.load_world_model()
        for _ in range(3):
            tid = wm.record_action(
                "llm_call",
                "LLM thinking call (adaptive retry policy)",
                "success: LLM responds within 2 attempt(s) × 90s budget",
                expected_source="daemon",
                parameters={"success": True, "last_error": None},
            )
            wm.complete_action(tid, "succeeded on attempt 1 (1.5s)")
        wm.save()

        asyncio.run(self._run_cycle(evolve_env, monkeypatch,
                                    populate_stats=True,
                                    fallback_cycles=0))

        wm = td.load_world_model()
        triples = [t for t in wm.data.get("action_triples", [])
                   if t["action_type"] == "llm_call"]
        t = triples[-1]
        assert "success or timeout" not in t["expected_outcome"], t["expected_outcome"]
        assert t["expected_outcome"].startswith("success:"), t["expected_outcome"]
        assert t["prediction_error"] == 0.15, t["prediction_error"]

    def test_healthy_tier_near_budget_successes_hedge(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Near-budget successes must hedge a healthy tier like failures do.

        Regression guard for the 2026-08-02 06:00:40 observation: a
        179.6s success against a 2×90s budget at 05:10 (99.8% of budget)
        left the hard-failure window clean, so the next cycle predicted a
        flat "success:" into a 2-attempt timeout and scored 0.85.  A
        success that consumed >= 85% of its total retry budget is a
        strain signal — with near-budget successes in the last
        _LLM_HEALTH_WINDOW calls, even a healthy tier must emit the
        disjunctive hedge and score the intermediate 0.4 (honest
        uncertainty), never a confident 0.85.
        """
        td = evolve_env["module"]
        wm = td.load_world_model()
        for _ in range(3):
            tid = wm.record_action(
                "llm_call",
                "LLM thinking call (adaptive retry policy)",
                "success: LLM responds within 2 attempt(s) × 90s budget",
                expected_source="daemon",
                parameters={"success": True, "last_error": None},
            )
            wm.complete_action(tid, "succeeded on attempt 2 (179.6s)")
        wm.save()

        asyncio.run(self._run_cycle(evolve_env, monkeypatch,
                                    populate_stats=True,
                                    fallback_cycles=0))

        wm = td.load_world_model()
        triples = [t for t in wm.data.get("action_triples", [])
                   if t["action_type"] == "llm_call"]
        assert len(triples) == 4, f"expected 4 llm_call triples, got {len(triples)}"
        t = triples[-1]
        assert "success or timeout" in t["expected_outcome"], t["expected_outcome"]
        assert "outage tier 'healthy'" in t["expected_outcome"], t["expected_outcome"]
        assert "3/10 recent calls near-budget" in t["expected_outcome"], t["expected_outcome"]
        assert t["prediction_error"] == 0.4, t["prediction_error"]

    def test_healthy_tier_strain_outside_failure_window_still_hedges(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Near-budget strain 6-10 calls back must still hedge a healthy tier.

        Regression guard for the 2026-08-03 12:33Z observation: a 94.9%-
        of-budget success at 10:52Z (near-budget strain, a LEADING
        degradation indicator) was followed by five clean cycles and then
        a 2-attempt timeout.  By the time the timeout hit, the strain sat
        6+ calls back — outside the 5-call hard-failure window — so the
        healthy tier saw a clean window, predicted flat "success:", and
        scored a confident 0.85.  The strain lookback (_LLM_STRAIN_WINDOW)
        is wider than the failure lookback (_LLM_HEALTH_WINDOW) so a
        success that ground near its budget 6-10 calls back still hedges.
        """
        td = evolve_env["module"]
        wm = td.load_world_model()
        # 7 clean, fast successes (would clear a 5-call failure window)...
        for _ in range(7):
            tid = wm.record_action(
                "llm_call",
                "LLM thinking call (adaptive retry policy)",
                "success: LLM responds within 2 attempt(s) × 90s budget",
                expected_source="daemon",
                parameters={"success": True, "last_error": None},
            )
            wm.complete_action(tid, "succeeded on attempt 1 (1.5s)")
        # ...then one near-budget strain (94.9% of 2×90s) 8 calls back
        tid = wm.record_action(
            "llm_call",
            "LLM thinking call (adaptive retry policy)",
            "success: LLM responds within 2 attempt(s) × 90s budget",
            expected_source="daemon",
            parameters={"success": True, "last_error": None},
        )
        wm.complete_action(tid, "succeeded on attempt 2 (170.9s)")
        wm.save()

        asyncio.run(self._run_cycle(evolve_env, monkeypatch,
                                    populate_stats=True,
                                    fallback_cycles=0))

        wm = td.load_world_model()
        triples = [t for t in wm.data.get("action_triples", [])
                   if t["action_type"] == "llm_call"]
        t = triples[-1]
        assert "success or timeout" in t["expected_outcome"], t["expected_outcome"]
        assert "1/10 recent calls near-budget" in t["expected_outcome"], t["expected_outcome"]
        assert t["prediction_error"] == 0.4, t["prediction_error"]

    def test_healthy_tier_fast_recovery_stays_confident(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 2-attempt success well under the total budget is NOT strain.

        Boundary guard for the near-budget hedge: "succeeded on attempt 2
        (128.4s)" against a 2×90s budget (71% of budget — an attempt-1
        timeout with a fast recovery) must not flip a healthy tier to the
        hedge; only consumption >= 85% of the total budget counts.
        """
        td = evolve_env["module"]
        wm = td.load_world_model()
        for _ in range(3):
            tid = wm.record_action(
                "llm_call",
                "LLM thinking call (adaptive retry policy)",
                "success: LLM responds within 2 attempt(s) × 90s budget",
                expected_source="daemon",
                parameters={"success": True, "last_error": None},
            )
            wm.complete_action(tid, "succeeded on attempt 2 (128.4s)")
        wm.save()

        asyncio.run(self._run_cycle(evolve_env, monkeypatch,
                                    populate_stats=True,
                                    fallback_cycles=0))

        wm = td.load_world_model()
        triples = [t for t in wm.data.get("action_triples", [])
                   if t["action_type"] == "llm_call"]
        t = triples[-1]
        assert "success or timeout" not in t["expected_outcome"], t["expected_outcome"]
        assert t["expected_outcome"].startswith("success:"), t["expected_outcome"]
        assert t["prediction_error"] == 0.15, t["prediction_error"]

    def test_llm_call_triple_budget_fraction_none_without_budget(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a cycle budget (persistent daemon mode), the triple records
        cycle_budget_s=None and budget_fraction=None — no phantom numbers."""
        td = evolve_env["module"]
        td._cycle_budget_seconds = None  # explicit: daemon mode default
        asyncio.run(self._run_cycle(evolve_env, monkeypatch, populate_stats=True))

        wm = td.load_world_model()
        triples = [t for t in wm.data.get("action_triples", [])
                   if t["action_type"] == "llm_call"]
        assert len(triples) == 1, f"expected 1 llm_call triple, got {len(triples)}"
        params = triples[0]["action_parameters"]
        assert params.get("cycle_budget_s") is None
        assert params.get("budget_fraction") is None


class TestAutoCreatePlanGuard:
    """The bootstrap auto-create must not duplicate or re-plan a goal that
    already has an active or completed plan.

    Regression (2026-07-31): the old guard only checked
    ``get_active_plan() is None``.  When the initial Gap 8 plan was
    completed, that condition became true and the next cycle auto-created a
    second *active* plan for the same already-done goal.  The duplicate
    shadowed the completed plan in ``format_plan_context`` and drove a
    re-planning fixation loop.  Failed plans are exempt — a goal that
    genuinely failed may legitimately be re-planned.
    """

    GOAL = "Complete Gap 8 — Self-directed evolution"

    async def _run_cycle(self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch) -> tuple:
        td = evolve_env["module"]

        async def _fake_llm(messages: list, task: str = "thinking") -> str:
            return ""  # empty → local-analysis fallback, still runs the guard

        monkeypatch.setattr(td, "_call_llm", _fake_llm)

        result = {"status": "ok", "tick_duration": 0, "insight": None, "error": None}
        ds = td.load_daemon_state()
        ds.setdefault(
            "cycle_stats",
            {"total": 0, "ok": 0, "error": 0, "parse_error": 0,
             "avg_duration": 0.0, "max_duration": 0.0},
        )

        # Local-analysis fallback executes an action via subprocess; mock it
        # (same pattern as TestCycleBodyEmptyLlmResponse) for hermeticity.
        import subprocess

        original_run = subprocess.run

        def _mock_run(*a, **kw):
            return type("_R", (), {"returncode": 0, "stdout": "mocked\n", "stderr": ""})()

        try:
            subprocess.run = _mock_run
            result = await td._run_cycle_body(result, ds)
        finally:
            subprocess.run = original_run

        return result, ds

    def _plans(self, td) -> list:
        return (td.load_timeline().get("future") or {}).get("plans", [])

    def test_completed_plan_same_goal_blocks_recreation(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A completed plan for the goal must NOT be re-created (fixation loop)."""
        from data_layer import create_plan, complete_plan, get_active_plan

        td = evolve_env["module"]
        plan_id = create_plan(self.GOAL, steps=[{"description": "d", "verification": "v"}])
        complete_plan(plan_id)
        assert get_active_plan() is None  # the exact condition that used to trigger re-creation

        asyncio.run(self._run_cycle(evolve_env, monkeypatch))

        assert get_active_plan() is None, "no new plan may shadow a completed goal"
        plans = self._plans(td)
        assert len(plans) == 1, f"expected 1 plan, got {len(plans)}"
        assert plans[0]["id"] == plan_id

    def test_active_plan_same_goal_blocks_duplicate(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An active plan for the goal must not be duplicated."""
        from data_layer import create_plan, get_active_plan

        td = evolve_env["module"]
        plan_id = create_plan(self.GOAL, steps=[{"description": "d", "verification": "v"}])
        assert get_active_plan()["id"] == plan_id

        asyncio.run(self._run_cycle(evolve_env, monkeypatch))

        plans = self._plans(td)
        assert len(plans) == 1, f"expected 1 plan, got {len(plans)}"
        assert plans[0]["id"] == plan_id

    def test_no_existing_plan_still_auto_creates(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no plans at all, the bootstrap plan is still auto-created."""
        from data_layer import get_active_plan

        assert get_active_plan() is None

        asyncio.run(self._run_cycle(evolve_env, monkeypatch))

        plan = get_active_plan()
        assert plan is not None
        assert plan["goal"] == self.GOAL

    def test_failed_plan_same_goal_allows_replan(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed plan for the goal is NOT a blocker — re-planning is legit."""
        from data_layer import create_plan, complete_plan, get_active_plan

        td = evolve_env["module"]
        create_plan(self.GOAL, steps=[{"description": "d", "verification": "v"}])
        for p in self._plans(td):
            complete_plan(p["id"], status="failed")
        assert get_active_plan() is None

        asyncio.run(self._run_cycle(evolve_env, monkeypatch))

        assert get_active_plan() is not None, "failed goal may be re-planned"
        assert len(self._plans(td)) == 2  # failed one + new active one


class TestPlanContinuityInvariant:
    """P3 (goal_20260803174119_0): the active-plan slot must never remain
    occupied by a finished plan nor empty after a plan completion.

    Regression (2026-08-03): update_plan_step marked steps complete but never
    transitioned the plan itself to "complete", so a 3/3-steps plan stayed
    "active" forever.  The new_plan guard (has_active) then read the stale
    in-memory timeline and rejected every same-response replacement plan —
    the active-plan slot was permanently occupied by a done plan and plan
    reinstantiation never happened.  Fix: update_plan_step auto-completes a
    plan whose steps are all complete, and _apply_insights reloads the
    timeline after a plan_action so the has_active guard sees fresh state.
    """

    def _make_state(self, wm=None) -> Dict[str, Any]:
        return {
            "daemon_state": {"tick_count": 5, "last_action_output": ""},
            "world_model": wm,
            "timeline": {
                "version": 1,
                "past": {"events": []},
                "present": {},
                "future": {"plans": []},
            },
            "self_model": {
                "identity": {"name": "test", "role": "test"},
                "state": {},
                "capabilities": {"strengths": [], "weaknesses": [], "unknown_areas": []},
                "commitments": {},
            },
            "orientation": {"vision": "Test", "phase": "test"},
        }

    def test_last_step_complete_accepts_same_response_new_plan(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Completing the final step frees the slot and a same-response
        new_plan is accepted (immediate plan reinstantiation)."""
        from data_layer import create_plan, get_active_plan, load_timeline_dict

        td = evolve_env["module"]
        steps = [
            {"description": "Step 1", "verification": "V1"},
            {"description": "Step 2", "verification": "V2"},
        ]
        old_pid = create_plan("Old plan", steps)
        # Mark step_1 complete so the plan is at 1/2 → still active
        from data_layer import update_plan_step

        update_plan_step(old_pid, "step_1", "complete")

        state = self._make_state()
        # Reload the timeline so _apply_insights starts from disk truth
        state["timeline"] = load_timeline_dict()

        result = {
            "action": None,
            "fallback": True,
            "insight": "plan completes",
            "focus_next": "continue",
            "confidence": 0.5,
            "reasoning": "test",
            "event_to_record": None,
            "outcome_to_record": None,
            "commitment": None,
            "prediction": None,
            "session_record": None,
            # Complete the LAST step in the same response that proposes the
            # replacement plan.
            "plan_action": {"step_id": "step_2", "new_status": "complete", "note": ""},
            "new_plan": {
                "goal": "Replacement plan",
                "steps": [{"description": "Run calibration check", "verification": "calibration report generated"}],
            },
            "new_goal": None,
            "goal_action": None,
            "search_query": None,
            "episodic_record": None,
            "semantic_record": None,
            "procedural_record": None,
            "self_model_update": {"weakness": None, "unknown": None, "new_commitment": None},
            "next_gap": None,
        }

        import subprocess

        original_run = subprocess.run

        def _mock_run(*a, **kw):
            return type("_R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        try:
            subprocess.run = _mock_run
            updates = td._apply_insights(result, state)
        finally:
            subprocess.run = original_run

        # Old plan: complete, no longer active (slot freed).
        assert get_active_plan() is not None, "replacement plan must be active"
        active = get_active_plan()
        assert active["goal"] == "Replacement plan"
        # The old plan must be persisted as complete in the timeline.
        data = load_timeline_dict()
        old = next(p for p in data["future"]["plans"] if p["goal"] == "Old plan")
        assert old["status"] == "complete"
        assert old["steps"][1]["status"] == "complete"

    def test_partial_completion_keeps_active_and_blocks_new_plan(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A plan with pending steps stays active; a new_plan is rejected
        (the slot is legitimately occupied)."""
        from data_layer import create_plan, get_active_plan, load_timeline_dict

        td = evolve_env["module"]
        steps = [
            {"description": "Step 1", "verification": "V1"},
            {"description": "Step 2", "verification": "V2"},
        ]
        old_pid = create_plan("Old plan", steps)

        state = self._make_state()
        state["timeline"] = load_timeline_dict()

        result = {
            "action": None,
            "fallback": True,
            "insight": "partial",
            "focus_next": "continue",
            "confidence": 0.5,
            "reasoning": "test",
            "event_to_record": None,
            "outcome_to_record": None,
            "commitment": None,
            "prediction": None,
            "session_record": None,
            # Only step_1 completes; step_2 pending → plan stays active.
            "plan_action": {"step_id": "step_1", "new_status": "complete", "note": ""},
            "new_plan": {
                "goal": "Replacement plan",
                "steps": [{"description": "Run calibration check", "verification": "calibration report generated"}],
            },
            "new_goal": None,
            "goal_action": None,
            "search_query": None,
            "episodic_record": None,
            "semantic_record": None,
            "procedural_record": None,
            "self_model_update": {"weakness": None, "unknown": None, "new_commitment": None},
            "next_gap": None,
        }

        import subprocess

        original_run = subprocess.run

        def _mock_run(*a, **kw):
            return type("_R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        try:
            subprocess.run = _mock_run
            td._apply_insights(result, state)
        finally:
            subprocess.run = original_run

        active = get_active_plan()
        assert active is not None
        assert active["id"] == old_pid, "pending plan must remain active"
        assert active["goal"] == "Old plan"
        assert active["progress"] == "1/2 steps"


class TestPlanContinuityMonitor:
    """``_record_plan_continuity_observation`` per-cycle invariant checks.

    P3 goal_20260803174119_0 closing evidence (plan_20260803184432 step_2):
    every successful daemon cycle appends an observation to
    ``<evolve>/plan_continuity_monitor.json``; 20 consecutive clean
    observations complete the verification window.  Checks:

      V1 — no active plan may have all steps complete (auto-complete trap).
      V2 — after a plan completion, the slot must not stay empty for two
           consecutive cycles (replacement never created).
    """

    @staticmethod
    def _plan(pid: str, status: str, steps: list) -> Dict[str, Any]:
        return {
            "id": pid,
            "goal": "Goal " + pid,
            "steps": steps,
            "status": status,
            "progress": f"{sum(1 for s in steps if s.get('status') == 'complete')}/{len(steps)} steps",
        }

    @staticmethod
    def _timeline(plans: list) -> Dict[str, Any]:
        return {
            "version": 1,
            "past": {"events": []},
            "present": {},
            "future": {"plans": plans},
        }

    def test_clean_cycle_with_active_plan_increments_streak(
        self, evolve_env: Dict
    ) -> None:
        td = evolve_env["module"]
        tl = self._timeline([
            self._plan("p1", "active", [
                {"id": "step_1", "status": "complete"},
                {"id": "step_2", "status": "pending"},
            ]),
        ])
        log = td._record_plan_continuity_observation(
            101, {"plan_action": None, "new_plan": None}, "p1", tl
        )
        assert log["consecutive_clean"] == 1
        assert log["window_complete"] is False
        assert log["pending_reinstantiation"] is False
        obs = log["observations"][-1]
        assert obs["ok"] is True
        assert obs["violations"] == []
        assert obs["active_plan_after"] == "p1"
        # Log persisted at the expected path.
        log_path = evolve_env["paths"]["evolve_dir"] / "plan_continuity_monitor.json"
        assert log_path.exists()

    def test_finished_active_plan_is_violation(
        self, evolve_env: Dict
    ) -> None:
        """V1: an active plan whose steps are all complete is a violation."""
        td = evolve_env["module"]
        tl = self._timeline([
            self._plan("p1", "active", [
                {"id": "step_1", "status": "complete"},
                {"id": "step_2", "status": "complete"},
            ]),
        ])
        log = td._record_plan_continuity_observation(
            102, {"plan_action": None, "new_plan": None}, "p1", tl
        )
        assert log["consecutive_clean"] == 0
        assert log["window_complete"] is False
        assert log["last_violation"]["tick"] == 102
        obs = log["observations"][-1]
        assert obs["ok"] is False
        assert any("all 2 steps complete" in v for v in obs["violations"])

    def test_completion_without_replacement_pending_then_violation(
        self, evolve_env: Dict
    ) -> None:
        """V2: completion frees the slot; one empty cycle is allowed for
        reinstantiation, a second consecutive empty cycle is a violation."""
        td = evolve_env["module"]
        # Cycle N: the active plan is completed (plan_action) and no
        # replacement is issued in the same response.
        tl_n = self._timeline([
            self._plan("p1", "complete", [
                {"id": "step_1", "status": "complete"},
                {"id": "step_2", "status": "complete"},
            ]),
        ])
        log = td._record_plan_continuity_observation(
            103,
            {
                "plan_action": {"step_id": "step_2", "new_status": "complete", "note": ""},
                "new_plan": None,
            },
            "p1",
            tl_n,
        )
        obs_n = log["observations"][-1]
        assert obs_n["ok"] is True, obs_n["violations"]  # deferred, not yet a violation
        assert obs_n["completed_this_cycle"] is True
        assert obs_n["slot_empty"] is True
        assert log["pending_reinstantiation"] is True
        assert log["consecutive_clean"] == 1

        # Cycle N+1: still no active plan and no replacement → violation.
        tl_n1 = self._timeline([])
        log = td._record_plan_continuity_observation(
            104, {"plan_action": None, "new_plan": None}, None, tl_n1
        )
        obs_n1 = log["observations"][-1]
        assert obs_n1["ok"] is False
        assert any("2+ consecutive cycles" in v for v in obs_n1["violations"])
        assert log["consecutive_clean"] == 0

    def test_completion_with_same_response_new_plan_is_clean(
        self, evolve_env: Dict
    ) -> None:
        """Same-response reinstantiation: completion + new_plan keeps the
        slot filled — clean, no pending state carried forward."""
        td = evolve_env["module"]
        tl = self._timeline([
            self._plan("p2", "active", [
                {"id": "step_1", "status": "pending"},
            ]),
        ])
        log = td._record_plan_continuity_observation(
            105,
            {
                "plan_action": {"step_id": "step_1", "new_status": "complete", "note": ""},
                "new_plan": {"goal": "Replacement", "steps": [{"description": "S", "verification": "V"}]},
            },
            "p1",
            tl,
        )
        obs = log["observations"][-1]
        assert obs["ok"] is True
        assert obs["new_plan_issued"] is True
        assert log["pending_reinstantiation"] is False
        assert log["consecutive_clean"] == 1

    def test_fallback_empty_slot_is_not_a_violation(
        self, evolve_env: Dict
    ) -> None:
        """Outage-aware V2: during a local-analysis fallback cycle the daemon
        cannot emit new_plan (hardcoded None), so an empty slot left by a
        plan completion is expected — it must NOT reset the clean streak.

        Without this, a plan completed by _local_plan_maintenance mid-outage
        would flag a violation on the next fallback cycle (2026-08-03:
        consecutive_fallback_cycles reached 5), punishing the daemon for
        LLM-provider downtime and blocking the 20-cycle window.
        """
        td = evolve_env["module"]
        # Cycle N: active plan completed; parsed carries fallback=True (the
        # local-analysis response shape).
        tl_n = self._timeline([
            self._plan("p1", "complete", [
                {"id": "step_1", "status": "complete"},
                {"id": "step_2", "status": "complete"},
            ]),
        ])
        parsed_fallback = {
            "fallback": True,
            "plan_action": {"step_id": "step_2", "new_status": "complete", "note": ""},
            "new_plan": None,
        }
        log = td._record_plan_continuity_observation(110, parsed_fallback, "p1", tl_n)
        obs_n = log["observations"][-1]
        assert obs_n["ok"] is True, obs_n["violations"]
        assert obs_n["llm_available"] is False
        assert log["pending_reinstantiation"] is True
        assert log["consecutive_clean"] == 1

        # Cycle N+1: still in fallback, slot still empty → still NO violation.
        log = td._record_plan_continuity_observation(
            111, {"fallback": True, "plan_action": None, "new_plan": None}, None,
            self._timeline([]),
        )
        obs_n1 = log["observations"][-1]
        assert obs_n1["ok"] is True, obs_n1["violations"]
        assert obs_n1["llm_available"] is False
        assert log["consecutive_clean"] == 2

    def test_llm_available_empty_slot_is_still_a_violation(
        self, evolve_env: Dict
    ) -> None:
        """The outage exemption must NOT weaken the invariant when the LLM
        is available: an LLM-backed cycle that leaves the slot empty after a
        completion is still a V2 violation (the LLM had capacity to create a
        replacement and did not)."""
        td = evolve_env["module"]
        # Cycle N: LLM-backed completion with no replacement → pending.
        tl_n = self._timeline([
            self._plan("p1", "complete", [
                {"id": "step_1", "status": "complete"},
            ]),
        ])
        parsed_llm = {
            "plan_action": {"step_id": "step_1", "new_status": "complete", "note": ""},
            "new_plan": None,
        }
        log = td._record_plan_continuity_observation(120, parsed_llm, "p1", tl_n)
        assert log["observations"][-1]["ok"] is True  # deferred
        assert log["pending_reinstantiation"] is True

        # Cycle N+1: LLM available, slot still empty → violation.
        log = td._record_plan_continuity_observation(
            121, {"plan_action": None, "new_plan": None}, None, self._timeline([])
        )
        obs = log["observations"][-1]
        assert obs["ok"] is False
        assert obs["llm_available"] is True
        assert any("2+ consecutive cycles" in v for v in obs["violations"])
        assert log["consecutive_clean"] == 0

    def test_window_completes_after_20_clean_observations(
        self, evolve_env: Dict
    ) -> None:
        td = evolve_env["module"]
        tl = self._timeline([
            self._plan("p1", "active", [
                {"id": "step_1", "status": "complete"},
                {"id": "step_2", "status": "pending"},
            ]),
        ])
        parsed = {"plan_action": None, "new_plan": None}
        for i in range(20):
            log = td._record_plan_continuity_observation(200 + i, parsed, "p1", tl)
            if i < 19:
                assert log["window_complete"] is False, f"premature at {i}"
            else:
                assert log["window_complete"] is True
        assert log["consecutive_clean"] == 20
        assert len(log["observations"]) == 20

    def test_observation_log_capped_at_40(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        tl = self._timeline([
            self._plan("p1", "active", [
                {"id": "step_1", "status": "pending"},
            ]),
        ])
        parsed = {"plan_action": None, "new_plan": None}
        for i in range(45):
            td._record_plan_continuity_observation(300 + i, parsed, "p1", tl)
        log = td._record_plan_continuity_observation(345, parsed, "p1", tl)
        assert len(log["observations"]) == 40
        assert log["observations"][0]["tick"] == 306


class TestCleanPatternCyclesCounter:
    """daemon_state.clean_pattern_cycles tracks consecutive pattern-free cycles.

    The llm_call error-reduction plan's closing gate is "no llm_call
    discrepancy pattern for 5 consecutive cycles"; without a counter that
    verification was a manual JSON read every cycle.  ``_run_cycle_body``
    step 7 now increments ``clean_pattern_cycles`` when the world model's
    live discrepancy_patterns list is empty and resets it to 0 on the
    first cycle a pattern appears (both llm-backed and local-analysis
    paths converge on the same code).
    """

    _VALID_JSON = json.dumps({
        "insight": "test insight",
        "self_model_update": [],
        "timeline_update": {"events": []},
        "goals_update": [],
        "search_query": "",
        "action": None,
        "prediction": None,
    })

    async def _run_cycle(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> tuple:
        td = evolve_env["module"]

        async def _fake_llm(messages: list, task: str = "thinking") -> str:
            return self._VALID_JSON

        monkeypatch.setattr(td, "_call_llm", _fake_llm)

        result = {"status": "ok", "tick_duration": 0, "insight": None, "error": None}
        ds = td.load_daemon_state()
        ds.setdefault(
            "cycle_stats",
            {"total": 0, "ok": 0, "error": 0, "parse_error": 0,
             "avg_duration": 0.0, "max_duration": 0.0},
        )
        # The LLM-backed path executes an action via subprocess; mock it
        # (same pattern as TestCycleBodyEmptyLlmResponse) for hermeticity.
        import subprocess

        original_run = subprocess.run

        def _mock_run(*a, **kw):
            return type("_R", (), {"returncode": 0, "stdout": "mocked\n", "stderr": ""})()

        try:
            subprocess.run = _mock_run
            result = await td._run_cycle_body(result, ds)
        finally:
            subprocess.run = original_run

        return result, ds

    def test_increments_on_clean_cycle(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cycle ending with zero discrepancy patterns bumps the counter."""
        td = evolve_env["module"]
        wm = td.load_world_model()
        assert wm.get_discrepancy_patterns() == []  # fresh store: no patterns

        _, ds = asyncio.run(self._run_cycle(evolve_env, monkeypatch))

        assert ds["clean_pattern_cycles"] == 1, ds
        assert ds["last_output"]["clean_pattern_cycles"] == 1, ds["last_output"]

    def test_resets_when_pattern_appears(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cycle where a discrepancy pattern exists resets the counter to 0."""
        td = evolve_env["module"]

        # Seed a real discrepancy: 3 identical unhedged confident-wrong
        # llm_call triples (recurrence >= 3 bypasses the ratio-decay
        # suppression, so the miner must surface a pattern).
        wm = td.load_world_model()
        for _ in range(3):
            tid = wm.record_action(
                "llm_call",
                "LLM thinking call (adaptive retry policy)",
                expected_outcome="success: LLM responds within 2 attempt(s) × 90s budget",
                expected_source="daemon",
                prediction_confidence=0.5,
            )
            wm.complete_action(tid, "failed: timeout after 2 attempt(s)")
        wm.save()
        patterns = wm.get_discrepancy_patterns()
        assert any(p.get("action_type") == "llm_call" for p in patterns), patterns

        # Pre-seed the counter at 4 so the reset is observable.
        ds0 = td.load_daemon_state()
        ds0["clean_pattern_cycles"] = 4
        td.save_daemon_state(ds0)

        _, ds = asyncio.run(self._run_cycle(evolve_env, monkeypatch))

        assert ds["clean_pattern_cycles"] == 0, ds

    def test_persisted_across_save(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The counter survives a daemon_state round-trip."""
        td = evolve_env["module"]
        _, ds = asyncio.run(self._run_cycle(evolve_env, monkeypatch))
        td.save_daemon_state(ds)

        reloaded = td.load_daemon_state()
        assert reloaded["clean_pattern_cycles"] == 1, reloaded

    def test_non_llm_pattern_does_not_reset(
        self, evolve_env: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A shell/api_call discrepancy pattern must not reset the llm_call
        verification streak.

        The counter exists to gate the llm_call error-reduction plan ("no
        llm_call discrepancy pattern for 5 consecutive cycles"); an
        unrelated pattern (e.g. a flaky shell command) would otherwise
        keep resetting the streak and block the plan's closing gate
        forever.  Scope the reset to action_type == "llm_call".
        """
        td = evolve_env["module"]

        # Seed a real NON-llm_call discrepancy: 3 identical unhedged shell
        # failures (recurrence >= 3 surfaces a shell pattern).
        wm = td.load_world_model()
        for _ in range(3):
            tid = wm.record_action(
                "shell",
                "State check: verify shell health",
                expected_outcome="exit=0: command succeeds",
                expected_source="daemon",
                prediction_confidence=0.5,
            )
            wm.complete_action(tid, "exit=1: command failed")
        wm.save()
        patterns = wm.get_discrepancy_patterns()
        assert any(p.get("action_type") == "shell" for p in patterns), patterns
        assert not any(p.get("action_type") == "llm_call" for p in patterns)

        # Pre-seed the counter at 4; the shell pattern must NOT reset it.
        ds0 = td.load_daemon_state()
        ds0["clean_pattern_cycles"] = 4
        td.save_daemon_state(ds0)

        _, ds = asyncio.run(self._run_cycle(evolve_env, monkeypatch))

        assert ds["clean_pattern_cycles"] == 5, ds

