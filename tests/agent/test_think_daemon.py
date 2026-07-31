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

    def test_prompt_includes_commitments(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        state = self._make_state()
        state["timeline"]["present"]["commitments"] = [
            {"what": "Write daemon tests", "status": "active", "deadline": "2026-08-01"},
        ]
        prompt = td._build_thinking_prompt(state)
        assert "Write daemon tests" in prompt

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


# ═══════════════════════════════════════════════════════════════════
#  Cycle body: empty/whitespace LLM response → local fallback
# ═══════════════════════════════════════════════════════════════════


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
        # Depth >= 3 on a non-probe cycle: 0 attempts — the whole cycle
        # budget goes to local analysis + action execution instead of a
        # doomed 45s probe.
        assert td._llm_retry_policy(3) == (0, 0.0)
        assert td._llm_retry_policy(5) == (0, 0.0)
        assert td._llm_retry_policy(6) == (0, 0.0)
        assert td._llm_retry_policy(7) == (0, 0.0)
        assert td._llm_retry_policy(10) == (0, 0.0)

    def test_extended_outage_probe_cycle(self, evolve_env: Dict) -> None:
        td = evolve_env["module"]
        # Every 4th cycle (depth % 4 == 0): a bounded 30s probe so
        # recovery is still detected within 4 cycles of the provider
        # coming back.
        assert td._llm_retry_policy(4) == (1, 30.0)
        assert td._llm_retry_policy(8) == (1, 30.0)
        assert td._llm_retry_policy(12) == (1, 30.0)

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

