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
                "capabilities": {"available_tools": ["shell", "write_file"], "weaknesses": [], "unknown_areas": []},
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
        state["timeline"]["future"]["goals"] = [
            "Achieve 90% test coverage",
            "Implement Gap 8",
        ]
        prompt = td._build_thinking_prompt(state)
        assert "90% test coverage" in prompt or "Achieve" in prompt

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
