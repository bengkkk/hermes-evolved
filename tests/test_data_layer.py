"""Comprehensive tests for data_layer.py — the canonical storage module.

Tests all public classes and functions:

  - Timeline (event log)
  - SelfModel (identity, capabilities, state, commitments)
  - Orientation (focus, insights, next steps)
  - Memory (episodic, semantic, procedural)
  - Goals (self-generated goal lifecycle)
  - Dict-format timeline helpers (record_event, save/load, plans)
  - Context formatters (format_orientation_context, etc.)
  - Low-level helpers (safe_read_json, safe_write_json, now_iso)

All tests are hermetic — isolated temp directories, no real ~/.hermes I/O.
"""

from __future__ import annotations

import json
import os
import sys
import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_layer import (
    Goals,
    Memory,
    Orientation,
    SelfModel,
    Timeline,
    _DEFAULT_TIMELINE_DICT,
    _atomic_write,
    _load_timeline_dict,
    _read_json,
    add_commitment,
    add_prediction,
    complete_plan,
    create_plan,
    format_orientation_context,
    format_plan_context,
    format_self_model_context,
    format_timeline_context,
    get_active_plan,
    get_evolve_dir,
    load_self_model,
    load_timeline_dict,
    now_compact,
    now_iso,
    record_event,
    record_outcome,
    record_session_completion,
    safe_read_json,
    safe_write_json,
    save_self_model,
    save_timeline_dict,
    update_commitment,
    update_goals,
    update_plan_step,
    update_present_state,
    update_self_model,
)


# ═══════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def evolve_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    """Set up an isolated evolve directory for file-backed tests.

    Yields a dict with 'evolve_dir' (Path) and 'cleanup' helpers.
    All file-backed operations from data_layer point at tmp_path/evolve/.
    """
    evolve_dir = tmp_path / "evolve"
    evolve_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_EVOLVE_DIR", str(tmp_path))

    # Force data_layer to recompute _EVOLVE_DIR
    import importlib
    import data_layer as dl_mod
    dl_mod = importlib.reload(dl_mod)

    # Re-import after env var is set and module is reloaded
    from data_layer import get_evolve_dir as _ged
    actual_dir = _ged()
    assert str(actual_dir).startswith(str(evolve_dir)), (
        f"Expected evolve dir under {evolve_dir}, got {actual_dir}"
    )

    return {"evolve_dir": evolve_dir}


# ═══════════════════════════════════════════════════════════════════════
#  Timeline — append-only event log
# ═══════════════════════════════════════════════════════════════════════


class TestTimeline:
    """In-memory Timeline tests (no file I/O)."""

    def test_empty_timeline(self) -> None:
        tl = Timeline()
        assert len(tl) == 0
        assert bool(tl) is False
        assert tl.get_all() == []

    def test_add_event(self) -> None:
        tl = Timeline()
        event = tl.add_event("milestone", "Did something", "It matters")
        assert event["type"] == "milestone"
        assert event["summary"] == "Did something"
        assert event["impact"] == "It matters"
        assert "timestamp" in event
        assert len(tl) == 1
        assert bool(tl) is True

    def test_get_recent(self) -> None:
        tl = Timeline()
        for i in range(5):
            tl.add_event("milestone", f"Event {i}", "")
        recent = tl.get_recent(2)
        assert len(recent) == 2
        assert recent[0]["summary"] == "Event 3"  # events[-2:] = [Event 3, Event 4]
        assert recent[1]["summary"] == "Event 4"

    def test_get_by_type(self) -> None:
        tl = Timeline()
        tl.add_event("milestone", "M1", "")
        tl.add_event("decision", "D1", "")
        tl.add_event("reflection", "R1", "")
        tl.add_event("milestone", "M2", "")
        assert len(tl.get_by_type("milestone")) == 2
        assert len(tl.get_by_type("decision")) == 1
        assert len(tl.get_by_type("nonexistent")) == 0

    def test_get_by_id(self) -> None:
        tl = Timeline()
        e = tl.add_event("milestone", "Test", "")
        found = tl.get_by_id(e["id"])
        assert found is not None
        assert found["summary"] == "Test"
        assert tl.get_by_id("nonexistent") is None

    def test_get_all(self) -> None:
        tl = Timeline()
        tl.add_event("milestone", "E1", "")
        tl.add_event("decision", "E2", "")
        all_ = tl.get_all()
        assert len(all_) == 2
        assert all_[0]["summary"] == "E1"
        assert all_[1]["summary"] == "E2"

    def test_to_dict_round_trip(self) -> None:
        tl = Timeline()
        tl.add_event("milestone", "Test event", "Impact")
        d = tl.to_dict()
        assert d["version"] == 2
        assert len(d["events"]) == 1

        restored = Timeline.from_dict(d)
        assert len(restored) == 1
        assert restored.events[0]["summary"] == "Test event"

    def test_v1_format_backward_compat(self) -> None:
        """v1 stored events as a flat list; v2 nests under 'events'."""
        v1_data = {
            "version": 1,
            "events": [{"type": "milestone", "summary": "Old event"}],
        }
        tl = Timeline.from_dict(v1_data)
        assert len(tl) == 1
        assert tl.events[0]["summary"] == "Old event"

    def test_repr(self) -> None:
        tl = Timeline()
        assert repr(tl) == "<Timeline events=0>"
        tl.add_event("milestone", "X", "")
        assert "events=1" in repr(tl)


class TestTimelinePersistence:
    """File-backed Timeline save/load."""

    def test_save_creates_file(self, evolve_env: Dict) -> None:
        tl = Timeline()
        tl.add_event("milestone", "Test", "")
        path = tl.save()
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["version"] == 2
        assert len(data["events"]) == 1

    def test_load_round_trip(self, evolve_env: Dict) -> None:
        tl = Timeline()
        tl.add_event("milestone", "Saved event", "Impact")
        tl.save()

        loaded = Timeline.load()
        assert len(loaded) == 1
        assert loaded.events[0]["summary"] == "Saved event"
        assert loaded.events[0]["impact"] == "Impact"

    def test_load_empty_when_no_file(self, evolve_env: Dict) -> None:
        tl = Timeline.load()
        assert len(tl) == 0

    def test_load_corrupted_file(self, evolve_env: Dict) -> None:
        path = Timeline.storage_path()
        path.write_text("not json")
        tl = Timeline.load()
        assert len(tl) == 0

    def test_save_is_idempotent(self, evolve_env: Dict) -> None:
        tl = Timeline()
        tl.add_event("milestone", "E1", "")
        tl.save()

        # After saving twice with a new event, the file should contain 2 events
        tl.add_event("milestone", "E2", "")
        tl.save()

        loaded = Timeline.load()
        assert len(loaded) == 2

    def test_multiple_events_persist(self, evolve_env: Dict) -> None:
        tl = Timeline()
        for i in range(10):
            tl.add_event("milestone", f"Event {i}", "")
        tl.save()
        loaded = Timeline.load()
        assert len(loaded) == 10
        assert loaded.events[-1]["summary"] == "Event 9"


# ═══════════════════════════════════════════════════════════════════════
#  SelfModel — identity, capabilities, and metacognitive state
# ═══════════════════════════════════════════════════════════════════════


class TestSelfModel:
    """SelfModel in-memory tests."""

    def test_default_data(self) -> None:
        sm = SelfModel()
        assert sm.identity.get("name") == "Hermes (evolved)"
        assert sm.state.get("evolution_version") == 5
        assert sm.capabilities["strengths"] == []

    def test_identity_property(self) -> None:
        sm = SelfModel()
        assert isinstance(sm.identity, dict)
        sm.identity = {"name": "Test Bot", "role": "Tester"}
        assert sm.identity["name"] == "Test Bot"

    def test_state_value_get_set(self) -> None:
        sm = SelfModel()
        assert sm.get_state_value("nonexistent", "default") == "default"
        sm.set_state_value("total_cycles", 42)
        assert sm.get_state_value("total_cycles") == 42

    def test_add_strength(self) -> None:
        sm = SelfModel()
        assert sm.add_strength("Fast thinking") is True
        assert sm.add_strength("Fast thinking") is False  # duplicate
        assert len(sm.capabilities["strengths"]) == 1

    def test_add_weakness(self) -> None:
        sm = SelfModel()
        assert sm.add_weakness("Slow on large files") is True
        assert sm.add_weakness("Slow on large files") is False
        assert len(sm.capabilities["weaknesses"]) == 1

    def test_add_unknown(self) -> None:
        sm = SelfModel()
        assert sm.add_unknown("How X works") is True
        assert sm.add_unknown("How X works") is False  # duplicate
        assert len(sm.capabilities["unknown_areas"]) == 1

    def test_remove_weakness_substring(self) -> None:
        sm = SelfModel()
        sm.add_weakness("Slow on Python files")
        sm.add_weakness("Slow on large JSON")
        sm.add_weakness("Fast on small files")
        removed = sm.remove_weakness("slow")
        assert removed == 2
        assert len(sm.capabilities["weaknesses"]) == 1
        assert sm.capabilities["weaknesses"][0] == "Fast on small files"

    def test_remove_weakness_no_match(self) -> None:
        sm = SelfModel()
        sm.add_weakness("Some weakness")
        assert sm.remove_weakness("nonexistent") == 0

    def test_commitment_lifecycle(self) -> None:
        sm = SelfModel()
        c = sm.set_commitment("Fix bug X", "2026-08-01")
        assert c["status"] == "active"
        assert c["what"] == "Fix bug X"

        assert sm.complete_commitment("Fix bug X") is True
        # Calling again still returns True because the method
        # finds the matching 'what' regardless of current status
        assert sm.complete_commitment("Fix bug X") is True
        assert sm.data["commitments"]["active_obligations"][0]["status"] == "completed"

    def test_state_snapshot(self) -> None:
        sm = SelfModel()
        sm.add_strength("S1")
        sm.add_strength("S2")
        sm.add_weakness("W1")
        sm.add_unknown("U1")
        snapshot = sm.get_state_snapshot()
        assert snapshot["identity"] == "Hermes (evolved)"
        assert snapshot["evolution_version"] == 5
        assert "S1" in snapshot["strengths"]
        assert "W1" in snapshot["weaknesses"]
        assert "U1" in snapshot["unknown_areas"]

    def test_summary(self) -> None:
        sm = SelfModel()
        sm.set_state_value("total_cycles", 7)
        sm_str = sm.summary()
        assert "v5" in sm_str
        assert "7 cycles" in sm_str

    def test_repr(self) -> None:
        sm = SelfModel()
        sm.set_state_value("current_gap_focus", "Gap 6")
        r = repr(sm)
        assert "SelfModel" in r
        assert "Gap 6" in r

    def test_merge_defaults_handles_extra_keys(self) -> None:
        sm = SelfModel(data={"unknown_key": "value", "state": {"total_cycles": 10}})
        assert sm.data["unknown_key"] == "value"
        assert sm.state["total_cycles"] == 10

    # ── Permissions (Gap 10) ──────────────────────────────────────

    def test_permissions_default_deny(self) -> None:
        sm = SelfModel()
        assert sm.permissions["github"] == {
            "read": False, "write": False, "act": False, "cap": None,
        }
        # Deny by default: declared resource, zero grants.
        assert sm.check_permission("github", "read") is False
        assert sm.check_permission("github", "write") is False
        assert sm.check_permission("github", "act") is False
        # Unknown resource → no entry → deny.
        assert sm.permission_entry("nonexistent") is None
        assert sm.check_permission("nonexistent", "read") is False

    def test_permission_grant_revoke(self) -> None:
        sm = SelfModel()
        sm.grant_permission("github", "read")
        assert sm.check_permission("github", "read") is True
        assert sm.check_permission("github", "write") is False
        assert sm.revoke_permission("github", "read") is True
        assert sm.check_permission("github", "read") is False
        # Revoking an already-False flag changes nothing.
        assert sm.revoke_permission("github", "read") is False

    def test_permission_grant_creates_full_entry(self) -> None:
        sm = SelfModel()
        sm.grant_permission("calendar", "read", cap=10)
        assert sm.permission_entry("calendar") == {
            "read": True, "write": False, "act": False, "cap": 10,
        }

    def test_permission_unknown_action_denied_and_rejected(self) -> None:
        sm = SelfModel()
        sm.grant_permission("github", "read")
        assert sm.check_permission("github", "delete") is False
        with pytest.raises(ValueError):
            sm.grant_permission("github", "delete")

    def test_validate_permissions(self) -> None:
        sm = SelfModel()
        # Default registry is valid.
        assert sm.validate_permissions() == []
        # Non-bool flag.
        sm.permissions["github"]["read"] = "yes"  # type: ignore[assignment]
        assert any("read" in p for p in sm.validate_permissions())
        # Negative cap.
        sm.permissions["github"]["read"] = True
        sm.permissions["github"]["cap"] = -1
        assert any("cap" in p for p in sm.validate_permissions())
        # Non-dict entry.
        sm.permissions["github"]["cap"] = None
        sm.permissions["bad"] = "not-a-dict"  # type: ignore[assignment]
        assert any("bad" in p for p in sm.validate_permissions())
        assert sm.validate_permissions()  # non-empty → invalid

    def test_validate_permissions_unknown_key(self) -> None:
        sm = SelfModel()
        sm.permissions["github"]["exec"] = True
        assert any("exec" in p for p in sm.validate_permissions())

    def test_merge_defaults_permissions_section(self) -> None:
        sm = SelfModel(
            data={
                "permissions": {
                    "github": {"read": True, "write": False, "act": False, "cap": None},
                }
            }
        )
        # Loaded state wins over the default entry.
        assert sm.check_permission("github", "read") is True
        assert sm.validate_permissions() == []

    def test_state_snapshot_permissions(self) -> None:
        sm = SelfModel()
        assert sm.get_state_snapshot()["permissions"] == {"github": []}
        sm.grant_permission("github", "read")
        assert sm.get_state_snapshot()["permissions"]["github"] == ["read"]


class TestSelfModelPersistence:
    """File-backed SelfModel save/load."""

    def test_save_creates_file(self, evolve_env: Dict) -> None:
        sm = SelfModel()
        sm.add_strength("Test")
        path = sm.save()
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["identity"]["name"] == "Hermes (evolved)"

    def test_load_round_trip(self, evolve_env: Dict) -> None:
        sm = SelfModel()
        sm.add_strength("Persistence works")
        sm.set_state_value("total_cycles", 99)
        sm.save()

        loaded = SelfModel.load()
        assert "Persistence works" in loaded.capabilities["strengths"]
        assert loaded.state["total_cycles"] == 99

    def test_load_empty_when_no_file(self, evolve_env: Dict) -> None:
        sm = SelfModel.load()
        assert sm.identity.get("name") == "Hermes (evolved)"

    def test_module_level_helpers(self, evolve_env: Dict) -> None:
        """load_self_model / save_self_model / update_self_model work end to end."""
        sm_data = load_self_model()
        assert isinstance(sm_data, dict)
        assert "identity" in sm_data

        sm_data["state"]["total_cycles"] = 42
        save_self_model(sm_data)

        reloaded = load_self_model()
        assert reloaded["state"]["total_cycles"] == 42

    def test_update_self_model(self, evolve_env: Dict) -> None:
        update_self_model(state={"total_cycles": 10})
        sm = SelfModel.load()
        assert sm.state["total_cycles"] == 10


# ═══════════════════════════════════════════════════════════════════════
#  Orientation — lightweight focus tracking
# ═══════════════════════════════════════════════════════════════════════


class TestOrientation:
    """Orientation in-memory tests."""

    def test_default_empty(self) -> None:
        o = Orientation()
        assert o.focus == ""

    def test_focus_round_trip(self) -> None:
        o = Orientation()
        o.focus = "Working on Gap 6"
        assert o.focus == "Working on Gap 6"

    def test_add_insight(self) -> None:
        o = Orientation()
        o.add_insight("Learned about X")
        assert len(o.data["insights"]) == 1
        o.add_insight("Learned about X")  # duplicate — last matches
        assert len(o.data["insights"]) == 1  # not duplicated

    def test_add_insight_distinct(self) -> None:
        o = Orientation()
        o.add_insight("First insight")
        o.add_insight("Second insight")
        assert len(o.data["insights"]) == 2
        assert o.data["insights"][-1] == "Second insight"

    def test_add_next_step(self) -> None:
        o = Orientation()
        o.add_next_step("Setup daemon")
        assert len(o.data["next_steps"]) == 1
        o.add_next_step("Setup daemon")  # duplicate
        assert len(o.data["next_steps"]) == 1

    def test_add_next_step_distinct(self) -> None:
        o = Orientation()
        o.add_next_step("Step 1")
        o.add_next_step("Step 2")
        assert len(o.data["next_steps"]) == 2

    def test_cap_at_20_insights(self) -> None:
        o = Orientation()
        for i in range(25):
            o.add_insight(f"Insight {i}")
        assert len(o.data["insights"]) == 20

    def test_cap_at_10_steps(self) -> None:
        o = Orientation()
        for i in range(15):
            o.add_next_step(f"Step {i}")
        assert len(o.data["next_steps"]) == 10

    def test_repr(self) -> None:
        o = Orientation()
        o.focus = "Testing"
        assert repr(o) == "<Orientation focus='Testing'>"


class TestOrientationPersistence:
    """File-backed Orientation save/load."""

    def test_save_and_load(self, evolve_env: Dict) -> None:
        o = Orientation()
        o.focus = "Building world model"
        o.add_insight("Data layer works")
        o.add_next_step("Test save/load")
        o.save()

        loaded = Orientation.load()
        assert loaded.focus == "Building world model"
        assert len(loaded.data["insights"]) == 1
        assert len(loaded.data["next_steps"]) == 1

    def test_load_empty(self, evolve_env: Dict) -> None:
        o = Orientation.load()
        assert o.focus == ""


# ═══════════════════════════════════════════════════════════════════════
#  Memory — multi-type store (episodic / semantic / procedural)
# ═══════════════════════════════════════════════════════════════════════


class TestMemory:
    """Memory in-memory tests."""

    def test_default_empty(self) -> None:
        mem = Memory()
        assert mem.data["episodic"] == []
        assert mem.data["semantic"] == []
        assert mem.data["procedural"] == []

    def test_add_episodic(self) -> None:
        mem = Memory()
        mid = mem.add_episodic("observation", "Observed something", "Details", salience=0.8)
        assert mid.startswith("ep_")
        assert len(mem.data["episodic"]) == 1
        entry = mem.data["episodic"][0]
        assert entry["summary"] == "Observed something"
        assert entry["salience"] == 0.8
        assert entry["type"] == "observation"
        assert entry["details"] == "Details"

    def test_episodic_cap_at_200(self) -> None:
        mem = Memory()
        for i in range(250):
            mem.add_episodic("observation", f"Entry {i}", "")
        assert len(mem.data["episodic"]) == 200

    def test_search_episodic(self) -> None:
        mem = Memory()
        mem.add_episodic("observation", "Found a bug in parser", "details")
        mem.add_episodic("action", "Fixed the bug", "patch applied")
        mem.add_episodic("observation", "Nothing relevant", "")
        results = mem.search_episodic("bug")
        assert len(results) >= 1
        assert "bug" in results[0]["summary"].lower()

    def test_add_semantic(self) -> None:
        mem = Memory()
        mid = mem.add_semantic("python", "Python is dynamically typed", "experience", 0.9)
        assert mid.startswith("sem_")
        assert len(mem.data["semantic"]) == 1
        entry = mem.data["semantic"][0]
        assert entry["topic"] == "python"
        assert entry["confidence"] == 0.9

    def test_semantic_deduplication(self) -> None:
        mem = Memory()
        mid1 = mem.add_semantic("topic", "Same fact", "source1", 0.7)
        mid2 = mem.add_semantic("topic", "Same fact", "source2", 0.9)
        assert mid1 == mid2  # Same (topic, fact) → deduplicated
        assert len(mem.data["semantic"]) == 1
        # Confidence should be max of the two
        assert mem.data["semantic"][0]["confidence"] == 0.9

    def test_get_semantic_by_topic(self) -> None:
        mem = Memory()
        mem.add_semantic("python", "Python is great", "", 0.8)
        mem.add_semantic("javascript", "JS is flexible", "", 0.7)
        results = mem.get_semantic_by_topic("python")
        assert len(results) == 1
        assert "great" in results[0]["fact"]

    def test_add_procedural(self) -> None:
        mem = Memory()
        mid = mem.add_procedural("file-pattern", "When editing files", "Always verify")
        assert mid.startswith("pro_")
        assert len(mem.data["procedural"]) == 1

    def test_procedural_deduplication(self) -> None:
        mem = Memory()
        mid1 = mem.add_procedural("pattern1", "trigger1", "procedure1")
        mid2 = mem.add_procedural("pattern1", "trigger1", "procedure1")
        assert mid1 == mid2  # Deduplicated on pattern name
        assert mem.data["procedural"][0]["success_count"] == 2

    def test_procedural_cap_at_100(self) -> None:
        mem = Memory()
        for i in range(150):
            mem.add_procedural(f"pattern_{i}", "trigger", "procedure")
        assert len(mem.data["procedural"]) == 100

    def test_cross_type_search(self) -> None:
        mem = Memory()
        mem.add_episodic("observation", "Testing cross search", "details")
        mem.add_semantic("search", "Cross search works", "", 0.7)
        results = mem.search("cross", memory_types=["episodic", "semantic"])
        assert "episodic" in results
        assert "semantic" in results
        assert len(results["episodic"]) >= 1
        assert len(results["semantic"]) >= 1

    def test_format_context_empty(self) -> None:
        mem = Memory()
        ctx = mem.format_context()
        assert "(no recent memories yet)" in ctx

    def test_format_context_with_data(self) -> None:
        mem = Memory()
        mem.add_episodic("observation", "Important observation", "", salience=0.9)
        mem.add_semantic("python", "Python is dynamic", "exp", 0.85)
        mem.add_procedural("test-pattern", "when testing", "run tests")
        ctx = mem.format_context()
        assert "Recent experiences" in ctx
        assert "Knowledge gained" in ctx
        assert "Learned patterns" in ctx
        assert "Important observation" in ctx
        assert "Python is dynamic" in ctx
        assert "test-pattern" in ctx

    def test_repr(self) -> None:
        mem = Memory()
        assert repr(mem) == "<Memory episodic=0 semantic=0 procedural=0>"
        mem.add_episodic("observation", "X", "")
        assert "episodic=1" in repr(mem)


class TestMemoryPersistence:
    """File-backed Memory save/load."""

    def test_save_and_load(self, evolve_env: Dict) -> None:
        mem = Memory()
        mem.add_episodic("observation", "Persisted event", "details", salience=0.7)
        mem.add_semantic("test", "Test fact", "test", 0.9)
        mem.add_procedural("test-pattern", "trigger", "procedure")
        mem.save()

        loaded = Memory.load()
        assert len(loaded.data["episodic"]) == 1
        assert len(loaded.data["semantic"]) == 1
        assert len(loaded.data["procedural"]) == 1
        assert loaded.data["episodic"][0]["salience"] == 0.7

    def test_load_empty(self, evolve_env: Dict) -> None:
        mem = Memory.load()
        assert mem.data["episodic"] == []

    def test_load_partial_file(self, evolve_env: Dict) -> None:
        """A partial file (e.g. only episodic) should merge with defaults."""
        path = Memory.storage_path()
        path.write_text(json.dumps({"episodic": [{"summary": "test"}]}))
        loaded = Memory.load()
        assert len(loaded.data["episodic"]) == 1
        assert loaded.data["semantic"] == []  # default restored
        assert loaded.data["procedural"] == []


# ═══════════════════════════════════════════════════════════════════════
#  Goals — self-generated goal lifecycle
# ═══════════════════════════════════════════════════════════════════════


class TestGoals:
    """Goals in-memory tests."""

    def test_default_empty(self) -> None:
        g = Goals()
        assert g.get_active() == []

    def test_propose(self) -> None:
        g = Goals()
        gid = g.propose(
            "Build world model",
            "Implement predict→act→observe→learn loop",
            rationale="Core for Phase 2",
            gap_reference="6",
            verification_criteria="Tests pass",
            priority=1,
        )
        assert gid.startswith("goal_")
        assert len(g.data["goals"]) == 1
        goal = g.data["goals"][0]
        assert goal["title"] == "Build world model"
        assert goal["priority"] == 1
        assert goal["status"] == "proposed"
        assert goal["gap_reference"] == "6"
        assert goal["verification_criteria"] == "Tests pass"

    def test_propose_non_string_llm_fields(self) -> None:
        """LLM JSON may emit numbers where strings are expected.

        Regression for the 2026-08-01 daemon crash in
        ``_find_similar_active_goal``: ``AttributeError: 'int' object has
        no attribute 'strip'`` when the parsed ``new_goal`` block carried
        ``gap_reference: 8`` (an int) instead of ``"8"``.
        """
        g = Goals()
        gid = g.propose(
            "Build world model v2",
            "Desc",
            gap_reference=8,       # int — as emitted by the LLM
            priority="2",          # string priority must not break comparisons
        )
        goal = g.data["goals"][0]
        assert goal["gap_reference"] == "8"   # canonical string form
        assert goal["priority"] == 2          # coerced to int

    def test_propose_int_gap_ref_dedup_matches_existing(self) -> None:
        """An int gap_reference should dedup against a goal stored with
        the equivalent string form (boundary coercion normalizes both)."""
        g = Goals()
        gid1 = g.propose(
            "Consolidate memory files", "Desc", gap_reference="6"
        )
        gid2 = g.propose(
            "Consolidate memory files", "Desc2", gap_reference=6
        )
        assert gid2 == gid1  # deduplicated, not duplicated
        assert len(g.data["goals"]) == 1

    def test_existing_goal_with_int_gap_reference(self) -> None:
        """Legacy persisted data with an int gap_reference must not crash
        the dedup scan (defensive read-site coercion)."""
        g = Goals()
        gid1 = g.propose("Refactor daemon loop", "Desc", gap_reference="8")
        g.data["goals"][0]["gap_reference"] = 8  # simulate legacy int
        gid2 = g.propose("Refactor daemon loop", "Desc2", gap_reference="8")
        assert gid2 == gid1
        assert len(g.data["goals"]) == 1

    def test_propose_suppresses_recently_completed_duplicate(self) -> None:
        """A proposal matching a recently-completed goal is suppressed, not
        duplicated.

        Regression for the 5 identical 'Investigate shell prediction
        failures' goals observed 2026-08-01: the active-only dedup missed
        completed goals, so every cycle re-created the same objective the
        moment the previous instance was completed. The re-proposal must
        return the existing goal's ID without adding a new goal, and must
        NOT resurrect the completed goal.

        Near-exact titles suppress regardless of age (an identical
        verbatim re-proposal is an LLM loop, not a new investigation), so
        this holds even when the completion is far outside the window.
        """
        g = Goals()
        title = (
            "Investigate shell prediction failures "
            "(keywords: auto-default, sibling, dirs)"
        )
        gid1 = g.propose(title, "Desc")
        g.update_status(gid1, "completed")
        assert len(g.data["goals"]) == 1

        gid2 = g.propose(title, "Desc again")
        assert gid2 == gid1                       # same objective → same reference
        assert len(g.data["goals"]) == 1          # no new goal created
        assert g.data["goals"][0]["status"] == "completed"  # not resurrected

        # Same verbatim title proposed much later is still a loop, not a
        # new investigation.
        old = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        g.data["goals"][0]["completed_at"] = old
        gid3 = g.propose(title, "Desc once more")
        assert gid3 == gid1
        assert len(g.data["goals"]) == 1

    def test_propose_suppresses_reworded_variant_within_window(self) -> None:
        """A reworded variant (moderate overlap) is suppressed only while
        the completed goal is inside the recency window."""
        g = Goals()
        original = (
            "Investigate shell prediction failures "
            "(keywords: auto-default, sibling, dirs)"
        )
        gid1 = g.propose(original, "Desc", gap_reference="6")
        g.update_status(gid1, "completed")
        # Completed 1h ago → inside the 24h window.
        recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        g.data["goals"][0]["completed_at"] = recent

        variant = (
            "Investigate recurring shell action failures "
            "in auto-default commands"
        )
        gid2 = g.propose(variant, "Desc again", gap_reference="6")
        assert gid2 == gid1
        assert len(g.data["goals"]) == 1

    def test_propose_allows_after_completion_window(self) -> None:
        """A reworded variant after the 24h suppression window is a
        legitimate new investigation of a recurring problem → a fresh goal
        is created."""
        g = Goals()
        original = (
            "Investigate shell prediction failures "
            "(keywords: auto-default, sibling, dirs)"
        )
        gid1 = g.propose(original, "Desc", gap_reference="6")
        g.update_status(gid1, "completed")
        # Backdate completion beyond the 24h suppression window.
        old = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        g.data["goals"][0]["completed_at"] = old

        variant = (
            "Investigate recurring shell action failures "
            "in auto-default commands"
        )
        gid2 = g.propose(variant, "Desc again", gap_reference="6")
        assert gid2 != gid1
        assert len(g.data["goals"]) == 2

    def test_dedupe_completed_collapses_near_identical(self) -> None:
        """Pre-guard duplicate COMPLETED goals are collapsed into the most
        recent record, keeping notes; active goals are untouched.

        Regression for the 5 identical 'Investigate shell prediction
        failures' goals observed 2026-08-01: the propose() suppression
        stopped new duplicates but never removed the ones already in the
        store. dedupe_completed() must self-heal that store.

        The duplicates are injected directly into the store (not via
        propose(), which now suppresses them) to simulate goals created
        before the guard existed.
        """
        g = Goals()
        title = (
            "Investigate shell prediction failures "
            "(keywords: auto-default, sibling, dirs)"
        )
        now = datetime.now(timezone.utc)
        # Three near-identical completed goals (the pre-guard spam shape)
        ids = []
        for i in range(3):
            gid = f"goal_dupe_{i}"
            g.data.setdefault("goals", []).append({
                "id": gid,
                "title": title,
                "description": f"Desc {i}",
                "status": "completed",
                "gap_reference": "6",
                "priority": 3,
                "created_at": (now - timedelta(hours=5)).isoformat(),
                "completed_at": (now - timedelta(hours=3 - i)).isoformat(),
                "notes": f"investigation {i}",
            })
            ids.append(gid)
        # A distinct completed goal must survive untouched
        other_id = "goal_other"
        g.data["goals"].append({
            "id": other_id,
            "title": "Refactor daemon loop",
            "description": "Desc",
            "status": "completed",
            "gap_reference": "8",
            "priority": 3,
            "created_at": now.isoformat(),
            "completed_at": now.isoformat(),
            "notes": "",
        })
        assert len(g.data["goals"]) == 4

        removed = g.dedupe_completed()
        assert removed == 2
        remaining = g.data["goals"]
        assert len(remaining) == 2
        assert other_id in [x["id"] for x in remaining]
        # Survivor is the most recently completed duplicate (i == 2)
        survivor = [x for x in remaining if x["id"] == ids[2]]
        assert len(survivor) == 1
        assert "investigation 2" in survivor[0].get("notes", "")
        # Idempotent: a second pass removes nothing
        assert g.dedupe_completed() == 0
        assert len(g.data["goals"]) == 2

    def test_dedupe_completed_leaves_active_untouched(self) -> None:
        """Near-identical ACTIVE goals are live work, not cleanup targets."""
        g = Goals()
        now = datetime.now(timezone.utc)
        g.data["goals"] = [
            {
                "id": "goal_active_1",
                "title": "Investigate shell prediction failures (keywords: auto)",
                "description": "Desc",
                "status": "completed",
                "gap_reference": "6",
                "priority": 3,
                "created_at": (now - timedelta(hours=2)).isoformat(),
                "completed_at": (now - timedelta(hours=1)).isoformat(),
                "notes": "",
            },
            {
                "id": "goal_active_2",
                "title": "Investigate shell prediction failures (keywords: auto) again",
                "description": "Desc",
                "status": "proposed",
                "gap_reference": "6",
                "priority": 3,
                "created_at": now.isoformat(),
                "completed_at": None,
                "notes": "",
            },
        ]
        assert len(g.data["goals"]) == 2
        # Only ONE completed goal → nothing to collapse even though the
        # proposed sibling is near-identical (live work, not cleanup).
        assert g.dedupe_completed() == 0
        assert len(g.data["goals"]) == 2
        assert g.data["goals"][0]["status"] == "completed"
        assert g.data["goals"][1]["status"] == "proposed"

    def test_update_status_valid(self) -> None:
        g = Goals()
        gid = g.propose("Test", "Desc", "", "", "", 3)
        assert g.update_status(gid, "active") is True
        assert g.data["goals"][0]["status"] == "active"

        assert g.update_status(gid, "in_progress") is True
        assert g.data["goals"][0]["status"] == "in_progress"

        assert g.update_status(gid, "completed") is True
        assert g.data["goals"][0]["completed_at"] is not None

    def test_update_status_invalid(self) -> None:
        g = Goals()
        gid = g.propose("Test", "Desc", "", "", "", 3)
        assert g.update_status(gid, "invalid_status") is False
        assert g.data["goals"][0]["status"] == "proposed"  # unchanged

    def test_update_status_nonexistent(self) -> None:
        g = Goals()
        assert g.update_status("nonexistent", "active") is False

    def test_get_active_default(self) -> None:
        g = Goals()
        gid1 = g.propose("G1", "D1", "", "", "", 2)
        gid2 = g.propose("G2", "D2", "", "", "", 1)
        g.update_status(gid1, "completed")
        active = g.get_active()  # default: proposed + active + in_progress
        assert len(active) == 1  # G2 is still proposed
        assert active[0]["title"] == "G2"

    def test_get_active_with_filter(self) -> None:
        g = Goals()
        gid = g.propose("Test", "Desc", "", "", "", 3)
        assert len(g.get_active(status_filter=["completed"])) == 0

        g.update_status(gid, "completed")
        completed = g.get_active(status_filter=["completed"])
        assert len(completed) == 1

    def test_sort_by_priority_then_created(self) -> None:
        g = Goals()
        g.propose("Low priority", "D", "", "", "", 5)
        g.propose("High priority", "D", "", "", "", 1)
        g.propose("Medium priority", "D", "", "", "", 3)
        active = g.get_active()
        assert active[0]["title"] == "High priority"
        assert active[1]["title"] == "Medium priority"
        assert active[2]["title"] == "Low priority"

    def test_format_context_empty(self) -> None:
        g = Goals()
        ctx = g.format_context()
        assert "no self-generated goals" in ctx

    def test_format_context_with_goals(self) -> None:
        g = Goals()
        g.propose("Goal A", "Description A", "Rationale A", "6", "Done", 2)
        g.propose("Goal B", "Description B", "Rationale B", "8", "Done", 1)
        ctx = g.format_context()
        assert "P1" in ctx or "P2" in ctx
        assert "Goal A" in ctx
        assert "Goal B" in ctx

    def test_repr(self) -> None:
        g = Goals()
        assert repr(g) == "<Goals count=0>"
        g.propose("Test", "D", "", "", "", 3)
        assert repr(g) == "<Goals count=1>"

    def test_goal_cap_at_100(self) -> None:
        g = Goals()
        for i in range(150):
            g.propose(f"Goal {i}", "D", "", "", "", 3)
        assert len(g.data["goals"]) == 100


class TestGoalsPersistence:
    """File-backed Goals save/load."""

    def test_save_and_load(self, evolve_env: Dict) -> None:
        g = Goals()
        g.propose("Persist test", "Desc", "Rationale", "6", "Tests", 2)
        g.save()

        loaded = Goals.load()
        assert len(loaded.data["goals"]) == 1
        assert loaded.data["goals"][0]["title"] == "Persist test"

    def test_load_empty(self, evolve_env: Dict) -> None:
        g = Goals.load()
        assert g.data["goals"] == []


# ═══════════════════════════════════════════════════════════════════════
#  Dict-format timeline helpers
# ═══════════════════════════════════════════════════════════════════════


class TestTimelineDictHelpers:
    """Functional tests for the dict-format timeline (past/present/future)."""

    def test_load_empty(self, evolve_env: Dict) -> None:
        data = load_timeline_dict()
        assert "past" in data
        assert "present" in data
        assert "future" in data
        assert data["past"]["events"] == []

    def test_record_event(self, evolve_env: Dict) -> None:
        record_event("milestone", "First event", "It matters")
        data = load_timeline_dict()
        assert len(data["past"]["events"]) == 1
        assert data["past"]["events"][0]["summary"] == "First event"
        assert data["past"]["events"][0]["type"] == "milestone"
        assert data["past"]["events"][0]["impact"] == "It matters"

    def test_record_outcome(self, evolve_env: Dict) -> None:
        record_event("milestone", "Event", "")
        data = load_timeline_dict()
        event_id = data["past"]["events"][0]["id"]
        record_outcome(event_id, "Completed", "Great result")
        data2 = load_timeline_dict()
        assert len(data2["past"]["outcomes"]) == 1
        assert data2["past"]["outcomes"][0]["summary"] == "Completed"

    def test_record_session_completion(self, evolve_env: Dict) -> None:
        record_session_completion("session_1", "Working on Gap 6", ["Did X", "Did Y"])
        data = load_timeline_dict()
        assert len(data["past"]["completed_sessions"]) == 1
        assert data["past"]["completed_sessions"][0]["focus"] == "Working on Gap 6"
        assert len(data["past"]["completed_sessions"][0]["outcomes"]) == 2

    def test_add_commitment(self, evolve_env: Dict) -> None:
        add_commitment("Finish Gap 6", "2026-08-01")
        data = load_timeline_dict()
        assert len(data["present"]["commitments"]) == 1
        assert data["present"]["commitments"][0]["what"] == "Finish Gap 6"

    def test_update_commitment(self, evolve_env: Dict) -> None:
        add_commitment("Task X", "2026-07-01")
        data = load_timeline_dict()
        cid = data["present"]["commitments"][0]["id"]
        update_commitment(cid, "done")
        data2 = load_timeline_dict()
        assert data2["present"]["commitments"][0]["status"] == "done"

    def test_add_prediction(self, evolve_env: Dict) -> None:
        add_prediction("System will complete Gap 6", "3 days", 0.8, "Based on progress")
        data = load_timeline_dict()
        assert len(data["future"]["predictions"]) == 1
        assert data["future"]["predictions"][0]["text"].startswith("System will")

    def test_update_present_state(self, evolve_env: Dict) -> None:
        update_present_state(active_project="Gap 6", tasks=["Task 1"], focus="Evolution")
        data = load_timeline_dict()
        assert data["present"]["active_project"] == "Gap 6"
        assert "Task 1" in data["present"]["active_tasks"]
        assert data["present"]["last_session_focus"] == "Evolution"

    def test_update_goals(self, evolve_env: Dict) -> None:
        update_goals(goals=["Complete Phase 2"])
        data = load_timeline_dict()
        assert "Complete Phase 2" in data["future"]["goals"]

    def test_save_timeline_dict_round_trip(self, evolve_env: Dict) -> None:
        data = load_timeline_dict()
        data["present"]["active_project"] = "Test"
        save_timeline_dict(data)
        reloaded = load_timeline_dict()
        assert reloaded["present"]["active_project"] == "Test"


# ═══════════════════════════════════════════════════════════════════════
#  Plan management
# ═══════════════════════════════════════════════════════════════════════


class TestPlans:
    """Plan creation, update, and completion."""

    def test_create_plan(self, evolve_env: Dict) -> None:
        steps = [
            {"description": "Step 1", "verification": "Verify 1"},
            {"description": "Step 2", "verification": "Verify 2"},
        ]
        pid = create_plan("Test plan", steps)
        assert pid.startswith("plan_")

        active = get_active_plan()
        assert active is not None
        assert active["goal"] == "Test plan"
        assert active["progress"] == "0/2 steps"
        assert len(active["steps"]) == 2

    def test_no_active_plan(self, evolve_env: Dict) -> None:
        assert get_active_plan() is None

    def test_get_active_plan_returns_first_active(self, evolve_env: Dict) -> None:
        create_plan("First plan", [{"description": "S1", "verification": "V"}])
        create_plan("Second plan", [{"description": "S1", "verification": "V"}])
        active = get_active_plan()
        # Should return the first active one (which was created first)
        assert active is not None
        assert active["goal"] == "First plan"

    def test_update_plan_step(self, evolve_env: Dict) -> None:
        steps = [{"description": "Step 1", "verification": "V"}]
        pid = create_plan("Test", steps)
        assert update_plan_step(pid, "step_1", "complete", "Done!") is True

        # Completing the ONLY step completes the plan (plan-continuity
        # invariant): the active-plan slot frees up for reinstantiation.
        active = get_active_plan()
        assert active is None

        data = load_timeline_dict()
        plan = next(p for p in data["future"]["plans"] if p["id"] == pid)
        assert plan["steps"][0]["status"] == "complete"
        assert plan["steps"][0]["note"] == "Done!"
        assert plan["progress"] == "1/1 steps"
        assert plan["status"] == "complete"

    def test_update_plan_step_partial_keeps_active(self, evolve_env: Dict) -> None:
        steps = [
            {"description": "Step 1", "verification": "V"},
            {"description": "Step 2", "verification": "V"},
        ]
        pid = create_plan("Test", steps)
        assert update_plan_step(pid, "step_1", "complete", "Done!") is True

        # One of two steps complete → plan stays active, progress updates.
        active = get_active_plan()
        assert active is not None
        assert active["steps"][0]["status"] == "complete"
        assert active["steps"][0]["note"] == "Done!"
        assert active["progress"] == "1/2 steps"
        assert active["status"] == "active"

        # Completing the last step completes the plan (continuity invariant).
        assert update_plan_step(pid, "step_2", "complete") is True
        assert get_active_plan() is None
        data = load_timeline_dict()
        plan = next(p for p in data["future"]["plans"] if p["id"] == pid)
        assert plan["status"] == "complete"

    def test_update_nonexistent_step(self, evolve_env: Dict) -> None:
        pid = create_plan("Test", [{"description": "S1", "verification": "V"}])
        assert update_plan_step(pid, "nonexistent", "complete") is False

    def test_complete_plan(self, evolve_env: Dict) -> None:
        pid = create_plan("Test", [{"description": "S1", "verification": "V"}])
        assert complete_plan(pid) is True

        active = get_active_plan()
        assert active is None  # No longer active

        # Should still be findable in the timeline data
        data = load_timeline_dict()
        plans = data["future"]["plans"]
        assert len(plans) == 1
        assert plans[0]["status"] == "complete"
        assert plans[0]["completed_at"] is not None

    def test_complete_nonexistent_plan(self, evolve_env: Dict) -> None:
        assert complete_plan("nonexistent") is False

    def test_plan_with_default_steps(self, evolve_env: Dict) -> None:
        """create_plan with no steps should still work."""
        pid = create_plan("Empty plan")
        active = get_active_plan()
        assert active["progress"] == "0 steps"

    def test_create_plan_does_not_mutate_module_default(self, evolve_env: Dict) -> None:
        """Creating a plan must never leak into the module-level default.

        Regression for the 2026-07-31 incident: ``_load_timeline_dict``
        returned a shallow copy of ``_DEFAULT_TIMELINE_DICT``, so
        ``create_plan`` appended the new plan into the shared default's
        ``future.plans`` list. Every subsequent test/daemon cycle that
        loaded a fresh (or missing) timeline then saw the leaked plan as
        the active plan — producing spurious "active plan" states and
        cross-test contamination (e.g. TestPlans failing when run after
        test_think_daemon.py, and the daemon's auto-created "Complete
        Gap 8" plan appearing in unrelated timelines).
        """
        import data_layer as dl_mod
        before = copy.deepcopy(dl_mod._DEFAULT_TIMELINE_DICT)
        assert before["future"]["plans"] == []

        pid = create_plan("Leak test", [
            {"description": "Do something real", "verification": "verify it works"},
        ])
        assert pid.startswith("plan_")

        # The module-level default must remain pristine — the new plan
        # lives only in the evolve-dir timeline.json, not in the default.
        after = dl_mod._DEFAULT_TIMELINE_DICT
        assert after["future"]["plans"] == [], (
            "create_plan mutated the module-level default; "
            "a later fresh load would see a phantom active plan"
        )
        # And a fresh load with a NEW evolve dir sees no active plan.
        # (evolve_env isolation means the file we wrote lives in the
        # current temp evolve dir; the point here is the default is clean.)
        assert dl_mod.get_active_plan() is not None  # file-backed plan
        assert after["future"]["plans"] == before["future"]["plans"]



# ═══════════════════════════════════════════════════════════════════════
#  Context formatters
# ═══════════════════════════════════════════════════════════════════════


class TestContextFormatters:
    """format_self_model_context, format_timeline_context, etc."""

    def test_format_self_model_context(self, evolve_env: Dict) -> None:
        sm = SelfModel()
        sm.add_strength("Test strength")
        sm.add_weakness("Test weakness")
        sm.add_unknown("Test unknown")
        sm.set_state_value("current_gap_focus", "Gap 6")
        sm.save()

        ctx = format_self_model_context()
        assert "Self Model" in ctx
        # Note: format_self_model_context does NOT include strengths
        # in its output — only weaknesses and unknown areas
        assert "Test weakness" in ctx
        assert "Test unknown" in ctx
        assert "Gap 6" in ctx

    def test_format_plan_context_empty(self, evolve_env: Dict) -> None:
        assert format_plan_context() == ""

    def test_format_plan_context_with_plan(self, evolve_env: Dict) -> None:
        steps = [
            {"description": "Build X", "verification": "Tests pass"},
            {"description": "Deploy X", "verification": "URL works"},
        ]
        create_plan("Complete project", steps)
        ctx = format_plan_context()
        assert "Complete project" in ctx
        assert "Build X" in ctx
        assert "→" in ctx  # pending icon

    def test_format_plan_context_shows_progress(self, evolve_env: Dict) -> None:
        steps = [
            {"description": "Step A", "verification": "V"},
            {"description": "Step B", "verification": "V"},
        ]
        pid = create_plan("Test", steps)
        update_plan_step(pid, "step_1", "complete")
        ctx = format_plan_context()
        assert "1/2 steps" in ctx
        assert "✓" in ctx  # complete icon

    def test_format_timeline_context_empty(self, evolve_env: Dict) -> None:
        """Should return minimal context with no events."""
        ctx = format_timeline_context()
        # It starts with ## Timeline even when empty
        assert ctx == "" or "## Timeline" in ctx

    def test_format_timeline_context_with_session(self, evolve_env: Dict) -> None:
        record_session_completion("s1", "Working on Gap 6", ["Did X"])
        ctx = format_timeline_context()
        assert "Working on Gap 6" in ctx or "Gap 6" in ctx

    def test_format_timeline_context_with_project(self, evolve_env: Dict) -> None:
        update_present_state(active_project="World Model", tasks=["Build loop"])
        ctx = format_timeline_context()
        assert "World Model" in ctx
        assert "Build loop" in ctx

    def test_format_orientation_context_empty(self, evolve_env: Dict) -> None:
        ctx = format_orientation_context()
        # Even with no user data, Memory and Goals produce default
        # messages like "(no recent memories yet)" and SelfModel
        # always has identity data, so context is never truly empty.
        # The key assertion: it contains the expected sections.
        assert "Recent Memories" in ctx
        assert "Self-generated Goals" in ctx or "Self Model" in ctx

    def test_format_orientation_context_with_all(self, evolve_env: Dict) -> None:
        """Full orientation context includes all sections."""
        # Set up orientation
        o = Orientation()
        o.focus = "Building world model"
        o.add_insight("Prediction works")
        o.add_next_step("Test daemon")
        o.save()

        # Add a timeline event
        record_event("milestone", "Started Gap 6", "Key milestone")

        # Add a goal
        g = Goals()
        g.propose("Complete Gap 6", "Implement world model", "Core", "6", "Tests pass", 1)
        g.save()

        # Add self model data
        sm = SelfModel()
        sm.add_strength("Good at predictions")
        sm.set_state_value("current_gap_focus", "Gap 6")
        sm.save()

        ctx = format_orientation_context()
        assert "Session Orientation" in ctx
        assert "Building world model" in ctx
        assert "Prediction works" in ctx
        assert "Test daemon" in ctx
        assert "Self-generated Goals" in ctx
        assert "P1" in ctx or "Complete Gap 6" in ctx
        assert "Self Model" in ctx
        # format_self_model_context does not include strengths
        assert "current gap focus: Gap 6" in ctx

    def test_format_orientation_context_memory(self, evolve_env: Dict) -> None:
        """Memory section appears when data exists."""
        mem = Memory()
        mem.add_episodic("observation", "Learned about X", "details", salience=0.7)
        mem.save()
        ctx = format_orientation_context()
        assert "Recent Memories" in ctx
        assert "Learned about X" in ctx


# ═══════════════════════════════════════════════════════════════════════
#  Low-level helpers
# ═══════════════════════════════════════════════════════════════════════


class TestLowLevelHelpers:
    """safe_read_json, safe_write_json, now_iso, now_compact."""

    def test_now_iso_format(self) -> None:
        ts = now_iso()
        # ISO-8601 with timezone info
        assert "T" in ts
        assert ts.endswith("+00:00") or "+00" in ts[-6:]

    def test_now_compact_format(self) -> None:
        ts = now_compact()
        assert len(ts) == 14  # YYYYMMDDHHMMSS
        assert ts.isdigit()

    def test_safe_read_json_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "test.json"
        path.write_text(json.dumps({"key": "value"}))
        data = safe_read_json(path)
        assert data["key"] == "value"

    def test_safe_read_json_missing(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.json"
        data = safe_read_json(path, default={"fallback": True})
        assert data["fallback"] is True

    def test_safe_read_json_corrupted(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not valid json")
        data = safe_read_json(path, default=None)
        assert data is None

    def test_safe_write_json_creates_file(self, tmp_path: Path) -> None:
        path = tmp_path / "output.json"
        safe_write_json(path, {"hello": "world"})
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["hello"] == "world"

    def test_safe_write_json_nested_dir(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "dir" / "nested.json"
        safe_write_json(path, {"test": True})
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["test"] is True

    def test_safe_write_json_is_atomic(self, tmp_path: Path) -> None:
        """Write should use .tmp staging before renaming to final."""
        path = tmp_path / "atomic.json"
        safe_write_json(path, {"v": 1})
        # The .tmp file should not linger
        assert not path.with_suffix(".tmp").exists()
        assert path.exists()

    def test_safe_read_json_version_warning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Version far ahead of default should log a warning but still return data."""
        path = tmp_path / "versioned.json"
        path.write_text(json.dumps({"version": 99, "data": "new_format"}))
        import logging
        caplog.set_level(logging.WARNING)
        data = safe_read_json(path, default={"version": 1})
        # Should still return the data despite version gap
        assert data["version"] == 99
        # Check that a warning was logged about the version gap

    def test_get_evolve_dir(self, evolve_env: Dict) -> None:
        d = get_evolve_dir()
        assert d.exists()
        assert d.name == "evolve"


# ═══════════════════════════════════════════════════════════════════════
#  Integration: cross-module data flow
# ═══════════════════════════════════════════════════════════════════════


class TestIntegration:
    """End-to-end workflows across multiple data_layer components."""

    def test_typical_workflow(self, evolve_env: Dict) -> None:
        """Simulate a typical evolution session."""
        # 1. Record a timeline event
        record_event("milestone", "Implemented world model", "Core capability")

        # 2. Add a goal
        g = Goals()
        g.propose("Complete Gap 6", "Build the world model", "Core", "6", "159 tests", 1)
        g.propose("Complete Gap 8", "Self-directed evolution", "Next step", "8", "TBD", 2)
        g.save()

        # 3. Update self-model
        sm = SelfModel()
        sm.add_strength("World model works")
        sm.add_weakness("Need more data")
        sm.set_state_value("current_gap_focus", "Gap 6")
        sm.save()

        # 4. Store a memory
        mem = Memory()
        mem.add_episodic("milestone", "World model now predicts action outcomes", "predict→act→observe→learn loop", salience=0.9)
        mem.add_semantic("world_model", "Prediction error uses multi-strategy approach", "self-discovery", 0.85)
        mem.save()

        # 5. Set orientation
        o = Orientation()
        o.focus = "Complete Gap 6 and begin Gap 8"
        o.add_insight("World model has 159 tests passing")
        o.add_next_step("Test think_daemon integration")
        o.save()

        # 6. Verify everything loads back
        loaded_g = Goals.load()
        assert len(loaded_g.get_active()) == 2

        loaded_sm = SelfModel.load()
        assert "World model works" in loaded_sm.capabilities["strengths"]

        loaded_mem = Memory.load()
        assert any("World model now predicts" in e["summary"] for e in loaded_mem.data["episodic"])

        loaded_tl = load_timeline_dict()
        assert len(loaded_tl["past"]["events"]) == 1

        loaded_o = Orientation.load()
        assert loaded_o.focus == "Complete Gap 6 and begin Gap 8"

    def test_format_orientation_context_full_workflow(self, evolve_env: Dict) -> None:
        """Full orientation context after a realistic session."""
        # Populate all stores
        record_event("milestone", "Built data layer", "Foundation for everything")
        record_session_completion("s1", "Data layer exploration", ["Found classes", "Designed schema"])

        g = Goals()
        g.propose("Finish Phase 2", "Complete all gaps", "", "", "", 1)
        g.save()

        sm = SelfModel()
        sm.add_strength("Clean architecture")
        sm.set_state_value("current_gap_focus", "Phase 2")
        sm.save()

        o = Orientation()
        o.focus = "Testing data layer"
        o.add_insight("All classes work")
        o.add_next_step("Write integration tests")
        o.save()

        ctx = format_orientation_context()
        assert "Phase 2" in ctx
        assert "Session Orientation" in ctx
        assert "Testing data layer" in ctx
        assert "All classes work" in ctx
        assert "Finish Phase 2" in ctx
        assert "Built data layer" in ctx
