"""Tests for think_daemon._local_analysis() — local fallback when LLM is unavailable.

Verifies that when the LLM API is unreachable, the daemon generates useful
insights from world model data directly, without external calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from world_model import WorldModel
from think_daemon import _local_analysis


# ── Fixtures ──────────────────────────────────────────────────────


def _make_wm_with_triples(count: int = 10, error_val: float = 0.15) -> WorldModel:
    """Build a WorldModel with *count* completed action triples."""
    wm = WorldModel()
    for i in range(count):
        atype = "shell" if i < count // 2 else "write_file"
        tid = wm.record_action(
            action_type=atype,
            action_description=f"test action {i} ({'shell cmd' if atype == 'shell' else 'write file'})",
            expected_outcome=f"expected outcome for action {i}",
        )
        wm.complete_action(
            triple_id=tid,
            actual_outcome=f"actual outcome for action {i}",
        )
        # Override prediction error to get consistent test values
        triples = wm.data["action_triples"]
        triples[-1]["prediction_error"] = error_val
    wm._update_accuracy_stats()
    return wm


def _make_state(
    wm: WorldModel,
    tick_count: int = 5,
    current_focus: str = "Gap 8: self-directed evolution",
    remaining_gaps: list | None = None,
) -> Dict[str, Any]:
    """Build a minimal state dict for _local_analysis."""
    from think_daemon import _DEFAULT_DAEMON_STATE

    ds = dict(_DEFAULT_DAEMON_STATE)
    ds["tick_count"] = tick_count
    ds["status"] = "running"

    sm = {
        "identity": {"name": "t", "role": "Self-evolving AI system"},
        "state": {
            "evolution_version": 6,
            "current_gap_focus": current_focus,
            "total_cycles": 45,
            "remaining_gaps": remaining_gaps or [],
        },
        "capabilities": {
            "strengths": [],
            "weaknesses": [],
            "unknown_areas": [],
        },
        "commitments": {"promised_features": [], "active_obligations": []},
    }

    tl = {
        "version": 2,
        "past": {"events": [], "outcomes": [], "completed_sessions": []},
        "present": {"commitments": []},
        "future": {"predictions": [], "plans": []},
    }

    return {
        "daemon_state": ds,
        "timeline": tl,
        "self_model": sm,
        "world_model": wm,
    }


# ══════════════════════════════════════════════════════════════════
#  Tests
# ══════════════════════════════════════════════════════════════════


class TestLocalAnalysis:
    """Verify _local_analysis produces valid output from world model data."""

    def test_returns_full_structure(self):
        """The returned dict has all expected keys matching LLM response schema."""
        wm = _make_wm_with_triples(count=6)
        state = _make_state(wm)
        result = _local_analysis(state)

        # Core fields
        assert isinstance(result["insight"], str)
        assert len(result["insight"]) > 20
        assert "[local-analysis]" in result["insight"]

        assert isinstance(result["focus_next"], str)
        assert len(result["focus_next"]) > 5

        assert isinstance(result["confidence"], (int, float))
        assert 0 <= result["confidence"] <= 1

        assert isinstance(result["reasoning"], str)

        # Event and session fields
        assert result["event_to_record"] is not None
        assert result["event_to_record"]["type"] == "observation"
        assert "LLM unavailable" in result["event_to_record"]["summary"]

        assert result["session_record"] is not None
        assert "focus" in result["session_record"]
        assert "outcomes_list" in result["session_record"]

        # Safe nulls for LLM-only fields
        assert result["prediction"] is None
        # action is non-None in fallback mode — _local_analysis produces
        # a rotating state-check command to keep collecting data
        assert result["action"] is not None
        assert "type" in result["action"]
        assert result["action"]["type"] == "shell" or result["action"]["type"] == "write_file" or result["action"]["type"] == "git_commit"
        assert result["commitment"] is None
        assert result["outcome_to_record"] is None
        assert result["new_plan"] is None
        assert result["plan_action"] is None
        assert result["search_query"] is None

        # Self-model update
        assert result["self_model_update"] is not None
        assert "weakness" in result["self_model_update"]

        # Episodic record
        assert result["episodic_record"] is not None
        assert result["episodic_record"]["mtype"] == "observation"

    def test_insight_contains_world_model_stats(self):
        """Insight string includes triple count and per-type breakdown."""
        wm = _make_wm_with_triples(count=10, error_val=0.15)
        state = _make_state(wm)
        result = _local_analysis(state)

        insight = result["insight"]
        # Should mention action triples count
        assert "10" in insight or "action triples" in insight
        # Should mention shell and write_file types
        assert "shell" in insight
        assert "write_file" in insight or "write" in insight

    def test_insight_shows_trend_stable_for_few_samples(self):
        """With <4 error_history entries, trend should be 'stable'."""
        wm = _make_wm_with_triples(count=3, error_val=0.15)
        state = _make_state(wm)
        result = _local_analysis(state)

        insight = result["insight"]
        assert "stable" in insight

    def test_insight_shows_trend_improving(self):
        """When recent errors are lower than older ones, trend is 'improving'."""
        wm = _make_wm_with_triples(count=15, error_val=0.15)
        # Manually craft error_history with improving trend
        error_history = [0.5, 0.5, 0.5, 0.4, 0.3, 0.2]
        wm.data["prediction_accuracy"]["error_history"] = error_history
        wm._update_accuracy_stats()
        state = _make_state(wm)
        result = _local_analysis(state)

        insight = result["insight"]
        assert "improving" in insight or "stable" in insight

    def test_insight_shows_trend_degrading(self):
        """When recent errors are higher than older ones, trend is 'degrading'."""
        wm = _make_wm_with_triples(count=15, error_val=0.15)
        error_history = [0.1, 0.1, 0.2, 0.3, 0.4, 0.5]
        wm.data["prediction_accuracy"]["error_history"] = error_history
        wm._update_accuracy_stats()
        state = _make_state(wm)
        result = _local_analysis(state)

        insight = result["insight"]
        assert "degrading" in insight

    def test_insight_trend_stable_despite_single_spike(self):
        """A single 0.5 spike in an otherwise-stable series must NOT flip to 'degrading'.

        Regression test: the old last-3-vs-first-3 multiplicative heuristic
        flagged 'degrading' on one late 0.5 outlier while the world model's
        canonical split-half trend said 'stable'.  The local analysis must
        agree with the world model — a single outlier is noise, not a trend.
        """
        wm = _make_wm_with_triples(count=15, error_val=0.15)
        error_history = [0.15] * 16 + [0.5]  # 17 entries, one late spike
        wm.data["prediction_accuracy"]["error_history"] = error_history
        wm._update_accuracy_stats()
        state = _make_state(wm)
        result = _local_analysis(state)

        insight = result["insight"]
        assert "degrading" not in insight
        assert "stable" in insight

    def test_confidence_with_high_error(self):
        """Confidence should be lower when avg prediction error is high."""
        wm = _make_wm_with_triples(count=8, error_val=0.8)
        state = _make_state(wm)
        result = _local_analysis(state)

        # High error + fallback discount = very low confidence
        assert result["confidence"] <= 0.35

    def test_confidence_with_low_error(self):
        """Confidence should be moderate when avg error is low."""
        wm = _make_wm_with_triples(count=8, error_val=0.1)
        state = _make_state(wm)
        result = _local_analysis(state)

        # Low error + fallback discount = ~0.6
        assert 0.2 <= result["confidence"] <= 0.7

    def test_self_model_weakness_included(self):
        """The self_model_update should note LLM unreliability."""
        wm = _make_wm_with_triples()
        state = _make_state(wm)
        result = _local_analysis(state)

        weakness = result["self_model_update"]["weakness"]
        assert weakness is not None
        assert "LLM" in weakness or "timeout" in weakness

    def test_next_gap_from_remaining_gaps(self):
        """next_gap should be the first remaining gap if available."""
        wm = _make_wm_with_triples()
        state = _make_state(wm, remaining_gaps=["Gap 7", "Gap 8", "Gap 9"])
        result = _local_analysis(state)

        assert result["next_gap"] == "Gap 7"

    def test_next_gap_none_when_empty(self):
        """next_gap should be None when no remaining gaps."""
        wm = _make_wm_with_triples()
        state = _make_state(wm, remaining_gaps=[])
        result = _local_analysis(state)

        assert result["next_gap"] is None

    def test_empty_world_model_still_produces_valid_result(self):
        """Even with no action triples, the function returns a valid structure."""
        wm = WorldModel()
        state = _make_state(wm, tick_count=1)
        result = _local_analysis(state)

        assert result["insight"] is not None
        assert result["confidence"] >= 0.1
        assert result["event_to_record"] is not None
        assert result["session_record"] is not None

    def test_tick_count_incremented(self):
        """The insight should reference the NEXT tick number (current + 1)."""
        wm = _make_wm_with_triples()
        state = _make_state(wm, tick_count=7)
        result = _local_analysis(state)

        assert "Cycle 8" in result["insight"]

    def test_action_follows_expected_schema(self):
        """Local analysis result can be consumed by _apply_insights without errors."""
        from think_daemon import _apply_insights

        wm = _make_wm_with_triples(count=5)
        state = _make_state(wm, tick_count=3)
        result = _local_analysis(state)

        # The fallback action type varies with the world model's per-type
        # sample counts (data-driven selection) — it must be one of the
        # supported rotation types.
        assert result["action"]["type"] in ("shell", "git_commit", "write_file")

        # Mock subprocess so _apply_insights never executes a REAL shell
        # command or git commit against the workspace repo. Without this,
        # a git_commit action (chosen when git_commit is the least-sampled
        # type, e.g. a fresh world model) runs `git add -A && git commit`
        # with cwd=_WORKSPACE_ROOT and sweeps uncommitted workspace changes
        # into an "Auto-sync" commit. Observed 2026-07-31 17:52 UTC.
        import subprocess

        original_run = subprocess.run
        try:
            def _mock_run(*a, **kw):
                return type("_R", (), {"returncode": 0, "stdout": "mocked\n", "stderr": ""})()

            subprocess.run = _mock_run

            # This should not raise
            updates = _apply_insights(result, state)
        finally:
            subprocess.run = original_run
        assert "timeline" in updates
        assert "self_model" in updates
        assert "orientation" in updates

        # Regression (2026-07-31): the fallback state-check write_file path
        # used to be baked from think_daemon's module-level EVOLVE_DIR,
        # frozen at import time BEFORE the hermetic HERMES_HOME fixture
        # applies. Executing the action then clobbered the LIVE daemon's
        # state_snapshot.txt with synthetic test data ("Tick: 4",
        # "Triples: 5/5"). The path must resolve inside the CURRENT
        # (hermetic) evolve dir instead.
        if result["action"]["type"] == "write_file":
            from data_layer import get_evolve_dir
            apath = Path(result["action"]["path"])
            assert apath.is_absolute(), f"write_file path not absolute: {apath}"
            assert str(apath).startswith(str(get_evolve_dir())), (
                f"fallback write_file path {apath} escapes hermetic evolve dir "
                f"{get_evolve_dir()}"
            )

    def test_cyclic_analysis_includes_per_type_breakdown(self):
        """With two action types, the insight includes both type summaries."""
        wm = WorldModel()
        # Add 3 shell and 3 write_file triples
        for i in range(3):
            tid = wm.record_action("shell", f"shell cmd {i}", f"expect shell {i}")
            wm.complete_action(tid, f"actual shell {i}")
            wm.data["action_triples"][-1]["prediction_error"] = 0.15
        for i in range(3):
            tid = wm.record_action("write_file", f"write file {i}", f"expect write {i}")
            wm.complete_action(tid, f"actual write {i}")
            wm.data["action_triples"][-1]["prediction_error"] = 0.2
        wm._update_accuracy_stats()

        state = _make_state(wm)
        result = _local_analysis(state)

        per_type_text = result["insight"]
        assert "shell" in per_type_text
        assert "write_file" in per_type_text or "write" in per_type_text

    def test_discrepancy_patterns_mentioned_when_present(self):
        """If discrepancy patterns exist, the insight mentions them."""
        wm = _make_wm_with_triples(count=12)
        # Manually add a discrepancy pattern
        wm.data["discrepancy_patterns"] = [
            {
                "action_type": "shell",
                "count": 3,
                "avg_error": 0.7,
                "common_themes": ["permission", "denied"],
                "description": "3/6 shell actions with high prediction error (avg 0.70)",
                "last_observed": "2026-07-29T14:08:31",
            }
        ]
        state = _make_state(wm)
        result = _local_analysis(state)

        assert "discrepancy" in result["insight"].lower() or "pattern" in result["insight"].lower()

    def test_active_predictions_mentioned(self):
        """If there are active (unverified) predictions, the insight mentions them."""
        wm = _make_wm_with_triples(count=4)
        wm.record_prediction(
            text="The think_daemon will run 10+ cycles this week",
            timeframe="7 days",
            confidence=0.6,
            basis="Current trend",
        )
        state = _make_state(wm)
        result = _local_analysis(state)

        assert "prediction" in result["insight"].lower() or "active" in result["insight"].lower()


# ══════════════════════════════════════════════════════════════════
#  Data-driven fallback action selection
# ══════════════════════════════════════════════════════════════════


class TestSelectStateCheckAction:
    """_select_state_check_action must diversify the world model's
    training data by preferring the least-sampled action type.

    Regression target: the blind ``tick_count % 5`` rotation gave shell
    3 of 5 slots, so shell accrued 52 of 77 triples while git_commit sat
    at 11 — calibration is bounded by the least-sampled type.
    """

    # Same shape as the rotation slots in _local_analysis
    _COMMANDS = [
        {"type": "shell", "command": "echo goals", "description": "goals"},
        {"type": "shell", "command": "echo self", "description": "self"},
        {"type": "shell", "command": "echo daemon", "description": "daemon"},
        {"type": "git_commit", "message": "sync", "description": "commit"},
        {"type": "write_file", "path": "/tmp/x", "content": "y", "description": "write"},
    ]

    def _wm_with_counts(self, counts: Dict[str, int]) -> WorldModel:
        """WorldModel with fabricated per-type sample counts."""
        wm = WorldModel()
        wm.data["per_type_accuracy"] = {
            t: {"count": c, "avg_error": 0.2, "min_error": 0.15, "max_error": 0.25}
            for t, c in counts.items()
        }
        return wm

    def test_picks_least_sampled_type(self) -> None:
        """git_commit (10 samples) is chosen over shell (50) and write_file (14)."""
        from think_daemon import _select_state_check_action

        wm = self._wm_with_counts({"shell": 50, "git_commit": 10, "write_file": 14})
        for tick in range(6):
            action = _select_state_check_action(wm, tick, self._COMMANDS)
            assert action["type"] == "git_commit", (
                f"tick {tick} picked {action['type']}, expected git_commit"
            )

    def test_rotates_through_tied_least_types(self) -> None:
        """Tied least-sampled types are rotated through, not biased to one."""
        from think_daemon import _select_state_check_action

        wm = self._wm_with_counts({"shell": 50, "git_commit": 10, "write_file": 10})
        picked = {
            _select_state_check_action(wm, t, self._COMMANDS)["type"]
            for t in range(8)
        }
        assert picked == {"git_commit", "write_file"}

    def test_falls_back_to_rotation_without_data(self) -> None:
        """No calibration data → plain tick rotation (old behavior)."""
        from think_daemon import _select_state_check_action

        wm = WorldModel()  # empty per_type_accuracy
        # tick 4 -> slot 4 (write_file); tick 5 -> slot 0 (goals shell)
        action = _select_state_check_action(wm, 4, self._COMMANDS)
        assert action["type"] == "write_file"
        action = _select_state_check_action(wm, 5, self._COMMANDS)
        assert action["type"] == "shell"
        assert "goals" in action["command"]

    def test_returns_copy_not_reference(self) -> None:
        """The returned dict is a copy; mutating it must not touch the catalog."""
        from think_daemon import _select_state_check_action

        wm = self._wm_with_counts({"shell": 50, "git_commit": 10, "write_file": 14})
        action = _select_state_check_action(wm, 0, self._COMMANDS)
        action["expected_outcome"] = "mutated"
        assert "expected_outcome" not in self._COMMANDS[3]

    def test_local_analysis_uses_data_driven_selection(self) -> None:
        """_local_analysis with an imbalanced world model picks the
        least-sampled type even when the tick would land on a shell slot."""
        from think_daemon import _select_state_check_action  # noqa: F401  (import check)

        wm = self._wm_with_counts({"shell": 50, "git_commit": 10, "write_file": 14})
        state = _make_state(wm, tick_count=0)
        result = _local_analysis(state)
        assert result["action"]["type"] == "git_commit"

    def test_local_analysis_empty_wm_keeps_rotation_semantics(self) -> None:
        """Hermetic/fresh world model → rotation semantics preserved:
        tick 4 (→5) hits slot 0 (goals), tick 5 (→6) slot 1 (self-model)."""
        wm = WorldModel()
        state = _make_state(wm, tick_count=4)
        result = _local_analysis(state)
        assert result["action"]["type"] == "shell"
        assert "Goals" in result["action"]["command"] or "goals" in result["action"]["command"]

        state = _make_state(wm, tick_count=5)
        result = _local_analysis(state)
        assert result["action"]["type"] == "shell"
        assert "Self Model" in result["action"]["command"]

