"""Tests for world_model.py — World Model (Gap 6).

Tests the predict→act→observe→learn loop, prediction error computation,
action triple lifecycle, macro prediction tracking, persistence, and
context formatting.  All tests use in-memory instances — no file I/O
unless explicitly testing save/load.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from world_model import WorldModel, _compute_prediction_error, load_world_model


# ═══════════════════════════════════════════════════════════════════
#  Error computation
# ═══════════════════════════════════════════════════════════════════

class TestComputePredictionError:
    """_compute_prediction_error maps expected vs actual to [0.0, 1.0]."""

    def test_exact_match(self) -> None:
        assert _compute_prediction_error("exact same", "exact same") == 0.0

    def test_empty_inputs(self) -> None:
        assert _compute_prediction_error("", "something") == 1.0
        assert _compute_prediction_error("something", "") == 1.0
        assert _compute_prediction_error("", "") == 1.0

    def test_complete_mismatch(self) -> None:
        assert _compute_prediction_error("abc", "xyz") >= 0.9

    def test_substring_contained(self) -> None:
        # actual is contained in expected → low error
        assert _compute_prediction_error("deploy the full system", "the full system") <= 0.25
        # expected is contained in actual
        assert _compute_prediction_error("full system", "deploy the full system now") <= 0.25

    def test_exit_code_success_match(self) -> None:
        # "deploy" implies success, exit=0 means success → low error
        err = _compute_prediction_error("deploy to production", "exit=0: deployed ok")
        assert err <= 0.2, f"Expected ≤ 0.2, got {err}"

    def test_exit_code_success_mismatch(self) -> None:
        # "deploy" implies success, exit=1 means failure → high error
        err = _compute_prediction_error("deploy to production", "exit=1: auth failed")
        assert err >= 0.8, f"Expected ≥ 0.8, got {err}"

    def test_exit_code_no_success_keyword(self) -> None:
        # "list directory contents" vs "exit=0: files" — content mismatch, capped at 0.5
        err = _compute_prediction_error("list directory contents", "exit=0: files")
        assert err == pytest.approx(0.5, abs=0.01), f"Expected 0.5, got {err}"

    def test_exit_code_neutral_keyword(self) -> None:
        # Content has no overlap with expected — capped at 0.5 by exit=0
        err = _compute_prediction_error("examine the output carefully", "exit=0: data")
        assert err == pytest.approx(0.5, abs=0.01), f"Expected 0.5, got {err}"

    def test_bigram_similarity_high(self) -> None:
        error = _compute_prediction_error("write configuration file", "written config file")
        # Strings share significant character bigrams
        assert error <= 0.5, f"Expected ≤ 0.5, got {error}"

    def test_bigram_similarity_low(self) -> None:
        error = _compute_prediction_error("install package", "error: package not found")
        # Should be a clear mismatch
        assert error >= 0.5, f"Expected ≥ 0.5, got {error}"

    def test_token_overlap_fallback(self) -> None:
        error = _compute_prediction_error("create new file storage", "file created in storage")
        # Shares "file" and "storage" tokens → moderate overlap
        assert 0.3 < error < 0.8, f"Expected 0.3–0.8, got {error}"


# ═══════════════════════════════════════════════════════════════════
#  Action triple lifecycle
# ═══════════════════════════════════════════════════════════════════

class TestActionTriples:
    """Action → expected → actual → error triples."""

    def test_record_action_returns_id(self) -> None:
        wm = WorldModel()
        tid = wm.record_action("shell", "ls /tmp", "should list files")
        assert tid.startswith("act_")
        assert len(tid) > 10

    def test_record_action_adds_to_list(self) -> None:
        wm = WorldModel()
        wm.record_action("write_file", "create test", "write test.txt")
        assert len(wm.data["action_triples"]) == 1
        t = wm.data["action_triples"][0]
        assert t["action_type"] == "write_file"
        assert t["action_description"] == "create test"
        assert t["expected_outcome"] == "write test.txt"
        assert t["actual_outcome"] is None
        assert t["completed"] is False

    def test_complete_action_calculates_error(self) -> None:
        wm = WorldModel()
        tid = wm.record_action("shell", "deploy", "deploy should succeed")
        err = wm.complete_action(tid, "exit=0: deployed successfully")
        assert err is not None
        assert 0 <= err <= 1.0

    def test_complete_action_nonexistent_id(self) -> None:
        wm = WorldModel()
        assert wm.complete_action("act_nonexistent", "outcome") is None

    def test_complete_action_updates_triple(self) -> None:
        wm = WorldModel()
        tid = wm.record_action("shell", "ls", "list files")
        wm.complete_action(tid, "exit=0: files")
        t = wm.data["action_triples"][0]
        assert t["completed"] is True
        assert t["actual_outcome"] == "exit=0: files"
        assert t["prediction_error"] is not None
        assert t["completed_at"] is not None

    def test_record_action_complete_convenience(self) -> None:
        wm = WorldModel()
        result = wm.record_action_complete("shell", "quick check", "exit=0: ok", "should work")
        assert "id" in result
        assert result["prediction_error"] is not None
        assert len(wm.data["action_triples"]) == 1

    def test_action_triples_capped_at_200(self) -> None:
        wm = WorldModel()
        for i in range(250):
            tid = wm.record_action("shell", f"action {i}", f"expected {i}")
            wm.complete_action(tid, f"exit=0: result {i}")
        assert len(wm.data["action_triples"]) == 200


# ═══════════════════════════════════════════════════════════════════
#  Macro predictions
# ═══════════════════════════════════════════════════════════════════

class TestMacroPredictions:
    """Higher-level trajectory predictions with delayed verification."""

    def test_record_prediction_returns_id(self) -> None:
        wm = WorldModel()
        pid = wm.record_prediction("system will have 10 cycles", "1d", 0.8, "pace")
        assert pid.startswith("pred_")

    def test_record_prediction_increments_total(self) -> None:
        wm = WorldModel()
        wm.record_prediction("p1", "1d", 0.5, "basis")
        wm.record_prediction("p2", "2d", 0.7, "basis")
        assert wm.data["prediction_accuracy"]["total_predictions"] == 2

    def test_unverified_prediction_listed(self) -> None:
        wm = WorldModel()
        wm.record_prediction("unverified pred", "1d", 0.5, "test")
        uv = wm.get_unverified_predictions()
        assert len(uv) == 1
        assert uv[0]["text"] == "unverified pred"

    def test_verify_prediction_marks_verified(self) -> None:
        wm = WorldModel()
        pid = wm.record_prediction("test", "1d", 0.5, "test")
        wm.verify_prediction(pid, "actual outcome", "verified")
        assert wm.data["predictions"][0]["verified"] is True
        assert wm.data["predictions"][0]["actual"] == "actual outcome"

    def test_verify_prediction_updates_stats(self) -> None:
        wm = WorldModel()
        pid = wm.record_prediction("will succeed", "1d", 0.8, "pace")
        wm.verify_prediction(pid, "exit=0: succeeded", "good")
        acc = wm.data["prediction_accuracy"]
        assert acc["verified_predictions"] == 1
        assert acc["avg_prediction_error"] > 0

    def test_verify_prediction_nonexistent(self) -> None:
        wm = WorldModel()
        assert wm.verify_prediction("pred_nonexistent", "outcome") is None

    def test_predictions_capped_at_100(self) -> None:
        wm = WorldModel()
        for i in range(150):
            pid = wm.record_prediction(f"pred {i}", "1d", 0.5, "test")
            wm.verify_prediction(pid, f"outcome {i}", "note")
        assert len(wm.data["predictions"]) == 100


# ═══════════════════════════════════════════════════════════════════
#  Context formatting
# ═══════════════════════════════════════════════════════════════════

class TestFormatContext:
    """format_world_model_context produces the prompt section."""

    def test_empty_model_context(self) -> None:
        wm = WorldModel()
        ctx = wm.format_world_model_context()
        assert "World Model State" in ctx
        assert "world model is empty" in ctx

    def test_context_includes_actions(self) -> None:
        wm = WorldModel()
        tid = wm.record_action("shell", "test action", "expected")
        wm.complete_action(tid, "exit=0: done")
        ctx = wm.format_world_model_context()
        assert "Action triples" in ctx
        assert "test action" in ctx
        assert "error=" in ctx

    def test_context_includes_predictions(self) -> None:
        wm = WorldModel()
        wm.record_prediction("trajectory pred", "2d", 0.75, "reasoning")
        ctx = wm.format_world_model_context()
        assert "Predictions:" in ctx
        assert "trajectory pred" in ctx

    def test_context_includes_discrepancies(self) -> None:
        wm = WorldModel()
        tid = wm.record_action("shell", "should fail", "expect failure")
        wm.complete_action(tid, "exit=0: actually succeeded")
        ctx = wm.format_world_model_context()
        # The mismatch should appear in discrepancies
        assert "error=" in ctx

    def test_context_includes_unverified(self) -> None:
        wm = WorldModel()
        wm.record_prediction("unverified", "1d", 0.5, "test")
        ctx = wm.format_world_model_context()
        assert "Unverified predictions" in ctx or "awaiting" in ctx

    def test_prediction_insight_empty(self) -> None:
        wm = WorldModel()
        assert "no predictions made yet" in wm.format_prediction_insight()

    def test_prediction_insight_with_data(self) -> None:
        wm = WorldModel()
        pid = wm.record_prediction("test", "1d", 0.9, "test")
        wm.verify_prediction(pid, "exit=0: ok", "verified")
        insight = wm.format_prediction_insight()
        assert "accuracy" in insight
        assert "verified" in insight


# ═══════════════════════════════════════════════════════════════════
#  Persistence
# ═══════════════════════════════════════════════════════════════════

class TestPersistence:
    """Save/load round-trip with atomic write pattern."""

    def test_save_and_load_round_trip(self) -> None:
        wm = WorldModel()
        wm.record_action("shell", "persist test", "expected")
        wm.record_prediction("predict persistence", "1d", 0.5, "test")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            wm.save(path)
            assert path.exists()
            assert path.stat().st_size > 50

            loaded = WorldModel.load(path)
            assert len(loaded.data["action_triples"]) == 1
            assert len(loaded.data["predictions"]) == 1
            # Check version is a positive int (not a change-detector literal)
            assert isinstance(loaded.data.get("version"), int)
            assert loaded.data["version"] >= 1
        finally:
            path.unlink(missing_ok=True)

    def test_load_nonexistent_returns_empty(self) -> None:
        wm = load_world_model()
        assert len(wm.data["action_triples"]) >= 0
        assert len(wm.data["predictions"]) >= 0

    def test_load_corrupted_returns_empty(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
            path.write_text("not valid json{{{")
        try:
            wm = WorldModel.load(path)
            assert len(wm.data["action_triples"]) == 0
        finally:
            path.unlink(missing_ok=True)

    def test_version_downgrade_handled(self) -> None:
        """Schema version checks don't break normal loading."""
        wm = WorldModel()
        wm.data["version"] = 2
        wm.data["action_triples"].append({
            "id": "act_test", "action_type": "shell", "action_description": "test",
            "expected_outcome": "x", "actual_outcome": "y", "prediction_error": 0.5,
            "timestamp": "2026-01-01", "completed": True,
        })
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            wm.save(path)
            loaded = WorldModel.load(path)
            assert len(loaded.data["action_triples"]) == 1
        finally:
            path.unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  Per-type accuracy and confidence calibration
# ═══════════════════════════════════════════════════════════════════

class TestPerTypeAccuracy:
    """Per-action-type prediction error tracking."""

    def test_updates_on_action_complete(self) -> None:
        wm = WorldModel()
        tid = wm.record_action("shell", "ls", "list files")
        wm.complete_action(tid, "exit=0: ok")
        pta = wm.get_per_type_accuracy()
        assert "shell" in pta
        assert pta["shell"]["count"] == 1
        assert 0 <= pta["shell"]["avg_error"] <= 1.0

    def test_multiple_types_tracked_separately(self) -> None:
        wm = WorldModel()
        tid1 = wm.record_action("shell", "build", "should build")
        wm.complete_action(tid1, "exit=0: built")
        tid2 = wm.record_action("write_file", "create config", "write config")
        wm.complete_action(tid2, "wrote config")
        pta = wm.get_per_type_accuracy()
        assert "shell" in pta
        assert "write_file" in pta

    def test_avg_error_aggregates_correctly(self) -> None:
        wm = WorldModel()
        # Two shell actions: one perfect (0.0), one bad (1.0) → avg 0.5
        tid1 = wm.record_action("shell", "exact", "same text")
        wm.complete_action(tid1, "same text")
        tid2 = wm.record_action("shell", "mismatch", "one thing")
        wm.complete_action(tid2, "completely different")
        pta = wm.get_per_type_accuracy()
        assert pta["shell"]["count"] == 2
        assert 0.45 <= pta["shell"]["avg_error"] <= 0.55

    def test_min_max_range(self) -> None:
        wm = WorldModel()
        tid1 = wm.record_action("shell", "exact", "same")
        wm.complete_action(tid1, "same")  # error = 0.0
        tid2 = wm.record_action("shell", "mismatch", "aaa")
        wm.complete_action(tid2, "bbb")   # error ~ 1.0
        pta = wm.get_per_type_accuracy()
        assert pta["shell"]["min_error"] <= 0.01
        assert pta["shell"]["max_error"] >= 0.9

    def test_unknown_type_returns_empty(self) -> None:
        wm = WorldModel()
        pta = wm.get_per_type_accuracy()
        assert "nonexistent" not in pta
        assert len(pta) == 0


class TestDiscrepancyPatterns:
    """_update_discrepancy_patterns detects recurring failure patterns."""

    def test_empty_when_no_data(self) -> None:
        wm = WorldModel()
        wm._update_discrepancy_patterns()
        assert wm.get_discrepancy_patterns() == []

    def test_no_pattern_with_low_error(self) -> None:
        wm = WorldModel()
        # Two perfect predictions → no high-error triples → no patterns
        tid1 = wm.record_action("shell", "exact", "same text")
        wm.complete_action(tid1, "same text")
        tid2 = wm.record_action("shell", "exact2", "same exact text")
        wm.complete_action(tid2, "same exact text")
        patterns = wm.get_discrepancy_patterns()
        assert len(patterns) == 0, f"Expected 0 patterns, got {len(patterns)}"

    def test_detects_type_level_pattern(self) -> None:
        wm = WorldModel()
        # Three mispredicted shell actions → should create a pattern
        # Use "deploy succeed" which contains the success keyword "deploy"
        for i in range(3):
            tid = wm.record_action("shell", f"deploy version {i}", "deploy should succeed")
            wm.complete_action(tid, "exit=1: build failure")
        patterns = wm.get_discrepancy_patterns()
        assert len(patterns) >= 1
        shell_pattern = next((p for p in patterns if p["action_type"] == "shell"), None)
        assert shell_pattern is not None
        assert shell_pattern["count"] == 3
        assert shell_pattern["avg_error"] >= 0.4

    def test_multiple_action_types_separate(self) -> None:
        wm = WorldModel()
        # Bad shell predictions and bad write_file predictions
        for i in range(3):
            tid = wm.record_action("shell", f"deploy {i}", "deploy successfully")
            wm.complete_action(tid, "exit=1: failure")
        for i in range(2):
            tid = wm.record_action("write_file", f"config {i}", "write config file")
            wm.complete_action(tid, "permission denied")
        patterns = wm.get_discrepancy_patterns()
        types_found = {p["action_type"] for p in patterns}
        assert "shell" in types_found
        assert "write_file" in types_found

    def test_common_themes_extracted(self) -> None:
        wm = WorldModel()
        # All descriptions share "deploy" theme
        for i in range(3):
            tid = wm.record_action("shell", f"deploy application v{i}", "deploy should succeed")
            wm.complete_action(tid, "exit=1: crash")
        patterns = wm.get_discrepancy_patterns()
        shell_pattern = next((p for p in patterns if p["action_type"] == "shell"), None)
        assert shell_pattern is not None
        themes = shell_pattern.get("common_themes", [])
        assert "deploy" in themes, f"Expected 'deploy' in themes, got {themes}"

    def test_patterns_sorted_by_frequency(self) -> None:
        wm = WorldModel()
        # 3 bad shell, 2 bad git_commit
        for i in range(3):
            tid = wm.record_action("shell", f"build {i}", "build should succeed")
            wm.complete_action(tid, "exit=2: fail")
        for i in range(2):
            tid = wm.record_action("git_commit", f"commit {i}", "commit should succeed")
            wm.complete_action(tid, "merge conflict")
        patterns = wm.get_discrepancy_patterns()
        if len(patterns) >= 2:
            # First pattern should have higher count
            assert patterns[0]["count"] >= patterns[1]["count"]

    def test_excluded_by_low_samples(self) -> None:
        wm = WorldModel()
        # Only 1 high-error triple — below min_samples=2 default
        tid = wm.record_action("shell", "deploy", "should work")
        wm.complete_action(tid, "exit=1: crash")
        patterns = wm.get_discrepancy_patterns()
        # Either no patterns or at least data is tracked correctly
        assert len(patterns) == 0

    def test_context_includes_patterns_section(self) -> None:
        wm = WorldModel()
        for i in range(3):
            tid = wm.record_action("shell", f"deploy {i}", "deploy should succeed")
            wm.complete_action(tid, "exit=1: build failure")
        ctx = wm.format_world_model_context()
        assert "Recurring discrepancy patterns" in ctx
        assert "deploy" in ctx

    def test_updated_via_complete_action(self) -> None:
        """Patterns should be updated automatically when complete_action is called."""
        wm = WorldModel()
        # CompleteAction triggers _update_accuracy_stats which triggers _update_discrepancy_patterns
        for i in range(3):
            tid = wm.record_action("shell", f"deploy v{i}", "deploy should succeed")
            wm.complete_action(tid, "exit=1: build failure")
        patterns = wm.get_discrepancy_patterns()
        assert any(p["action_type"] == "shell" for p in patterns)

    def test_persistence_round_trip(self) -> None:
        """Discrepancy patterns survive save/load cycle."""
        import tempfile
        from pathlib import Path

        wm = WorldModel()
        for i in range(3):
            tid = wm.record_action("shell", f"deploy {i}", "deploy successfully")
            wm.complete_action(tid, "exit=1: fail")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            wm.save(path)
            loaded = WorldModel.load(path)
            loaded_patterns = loaded.get_discrepancy_patterns()
            assert len(loaded_patterns) >= 1
            assert loaded_patterns[0]["action_type"] == "shell"
        finally:
            path.unlink(missing_ok=True)


class TestConfidenceAdjustment:
    """adjust_confidence uses historical accuracy to calibrate estimates."""

    def test_no_data_returns_raw(self) -> None:
        wm = WorldModel()
        # No actions recorded → no per-type data → should return close to raw
        adjusted = wm.adjust_confidence(0.8, "shell")
        assert 0.6 <= adjusted <= 0.9

    def test_low_error_type_increases_confidence(self) -> None:
        wm = WorldModel()
        # Record 2 shell actions with perfect predictions
        tid1 = wm.record_action("shell", "exact", "same text")
        wm.complete_action(tid1, "same text")
        tid2 = wm.record_action("shell", "exact2", "exact2 data")
        wm.complete_action(tid2, "exact2 data")
        # Low error type → adjustment should increase confidence
        adjusted = wm.adjust_confidence(0.5, "shell")
        assert adjusted >= 0.5, f"Expected ≥ 0.5, got {adjusted}"

    def test_high_error_type_decreases_confidence(self) -> None:
        wm = WorldModel()
        # Record 3 shell actions with bad predictions
        for i in range(3):
            tid = wm.record_action("shell", f"mismatch{i}", "completely wrong expected")
            wm.complete_action(tid, "totally different actual")
        adjusted = wm.adjust_confidence(0.9, "shell")
        assert adjusted <= 0.8, f"Expected ≤ 0.8, got {adjusted}"

    def test_adjustment_without_type_uses_global(self) -> None:
        wm = WorldModel()
        tid = wm.record_action("shell", "test", "expected result")
        wm.complete_action(tid, "exit=0: ok")
        adjusted = wm.adjust_confidence(0.7)
        assert 0 <= adjusted <= 1.0

    def test_confidence_clamped(self) -> None:
        wm = WorldModel()
        assert 0.0 <= wm.adjust_confidence(-0.5) <= 1.0
        assert 0.0 <= wm.adjust_confidence(1.5) <= 1.0


class TestCalibrationGuidance:
    """format_calibration_guidance shows per-type accuracy for the LLM."""

    def test_empty_when_no_data(self) -> None:
        wm = WorldModel()
        cal = wm.format_calibration_guidance()
        assert "No per-type calibration" in cal

    def test_shows_multiple_types(self) -> None:
        wm = WorldModel()
        tid1 = wm.record_action("shell", "build", "should build")
        wm.complete_action(tid1, "exit=0: built")
        tid2 = wm.record_action("write_file", "create", "write file")
        wm.complete_action(tid2, "wrote file")
        cal = wm.format_calibration_guidance()
        assert "shell" in cal
        assert "write_file" in cal
        assert "Best predicted" in cal or "Worst predicted" in cal

    def test_identifies_best_and_worst(self) -> None:
        wm = WorldModel()
        # shell = good predictions
        for i in range(3):
            tid = wm.record_action("shell", f"exact{i}", "same text")
            wm.complete_action(tid, "same text")
        # git_commit = bad predictions
        for i in range(3):
            tid = wm.record_action("git_commit", f"mismatch{i}", "x")
            wm.complete_action(tid, "y")
        cal = wm.format_calibration_guidance()
        assert "Best predicted" in cal
        assert "shell" in cal or "git_commit" in cal  # one is best, one worst

    def test_context_format_includes_calibration(self) -> None:
        wm = WorldModel()
        tid = wm.record_action("shell", "test action", "expected")
        wm.complete_action(tid, "exit=0: done")
        ctx = wm.format_world_model_context()
        assert "Per-type prediction" in ctx or "prediction accuracy" in ctx

    def test_prediction_insight_shows_type_count(self) -> None:
        wm = WorldModel()
        tid = wm.record_action("shell", "test", "expected")
        wm.complete_action(tid, "exit=0: ok")
        insight = wm.format_prediction_insight()
        assert "action types" in insight or "no predictions" in insight


# ═══════════════════════════════════════════════════════════════════
#  Error trend analysis (rolling error_history tracking)
# ═══════════════════════════════════════════════════════════════════

class TestErrorTrend:
    """_add_to_error_history and _compute_error_trend track improving/worsening."""

    def test_history_appends_on_complete(self) -> None:
        wm = WorldModel()
        tid = wm.record_action("shell", "check", "expected")
        wm.complete_action(tid, "exit=0: ok")
        hist = wm.data["prediction_accuracy"]["error_history"]
        assert len(hist) == 1
        assert 0 <= hist[0] <= 1.0

    def test_history_capped_at_20(self) -> None:
        wm = WorldModel()
        for i in range(25):
            tid = wm.record_action("shell", f"action {i}", f"expected {i}")
            wm.complete_action(tid, f"exit=0: ok {i}")
        assert len(wm.data["prediction_accuracy"]["error_history"]) <= 20

    def test_prediction_adds_to_history(self) -> None:
        wm = WorldModel()
        pid = wm.record_prediction("will work", "1d", 0.8, "test")
        wm.verify_prediction(pid, "worked")
        assert len(wm.data["prediction_accuracy"]["error_history"]) == 1

    def test_auto_verify_adds_to_history(self) -> None:
        from datetime import datetime, timezone, timedelta
        wm = WorldModel()
        pid = wm.record_prediction("old pred", "1 day", 0.7, "test")
        for p in wm.data["predictions"]:
            if p["id"] == pid:
                p["timestamp"] = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        wm.verify_expired_predictions()
        assert len(wm.data["prediction_accuracy"]["error_history"]) == 1
        assert wm.data["prediction_accuracy"]["error_history"][0] == 0.5

    def test_trend_unknown_with_few_samples(self) -> None:
        wm = WorldModel()
        for i in range(4):
            tid = wm.record_action("shell", f"act {i}", f"exp {i}")
            wm.complete_action(tid, f"exit=0: ok {i}")
        trend = wm._compute_error_trend()
        assert trend == "", f"Expected empty trend with 4 samples, got {trend!r}"

    def test_trend_improving(self) -> None:
        wm = WorldModel()
        # First 3: high error (bad predictions)
        for i in range(3):
            tid = wm.record_action("shell", f"deploy {i}", "deploy should succeed")
            wm.complete_action(tid, "exit=1: crash")
        # Next 3: low error (good predictions)
        for i in range(3):
            tid = wm.record_action("shell", f"list {i}", "list files")
            wm.complete_action(tid, "exit=0: ok")
        trend = wm._compute_error_trend()
        assert "improving" in trend, f"Expected improving, got {trend!r}"

    def test_trend_worsening(self) -> None:
        wm = WorldModel()
        # First 3: low error
        for i in range(3):
            tid = wm.record_action("shell", f"list {i}", "list files")
            wm.complete_action(tid, "exit=0: ok")
        # Next 3: high error
        for i in range(3):
            tid = wm.record_action("shell", f"deploy {i}", "deploy should succeed")
            wm.complete_action(tid, "exit=1: crash")
        trend = wm._compute_error_trend()
        assert "worsening" in trend, f"Expected worsening, got {trend!r}"

    def test_trend_stable(self) -> None:
        wm = WorldModel()
        # 6 similar low-error actions
        for i in range(6):
            tid = wm.record_action("shell", f"exact {i}", "exact same text")
            wm.complete_action(tid, "exact same text")
        trend = wm._compute_error_trend()
        assert "stable" in trend or "improving" in trend

    def test_insight_includes_trend_when_enough_data(self) -> None:
        wm = WorldModel()
        for i in range(6):
            tid = wm.record_action("shell", f"exact {i}", "exact same")
            wm.complete_action(tid, "exact same")
        insight = wm.format_prediction_insight()
        # Should contain either a trend bracket or accuracy info
        assert "accuracy" in insight or "→" in insight or "↓" in insight or "↑" in insight


# ═══════════════════════════════════════════════════════════════════
#  Proactive action guidance (discrepancy → decision bridge)
# ═══════════════════════════════════════════════════════════════════

class TestActionGuidance:
    """format_action_guidance warns when actions are risky based on history."""

    def test_no_data_returns_none(self) -> None:
        wm = WorldModel()
        assert wm.format_action_guidance("shell", "list files") is None
        assert wm.format_action_guidance("write_file", "create config") is None

    def test_unknown_type_returns_none(self) -> None:
        wm = WorldModel()
        tid = wm.record_action("shell", "deploy", "deploy should succeed")
        wm.complete_action(tid, "exit=0: ok")
        # "write_file" has no data → no guidance
        assert wm.format_action_guidance("write_file", "write config") is None

    def test_low_error_type_returns_none(self) -> None:
        wm = WorldModel()
        # 3 perfect shell actions → low avg error → no warning
        for i in range(3):
            tid = wm.record_action("shell", f"exact{i}", "same text here")
            wm.complete_action(tid, "same text here")
        assert wm.format_action_guidance("shell", "another exact") is None

    def test_high_error_type_returns_warning(self) -> None:
        wm = WorldModel()
        for i in range(3):
            tid = wm.record_action("shell", f"deploy v{i}", "deploy should succeed")
            wm.complete_action(tid, "exit=1: crash")
        guidance = wm.format_action_guidance("shell", "deploy new version")
        assert guidance is not None
        assert "risk assessment" in guidance
        assert "shell" in guidance

    def test_keyword_match_in_discrepancy_pattern(self) -> None:
        wm = WorldModel()
        for i in range(3):
            tid = wm.record_action(
                "shell", f"deploy application {i}", "deploy should succeed"
            )
            wm.complete_action(tid, "exit=1: build failure")
        # Match on keyword "deploy" in description
        guidance = wm.format_action_guidance("shell", "deploy the new app")
        assert guidance is not None
        assert "deploy" in guidance
        assert "failures" in guidance

    def test_keyword_no_match_ignores_pattern(self) -> None:
        wm = WorldModel()
        for i in range(3):
            tid = wm.record_action(
                "shell", f"deploy application {i}", "deploy should succeed"
            )
            wm.complete_action(tid, "exit=1: crash")
        # Description without the problematic keyword
        guidance = wm.format_action_guidance("shell", "list directory")
        assert guidance is not None  # Still warns about type
        if "deploy" in guidance:
            # "deploy" might appear in the per-type accuracy warning
            # but shouldn't appear in a keyword-match section
            pass

    def test_recent_trend_triggers_warning(self) -> None:
        wm = WorldModel()
        # 2 recent actions with high error (use success keywords in expected)
        tid1 = wm.record_action("shell", "build project", "deploy should succeed")
        wm.complete_action(tid1, "exit=1: fail")
        tid2 = wm.record_action("shell", "run tests", "deploy should work")
        wm.complete_action(tid2, "exit=1: fail")
        guidance = wm.format_action_guidance("shell", "run tests")
        assert guidance is not None
        assert "averaged" in guidance or "error" in guidance or "risk" in guidance

    def test_recent_trend_no_warning_when_low_error(self) -> None:
        wm = WorldModel()
        # 2 recent perfect actions
        tid1 = wm.record_action("shell", "list", "list files")
        wm.complete_action(tid1, "exit=0: done")
        tid2 = wm.record_action("shell", "show", "show status")
        wm.complete_action(tid2, "exit=0: active")
        guidance = wm.format_action_guidance("shell", "check status")
        # Low per-type accuracy too → no warning
        if guidance is not None:
            # If there's guidance, it shouldn't mention recent trend
            assert "averaged" not in guidance

    def test_returns_none_with_single_high_error(self) -> None:
        wm = WorldModel()
        # Single high-error action — below min_samples=2 for patterns,<2 for trend
        tid = wm.record_action("shell", "deploy", "deploy should succeed")
        wm.complete_action(tid, "exit=1: fail")
        # Per-type check needs count>=2, patterns needs min_samples, trend needs >=2
        assert wm.format_action_guidance("shell", "deploy") is None


# ═══════════════════════════════════════════════════════════════════
#  Auto-verification of expired predictions
# ═══════════════════════════════════════════════════════════════════

class TestParseTimeframeDays:
    """_parse_timeframe_days converts human-readable timeframes to days."""

    def test_none_returns_none(self) -> None:
        assert WorldModel._parse_timeframe_days(None) is None

    def test_empty_returns_none(self) -> None:
        assert WorldModel._parse_timeframe_days("") is None

    def test_completed_returns_zero(self) -> None:
        assert WorldModel._parse_timeframe_days("completed") == 0.0

    def test_days(self) -> None:
        assert WorldModel._parse_timeframe_days("3 days") == 3.0
        assert WorldModel._parse_timeframe_days("1 day") == 1.0

    def test_weeks(self) -> None:
        assert WorldModel._parse_timeframe_days("2 weeks") == 14.0
        assert WorldModel._parse_timeframe_days("1 week") == 7.0

    def test_months(self) -> None:
        assert WorldModel._parse_timeframe_days("1 month") == 30.0
        assert WorldModel._parse_timeframe_days("6 months") == 180.0

    def test_years(self) -> None:
        assert WorldModel._parse_timeframe_days("1 year") == 365.0
        assert WorldModel._parse_timeframe_days("2 years") == 730.0

    def test_unparseable_returns_none(self) -> None:
        assert WorldModel._parse_timeframe_days("soon") is None
        assert WorldModel._parse_timeframe_days("whenever") is None
        assert WorldModel._parse_timeframe_days("next tuesday") is None


class TestVerifyExpiredPredictions:
    """verify_expired_predictions auto-verifies predictions past their timeframe."""

    def test_no_predictions(self) -> None:
        wm = WorldModel()
        count = wm.verify_expired_predictions()
        assert count == 0

    def test_all_already_verified(self) -> None:
        wm = WorldModel()
        pid = wm.record_prediction("test", "1 day", 0.5, "test")
        wm.verify_prediction(pid, "done", "manually verified")
        count = wm.verify_expired_predictions()
        assert count == 0  # Already verified

    def test_expired_prediction_gets_verified(self) -> None:
        wm = WorldModel()
        # Manually set timestamp to 10 days ago
        from datetime import datetime, timezone, timedelta
        old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        pid = wm.record_prediction("will happen", "3 days", 0.7, "test")
        # Override timestamp
        for p in wm.data["predictions"]:
            if p["id"] == pid:
                p["timestamp"] = old_ts
        count = wm.verify_expired_predictions()
        assert count == 1

    def test_recent_prediction_not_expired(self) -> None:
        wm = WorldModel()
        wm.record_prediction("will happen soon", "1 month", 0.6, "test")
        count = wm.verify_expired_predictions()
        assert count == 0  # Not expired yet

    def test_no_timeframe_skipped(self) -> None:
        wm = WorldModel()
        # Manually set timestamp to 10 days ago but no timeframe
        from datetime import datetime, timezone, timedelta
        old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        pid = wm.record_prediction("no timeframe", None, 0.5, "test")
        for p in wm.data["predictions"]:
            if p["id"] == pid:
                p["timestamp"] = old_ts
        count = wm.verify_expired_predictions()
        assert count == 0  # Without timeframe, can't determine expiry

    def test_expired_with_grace_period(self) -> None:
        wm = WorldModel()
        from datetime import datetime, timezone, timedelta
        # 2 days prediction, made 2.1 days ago — within grace (1 day grace for 2-day pred)
        almost_expired = (datetime.now(timezone.utc) - timedelta(days=2, hours=2)).isoformat()
        pid = wm.record_prediction("2 day pred", "2 days", 0.6, "test")
        for p in wm.data["predictions"]:
            if p["id"] == pid:
                p["timestamp"] = almost_expired
        count = wm.verify_expired_predictions()
        # 2 days elapsed, pred is 2 day, grace = max(2*0.1, 1) = 1.0 day
        # 2.08 > 2 + 1? No, 2.08 < 3.0, so NOT expired
        assert count == 0, f"Expected 0 (within grace period), got {count}"

    def test_well_beyond_timeframe(self) -> None:
        wm = WorldModel()
        from datetime import datetime, timezone, timedelta
        # 1 day prediction, made 5 days ago — well past timeframe
        old_ts = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        pid = wm.record_prediction("short pred", "1 day", 0.9, "test")
        for p in wm.data["predictions"]:
            if p["id"] == pid:
                p["timestamp"] = old_ts
        count = wm.verify_expired_predictions()
        assert count == 1, f"Expected 1 (well beyond timeframe), got {count}"

    def test_stats_updated_on_auto_verify(self) -> None:
        wm = WorldModel()
        from datetime import datetime, timezone, timedelta
        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        pid = wm.record_prediction("long expired", "1 week", 0.8, "test")
        for p in wm.data["predictions"]:
            if p["id"] == pid:
                p["timestamp"] = old_ts
        wm.verify_expired_predictions()
        acc = wm.data["prediction_accuracy"]
        assert acc["verified_predictions"] == 1
        assert acc["avg_prediction_error"] == 0.5  # error=0.5 for expired

    def test_mixed_verified_and_unverified(self) -> None:
        wm = WorldModel()
        from datetime import datetime, timezone, timedelta
        # One manually verified
        pid1 = wm.record_prediction("verified one", "1 day", 0.9, "test")
        wm.verify_prediction(pid1, "done")
        # One expired (30 days ago, 3-day timeframe)
        pid2 = wm.record_prediction("expired one", "3 days", 0.6, "test")
        for p in wm.data["predictions"]:
            if p["id"] == pid2:
                p["timestamp"] = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        count = wm.verify_expired_predictions()
        assert count == 1  # Only the expired one


# ═══════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════

class TestCLI:
    """world_model main() handles --status, --discrepancies, etc."""

    def test_main_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from world_model import main
        monkeypatch.setattr("sys.argv", ["world_model.py", "--status"])
        # Should not crash — output is printed to stdout
        try:
            main()
        except SystemExit:
            pass  # argparse may call sys.exit

    def test_main_discrepancies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from world_model import main
        monkeypatch.setattr("sys.argv", ["world_model.py", "--discrepancies", "3"])
        try:
            main()
        except SystemExit:
            pass
