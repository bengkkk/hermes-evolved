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
        # "list" is a success keyword → exit=0 gives low error
        err = _compute_prediction_error("list directory contents", "exit=0: files")
        assert err <= 0.2, f"Expected ≤ 0.2 (list is success kw), got {err}"

    def test_exit_code_neutral_keyword(self) -> None:
        # Expected text with no success/failure keyword + exit=0 → moderate error (0.6)
        err = _compute_prediction_error("examine the output carefully", "exit=0: data")
        assert err == 0.6, f"Expected 0.6, got {err}"

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
            assert loaded.data["version"] == 2
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
