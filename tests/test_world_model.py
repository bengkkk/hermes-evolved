"""Comprehensive tests for world_model.py — Gap 6 (World Model).

Tests the full predict → act → observe → learn pipeline:

  - _compute_prediction_error edge cases
  - Action triple lifecycle (record → complete → error calculation)
  - Macro prediction lifecycle (record → verify → accuracy stats)
  - Per-type accuracy tracking
  - Discrepancy pattern detection
  - Confidence adjustment calibration
  - Action risk guidance
  - Timeframe parsing
  - Expired prediction auto-verification
  - Trend computation
  - Persistence (save/load round-trip)
  - Context formatting
  - Calibration guidance formatting

All tests are hermetic — no file I/O or API calls.
"""

from __future__ import annotations

import copy
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

import pytest

# Ensure project root is importable
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from world_model import (
    WorldModel,
    _compute_prediction_error,
    _DEFAULT_WORLD_MODEL,
)


# ══════════════════════════════════════════════════════════════════════
#  _compute_prediction_error — unit tests
# ══════════════════════════════════════════════════════════════════════


class TestComputePredictionError:
    """Edge cases for the multi-strategy prediction error calculator."""

    def test_exact_match(self):
        """Exact string match → error 0.0."""
        assert _compute_prediction_error("hello world", "hello world") == 0.0

    def test_exact_match_trimmed(self):
        """Whitespace-trimmed match → error 0.0."""
        assert _compute_prediction_error("hello", "  hello  ") == 0.0

    def test_case_insensitive_match(self):
        """Case-insensitive match → error 0.0."""
        assert _compute_prediction_error("Hello World", "hello world") == 0.0

    def test_one_contains_the_other(self):
        """One string contains the other → error 0.25."""
        err = _compute_prediction_error("test file", "this is a test file")
        assert err == pytest.approx(0.25, abs=0.01)

    def test_empty_expected(self):
        """Empty expected string → error 1.0."""
        assert _compute_prediction_error("", "some output") == 1.0

    def test_empty_actual(self):
        """Empty actual string → error 1.0."""
        assert _compute_prediction_error("expected output", "") == 1.0

    def test_both_empty(self):
        """Both empty → error 1.0."""
        assert _compute_prediction_error("", "") == 1.0

    def test_non_str_expected_coerced(self):
        """LLM can emit ints where strings are expected — must not crash.

        Regression for the 2026-08-01 crash class ('int' object has no
        attribute 'strip'/'lower'): an int expected_outcome used to
        explode inside complete_action's except-path re-entry and kill
        the daemon cycle.  After coercion the comparison proceeds and a
        bounded error is returned.
        """
        err = _compute_prediction_error(123, "exit=0: 123")
        assert isinstance(err, float)
        assert 0.0 <= err <= 1.0

    def test_non_str_actual_coerced(self):
        """Non-string actual outcome must not crash either."""
        err = _compute_prediction_error("exit=0: 123", 123)
        assert isinstance(err, float)
        assert 0.0 <= err <= 1.0

    def test_record_action_int_expected_roundtrip(self):
        """record_action → complete_action with an int expected survives.

        Simulates the daemon path end-to-end: an int expected_outcome
        must be stringified before storage so complete_action can score
        it without raising.
        """
        wm = WorldModel()
        tid = wm.record_action("shell", "run test", 123, expected_source="llm")
        err = wm.complete_action(tid, "exit=0: output")
        assert err is not None
        assert 0.0 <= err <= 1.0
        triple = next(t for t in wm.data["action_triples"] if t["id"] == tid)
        assert isinstance(triple["expected_outcome"], str)

    # ── Write-file heuristics ──

    def test_write_file_match_with_path(self):
        """Write-file output naming the expected path → low error (0.15)."""
        err = _compute_prediction_error("write config to main.py", "Wrote main.py (245 bytes)")
        assert err == pytest.approx(0.15, abs=0.01)

    def test_write_file_match_created_variant(self):
        """'Created' variant — file path must appear in expected for 0.15."""
        # "config.yaml" NOT in expected "create config" → 0.25 (file written, path not named)
        err = _compute_prediction_error("create config", "Created config.yaml (120 bytes)")
        assert err == pytest.approx(0.25, abs=0.01)

    def test_write_file_match_created_with_path(self):
        """'Created' variant with path in expected → 0.15."""
        err = _compute_prediction_error("create config.yaml", "Created config.yaml (120 bytes)")
        assert err == pytest.approx(0.15, abs=0.01)

    def test_write_file_no_path_match(self):
        """Write output not naming the expected path → error 0.25."""
        err = _compute_prediction_error("write some code", "Wrote main.py (245 bytes)")
        assert err == pytest.approx(0.25, abs=0.01)

    def test_write_file_overwrote_variant(self):
        """'Overwrote' variant — path not named → 0.25."""
        err = _compute_prediction_error("update config", "Overwrote config.yaml (30 bytes)")
        assert err == pytest.approx(0.25, abs=0.01)

    def test_write_file_overwrote_with_path(self):
        """'Overwrote' with path in expected → 0.15."""
        err = _compute_prediction_error("overwrite config.yaml", "Overwrote config.yaml (30 bytes)")
        assert err == pytest.approx(0.15, abs=0.01)

    # ── Exit code heuristics ──

    def test_exit_zero_success(self):
        """exit=0 with content similarity → error based on content, capped at 0.5."""
        err = _compute_prediction_error("deploy should succeed", "exit=0: deployed successfully")
        # Bigram jaccard ~0.41 → 0.4, capped by exit=0 at min(0.4, 0.5)=0.4
        assert err == pytest.approx(0.4, abs=0.01)

    def test_exit_zero_neutral(self):
        """exit=0 with completely different content → error 0.5 (exit=0 cap)."""
        err = _compute_prediction_error("list directory", "exit=0: file1.txt  file2.txt")
        # Token overlap minimal → 1.0, capped by exit=0 at min(1.0, 0.5)=0.5
        assert err == pytest.approx(0.5, abs=0.01)

    def test_exit_one_failure_expected_success(self):
        """exit=1 when success was expected → high error (0.85)."""
        err = _compute_prediction_error("deploy should succeed", "exit=1: build failure — dependency not found")
        assert err == pytest.approx(0.85, abs=0.01)

    def test_exit_one_with_neutral_expected(self):
        """exit=1 with expected text containing 'check' keyword → 0.85
        because 'check' is in the success_expected keyword list."""
        # 'check' keyword triggers success_expected=True → exit=1 → 0.85
        err = _compute_prediction_error("check something", "exit=1: command not found")
        assert err == pytest.approx(0.85, abs=0.01)

    def test_exit_one_truly_neutral(self):
        """exit=1 with expected having NO success keywords → content-based error floored at 0.5."""
        err = _compute_prediction_error("inspect something", "exit=1: some issue occurred")
        # Bigram jaccard ~0.11 → 0.8 (low overlap), floored by exit≠0 at max(0.8, 0.5)=0.8
        assert err == pytest.approx(0.8, abs=0.01)

    # Note: exit code regex captures "exit=N:" pattern specifically.
    # A message like "Process exited with code 1" is NOT matched
    # by the current regex (no colon after exit=N).

    # ── Tool failure/success keywords ──

    def test_tool_failure_keyword(self):
        """Failure keyword without success keyword → 0.85."""
        err = _compute_prediction_error("do something", "Traceback: Permission denied")
        assert err == pytest.approx(0.85, abs=0.01)

    def test_tool_success_keyword_only(self):
        """Success keyword without failure keyword → 0.15."""
        err = _compute_prediction_error("do something", "all tasks ok")
        assert err == pytest.approx(0.15, abs=0.01)

    def test_tool_both_keywords_passed_failed(self):
        """Both success ('passed') and failure ('failed') keywords present.
        Falls through to exit code/bigram, and since neither matches here,
        falls to bigram/token level."""
        err = _compute_prediction_error("do something", "3 passed, 1 failed")
        # "passed" → tool_success ✓, "failed" → tool_failure ✓
        # Both true → neither branch fires → falls through
        # exit= not found, bigram is low → token level, no overlap → 1.0
        assert err == pytest.approx(1.0, abs=0.01)

    def test_tool_zero_failed_is_success(self):
        """'0 failed' is a success count, not a failure keyword."""
        err = _compute_prediction_error(
            "tests will pass", "556/556 tests passed, 0 failed, exit=0"
        )
        # 'passed' → tool_success ✓; '0 failed' neutralized → no failure
        # keyword → 0.15, then exit=0 caps at 0.5 (min → stays 0.15)
        assert err == pytest.approx(0.15, abs=0.01)

    def test_tool_nonzero_failed_is_failure(self):
        """A non-zero failure count remains a failure keyword."""
        err = _compute_prediction_error("tests will pass", "1 failed, exit=1")
        # 'failed' → tool_failure ✓, no success keyword → 0.85,
        # then exit=1 floors at 0.5 (max → stays 0.85)
        assert err == pytest.approx(0.85, abs=0.01)

    def test_tool_both_succeeded_failed(self):
        """'succeeded' (success) and 'failed' (failure) — both true."""
        err = _compute_prediction_error("do something", "succeeded but also failed")
        # Both tool_failure and tool_success are True → falls through
        # No exit code, low bigram → token level → likely 1.0
        assert err >= 0.5

    # ── Bigram similarity ──

    def test_high_bigram_similarity(self):
        """High bigram overlap → medium-low error (0.4 via bigram jaccard ~0.5)."""
        err = _compute_prediction_error("install flask package", "install flask in venv")
        # Bigram Jaccard ≈ 0.5 → returns 0.4
        assert err == pytest.approx(0.4, abs=0.05)

    def test_medium_bigram_similarity(self):
        """Medium bigram overlap."""
        err = _compute_prediction_error("install package for project", "install project dependencies")
        # Shared bigrams raise jaccard above 0.35 → returns 0.4
        assert err == pytest.approx(0.4, abs=0.05)

    def test_no_bigram_overlap(self):
        """No bigram overlap → higher error (falls to token level)."""
        err = _compute_prediction_error("abc", "xyz")
        assert err == pytest.approx(1.0, abs=0.01)

    # ── Token overlap (word-level) ──

    def test_partial_token_overlap(self):
        """Some word overlap caught by bigram jaccard > 0 → 0.8 (code uses bigram for all jaccard>0)."""
        err = _compute_prediction_error("foo bar baz", "foo qux bar")
        # Bigram Jaccard ≈ 0.43 → returns 0.4
        assert err == pytest.approx(0.4, abs=0.05)

    def test_token_low_overlap(self):
        """Minimal overlap caught by bigram jaccard > 0 → 0.6."""
        err = _compute_prediction_error("alpha beta gamma", "alpha delta epsilon")
        # Bigram Jaccard ≈ 0.22 → returns 0.6
        assert err == pytest.approx(0.6, abs=0.05)

    # ── Realistic Hermes output patterns ──

    def test_git_commit_success(self):
        """Successful git commit — content differs from expected, capped at 0.5."""
        err = _compute_prediction_error("commit changes", "exit=0: 1 file changed, 1 insertion(+)")
        assert err == pytest.approx(0.5, abs=0.01)

    def test_install_package_already_installed(self):
        """pip install with already-satisfied output.
        'Requirement already satisfied' is now recognized as an
        informational non-error (0.15), matching the semantic success
        of the install command.
        """
        err = _compute_prediction_error("install pytest", "Requirement already satisfied: pytest")
        assert err == pytest.approx(0.15, abs=0.01)

    def test_shell_list_dir(self):
        """Shell directory listing — content different from expected, capped at 0.5."""
        err = _compute_prediction_error("list workspace files", "exit=0: main.py  utils.py  tests/")
        assert err == pytest.approx(0.5, abs=0.01)

    def test_shell_failure_permission(self):
        """Shell permission denied."""
        err = _compute_prediction_error("read protected file", "exit=1: cat: Permission denied")
        assert err >= 0.5

    def test_write_file_permission_denied(self):
        """Write file that fails with permission error."""
        err = _compute_prediction_error("write critical config", "permission denied: /etc/config.yaml")
        assert err >= 0.5

    def test_tool_output_with_exit_0_no_stdout(self):
        """exit=0 with empty stdout → 0.5 (capped by exit=0)."""
        err = _compute_prediction_error("run quiet command", "exit=0: ")
        # "exit=0: " stripped → "exit=0:" → no bigram overlap with expected
        # falls to token level → 1.0, capped by exit=0 at 0.5
        assert err == pytest.approx(0.5, abs=0.01)

    def test_error_contains_expected_phrase(self):
        """Actual contains the expected phrase as substring."""
        err = _compute_prediction_error("file should be found", "file should be found in /tmp")
        assert err == pytest.approx(0.25, abs=0.01)

    def test_complete_mismatch(self):
        """No overlap at all between expected and actual."""
        err = _compute_prediction_error("write python script", "exit=1: npm ERR! code E404")
        assert err >= 0.5

    def test_none_inputs(self):
        """Edge case: empty string fallback."""
        err = _compute_prediction_error("something", "")
        assert err == 1.0

    # ── Exit code vs keyword precedence ──

    def test_exit_0_overrides_failure_keywords(self):
        """exit=0 with failure keywords in output → capped at 0.5 (command succeeded).
        The exit code says the command worked, but content differs from expected."""
        err = _compute_prediction_error(
            "resolve hostname",
            "exit=0: (no output — host not found)",
        )
        # tool_failure triggered (not found), _error=0.85, capped by exit=0 at 0.5
        assert err == pytest.approx(0.5, abs=0.01)

    def test_exit_1_still_high_with_success_keywords(self):
        """exit=1 still gives high error (0.85) when expected text
        contains success keywords — the command really did fail."""
        err = _compute_prediction_error(
            "deploy should succeed",
            "exit=1: Crash: permission denied",
        )
        assert err == pytest.approx(0.85, abs=0.01)

    def test_exit_0_with_traceback_keyword(self):
        """exit=0 with 'Traceback' in output → capped at 0.5."""
        err = _compute_prediction_error(
            "run script",
            "exit=0: Traceback printed but exit was 0",
        )
        # tool_failure triggered (traceback), _error=0.85, capped by exit=0 at 0.5
        assert err == pytest.approx(0.5, abs=0.01)

    # ── Pathological exit code edge cases ──

    def test_no_exit_code_uses_keyword_heuristic(self):
        """Without an exit= marker, the keyword heuristic still
        fires correctly for clearly failed commands."""
        err = _compute_prediction_error(
            "do something",
            "Permission denied: /etc/config.yaml",
        )
        assert err >= 0.5

    def test_exit_0_always_low(self):
        """exit=0 with content mismatch → capped at 0.5 (not the old hardcoded 0.15)."""
        err = _compute_prediction_error("this will definitely fail", "exit=0: it worked")
        # Content completely different → 1.0, capped by exit=0 at 0.5
        assert err == pytest.approx(0.5, abs=0.01)

    def test_short_expected_and_actual_no_match(self):
        """Very short strings with no overlap."""
        err = _compute_prediction_error("abc", "def")
        assert err == pytest.approx(1.0, abs=0.01)

    def test_version_numbers_in_output(self):
        """Output contains version numbers — content mismatch, capped at 0.5."""
        err = _compute_prediction_error("check version", "exit=0: Python 3.11.5")
        assert err == pytest.approx(0.5, abs=0.01)


# ══════════════════════════════════════════════════════════════════════
#  Timeframe parsing
# ══════════════════════════════════════════════════════════════════════


class TestParseTimeframeDays:
    """Test the static timeframe parser used by verify_expired_predictions."""

    def test_none_returns_none(self):
        assert WorldModel._parse_timeframe_days(None) is None

    def test_empty_string_returns_none(self):
        assert WorldModel._parse_timeframe_days("") is None

    def test_days(self):
        assert WorldModel._parse_timeframe_days("3 days") == 3.0

    def test_day_singular(self):
        assert WorldModel._parse_timeframe_days("1 day") == 1.0

    def test_weeks(self):
        assert WorldModel._parse_timeframe_days("2 weeks") == 14.0

    def test_week_singular(self):
        assert WorldModel._parse_timeframe_days("1 week") == 7.0

    def test_months(self):
        assert WorldModel._parse_timeframe_days("2 months") == 60.0

    def test_years(self):
        assert WorldModel._parse_timeframe_days("1 year") == 365.0

    def test_completed(self):
        assert WorldModel._parse_timeframe_days("completed") == 0.0

    def test_a_day(self):
        assert WorldModel._parse_timeframe_days("a day") == 1.0

    def test_one_week(self):
        assert WorldModel._parse_timeframe_days("one week") == 7.0

    def test_unparsable_returns_none(self):
        assert WorldModel._parse_timeframe_days("soon") is None

    def test_decimal_days(self):
        assert WorldModel._parse_timeframe_days("0.5 days") == 0.5

    def test_every_phrase_does_not_match(self):
        """'every 2h' style is not supported by timeframe parser."""
        assert WorldModel._parse_timeframe_days("every 2h") is None

    def test_immediate_returns_zero(self):
        """'immediate' timeframe → 0.0 days (auto-verified on first check)."""
        assert WorldModel._parse_timeframe_days("immediate") == 0.0

    def test_immediately_returns_zero(self):
        """'immediately' timeframe → 0.0 days (same as immediate)."""
        assert WorldModel._parse_timeframe_days("immediately") == 0.0


# ══════════════════════════════════════════════════════════════════════
#  WorldModel — action triple lifecycle
# ══════════════════════════════════════════════════════════════════════


class TestActionTripleLifecycle:
    """Test record_action → complete_action → error calculation."""

    def test_record_action_creates_triple(self):
        wm = WorldModel()
        tid = wm.record_action("shell", "test action", "should work")
        assert tid.startswith("act_")
        triples = wm.data["action_triples"]
        assert len(triples) == 1
        assert triples[0]["completed"] is False
        assert triples[0]["action_type"] == "shell"
        assert triples[0]["expected_outcome"] == "should work"
        assert triples[0]["actual_outcome"] is None
        assert triples[0]["prediction_error"] is None

    def test_record_action_truncates_large_parameter_values(self):
        wm = WorldModel()
        big_content = "x" * 5000
        tid = wm.record_action(
            "write_file",
            "write large file",
            "file written",
            parameters={"path": "/tmp/big.txt", "content": big_content},
        )
        stored = wm.data["action_triples"][0]["action_parameters"]
        # Content is capped; short values (path) pass through untouched.
        assert stored["path"] == "/tmp/big.txt"
        assert len(stored["content"]) < 400
        assert stored["content"].endswith("...[truncated]")
        # The stored prefix is still usable for token matching.
        assert stored["content"].startswith("x" * 300)

    def test_record_action_keeps_small_parameter_values(self):
        wm = WorldModel()
        tid = wm.record_action(
            "shell",
            "list dir",
            "lists files",
            parameters={"command": "ls -la /tmp"},
        )
        stored = wm.data["action_triples"][0]["action_parameters"]
        assert stored["command"] == "ls -la /tmp"

    def test_complete_action_fills_triple(self):
        wm = WorldModel()
        tid = wm.record_action("write_file", "write config", "should succeed")
        err = wm.complete_action(tid, "Wrote config.yaml (50 bytes)")
        assert err is not None
        assert 0.0 <= err <= 1.0
        triple = wm.data["action_triples"][0]
        assert triple["completed"] is True
        assert triple["actual_outcome"] == "Wrote config.yaml (50 bytes)"
        assert triple["prediction_error"] is not None
        assert triple["completed_at"] is not None

    def test_complete_nonexistent_id_returns_none(self):
        wm = WorldModel()
        err = wm.complete_action("nonexistent_id", "output")
        assert err is None

    def test_double_complete_returns_none(self):
        wm = WorldModel()
        tid = wm.record_action("shell", "test", "ok")
        wm.complete_action(tid, "exit=0: done")
        err2 = wm.complete_action(tid, "exit=0: done again")
        assert err2 is None  # Already completed

    def test_record_action_complete_convenience(self):
        wm = WorldModel()
        result = wm.record_action_complete("shell", "test", "exit=0: done", "should work")
        assert "id" in result
        assert "prediction_error" in result
        assert result["prediction_error"] is not None
        assert len(wm.data["action_triples"]) == 1
        assert wm.data["action_triples"][0]["completed"] is True

    def test_multiple_actions_tracked(self):
        wm = WorldModel()
        tid1 = wm.record_action("shell", "first", "ok")
        tid2 = wm.record_action("write_file", "second", "ok")
        wm.complete_action(tid1, "exit=0: done")
        wm.complete_action(tid2, "Wrote file.txt (10 bytes)")
        assert len(wm.data["action_triples"]) == 2

    def test_action_triple_limit_200(self):
        wm = WorldModel()
        for i in range(250):
            wm.record_action("shell", f"action {i}", "ok")
        assert len(wm.data["action_triples"]) == 200

    def test_correct_prediction_error_computed(self):
        wm = WorldModel()
        tid = wm.record_action("shell", "deploy", "deploy should succeed")
        wm.complete_action(tid, "exit=1: build failed")
        err = wm.data["action_triples"][0]["prediction_error"]
        assert err >= 0.5  # exit=1 with success-expected keywords → high error

    def test_stats_updated_after_complete(self):
        wm = WorldModel()
        tid = wm.record_action("shell", "test", "ok")
        wm.complete_action(tid, "exit=0: done")
        acc = wm.data["prediction_accuracy"]
        assert acc["total_triples"] == 1
        assert acc["avg_triple_error"] > 0

    def test_per_type_accuracy_after_complete(self):
        wm = WorldModel()
        tid = wm.record_action("write_file", "test", "write file")
        wm.complete_action(tid, "Wrote test.py (10 bytes)")
        pta = wm.get_per_type_accuracy()
        assert "write_file" in pta
        assert pta["write_file"]["count"] == 1

    def test_low_error_action_type(self):
        """Multiple actions of same type — content mismatch capped at 0.5 by exit=0."""
        wm = WorldModel()
        for i in range(3):
            tid = wm.record_action("install_package", f"install pkg{i}", "should install")
            wm.complete_action(tid, "exit=0: Successfully installed")
        pta = wm.get_per_type_accuracy()
        # Each action: bigram overlap gives 0.6, capped by exit=0 at min(0.6, 0.5)=0.5
        assert abs(pta["install_package"]["avg_error"] - 0.5) < 0.1

    def test_high_error_action_type(self):
        """Multiple failing actions → high avg error."""
        wm = WorldModel()
        for i in range(3):
            tid = wm.record_action("deploy", f"deploy v{i}", "should succeed")
            wm.complete_action(tid, "exit=1: Deployment failed")
        pta = wm.get_per_type_accuracy()
        assert pta["deploy"]["avg_error"] >= 0.5

    def test_action_from_different_types(self):
        wm = WorldModel()
        wm.record_action_complete("shell", "list files", "exit=0: done", "ok")
        wm.record_action_complete("write_file", "write file", "Wrote f.txt", "ok")
        wm.record_action_complete("git_commit", "commit", "exit=0: committed", "ok")
        wm.record_action_complete("install_package", "install", "exit=0: installed", "ok")
        pta = wm.get_per_type_accuracy()
        assert len(pta) == 4


# ══════════════════════════════════════════════════════════════════════
#  WorldModel — macro predictions
# ══════════════════════════════════════════════════════════════════════


class TestMacroPredictions:
    """Test record_prediction → verify_prediction → accuracy stats."""

    def test_record_prediction(self):
        wm = WorldModel()
        pid = wm.record_prediction("system will grow", "3 days", 0.7, "trend")
        assert pid.startswith("pred_")
        assert wm.data["prediction_accuracy"]["total_predictions"] == 1

    def test_verify_correct_prediction_low_error(self):
        """Prediction where actual closely matches expected → low error."""
        wm = WorldModel()
        pid = wm.record_prediction("will have many cycles", "1 day", 0.7, "rate")
        err = wm.verify_prediction(pid, "will have many cycles now — 50")
        # "will have many cycles" is a substring of the actual → 0.25
        assert err is not None
        assert err <= 0.4

    def test_verify_incorrect_prediction(self):
        wm = WorldModel()
        pid = wm.record_prediction("will succeed", "1 day", 0.9, "confidence")
        err = wm.verify_prediction(pid, "exit=1: completely failed")
        assert err is not None
        acc = wm.data["prediction_accuracy"]
        assert acc["verified_predictions"] == 1
        assert err >= 0.5  # Mismatch should be high

    def test_verify_nonexistent_prediction_returns_none(self):
        wm = WorldModel()
        assert wm.verify_prediction("nonexistent", "outcome") is None

    def test_double_verify_returns_none(self):
        wm = WorldModel()
        pid = wm.record_prediction("test", "1 day", 0.5, "test")
        wm.verify_prediction(pid, "outcome")
        assert wm.verify_prediction(pid, "outcome again") is None

    def test_prediction_limit_100(self):
        wm = WorldModel()
        for i in range(150):
            wm.record_prediction(f"pred {i}", None, 0.5, "")
        assert len(wm.data["predictions"]) == 100

    def test_correct_prediction_tracking(self):
        wm = WorldModel()
        pid = wm.record_prediction("will work", "1 day", 0.8, "")
        wm.verify_prediction(pid, "exit=0: success")
        acc = wm.data["prediction_accuracy"]
        # Error 0.5 (content mismatch capped by exit=0) → counts as incorrect
        assert acc["correct_predictions"] == 0
        assert acc["incorrect_predictions"] == 1

    def test_incorrect_prediction_tracking(self):
        wm = WorldModel()
        pid = wm.record_prediction("will work", "1 day", 0.8, "")
        wm.verify_prediction(pid, "exit=1: failed")
        acc = wm.data["prediction_accuracy"]
        assert acc["incorrect_predictions"] == 1

    def test_avg_prediction_error_updates(self):
        wm = WorldModel()
        pid = wm.record_prediction("test", "1 day", 0.5, "")
        wm.verify_prediction(pid, "some outcome")
        acc = wm.data["prediction_accuracy"]
        assert acc["avg_prediction_error"] > 0

    def test_unverified_predictions(self):
        wm = WorldModel()
        wm.record_prediction("future event", "3 days", 0.5, "thinking")
        wm.record_prediction("past event", "1 day", 0.5, "thinking")
        pid2 = wm.record_prediction("verified one", "1 day", 0.5, "thinking")
        wm.verify_prediction(pid2, "happened")
        unverified = wm.get_unverified_predictions()
        assert len(unverified) == 2  # first two not verified


class TestPredictionStatsReconciliation:
    """Counters must stay consistent when predictions age out of the 100 cap."""

    def test_counters_reconcile_after_cap_trim(self):
        """Verified predictions dropped by the 100-cap no longer count."""
        wm = WorldModel()
        # Fill past the cap; the first 50 will be trimmed away.
        pids = []
        for i in range(150):
            pids.append(wm.record_prediction(f"pred {i}", "1 day", 0.8, ""))
        assert len(wm.data["predictions"]) == 100
        # The trimmed 50 were never verified — but simulate the drift:
        # pre-fix, total_predictions would read 150.  Post-fix it must
        # equal the retained list length.
        acc = wm.data["prediction_accuracy"]
        assert acc["total_predictions"] == 100
        assert acc["verified_predictions"] == 0

    def test_verified_prediction_trimmed_no_longer_counts(self):
        """The key regression: a verified prediction that falls off the
        cap must not keep inflating verified/correct counters."""
        wm = WorldModel()
        # Verify the first prediction, then push it out of the cap.
        pid0 = wm.record_prediction("will work", "1 day", 0.8, "")
        wm.verify_prediction(pid0, "exit=0: success")
        acc = wm.data["prediction_accuracy"]
        assert acc["verified_predictions"] == 1
        assert acc["incorrect_predictions"] == 1
        # Now record 100 more — pid0 ages out of the retained list.
        for i in range(100):
            wm.record_prediction(f"filler {i}", None, 0.5, "")
        assert len(wm.data["predictions"]) == 100
        assert all(p["id"] != pid0 for p in wm.data["predictions"])
        acc = wm.data["prediction_accuracy"]
        assert acc["verified_predictions"] == 0
        assert acc["incorrect_predictions"] == 0
        assert acc["total_predictions"] == 100

    def test_uncertain_expiry_classification_survives_reconcile(self):
        """Auto-verified (uncertain) predictions stay out of correct/incorrect."""
        from datetime import datetime, timezone, timedelta

        wm = WorldModel()
        pid = wm.record_prediction("long expired", "1 week", 0.8, "test")
        for p in wm.data["predictions"]:
            if p["id"] == pid:
                p["timestamp"] = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        wm.verify_expired_predictions()
        acc = wm.data["prediction_accuracy"]
        assert acc["verified_predictions"] == 1
        assert acc["correct_predictions"] == 0
        assert acc["incorrect_predictions"] == 0
        # Explicit class stamp present; reconcile preserves it
        pred = next(p for p in wm.data["predictions"] if p["id"] == pid)
        assert pred["outcome_class"] == "uncertain"
        wm._reconcile_prediction_stats()
        acc = wm.data["prediction_accuracy"]
        assert acc["verified_predictions"] == 1
        assert acc["correct_predictions"] == 0
        assert acc["incorrect_predictions"] == 0
        assert acc["avg_prediction_error"] == 0.5

    def test_legacy_records_without_stamp_classified_by_rules(self):
        """Records written before outcome_class existed are inferred correctly."""
        wm = WorldModel()
        pid = wm.record_prediction("will work", "1 day", 0.8, "")
        wm.verify_prediction(pid, "exit=0: success")  # error 0.5 → incorrect
        pred = next(p for p in wm.data["predictions"] if p["id"] == pid)
        # Simulate a legacy record: drop the stamp entirely
        del pred["outcome_class"]
        # Also add a legacy uncertain record (expiry-style note, no stamp)
        pid2 = wm.record_prediction("expired thing", "1 week", 0.7, "")
        p2 = next(p for p in wm.data["predictions"] if p["id"] == pid2)
        p2["verified"] = True
        p2["error"] = 0.5
        p2["verification_note"] = "Auto-verified: timeframe '1 week' (7 days) expired 3.0 days ago"
        p2["verified_at"] = "2026-07-31T00:00:00+00:00"
        wm._reconcile_prediction_stats()
        acc = wm.data["prediction_accuracy"]
        assert acc["verified_predictions"] == 2
        assert acc["incorrect_predictions"] == 1  # the 0.5 manual one
        assert acc["correct_predictions"] == 0
        # uncertain (expiry note) contributes to neither counter
        assert acc["verified_predictions"] == acc["correct_predictions"] + acc["incorrect_predictions"] + 1


# ══════════════════════════════════════════════════════════════════════
#  Expired prediction auto-verification
# ══════════════════════════════════════════════════════════════════════


class TestExpiredPredictions:
    """Test verify_expired_predictions with various timeframe scenarios."""

    def test_no_expired_predictions(self):
        wm = WorldModel()
        count = wm.verify_expired_predictions()
        assert count == 0

    def test_expired_prediction_auto_verified(self):
        """A prediction made 10 days ago with a 1-day timeframe should be expired."""
        wm = WorldModel()
        old_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        wm.data["predictions"].append({
            "id": "pred_old",
            "text": "will complete soon",
            "timeframe": "1 day",
            "confidence": 0.7,
            "basis": "guess",
            "verified": False,
            "timestamp": old_time,
        })
        count = wm.verify_expired_predictions()
        assert count == 1
        pred = wm.data["predictions"][0]
        assert pred["verified"] is True
        assert pred["error"] == 0.5  # Uncertain
        assert "timeframe expired" in pred["actual"]

    def test_recent_prediction_not_expired(self):
        """A prediction made 1 hour ago with a 3-day timeframe should NOT be expired."""
        wm = WorldModel()
        recent_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        wm.data["predictions"].append({
            "id": "pred_recent",
            "text": "will happen in future",
            "timeframe": "3 days",
            "confidence": 0.5,
            "basis": "",
            "verified": False,
            "timestamp": recent_time,
        })
        count = wm.verify_expired_predictions()
        assert count == 0

    def test_prediction_no_timeframe_skipped(self):
        """Predictions without a timeframe should not be auto-verified."""
        wm = WorldModel()
        old_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        wm.data["predictions"].append({
            "id": "pred_no_tf",
            "text": "something",
            "timeframe": None,
            "confidence": 0.5,
            "basis": "",
            "verified": False,
            "timestamp": old_time,
        })
        count = wm.verify_expired_predictions()
        assert count == 0  # No timeframe → skipped

    def test_already_verified_skipped(self):
        """Already-verified predictions should not be re-verified."""
        wm = WorldModel()
        old_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        wm.data["predictions"].append({
            "id": "pred_verified",
            "text": "old prediction",
            "timeframe": "1 day",
            "confidence": 0.7,
            "basis": "",
            "verified": True,
            "timestamp": old_time,
            "actual": "already verified",
            "error": 0.2,
        })
        count = wm.verify_expired_predictions()
        assert count == 0

    def test_expired_prediction_accuracy_stats_updated(self):
        wm = WorldModel()
        old_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        wm.data["predictions"].append({
            "id": "pred_exp",
            "text": "will complete soon",
            "timeframe": "1 day",
            "confidence": 0.6,
            "basis": "",
            "verified": False,
            "timestamp": old_time,
        })
        wm.verify_expired_predictions()
        acc = wm.data["prediction_accuracy"]
        assert acc["verified_predictions"] >= 1
        assert acc["avg_prediction_error"] == pytest.approx(0.5, abs=0.01)

    def test_expired_prediction_does_not_pollute_calibration(self):
        """Auto-verified predictions deliberately skip _update_calibration.

        Auto-verified predictions always get error=0.5 (uncertain), which would
        pollute calibration buckets with systematic bias — every entry in a given
        confidence bucket would show error=0.5 regardless of accuracy.
        Calibration data should only reflect predictions where the system
        actually observed the outcome (via verify_prediction).
        """
        wm = WorldModel()
        old_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        wm.data["predictions"].append({
            "id": "pred_cal",
            "text": "will complete soon",
            "timeframe": "1 day",
            "confidence": 0.6,
            "basis": "",
            "verified": False,
            "timestamp": old_time,
        })
        wm.verify_expired_predictions()
        buckets = wm.data["prediction_accuracy"]["calibration_buckets"]
        # Calibration buckets should remain empty — auto-verification does not
        # pollute calibration data with uncertain (0.5) outcomes.
        assert len(buckets) == 0, (
            "Calibration buckets must not be populated by auto-verification "
            "(intentionally skipped to avoid systematic 0.5 bias)"
        )


# ══════════════════════════════════════════════════════════════════════
#  Pending prediction evidence verification
# ══════════════════════════════════════════════════════════════════════


class TestPendingPredictions:
    """verify_pending_predictions resolves non-expired predictions with evidence."""

    def test_no_predictions_returns_zero(self):
        wm = WorldModel()
        assert wm.verify_pending_predictions() == 0

    def test_unexpired_with_evidence_resolved(self):
        """A fresh prediction with decisive action-triple evidence is resolved now."""
        wm = WorldModel()
        wm.record_action_complete(
            "shell",
            "locate think_daemon.py via filesystem search",
            "exit=0: found /workspace/hermes-evolved/think_daemon.py",
            "expect to find the daemon script",
        )
        wm.record_prediction(
            "A filesystem search will reveal think_daemon.py under /workspace or /root",
            "1 day",
            0.8,
            "test basis",
        )
        count = wm.verify_pending_predictions()
        assert count == 1
        pred = wm.data["predictions"][0]
        assert pred["verified"] is True
        assert pred["error"] == pytest.approx(0.15, abs=0.01)
        # Evidence-resolved predictions are real observations → feed calibration
        acc = wm.data["prediction_accuracy"]
        assert acc["verified_predictions"] == 1
        assert acc["correct_predictions"] == 1
        assert acc["calibration_buckets"][4]["count"] == 1  # confidence 0.8 → bucket 4

    def test_contradicted_evidence_resolved_high_error(self):
        """Evidence showing failure resolves the prediction with error 0.85."""
        wm = WorldModel()
        wm.record_action_complete(
            "shell",
            "compile the world model module",
            "exit=1: SyntaxError in world_model.py line 42",
            "expect clean compile",
        )
        wm.record_prediction(
            "world_model.py will compile cleanly this cycle",
            "1 day",
            0.7,
            "test",
        )
        count = wm.verify_pending_predictions()
        assert count == 1
        pred = wm.data["predictions"][0]
        assert pred["verified"] is True
        assert pred["error"] == pytest.approx(0.85, abs=0.01)

    def test_unexpired_without_evidence_left_pending(self):
        """No decisive evidence → prediction stays unverified for a later cycle."""
        wm = WorldModel()
        wm.record_prediction("The sky will turn green tomorrow", "1 day", 0.5, "test")
        assert wm.verify_pending_predictions() == 0
        assert len(wm.get_unverified_predictions()) == 1

    def test_expired_left_for_expired_verifier(self):
        """Expired predictions are not touched — verify_expired_predictions owns them."""
        wm = WorldModel()
        old_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        wm.data["predictions"].append({
            "id": "pred_old",
            "text": "old prediction about think_daemon.py",
            "timeframe": "1 day",
            "confidence": 0.5,
            "basis": "",
            "verified": False,
            "timestamp": old_time,
        })
        wm.record_action_complete(
            "shell",
            "locate think_daemon.py",
            "exit=0: found think_daemon.py",
            "expect path",
        )
        assert wm.verify_pending_predictions() == 0
        assert len(wm.get_unverified_predictions()) == 1
        # The expired verifier still owns the old prediction
        assert wm.verify_expired_predictions() == 1
        assert len(wm.get_unverified_predictions()) == 0


# ══════════════════════════════════════════════════════════════════════
#  Discrepancy pattern detection
# ══════════════════════════════════════════════════════════════════════


class TestDiscrepancyPatterns:
    """Test automatic detection of recurring prediction failure patterns."""

    def test_no_patterns_with_few_samples(self):
        wm = WorldModel()
        wm.record_action_complete("shell", "test", "exit=0: done")
        patterns = wm.get_discrepancy_patterns()
        assert len(patterns) == 0  # Not enough high-error samples

    def test_pattern_detected_for_high_error_type(self):
        wm = WorldModel()
        for i in range(3):
            wm.record_action_complete(
                "shell", f"deploy to staging v{i}",
                "exit=1: Deployment failed horribly",
                "deploy should succeed"
            )
        patterns = wm.get_discrepancy_patterns()
        assert len(patterns) >= 1
        shell_pattern = next((p for p in patterns if p["action_type"] == "shell"), None)
        assert shell_pattern is not None
        assert shell_pattern["count"] >= 2  # At least 2 high-error
        assert "deploy" in shell_pattern.get("common_themes", [])

    def test_multiple_type_patterns(self):
        wm = WorldModel()
        for i in range(3):
            wm.record_action_complete("shell", f"deploy v{i}", "exit=1: failed", "should succeed")
        for i in range(3):
            wm.record_action_complete("write_file", f"write to /etc/config_{i}", "Permission denied", "should write")
        patterns = wm.get_discrepancy_patterns()
        types_found = {p["action_type"] for p in patterns}
        assert "shell" in types_found
        assert "write_file" in types_found

    def test_low_error_actions_dont_create_patterns(self):
        wm = WorldModel()
        for i in range(5):
            wm.record_action_complete("shell", f"successful cmd {i}", "exit=0: done", "should work")
        patterns = wm.get_discrepancy_patterns()
        assert len(patterns) == 0  # All low-error

    def test_pattern_description(self):
        wm = WorldModel()
        for i in range(3):
            wm.record_action_complete(
                "shell", f"attempt {i}", "exit=1: failed", "should succeed"
            )
        patterns = wm.get_discrepancy_patterns()
        if patterns:
            desc = patterns[0].get("description", "")
            assert "shell" in desc

    def test_pattern_last_observed_tracking(self):
        wm = WorldModel()
        for i in range(3):
            wm.record_action_complete(
                "shell", "deploy v{i}", "exit=1: fail", "should succeed"
            )
        patterns = wm.get_discrepancy_patterns()
        if patterns:
            assert "last_observed" in patterns[0]
            assert patterns[0]["last_observed"]  # Non-empty


# ══════════════════════════════════════════════════════════════════════
#  Confidence adjustment calibration
# ══════════════════════════════════════════════════════════════════════


class TestConfidenceAdjustment:
    """Test adjust_confidence with various historical data scenarios."""

    def test_no_data_returns_raw(self):
        wm = WorldModel()
        result = wm.adjust_confidence(0.8, "unknown_type")
        assert result == 0.8  # No data → unchanged

    def test_no_type_uses_global(self):
        wm = WorldModel()
        wm.record_action_complete("shell", "test", "exit=0: ok", "will succeed")
        # Has global data but no per-type for "different_type"
        result = wm.adjust_confidence(0.8, "different_type")
        assert result != 0.8  # Adjusted using global avg

    def test_low_error_type_increases_confidence(self):
        wm = WorldModel()
        for i in range(5):
            wm.record_action_complete(
                "install_package", f"install {i}", "exit=0: done", "should install"
            )
        raw = 0.5
        adj = wm.adjust_confidence(raw, "install_package")
        # Low error → confidence should increase or stay very close
        assert adj >= raw or abs(adj - raw) < 0.05

    def test_high_error_type_decreases_confidence(self):
        wm = WorldModel()
        for i in range(5):
            wm.record_action_complete(
                "deploy", f"deploy v{i}", "exit=1: crashed", "should succeed"
            )
        raw = 0.9
        adj = wm.adjust_confidence(raw, "deploy")
        # High error → confidence should decrease
        assert adj < raw

    def test_small_sample_blends_with_global(self):
        wm = WorldModel()
        # 1 sample of low-error type
        wm.record_action_complete("rare_type", "test", "exit=0: ok", "will work")
        # 5 samples of high-error type to affect global avg
        for i in range(5):
            wm.record_action_complete("deploy", f"deploy {i}", "exit=1: fail", "should succeed")
        raw = 0.5
        adj = wm.adjust_confidence(raw, "rare_type")
        assert 0.0 <= adj <= 1.0

    def test_confidence_clamped(self):
        wm = WorldModel()
        result = wm.adjust_confidence(1.5)  # Outside range
        assert 0.0 <= result <= 1.0
        result = wm.adjust_confidence(-0.5)
        assert 0.0 <= result <= 1.0

    def test_adjustment_vs_type_unknown(self):
        """Type with no data but global data available."""
        wm = WorldModel()
        wm.record_action_complete("shell", "test", "exit=1: fail", "should work")
        # Global avg_triple_error will be high (one triple, high error)
        result = wm.adjust_confidence(0.8, "rare_type")
        # Should decrease due to high global error
        assert 0.0 <= result <= 1.0


# ══════════════════════════════════════════════════════════════════════
#  Error history and trend
# ══════════════════════════════════════════════════════════════════════


class TestErrorHistoryAndTrend:
    """Test rolling error history and trend computation."""

    def test_error_history_appends(self):
        wm = WorldModel()
        wm._add_to_error_history(0.5)
        wm._add_to_error_history(0.3)
        history = wm.data["prediction_accuracy"]["error_history"]
        assert len(history) == 2
        assert history == [0.5, 0.3]

    def test_error_history_bounded_at_20(self):
        wm = WorldModel()
        for i in range(30):
            wm._add_to_error_history(0.5)
        history = wm.data["prediction_accuracy"]["error_history"]
        assert len(history) == 20

    def test_no_trend_with_few_samples(self):
        wm = WorldModel()
        for err in [0.5, 0.3]:
            wm._add_to_error_history(err)
        trend = wm._compute_error_trend()
        assert trend == ""  # Need >= 6 samples

    def test_improving_trend(self):
        wm = WorldModel()
        for err in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
            wm._add_to_error_history(err)
        trend = wm._compute_error_trend()
        assert "improving" in trend

    def test_worsening_trend(self):
        wm = WorldModel()
        for err in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            wm._add_to_error_history(err)
        trend = wm._compute_error_trend()
        assert "worsening" in trend

    def test_stable_trend(self):
        wm = WorldModel()
        for _ in range(8):
            wm._add_to_error_history(0.4)
        trend = wm._compute_error_trend()
        assert "stable" in trend or trend == ""

    def test_error_history_via_complete_action(self):
        wm = WorldModel()
        tid = wm.record_action("shell", "test", "ok")
        wm.complete_action(tid, "exit=0: done")
        assert len(wm.data["prediction_accuracy"]["error_history"]) == 1

    def test_error_history_via_verify_prediction(self):
        wm = WorldModel()
        pid = wm.record_prediction("test", "1 day", 0.5, "")
        wm.verify_prediction(pid, "some outcome")
        assert len(wm.data["prediction_accuracy"]["error_history"]) == 1

    def test_trend_with_exact_data_points(self):
        wm = WorldModel()
        for err in [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]:
            wm._add_to_error_history(err)
        trend = wm._compute_error_trend()
        assert "stable" in trend or trend == ""


# ══════════════════════════════════════════════════════════════════════
#  Calibration buckets
# ══════════════════════════════════════════════════════════════════════


class TestCalibrationBuckets:
    """Test confidence vs accuracy tracking in calibration buckets."""

    def test_bucket_created_for_confidence(self):
        wm = WorldModel()
        wm._update_calibration(0.3, 0.5)
        buckets = wm.data["prediction_accuracy"]["calibration_buckets"]
        assert len(buckets) >= 2  # bucket_idx = int(0.3 * 5) = 1
        assert buckets[1]["count"] == 1
        assert buckets[1]["total_error"] == 0.5
        assert buckets[1]["avg_error"] == 0.5

    def test_bucket_high_confidence(self):
        wm = WorldModel()
        wm._update_calibration(0.9, 0.2)
        buckets = wm.data["prediction_accuracy"]["calibration_buckets"]
        # bucket_idx = min(int(0.9 * 5), 4) = 4
        assert len(buckets) >= 5
        assert buckets[4]["count"] == 1

    def test_bucket_accumulation(self):
        wm = WorldModel()
        for _ in range(3):
            wm._update_calibration(0.5, 0.2)
        buckets = wm.data["prediction_accuracy"]["calibration_buckets"]
        # bucket_idx = int(0.5 * 5) = 2
        assert buckets[2]["count"] == 3
        assert buckets[2]["total_error"] == pytest.approx(0.6, abs=0.01)
        assert buckets[2]["avg_error"] == pytest.approx(0.2, abs=0.01)

    def test_zero_confidence_bucket(self):
        wm = WorldModel()
        wm._update_calibration(0.0, 0.9)
        buckets = wm.data["prediction_accuracy"]["calibration_buckets"]
        assert len(buckets) >= 1
        assert buckets[0]["count"] == 1

    def test_max_confidence_bucket(self):
        wm = WorldModel()
        wm._update_calibration(1.0, 0.0)
        buckets = wm.data["prediction_accuracy"]["calibration_buckets"]
        assert len(buckets) >= 5
        assert buckets[4]["count"] == 1


# ══════════════════════════════════════════════════════════════════════
#  Action guidance
# ══════════════════════════════════════════════════════════════════════


class TestActionGuidance:
    """Test format_action_guidance risk assessment."""

    def test_no_guidance_for_safe_type(self):
        wm = WorldModel()
        guidance = wm.format_action_guidance("unknown_type", "safe action")
        assert guidance is None

    def test_guidance_for_high_error_type(self):
        wm = WorldModel()
        for i in range(3):
            wm.record_action_complete(
                "shell", f"risky deploy {i}", "exit=1: failed", "should succeed"
            )
        guidance = wm.format_action_guidance("shell", "another risky deploy")
        assert guidance is not None
        indicator_words = ["risk", "error", "warning", "elevated"]
        assert any(ind in guidance.lower() for ind in indicator_words)

    def test_no_guidance_with_insufficient_data(self):
        """Single high-error action shouldn't trigger guidance (needs ≥2)."""
        wm = WorldModel()
        wm.record_action_complete("shell", "one fail", "exit=1: fail", "should work")
        guidance = wm.format_action_guidance("shell", "another action")
        # With only 1 sample, per-type check needs count >= 2
        assert guidance is None

    def test_guidance_via_discrepancy_pattern(self):
        """Multiple high-error actions of same type should create pattern match."""
        wm = WorldModel()
        for i in range(4):
            wm.record_action_complete(
                "shell", f"deploy to staging {i}", "exit=1: failed", "should succeed"
            )
        guidance = wm.format_action_guidance("shell", "deploy to staging")
        assert guidance is not None

    def test_no_guidance_for_other_type(self):
        """High error in one type doesn't affect another type."""
        wm = WorldModel()
        for i in range(4):
            wm.record_action_complete(
                "shell", f"deploy {i}", "exit=1: failed", "should succeed"
            )
        # write_file has no history at all
        guidance = wm.format_action_guidance("write_file", "write something")
        assert guidance is None

    def test_recurring_command_does_not_poison_generic_keywords(self):
        """A command repeated N times must not turn its generic words into
        themes that flag unrelated actions (observed false positive 2026-08-01:
        the word "list" from a recurring "State check: list current goals"
        command flagged a safe "List saved /tmp evidence files" action)."""
        wm = WorldModel()
        for i in range(8):
            wm.record_action_complete(
                "shell", f"safe maintenance task {i}",
                "exit=0: done", "should succeed",
            )
        for _ in range(4):
            wm.record_action_complete(
                "shell", "State check: list current goals and their statuses",
                "exit=1: failed", "should succeed",
            )
        for i in range(2):
            wm.record_action_complete(
                "shell", f"post maintenance task {i}",
                "exit=0: done", "should succeed",
            )
        # The exact recurring command is still flagged via recurring_description.
        g1 = wm.format_action_guidance(
            "shell", "State check: list current goals and their statuses"
        )
        assert g1 is not None
        assert "Recurring failing command" in g1
        # ...but an unrelated action merely containing the generic word "list"
        # is NOT flagged, and no generic keyword themes survive.
        g2 = wm.format_action_guidance("shell", "List saved /tmp evidence files")
        assert g2 is None
        themes = [
            w for p in wm.get_discrepancy_patterns()
            for w in p.get("common_themes", [])
        ]
        assert "list" not in themes


# ══════════════════════════════════════════════════════════════════════
#  Formatting methods
# ══════════════════════════════════════════════════════════════════════


class TestContextFormatting:
    """Test format_world_model_context and related formatting."""

    def test_empty_context(self):
        wm = WorldModel()
        ctx = wm.format_world_model_context()
        assert isinstance(ctx, str)
        assert "World Model State" in ctx

    def test_context_includes_recent_actions(self):
        wm = WorldModel()
        tid = wm.record_action("shell", "test action", "should work")
        wm.complete_action(tid, "exit=0: done")
        ctx = wm.format_world_model_context()
        assert "Recent action" in ctx

    def test_context_includes_accuracy(self):
        wm = WorldModel()
        tid = wm.record_action("shell", "test", "ok")
        wm.complete_action(tid, "exit=0: done")
        ctx = wm.format_world_model_context()
        assert "Action triples" in ctx

    def test_prediction_insight_no_predictions(self):
        wm = WorldModel()
        insight = wm.format_prediction_insight()
        assert "no predictions" in insight.lower()

    def test_prediction_insight_with_predictions(self):
        wm = WorldModel()
        pid = wm.record_prediction("will happen", "1 day", 0.7, "test")
        wm.verify_prediction(pid, "it happened")
        insight = wm.format_prediction_insight()
        assert "accuracy" in insight.lower()

    def test_calibration_guidance_empty(self):
        wm = WorldModel()
        cal = wm.format_calibration_guidance()
        assert "No per-type calibration data yet" in cal

    def test_calibration_guidance_with_data(self):
        wm = WorldModel()
        wm.record_action_complete("shell", "test", "exit=0: ok", "will work")
        wm.record_action_complete("write_file", "test2", "Wrote f.txt", "will write")
        cal = wm.format_calibration_guidance()
        assert "Per-type prediction accuracy" in cal
        assert "Best predicted" in cal
        assert "Worst predicted" in cal

    def test_context_shows_discrepancy_patterns(self):
        wm = WorldModel()
        for i in range(4):
            wm.record_action_complete(
                "shell", f"deploy {i}", "exit=1: failed", "should succeed"
            )
        ctx = wm.format_world_model_context()
        # Should show discrepancy patterns section
        assert "Recurring discrepancy" in ctx or "Biggest prediction error" in ctx

    def test_context_shows_unverified_predictions(self):
        wm = WorldModel()
        wm.record_prediction("future outcome", "3 days", 0.6, "thinking")
        ctx = wm.format_world_model_context()
        assert "Unverified predictions" in ctx


# ══════════════════════════════════════════════════════════════════════
#  Persistence
# ══════════════════════════════════════════════════════════════════════


class TestPersistence:
    """Test save/load round-trip for WorldModel."""

    def test_save_and_load_round_trip(self, tmp_path):
        wm = WorldModel()
        wm.record_action_complete("shell", "test", "exit=0: done", "should work")
        wm.record_prediction("future", "3 days", 0.7, "test")
        path = tmp_path / "test_world_model.json"
        wm.save(path)

        assert path.exists()
        data = json.loads(path.read_text())
        assert "action_triples" in data
        assert len(data["action_triples"]) == 1
        assert "predictions" in data
        assert len(data["predictions"]) == 1

        wm2 = WorldModel.load(path)
        assert len(wm2.data["action_triples"]) == 1
        assert len(wm2.data["predictions"]) == 1

    def test_load_nonexistent_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        wm = WorldModel.load(path)
        assert isinstance(wm, WorldModel)
        assert len(wm.data["action_triples"]) == 0

    def test_load_corrupted_file(self, tmp_path):
        path = tmp_path / "corrupted.json"
        path.write_text("{invalid json")
        wm = WorldModel.load(path)
        assert isinstance(wm, WorldModel)

    def test_load_version_absorbed_by_defaults(self):
        """When loading data with a version field, _merge_defaults overwrites
        version with the code's _DEFAULT_WORLD_MODEL version because 'version'
        is one of the built-in default keys (always reset to current schema)."""
        wm = WorldModel(data={"version": 99})
        # Version comes from _DEFAULT_WORLD_MODEL, not from loaded data
        assert wm.data["version"] == _DEFAULT_WORLD_MODEL["version"]

    def test_save_creates_file(self, tmp_path):
        wm = WorldModel()
        path = tmp_path / "test_save.json"
        wm.save(path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_storage_path_uses_evolve_dir(self, monkeypatch, tmp_path):
        """Storage path should be under evolve directory."""
        import data_layer
        monkeypatch.setattr(data_layer, '_resolve_evolve_dir', lambda: tmp_path)
        path = WorldModel.storage_path()
        assert "world_model.json" in str(path)

    # Note: this test needs tmp_path fixture but it's a method
    # Let me redefine it properly
    def test_storage_path_format(self):
        path = WorldModel.storage_path()
        assert path.name == "world_model.json"

    def test_load_recomputes_per_type_accuracy(self, tmp_path):
        wm = WorldModel()
        wm.record_action_complete("shell", "test", "exit=0: ok", "should work")
        wm.save(tmp_path / "wm.json")
        # Load should recompute per-type accuracy
        wm2 = WorldModel.load(tmp_path / "wm.json")
        pta = wm2.get_per_type_accuracy()
        assert "shell" in pta

    def test_save_default_path_merges_concurrent_records(self, monkeypatch, tmp_path):
        """A stale in-memory model must not destroy newer on-disk records.

        Regression test for the 2026-07-31 data-loss incident: a session
        holding a stale copy of the world model saved over data the daemon
        had recorded meanwhile, destroying 17 triples and 6 predictions.
        Save-to-default-path must union-merge by record id instead.
        """
        import data_layer
        monkeypatch.setattr(data_layer, "_resolve_evolve_dir", lambda: tmp_path)

        # Writer 1 records one triple and saves
        wm1 = WorldModel()
        wm1.record_action_complete("shell", "alpha action", "exit=0: done", "ok")
        wm1.save()

        # Writer 2 holds a stale model (never saw alpha), records its own
        # triple, and saves — the merge must keep BOTH triples.
        wm2 = WorldModel()
        wm2.record_action_complete("shell", "beta action", "exit=0: done", "ok")
        wm2.save()

        wm3 = WorldModel.load()
        descs = {t["action_description"] for t in wm3.data["action_triples"]}
        assert "alpha action" in descs
        assert "beta action" in descs
        assert len(wm3.data["action_triples"]) == 2

    def test_save_explicit_path_writes_verbatim(self, tmp_path):
        """Explicit paths (tests, exports) are not merge-affected."""
        wm = WorldModel()
        wm.record_action_complete("shell", "explicit", "exit=0", "ok")
        path = tmp_path / "explicit.json"
        wm.save(path)
        data = json.loads(path.read_text())
        assert len(data["action_triples"]) == 1


# ══════════════════════════════════════════════════════════════════════
#  Edge cases & stress
# ══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Stress and edge case tests."""

    def test_very_long_descriptions(self):
        wm = WorldModel()
        long_desc = "a" * 10000
        tid = wm.record_action("shell", long_desc, "ok")
        err = wm.complete_action(tid, "exit=0: done")
        assert err is not None
        assert 0.0 <= err <= 1.0

    def test_special_characters_in_action(self):
        wm = WorldModel()
        tid = wm.record_action("shell", "test with unicode: ñoño 🎉", "ok")
        err = wm.complete_action(tid, "exit=0: unicode: ñoño 🎉")
        assert err is not None

    def test_no_side_effects_between_instances(self):
        wm1 = WorldModel()
        wm2 = WorldModel()
        wm1.record_action("shell", "test", "ok")
        assert len(wm2.data["action_triples"]) == 0  # Isolated

    def test_unique_ids_different(self):
        wm = WorldModel()
        tid1 = wm.record_action("shell", "a", "ok")
        tid2 = wm.record_action("shell", "b", "ok")
        assert tid1 != tid2

    def test_repr(self):
        wm = WorldModel()
        assert "WorldModel" in repr(wm)
        assert "triples=0" in repr(wm)

    def test_repr_after_actions(self):
        wm = WorldModel()
        wm.record_action_complete("shell", "test", "exit=0: done")
        r = repr(wm)
        assert "triples=1" in r

    def test_format_world_model_context_after_many_actions(self):
        wm = WorldModel()
        for i in range(20):
            outcome = "exit=0: done" if i % 2 == 0 else "exit=1: failed"
            expected = "should succeed" if i % 2 == 0 else "should work"
            wm.record_action_complete("shell", f"action {i}", outcome, expected)
        ctx = wm.format_world_model_context()
        assert len(ctx) > 100

    def test_default_world_model_version(self):
        wm = WorldModel()
        assert wm.data["version"] == _DEFAULT_WORLD_MODEL["version"]

    def test_merge_defaults_handles_extra_keys(self):
        wm = WorldModel(data={"extra_key": "value"})
        assert wm.data["extra_key"] == "value"
        assert wm.data["version"] == _DEFAULT_WORLD_MODEL["version"]

    def test_format_action_guidance_no_crash_on_empty_type(self):
        wm = WorldModel()
        guidance = wm.format_action_guidance("", "")
        assert guidance is None or isinstance(guidance, str)

    def test_concurrent_record_and_complete(self):
        """Interleaved record/complete calls should work correctly."""
        wm = WorldModel()
        t1 = wm.record_action("shell", "action 1", "ok")
        t2 = wm.record_action("shell", "action 2", "ok")
        t3 = wm.record_action("shell", "action 3", "ok")
        wm.complete_action(t1, "exit=0: done 1")
        wm.complete_action(t3, "exit=0: done 3")  # Complete out of order
        wm.complete_action(t2, "exit=0: done 2")
        assert wm.data["action_triples"][0]["completed"] is True
        assert wm.data["action_triples"][1]["completed"] is True
        assert wm.data["action_triples"][2]["completed"] is True

    def test_prediction_without_confidence_defaults(self):
        wm = WorldModel()
        pid = wm.record_prediction("test", "1 day")  # No confidence provided
        assert pid is not None
        assert wm.data["predictions"][0]["confidence"] == 0.5  # Default

    def test_prediction_without_timeframe(self):
        wm = WorldModel()
        pid = wm.record_prediction("test", None)
        assert pid is not None
        assert wm.data["predictions"][0]["timeframe"] is None

    def test_format_prediction_insight_with_trend(self):
        wm = WorldModel()
        for err in [0.5, 0.4, 0.3, 0.2, 0.1, 0.1]:
            wm._add_to_error_history(err)
        wm.record_prediction("test", "1 day", 0.5, "")
        # Should have trend info
        # But format_prediction_insight checks predictions, not action triples
        insight = wm.format_prediction_insight()
        assert isinstance(insight, str)

    def test_error_history_initial_state(self):
        wm = WorldModel()
        assert wm.data["prediction_accuracy"]["error_history"] == []

    def test_generate_improvement_suggestions_empty(self):
        """Empty world model → no suggestions."""
        wm = WorldModel()
        suggestions = wm.generate_improvement_suggestions()
        assert suggestions == []

    def test_format_improvement_context_empty(self):
        """Empty world model → empty string."""
        wm = WorldModel()
        ctx = wm.format_improvement_context()
        assert ctx == ""

    def test_generate_improvement_suggestions_high_error_type(self):
        wm = WorldModel()
        # Add high-error actions for a type
        tid = wm.record_action("shell", "broken deploy", "should succeed")
        wm.complete_action(tid, "exit=1: build failure")
        tid = wm.record_action("shell", "another fail", "should pass")
        wm.complete_action(tid, "exit=1: timeout")
        suggestions = wm.generate_improvement_suggestions()
        calibrate = [s for s in suggestions if s["type"] == "calibrate"]
        assert len(calibrate) >= 1
        assert any("shell" in c["title"] for c in calibrate)

    def test_generate_improvement_suggestions_discrepancy_pattern(self):
        wm = WorldModel()
        # Add several high-error actions of same type to trigger pattern
        for desc in ["read protected file", "resolve unknown host", "deploy staging"]:
            tid = wm.record_action("shell", desc, "expected ok")
            wm.complete_action(tid, "exit=1: permission denied")
        # Force pattern update
        wm._update_discrepancy_patterns(min_samples=2)
        suggestions = wm.generate_improvement_suggestions()
        investigate = [s for s in suggestions if s["type"] == "investigate"]
        assert len(investigate) >= 1

    def test_generate_improvement_suggestions_insufficient_data(self):
        wm = WorldModel()
        # Add a type with only 1 sample
        tid = wm.record_action("new_type", "first try", "expected")
        wm.complete_action(tid, "exit=0: ok")
        suggestions = wm.generate_improvement_suggestions()
        collect = [s for s in suggestions if s["type"] == "collect_data"]
        assert any("new_type" in c["title"] for c in collect)

    def test_generate_improvement_suggestions_capped_at_eight(self):
        wm = WorldModel()
        # Add many high-error types to fill suggestion list
        for i in range(6):
            atype = f"type_{i}"
            tid = wm.record_action(atype, "fail", "expected ok")
            wm.complete_action(tid, "exit=1: failed")
            tid = wm.record_action(atype, "fail2", "expected ok")
            wm.complete_action(tid, "exit=1: fail")
        suggestions = wm.generate_improvement_suggestions()
        assert len(suggestions) <= 8

    def test_format_improvement_context_with_suggestions(self):
        wm = WorldModel()
        tid = wm.record_action("shell", "risky op", "should work")
        wm.complete_action(tid, "exit=1: failed")
        tid = wm.record_action("shell", "risky op 2", "should work")
        wm.complete_action(tid, "exit=1: error")
        ctx = wm.format_improvement_context()
        assert "Improvement Suggestions" in ctx
        assert "calibrate" in ctx.lower() or "investigate" in ctx.lower() or "shell" in ctx.lower()
        assert "new_goal" in ctx or "gap_reference" in ctx

    def test_suggestions_prioritized_correctly(self):
        """High-error (priority 2) before medium-error (priority 3) before data-collection (priority 4)."""
        wm = WorldModel()
        # Create a high-error calibration suggestion
        tid = wm.record_action("shell", "deploy prod", "expected ok")
        wm.complete_action(tid, "exit=1: crash")
        tid = wm.record_action("shell", "deploy test", "expected ok")
        wm.complete_action(tid, "exit=1: timeout")
        suggestions = wm.generate_improvement_suggestions()
        priorities = [s["priority"] for s in suggestions]
        assert priorities == sorted(priorities)


# ══════════════════════════════════════════════════════════════════════
#  predict_action_outcome — data-driven prediction
# ══════════════════════════════════════════════════════════════════════


class TestPredictActionOutcome:
    """Test predict_action_outcome — data-driven outcome prediction."""

    def test_no_data_returns_defaults(self):
        """No historical data for action type → returns default with None predicted_outcome."""
        wm = WorldModel()
        result = wm.predict_action_outcome("shell", "run some command")
        assert result["predicted_outcome"] is None
        assert result["confidence"] == 0.0
        assert result["sample_count"] == 0
        assert result["avg_error"] is None
        assert result["risk_level"] == "unknown"
        assert result["success_probability"] == 0.5

    def test_data_but_low_sample_count_blends(self):
        """Fewer than 3 samples → blends success_probability toward neutral (0.5)."""
        wm = WorldModel()
        tid = wm.record_action("shell", "list files", "should list dir")
        wm.complete_action(tid, "exit=0: file1.txt")
        result = wm.predict_action_outcome("shell", "list files again")
        assert result["sample_count"] == 1
        assert result["avg_error"] is not None
        # With 1 sample at low error: raw=~0.85, blend=(0.85*0.33+0.5*0.67)=0.62
        assert 0.4 <= result["success_probability"] <= 0.75
        assert result["confidence"] > 0.0
        assert result["risk_level"] in ("low", "medium")

    def test_good_data_high_confidence(self):
        """Many successful actions → high success probability, low risk."""
        wm = WorldModel()
        for i in range(5):
            tid = wm.record_action("write_file", f"write file {i}", "should succeed")
            wm.complete_action(tid, f"Wrote file_{i}.py (10 bytes)")
        result = wm.predict_action_outcome("write_file", "write new file")
        assert result["sample_count"] == 5
        assert result["success_probability"] >= 0.7
        assert result["risk_level"] == "low"
        assert result["confidence"] > 0.0

    def test_poor_data_high_risk(self):
        """Many failed actions → low success probability, high risk."""
        wm = WorldModel()
        for i in range(5):
            tid = wm.record_action("deploy", f"deploy v{i}", "should succeed")
            wm.complete_action(tid, "exit=1: deployment failed")
        result = wm.predict_action_outcome("deploy", "deploy new version")
        assert result["sample_count"] == 5
        assert result["success_probability"] <= 0.5
        assert result["risk_level"] == "high"
        assert result["predicted_outcome"] is not None
        # predicted_outcome now contains actual output text of most similar action;
        # the meta-description moved to prediction_rationale.
        assert "likely to fail" in result["prediction_rationale"]

    def test_risk_level_medium(self):
        """Mixed outcomes → medium risk."""
        wm = WorldModel()
        for i in range(4):
            outcome = "exit=0: ok" if i % 2 == 0 else "exit=1: fail"
            expected = "should work" if i % 2 == 0 else "should succeed"
            tid = wm.record_action("shell", f"mixed op {i}", expected)
            wm.complete_action(tid, outcome)
        result = wm.predict_action_outcome("shell", "another mixed op")
        assert result["risk_level"] in ("low", "medium")
        # With 2/4 success, avg_err ~0.5 → success_prob ~0.5 → at least medium
        assert result["success_probability"] >= 0.3

    def test_unknown_action_type_returns_defaults(self):
        """Unknown action type → no crash, returns defaults."""
        wm = WorldModel()
        wm.record_action_complete("shell", "test", "exit=0: ok", "ok")
        result = wm.predict_action_outcome("nonexistent_type", "anything")
        assert result["predicted_outcome"] is None
        assert result["sample_count"] == 0
        assert result["risk_level"] == "unknown"

    def test_different_types_have_independent_stats(self):
        """Different action types should have independent predictions."""
        wm = WorldModel()
        # shell: all successful
        wm.record_action_complete("shell", "ok1", "exit=0: done", "expected ok")
        wm.record_action_complete("shell", "ok2", "exit=0: done", "expected ok")
        # write_file: all failed
        wm.record_action_complete("write_file", "fail1", "exit=1: perm denied", "expected ok")
        wm.record_action_complete("write_file", "fail2", "exit=1: disk full", "expected ok")
        shell_result = wm.predict_action_outcome("shell", "test")
        write_result = wm.predict_action_outcome("write_file", "test")
        assert shell_result["success_probability"] > write_result["success_probability"]
        assert shell_result["risk_level"] == "low"
        assert write_result["risk_level"] == "high"

    def test_predicted_outcome_format(self):
        """prediction_rationale should contain type, label, stats."""
        wm = WorldModel()
        wm.record_action_complete("git_commit", "fix: typo", "exit=0: committed", "ok")
        wm.record_action_complete("git_commit", "feat: add", "exit=0: committed", "ok")
        result = wm.predict_action_outcome("git_commit", "fix: bug")
        assert result["predicted_outcome"] is not None
        # predicted_outcome now contains actual output text (most similar action's output)
        assert "committed" in result["predicted_outcome"]
        # The meta-description moved to prediction_rationale
        assert "data-driven" in result["prediction_rationale"]
        assert "git_commit" in result["prediction_rationale"]
        assert "success" in result["prediction_rationale"].lower()

    def test_similar_actions_returned(self):
        """Similar past actions should appear in the result."""
        wm = WorldModel()
        wm.record_action_complete("shell", "deploy to production", "exit=0: done", "ok")
        wm.record_action_complete("shell", "deploy to staging", "exit=1: failed", "ok")
        wm.record_action_complete("write_file", "write readme", "Wrote readme.md", "ok")
        result = wm.predict_action_outcome("shell", "deploy to staging")
        assert len(result["similar_actions"]) >= 1
        # The similar actions should be shell type (not write_file)
        for sa in result["similar_actions"]:
            assert "deploy" in sa["description"].lower()

    def test_similar_actions_empty_for_unmatched_type(self):
        """No similar actions when description has no keyword overlap."""
        wm = WorldModel()
        wm.record_action_complete("shell", "deploy app", "exit=0: ok", "ok")
        result = wm.predict_action_outcome("shell", "xyzzy_qwerty")
        # No keyword overlap
        assert result["similar_actions"] == []

    def test_predict_action_outcome_edge_cases(self):
        """Edge cases: empty description, very long description."""
        wm = WorldModel()
        wm.record_action_complete("shell", "test", "exit=0: ok", "expected")
        # Empty description — should not crash
        result_empty = wm.predict_action_outcome("shell", "")
        assert result_empty is not None
        # Very long description — should not crash
        result_long = wm.predict_action_outcome("shell", "a" * 5000)
        assert result_long is not None

    def test_stale_failure_not_reused_when_command_changed(self):
        """A past FAILURE of a *different* command must not be predicted for the
        current action.

        Regression for the 2026-08-01 'sibling dirs check' bug: the auto-default
        shell command had broken quoting and failed once; after the quoting was
        fixed (command text changed), the predictor kept reusing the old
        'Syntax error' outcome as the prediction — manufacturing ~0.5 error
        every cycle and feeding the shell discrepancy pattern.
        """
        wm = WorldModel()
        broken_cmd = 'python3 -c "import pathlib; print(1)"'
        fixed_cmd = "python3 -c 'import pathlib; print(1)'"
        tid = wm.record_action(
            "shell", "Auto-default: sibling dirs check",
            "exit=0: listing", parameters={"command": broken_cmd},
        )
        wm.complete_action(tid, "exit=2: Syntax error: Unterminated quoted string")

        # Same description, but the command text changed (quoting fixed):
        result = wm.predict_action_outcome(
            "shell", "Auto-default: sibling dirs check",
            parameters={"command": fixed_cmd},
        )
        assert result["predicted_outcome"] is not None
        assert "Syntax error" not in result["predicted_outcome"]
        # Falls back to the generic shell prediction instead of the stale failure
        assert result["predicted_outcome"].startswith("exit=0")
        assert "STALE template" in result.get("prediction_rationale", "")

    def test_same_command_reuses_past_outcome(self):
        """An unchanged command may reuse the past action's outcome as template —
        even when that past action's own prediction error was at the 0.5 gate."""
        wm = WorldModel()
        cmd = "ls -la /tmp"
        tid = wm.record_action("shell", "list tmp", "exit=0: files",
                               parameters={"command": cmd})
        wm.complete_action(tid, "exit=0: file1 file2")
        result = wm.predict_action_outcome("shell", "list tmp",
                                           parameters={"command": cmd})
        assert "file1" in result["predicted_outcome"]
        assert "STALE template" not in result.get("prediction_rationale", "")

    def test_truncated_stored_command_keeps_copy_behavior(self):
        """Long commands are stored truncated ('...[truncated]') and cannot be
        compared reliably — the predictor must keep the existing copy behavior
        instead of flagging every long command as changed."""
        wm = WorldModel()
        long_cmd = "echo " + "x" * 400  # exceeds _MAX_PARAM_VALUE_LEN (300)
        tid = wm.record_action("shell", "long echo", "exit=0: x",
                               parameters={"command": long_cmd})
        wm.complete_action(tid, "exit=0: xxxx")
        result = wm.predict_action_outcome("shell", "long echo",
                                           parameters={"command": long_cmd})
        assert "xxxx" in result["predicted_outcome"]
        assert "STALE template" not in result.get("prediction_rationale", "")

    def test_non_shell_ignores_stale_check(self):
        """write_file content/path legitimately vary between runs — copying the
        most similar past outcome stays (no 'command' key, so no stale guard)."""
        wm = WorldModel()
        tid = wm.record_action(
            "write_file", "write snapshot", "Wrote file",
            parameters={"path": "/tmp/a.txt", "content": "v1"},
        )
        wm.complete_action(tid, "Wrote /tmp/a.txt (10 bytes)")
        result = wm.predict_action_outcome(
            "write_file", "write snapshot",
            parameters={"path": "/tmp/a.txt", "content": "v2-different"},
        )
        assert "10 bytes" in result["predicted_outcome"]
        assert "STALE template" not in result.get("prediction_rationale", "")


class TestCalibrationBucketCleanup:
    """Tests for _clean_stale_calibration_buckets on load."""

    def test_clean_stale_buckets_legacy_expired(self):
        """Legacy data: all buckets are 0.5 error, nothing verified — cleaned."""
        wm = WorldModel()
        wm.data["prediction_accuracy"]["calibration_buckets"] = [
            {"count": 8, "total_error": 4.0, "avg_error": 0.5}
        ]
        wm.data["prediction_accuracy"]["correct_predictions"] = 0
        wm.data["prediction_accuracy"]["incorrect_predictions"] = 0
        wm._clean_stale_calibration_buckets()
        assert wm.data["prediction_accuracy"]["calibration_buckets"] == []

    def test_clean_stale_buckets_excess_count(self):
        """Bucket count > actual verified predictions — cleaned."""
        wm = WorldModel()
        wm.data["prediction_accuracy"]["calibration_buckets"] = [
            {"count": 5, "total_error": 2.0, "avg_error": 0.4}
        ]
        wm.data["prediction_accuracy"]["correct_predictions"] = 2
        wm.data["prediction_accuracy"]["incorrect_predictions"] = 1
        # bucket_total=5 > actually_verified=3 → stale
        wm._clean_stale_calibration_buckets()
        assert wm.data["prediction_accuracy"]["calibration_buckets"] == []

    def test_preserve_legitimate_calibration_data(self):
        """Bucket count matches verified predictions — preserved."""
        wm = WorldModel()
        wm.data["prediction_accuracy"]["calibration_buckets"] = [
            {"count": 3, "total_error": 0.6, "avg_error": 0.2}
        ]
        wm.data["prediction_accuracy"]["correct_predictions"] = 3
        wm.data["prediction_accuracy"]["incorrect_predictions"] = 0
        wm._clean_stale_calibration_buckets()
        # Should NOT be cleaned — data is consistent
        assert len(wm.data["prediction_accuracy"]["calibration_buckets"]) == 1
        assert wm.data["prediction_accuracy"]["calibration_buckets"][0]["count"] == 3

    def test_preserve_empty_buckets(self):
        """No calibration data — no-op."""
        wm = WorldModel()
        wm._clean_stale_calibration_buckets()
        assert wm.data["prediction_accuracy"]["calibration_buckets"] == []

    def test_cleanup_invoked_on_load_legacy_data(self):
        """Verify _clean_stale_calibration_buckets runs during WorldModel.load()."""
        import tempfile, json
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_wm.json"
            # Write legacy dirty data
            dirty = {
                "version": 4,
                "action_triples": [],
                "predictions": [],
                "prediction_accuracy": {
                    "total_predictions": 8,
                    "verified_predictions": 8,
                    "correct_predictions": 0,
                    "incorrect_predictions": 0,
                    "avg_prediction_error": 0.5,
                    "total_triples": 0,
                    "avg_triple_error": 0.0,
                    "calibration_buckets": [
                        {"count": 8, "total_error": 4.0, "avg_error": 0.5}
                    ],
                    "error_history": [0.5] * 8,
                },
                "discrepancy_patterns": [],
                "per_type_accuracy": {},
            }
            path.write_text(json.dumps(dirty), encoding="utf-8")
            # Load should clean the buckets
            loaded = WorldModel.load(path)
            assert loaded.data["prediction_accuracy"]["calibration_buckets"] == [], (
                f"Expected empty buckets after load with legacy data, "
                f"got {loaded.data['prediction_accuracy']['calibration_buckets']}"
            )

