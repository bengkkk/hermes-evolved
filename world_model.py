"""World Model — action prediction, outcome tracking, and discrepancy learning (Gap 6).

The world model is the system's ability to PREDICT the effects of its own actions
BEFORE executing them, then COMPARE with what actually happened, and LEARN from
the discrepancy to improve future predictions.

This implements the full predict → act → observe → learn loop:

  1. PREDICT: Before any action, log the expected outcome
  2. ACT: Execute the action
  3. OBSERVE: Record the actual outcome
  4. COMPARE: Calculate prediction error (0.0 = exact match, 1.0 = complete mismatch)
  5. LEARN: Use discrepancies to update internal model (adaptive confidence calibration)

Two levels of prediction:
  - MICRO (action triples): Immediate effect of a shell/write/git action
  - MACRO (trajectory predictions): Higher-level "what will happen this week" forecasts

Stored as a JSON dict under evolve/world_model.json alongside the other
evolve data files (timeline.json, self_model.json, etc.).
"""

from __future__ import annotations

import copy
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data_layer import get_evolve_dir, safe_write_json, safe_read_json, now_compact, now_iso

logger = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────

# Monotonically incrementing counter for unique IDs within the same timestamp
_id_counter: int = 0


def _unique_id(prefix: str) -> str:
    """Generate a unique ID with timestamp and monotonic counter.

    Uses a module-level counter to guarantee uniqueness even when
    multiple actions/predictions are recorded in the same second.
    """
    global _id_counter
    _id_counter += 1
    return f"{prefix}_{now_compact()}_{_id_counter}"


_DEFAULT_WORLD_MODEL: Dict[str, Any] = {
    "version": 4,
    "action_triples": [],       # List[ActionTriple]
    "predictions": [],          # List[Prediction]
    "prediction_accuracy": {    # Running statistics
        "total_predictions": 0,
        "verified_predictions": 0,
        "correct_predictions": 0,
        "incorrect_predictions": 0,
        "avg_prediction_error": 0.0,
        "total_triples": 0,
        "avg_triple_error": 0.0,
        "calibration_buckets": [],  # confidence vs accuracy per bucket
        "error_history": [],        # Rolling list of last 20 prediction errors for trend
    },
    "discrepancy_patterns": [],  # Recurring categories of prediction failure
    "per_type_accuracy": {},     # Action-type → stats for calibration
}


# ═══════════════════════════════════════════════════════════════════
#  Error calculation
# ═══════════════════════════════════════════════════════════════════

def _compute_prediction_error(
    expected: str,
    actual: str,
) -> float:
    """Compute a prediction error score between 0.0 and 1.0.

    0.0 = exact match / clearly succeeded as expected
    0.25 = partial success (mostly worked with minor issues)
    0.5 = mixed (some expected parts realized, some didn't)
    0.75 = mostly wrong (significant discrepancy)
    1.0 = complete mismatch, failure, or empty

    Uses a multi-strategy approach:
    1. Exact or trimmed match → 0.0
    2. One contains the other → 0.25
    3. Exit code analysis (exit=0 vs exit=non-zero) → maps to success/failure
    4. Character bigram similarity for robust short-text comparison
    5. Keyword overlap fallback
    """
    if not expected or not actual:
        return 1.0

    e_lower = expected.lower().strip()
    a_lower = actual.lower().strip()

    # Exact or trimmed match
    if e_lower == a_lower:
        return 0.0

    # One contains the other
    if e_lower in a_lower or a_lower in e_lower:
        return 0.25

    # ── Write-file outcome heuristic ──
    # Hermes's write_file tool output: "Wrote <path> (<N> bytes)" or
    # "Created <path>", "Written <N> bytes to <path>"
    # If actual says the file was written and expected mentions the filename,
    # treat as a successful write (low error).
    file_write_match = re.match(
        r'(wrote|created|written|appended|overwrote)\s+(.+?)(?:\s*\(|$)',
        a_lower,
    )
    if file_write_match:
        file_path = file_write_match.group(2).strip().rstrip(".")
        # If expected mentions the file path, it's a successful write
        if file_path and file_path in e_lower:
            return 0.15
        # File was written but expected didn't name it — partial match
        return 0.25
    # ── Informational non-error messages ──
    # Some commands signal "no work to do" with non-zero exit codes
    # (e.g. "exit=1: nothing to commit, working tree clean").
    # These are semantically successful even though the exit code is
    # non-zero.  Check this BEFORE the exit-code check below so that
    # benign informational patterns are still scored low.
    informational_nonerror = bool(re.search(
        r"(nothing to commit|working tree clean|already up.to.date|"
        r"no changes|nothing changed|nothing to do|"
        r"0 files changed|0 insertions|0 deletions)",
        a_lower,
    ))
    if informational_nonerror:
        return 0.15

    # ── Exit-code aware comparison (ground truth) ──
    # The exit code IS the ground truth: exit=0 always means success,
    # exit != 0 always means failure.  This check MUST come BEFORE the
    # tool-keyword heuristic below but AFTER the informational pattern
    # check above, because content like "host not found" or "Traceback"
    # can appear in an otherwise successful command (exit=0).
    exit_match = re.search(r'exit=(\d+)', a_lower)
    if exit_match:
        exit_code = int(exit_match.group(1))
        if exit_code == 0:
            # Exit code 0 = command succeeded → low error regardless
            return 0.15
        # exit != 0 — check if the expected text suggests success
        success_expected = any(kw in e_lower for kw in
            ('success', 'complete', 'create', 'write', 'deploy', 'install',
             'commit', 'push', 'run', 'list', 'show', 'print',
             'pass', 'test', 'status', 'check', 'git', 'file',
             'read', 'done', 'build', 'fix', 'fetch', 'merge',
             'pull', 'add', 'update', 'find', 'search', 'info',
             'log', 'clean', 'set', 'get', 'patch', 'branch', 'diff'))
        if success_expected:
            return 0.85  # Expected success but got failure
        else:
            return 0.7   # Neutral expected but command failed

    # ── Shell/Tool success/failure keywords ──
    # Content-based heuristics for outputs that lack an explicit exit=
    # marker.  These are less reliable than exit code but better than
    # pure string similarity.
    tool_failure = bool(re.search(
        r'\b(traceback|error|failed|permission denied|not found|no such)\b',
        a_lower,
    ))
    tool_success = bool(re.search(
        r'\b(passed|succeeded|ok|complete|done)\b',
        a_lower,
    ))
    if tool_failure and not tool_success:
        return 0.85  # Tool reported failure
    if tool_success and not tool_failure:
        return 0.15  # Tool reported success

    # ── Character bigram similarity ──
    # For short strings, bigram overlap handles word variations better
    def _bigrams(s: str) -> set:
        return {s[i:i + 2] for i in range(len(s) - 1)}

    e_bg = _bigrams(e_lower)
    a_bg = _bigrams(a_lower)

    if e_bg and a_bg:
        intersection = e_bg & a_bg
        union = e_bg | a_bg
        jaccard = len(intersection) / len(union)
        if jaccard >= 0.6:
            return 0.15
        elif jaccard >= 0.35:
            return 0.4
        elif jaccard >= 0.15:
            return 0.6
        elif jaccard > 0.0:
            return 0.8

    # ── Token overlap (word-level, last resort) ──
    def _tokenize(s: str) -> set:
        return set(re.findall(r"[a-z0-9]+", s))

    e_tokens = _tokenize(e_lower)
    a_tokens = _tokenize(a_lower)

    if not e_tokens or not a_tokens:
        return 1.0

    intersection = e_tokens & a_tokens
    union = e_tokens | a_tokens
    jaccard = len(intersection) / len(union)

    if jaccard >= 0.5:
        return 0.5
    elif jaccard > 0.0:
        return 0.75
    else:
        return 1.0


# ═══════════════════════════════════════════════════════════════════
#  WorldModel class
# ═══════════════════════════════════════════════════════════════════

class WorldModel:
    """World model tracking action→outcome triples and prediction accuracy.

    Two-tier prediction system:
      - **Action triples** (micro): Specific action → expected outcome → actual outcome.
        Recorded automatically after every daemon action execution.
      - **Predictions** (macro): Higher-level forecasts about system trajectory.
        Verified later when the timeframe passes.

    File-backed: stored at ``evolve/world_model.json``.
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self.data: Dict[str, Any] = (
            self._merge_defaults(data) if data
            else copy.deepcopy(_DEFAULT_WORLD_MODEL)
        )

    @staticmethod
    def _merge_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
        merged = copy.deepcopy(_DEFAULT_WORLD_MODEL)
        for section in ("action_triples", "predictions", "discrepancy_patterns"):
            if section in data and isinstance(data[section], list):
                merged[section] = data[section]
        if "prediction_accuracy" in data and isinstance(data["prediction_accuracy"], dict):
            merged["prediction_accuracy"].update(data["prediction_accuracy"])
        # Absorb any extra top-level keys
        for k, v in data.items():
            if k not in merged:
                merged[k] = v
        return merged

    # ── Action triples (micro-level) ──────────────────────────────

    def record_action(
        self,
        action_type: str,
        action_description: str,
        expected_outcome: str = "",
    ) -> str:
        """Record an action BEFORE execution, returning the triple ID.

        Call this BEFORE executing the action to capture the *expected* outcome.
        Then call :meth:`complete_action` AFTER execution with the actual outcome.

        Args:
            action_type: ``write_file`` | ``shell`` | ``git_commit`` | ``install_package``
            action_description: Human-readable description of what the action does.
            expected_outcome: What the system expects will happen (from LLM prediction).

        Returns:
            Triple ID to pass to :meth:`complete_action`.
        """
        triple_id = _unique_id("act")
        triple: Dict[str, Any] = {
            "id": triple_id,
            "action_type": action_type,
            "action_description": action_description,
            "expected_outcome": expected_outcome or "unknown",
            "actual_outcome": None,  # filled in by complete_action
            "prediction_error": None,
            "timestamp": now_iso(),
            "completed": False,
        }
        self.data.setdefault("action_triples", []).append(triple)
        self.data["action_triples"] = self.data["action_triples"][-200:]
        self._update_accuracy_stats()
        return triple_id

    def complete_action(
        self,
        triple_id: str,
        actual_outcome: str,
    ) -> Optional[float]:
        """Record the actual outcome of a previously-recorded action.

        Calculates prediction error between expected and actual outcome.

        Args:
            triple_id: The ID returned by :meth:`record_action`.
            actual_outcome: What actually happened.

        Returns:
            Prediction error (0.0–1.0), or None if triple_id wasn't found.
        """
        for triple in self.data.get("action_triples", []):
            if triple.get("id") == triple_id and not triple.get("completed"):
                triple["actual_outcome"] = actual_outcome
                triple["completed"] = True

                expected = triple.get("expected_outcome", "")
                error = _compute_prediction_error(expected, actual_outcome)
                triple["prediction_error"] = error
                triple["completed_at"] = now_iso()

                self._add_to_error_history(error)
                self._update_accuracy_stats()
                return error
        return None

    def record_action_complete(
        self,
        action_type: str,
        action_description: str,
        action_output: str,
        expected_outcome: str = "",
    ) -> Dict[str, Any]:
        """Convenience: record and complete an action in one call.

        Use this when you don't need to record the expected outcome before
        the action executes (e.g. when you only know the outcome after the fact
        and want to retroactively log the triple).

        For the full predict→observe→compare cycle, use
        :meth:`record_action` + :meth:`complete_action` instead.
        """
        triple_id = self.record_action(action_type, action_description, expected_outcome)
        error = self.complete_action(triple_id, action_output)
        return {
            "id": triple_id,
            "prediction_error": error,
        }

    # ── Macro predictions ─────────────────────────────────────────

    def record_prediction(
        self,
        text: str,
        timeframe: Optional[str] = None,
        confidence: float = 0.5,
        basis: str = "",
    ) -> str:
        """Record a trajectory prediction for later verification.

        Args:
            text: What the prediction says (e.g., "timeline will have 50 events").
            timeframe: When this prediction should be evaluated (e.g. "3 days", "1 week").
            confidence: 0.0–1.0 how confident the system is.
            basis: Why this prediction is being made.

        Returns:
            Prediction ID.
        """
        pred_id = _unique_id("pred")
        pred: Dict[str, Any] = {
            "id": pred_id,
            "text": text,
            "timeframe": timeframe,
            "confidence": confidence,
            "basis": basis,
            "verified": False,
            "actual": None,
            "error": None,
            "verification_note": "",
            "timestamp": now_iso(),
        }
        self.data.setdefault("predictions", []).append(pred)
        self.data["predictions"] = self.data["predictions"][-100:]
        acc = self.data.setdefault("prediction_accuracy", {})
        acc["total_predictions"] = acc.get("total_predictions", 0) + 1
        return pred_id

    def verify_prediction(
        self,
        pred_id: str,
        actual_outcome: str,
        note: str = "",
    ) -> Optional[float]:
        """Verify a past prediction against what actually happened.

        Args:
            pred_id: The ID from :meth:`record_prediction`.
            actual_outcome: Description of what actually happened.
            note: Optional note about the verification.

        Returns:
            Error score (0.0–1.0) or None if not found.
        """
        for pred in self.data.get("predictions", []):
            if pred.get("id") == pred_id and not pred.get("verified"):
                pred["verified"] = True
                pred["actual"] = actual_outcome
                pred["verification_note"] = note

                error = _compute_prediction_error(pred.get("text", ""), actual_outcome)
                pred["error"] = error
                pred["verified_at"] = now_iso()

                # Update accuracy stats
                acc = self.data.setdefault("prediction_accuracy", {})
                acc["verified_predictions"] = acc.get("verified_predictions", 0) + 1
                if error <= 0.3:
                    acc["correct_predictions"] = acc.get("correct_predictions", 0) + 1
                else:
                    acc["incorrect_predictions"] = acc.get("incorrect_predictions", 0) + 1

                # Update running average error
                total_verified = acc.get("verified_predictions", 1)
                prev_avg = acc.get("avg_prediction_error", 0.0)
                acc["avg_prediction_error"] = round(
                    (prev_avg * (total_verified - 1) + error) / total_verified, 4
                )

                self._update_calibration(pred.get("confidence", 0.5), error)
                self._add_to_error_history(error)
                return error
        return None

    # ── Auto-verification of expired predictions ──────────────────

    @staticmethod
    def _parse_timeframe_days(timeframe: Optional[str]) -> Optional[float]:
        """Parse a human-readable timeframe string into days.

        Supported formats:
          - ``"3 days"`` / ``"3 day"`` → 3.0
          - ``"1 week"`` / ``"2 weeks"`` → 7.0 / 14.0
          - ``"1 month"`` / ``"2 months"`` → 30.0 / 60.0
          - ``"1 year"`` / ``"2 years"`` → 365.0 / 730.0
          - ``"1 hour"`` / ``"2 hours"`` → 0.042 / 0.083
          - ``"1 cycle"`` / ``"2 cycles"`` → 0.01 / 0.02 (≈15 min per cycle)
          - ``"this cycle"`` → 0.0 (effectively immediate)
          - ``"completed"`` → 0.0 (already past)
          - ``None`` / empty → None (can't determine)

        Args:
            timeframe: Human-readable duration string.

        Returns:
            Number of days as float, or None if unparseable.
        """
        if not timeframe:
            return None

        tf = timeframe.strip().lower()

        # Special cases that mean "immediately" / "already"
        if tf in ("completed", "this cycle", "this turn", "same cycle", "now"):
            return 0.0

        # Try "N <unit>" pattern
        import re as _re
        m = _re.match(r"(\d+\.?\d*)\s*(hour|hours|cycle|cycles|day|days|week|weeks|month|months|year|years)", tf)
        if m:
            value = float(m.group(1))
            unit = m.group(2)
            multipliers = {
                "hour": 1 / 24, "hours": 1 / 24,
                "cycle": 0.01, "cycles": 0.01,  # ~15 min per cycle
                "day": 1, "days": 1,
                "week": 7, "weeks": 7,
                "month": 30, "months": 30,
                "year": 365, "years": 365,
            }
            return value * multipliers.get(unit, 1)

        # Handle singular forms without digits
        singular_map = {
            "a day": 1, "a week": 7, "a month": 30, "a year": 365,
            "one day": 1, "one week": 7, "one month": 30, "one year": 365,
            "an hour": 1 / 24, "one hour": 1 / 24,
            "a cycle": 0.01, "one cycle": 0.01,
        }
        if tf in singular_map:
            return float(singular_map[tf])

        return None

    def verify_expired_predictions(self) -> int:
        """Auto-verify predictions whose timeframe has expired.

        Checks all unverified predictions against the current time.
        If a prediction has a timeframe that has passed, it is
        auto-verified with outcome ``"timeframe expired — no confirmation"``
        and error 0.5 (uncertain — could be right or wrong).

        Predictions without a parseable timeframe are left alone.

        Returns:
            Number of predictions auto-verified.
        """
        from datetime import timezone as _tz

        now = datetime.now(_tz.utc)
        verified_count = 0

        for pred in self.data.get("predictions", []):
            if pred.get("verified"):
                continue

            tf = pred.get("timeframe")
            days = self._parse_timeframe_days(tf)
            if days is None:
                continue  # Can't determine expiry — leave it

            # Parse prediction timestamp
            ts_str = pred.get("timestamp")
            if not ts_str:
                continue
            try:
                pred_time = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                continue

            # If prediction was made in the future (clock skew), skip
            if pred_time > now:
                continue

            # Calculate elapsed days
            elapsed_days = (now - pred_time).total_seconds() / 86400.0

            # Allow a grace period: 10% of the timeframe or 1 hour, whichever is larger.
            # For sub-day timeframes like "this cycle" or "1 hour", the 1-hour floor
            # ensures at least one daemon cycle worth of tolerance, while keeping the
            # verification prompt enough that cycle-based predictions don't linger forever.
            grace = max(days * 0.1, 1.0 / 24.0)  # at least ~1 hour
            if elapsed_days >= days + grace:
                # Timeframe has expired — auto-verify as uncertain
                pred["verified"] = True
                pred["actual"] = "timeframe expired — no confirmation"
                pred["error"] = 0.5
                pred["verification_note"] = (
                    f"Auto-verified: timeframe '{tf}' ({days} days) "
                    f"expired {elapsed_days - days:.1f} days ago"
                )
                pred["verified_at"] = now_iso()

                # Update accuracy stats
                acc = self.data.setdefault("prediction_accuracy", {})
                acc["verified_predictions"] = acc.get("verified_predictions", 0) + 1
                # error=0.5 is neither clearly correct nor incorrect
                # Neither correct_predictions nor incorrect_predictions gets
                # incremented — it's truly uncertain.

                # Update running average error
                total_verified = acc.get("verified_predictions", 1)
                prev_avg = acc.get("avg_prediction_error", 0.0)
                acc["avg_prediction_error"] = round(
                    (prev_avg * (total_verified - 1) + 0.5) / total_verified, 4
                )

                self._update_calibration(pred.get("confidence", 0.5), 0.5)
                self._add_to_error_history(0.5)
                verified_count += 1

        return verified_count

    # ── Discrepancy analysis ──────────────────────────────────────

    def analyze_recent_discrepancies(self, count: int = 10) -> List[Dict[str, Any]]:
        """Return the *count* triples with the highest prediction error.

        These are the cases where the system's prediction was most wrong,
        and thus the highest-value learning opportunities.
        """
        triples = [
            t for t in self.data.get("action_triples", [])
            if t.get("prediction_error") is not None
        ]
        triples.sort(key=lambda t: t.get("prediction_error", 0), reverse=True)
        return triples[:count]

    def get_unverified_predictions(self) -> List[Dict[str, Any]]:
        """Return predictions that haven't been verified yet."""
        return [
            p for p in self.data.get("predictions", [])
            if not p.get("verified")
        ]

    # ── Context formatting ─────────────────────────────────────────

    def format_world_model_context(self) -> str:
        """Format a context block about the world model for the thinking prompt.

        Includes:
          - Recent action triples (last 5)
          - Prediction accuracy stats
          - Biggest discrepancies (top 3)
          - Unverified predictions (up to 3)
        """
        parts: List[str] = []
        acc = self.data.get("prediction_accuracy", {})

        # ── Accuracy summary ──
        total_preds = acc.get("total_predictions", 0)
        verified = acc.get("verified_predictions", 0)
        correct = acc.get("correct_predictions", 0)
        avg_err = acc.get("avg_prediction_error", 0.0)
        total_trips = acc.get("total_triples", 0)
        avg_trip_err = acc.get("avg_triple_error", 0.0)

        parts.append("## World Model State")
        if total_preds > 0:
            # Count uncertain auto-verifications
            uncertain = sum(
                1 for p in self.data.get("predictions", [])
                if p.get("verified") and p.get("actual", "").startswith("timeframe expired")
            )
            decided = verified - uncertain
            if decided > 0 and correct >= 0:
                pct = round(correct / max(decided, 1) * 100)
                accuracy_str = f"{pct}% accuracy ({correct}/{decided} decided"
                if uncertain > 0:
                    accuracy_str += f", {uncertain} expired uncertain"
                accuracy_str += ")"
            elif uncertain > 0:
                accuracy_str = f"all {uncertain} auto-verified as uncertain (expired)"
            else:
                accuracy_str = f"avg error: {avg_err:.2f} (no decided yet)"
            parts.append(
                f"  Predictions: {total_preds} total, {verified} verified, "
                f"{accuracy_str}"
            )
        else:
            parts.append("  Predictions: (none yet)")

        if total_trips > 0:
            parts.append(
                f"  Action triples: {total_trips} logged, "
                f"avg prediction error: {avg_trip_err:.2f}"
            )
        else:
            parts.append("  Action triples: (none yet)")

        # ── Recent action triples ──
        triples = self.data.get("action_triples", [])
        recent = [t for t in triples if t.get("completed")][-5:]
        if recent:
            parts.append("  Recent action outcomes:")
            for t in reversed(recent):
                err = t.get("prediction_error", 1.0)
                icon = "✓" if err <= 0.3 else ("△" if err <= 0.6 else "✗")
                desc = t.get("action_description", "")[:50]
                parts.append(f"    {icon} {desc} [error={err:.2f}]")

        # ── Biggest discrepancies ──
        discrepancies = self.analyze_recent_discrepancies(3)
        if discrepancies:
            parts.append("  Biggest prediction errors (learning opportunities):")
            for t in discrepancies:
                err = t.get("prediction_error", 0)
                expected = t.get("expected_outcome", "")[:60]
                actual = t.get("actual_outcome", "")[:60]
                parts.append(
                    f"    ✗ error={err:.2f}: expected \"{expected}\" "
                    f"→ got \"{actual}\""
                )

        # ── Unverified predictions ──
        unverified = self.get_unverified_predictions()[:3]
        if unverified:
            parts.append("  Unverified predictions (awaiting outcome):")
            for p in unverified:
                conf = p.get("confidence", 0)
                timeframe = p.get("timeframe", "?")
                text = p.get("text", "")[:60]
                parts.append(f"    ○ [{conf:.0%} conf, {timeframe}] {text}")

        if not any([total_preds, total_trips, recent, discrepancies, unverified]):
            parts.append("  (world model is empty — start by taking actions and making predictions)")

        # ── Calibration guidance (per-type accuracy) ──
        cal = self.format_calibration_guidance()
        if cal and "No per-type" not in cal:
            parts.append("")
            parts.append(cal)

        # ── Discrepancy patterns ──
        patterns = self.get_discrepancy_patterns()
        if patterns:
            parts.append("")
            parts.append("  Recurring discrepancy patterns:")
            for p in patterns:
                themes = ""
                if p.get("common_themes"):
                    themes = f" [themes: {', '.join(p['common_themes'][:3])}]"
                parts.append(
                    f"    ● {p['description']}{themes}"
                )

        # ── Improvement suggestions (bridges Gap 6 → Gap 4) ──
        imp = self.format_improvement_context()
        if imp:
            parts.append("")
            parts.append(imp)

        return "\n".join(parts)

    def format_prediction_insight(self) -> str:
        """A compact summary of prediction accuracy with error trend.

        Shows three categories:
          - Clearly correct predictions (error ≤ 0.3)
          - Clearly incorrect predictions (error > 0.3, not auto-verified expired)
          - Uncertain auto-verifications (expired timeframe, error = 0.5)
        """
        acc = self.data.get("prediction_accuracy", {})
        total = acc.get("total_predictions", 0)
        verified = acc.get("verified_predictions", 0)
        correct = acc.get("correct_predictions", 0)
        incorrect = acc.get("incorrect_predictions", 0)
        per_type = self.get_per_type_accuracy()
        types_count = len(per_type)

        # Count uncertain auto-verifications from the predictions list
        uncertain = 0
        for p in self.data.get("predictions", []):
            if p.get("verified") and p.get("error") is not None:
                if p.get("actual", "").startswith("timeframe expired"):
                    uncertain += 1

        # Compute trend from rolling error history
        trend = self._compute_error_trend()

        if verified > 0:
            # Only compute accuracy percentage from clearly decided predictions
            decided = correct + incorrect
            if decided > 0:
                pct = round(correct / decided * 100)
                parts = [f"Prediction accuracy: {pct}% ({correct}/{decided} decided"]
            else:
                parts = [f"Prediction accuracy: (undetermined — all {verified} verified are uncertain)"]

            if uncertain > 0:
                parts.append(f"{uncertain} uncertain expired")
            parts.append(f"{verified} verified")
            parts.append(f"{total} total")

            base = ", ".join(parts)
            if types_count > 0:
                base += f", tracked {types_count} action types for calibration"
            if trend:
                base += f" [{trend}]"
            return base
        if total > 0:
            return f"Prediction accuracy: {total} predictions made, none verified yet"
        return "Prediction accuracy: no predictions made yet"

    def _compute_error_trend(self) -> str:
        """Compare recent prediction errors vs earlier ones to detect trend.

        Uses the rolling ``error_history`` list: splits the available history
        in half and compares the average of the more recent half vs. the
        earlier half. Requires at least 6 data points for a meaningful signal.

        Returns:
            A short string like ``\"↓ improving\"``, ``\"↑ worsening\"``,
            ``\"→ stable\"``, or empty string if insufficient data.
        """
        acc = self.data.get("prediction_accuracy", {})
        history = acc.get("error_history", [])
        if len(history) < 6:
            return ""
        mid = len(history) // 2
        earlier = history[:mid]
        recent = history[mid:]
        avg_earlier = sum(earlier) / len(earlier)
        avg_recent = sum(recent) / len(recent)
        delta = avg_recent - avg_earlier  # positive = getting worse
        if delta > 0.08:
            return "↑ worsening"
        elif delta < -0.08:
            return "↓ improving"
        else:
            return "→ stable"

    # ── Internal helpers ──────────────────────────────────────────

    def _update_accuracy_stats(self) -> None:
        """Recalculate running statistics from current triples."""
        acc = self.data.setdefault("prediction_accuracy", {})
        triples = self.data.get("action_triples", [])
        completed = [t for t in triples if t.get("prediction_error") is not None]

        acc["total_triples"] = len(triples)

        if completed:
            errors = [t["prediction_error"] for t in completed]
            acc["avg_triple_error"] = round(sum(errors) / len(errors), 4)

        # Also refresh per-type accuracy and discrepancy patterns whenever stats are recalculated
        self._update_per_type_accuracy()
        self._update_discrepancy_patterns()

    def _add_to_error_history(self, error: float) -> None:
        """Append a prediction error to the rolling history for trend analysis.

        Keeps the last 20 errors. The ``error_history`` list lives inside
        ``prediction_accuracy.error_history`` and is persisted with the rest
        of the world model data.

        Called automatically from :meth:`complete_action`,
        :meth:`verify_prediction`, and expired-prediction auto-verification.
        """
        acc = self.data.setdefault("prediction_accuracy", {})
        history = acc.setdefault("error_history", [])
        history.append(error)
        acc["error_history"] = history[-20:]

    def _update_calibration(self, confidence: float, error: float) -> None:
        """Track confidence vs accuracy for calibration curve."""
        acc = self.data.setdefault("prediction_accuracy", {})
        buckets = acc.setdefault("calibration_buckets", [])

        # Find the right confidence bucket (0.0-0.2, 0.2-0.4, etc.)
        bucket_idx = min(int(confidence * 5), 4)  # 0-4
        while len(buckets) <= bucket_idx:
            buckets.append({"count": 0, "total_error": 0.0, "avg_error": 0.0})

        bucket = buckets[bucket_idx]
        bucket["count"] = bucket.get("count", 0) + 1
        bucket["total_error"] = bucket.get("total_error", 0.0) + error
        bucket["avg_error"] = round(
            bucket["total_error"] / bucket["count"], 4
        )

    # ── Per-type accuracy (for adaptive confidence calibration) ─────

    def _update_per_type_accuracy(self) -> None:
        """Recalculate per-action-type prediction error statistics.

        Called automatically after every :meth:`complete_action` and
        :meth:`verify_prediction`. Groups completed action triples by
        ``action_type`` and computes count, avg_error, min, max for each.
        Stores results in ``data["per_type_accuracy"]``.
        """
        triples = [
            t for t in self.data.get("action_triples", [])
            if t.get("completed") and t.get("prediction_error") is not None
        ]
        by_type: Dict[str, Dict[str, Any]] = {}
        for t in triples:
            atype = t.get("action_type", "unknown")
            err = t["prediction_error"]
            if atype not in by_type:
                by_type[atype] = {
                    "count": 0,
                    "sum_error": 0.0,
                    "min_error": 1.0,
                    "max_error": 0.0,
                }
            s = by_type[atype]
            s["count"] += 1
            s["sum_error"] += err
            if err < s["min_error"]:
                s["min_error"] = err
            if err > s["max_error"]:
                s["max_error"] = err

        result: Dict[str, Dict[str, Any]] = {}
        for atype, stats in by_type.items():
            c = stats["count"]
            result[atype] = {
                "count": c,
                "avg_error": round(stats["sum_error"] / c, 4),
                "min_error": round(stats["min_error"], 4),
                "max_error": round(stats["max_error"], 4),
            }
        self.data["per_type_accuracy"] = result

    # ── Discrepancy pattern detection (Gap 6) ────────────────────

    def _update_discrepancy_patterns(self, min_samples: int = 2) -> None:
        """Analyze completed triples with high prediction error for recurring patterns.

        Groups high-error triples (prediction_error >= 0.4) by action type and
        identifies common themes in action descriptions. Updates
        ``data["discrepancy_patterns"]`` for use in context formatting and
        LLM calibration guidance.

        Only runs when there are at least *min_samples* completed triples
        (default 2) to avoid noise from single outliers.

        Patterns include:
          - Action type with chronic high error (systematic prediction bias)
          - Frequent keywords in mispredicted actions (topic-level bias)
          - Number of affected triples and avg error for prioritization

        Called automatically from :meth:`_update_accuracy_stats`.
        """
        patterns: List[Dict[str, Any]] = []
        completed = [
            t for t in self.data.get("action_triples", [])
            if t.get("completed") and t.get("prediction_error") is not None
        ]
        if len(completed) < min_samples:
            self.data["discrepancy_patterns"] = patterns
            return

        # ─ Group high-error triples (>= 0.4) by action type ──
        high_error = [t for t in completed if t["prediction_error"] >= 0.4]
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for t in high_error:
            atype = t.get("action_type", "unknown")
            by_type.setdefault(atype, []).append(t)

        for atype, triples in by_type.items():
            if len(triples) < min_samples:
                continue
            avg_err = sum(t["prediction_error"] for t in triples) / len(triples)
            timestamps = [t.get("timestamp", "") for t in triples if t.get("timestamp")]
            last_obs = max(timestamps) if timestamps else ""

            # Extract common keywords from action descriptions (words that
            # appear in >30% of the high-error triples of this type)
            all_words: List[str] = []
            for t in triples:
                desc = (t.get("action_description", "") or "").lower()
                all_words.extend(
                    w for w in desc.split()
                    if len(w) > 3 and w not in ("with", "from", "that", "this", "into")
                )

            word_counts: Dict[str, int] = {}
            for w in all_words:
                word_counts[w] = word_counts.get(w, 0) + 1
            threshold = max(1, len(triples) * 0.3)
            common_themes = sorted(
                [w for w, c in word_counts.items() if c >= threshold],
                key=lambda w: word_counts[w],
                reverse=True,
            )

            patterns.append({
                "action_type": atype,
                "count": len(triples),
                "total_completed": sum(1 for t in completed if t.get("action_type") == atype),
                "avg_error": round(avg_err, 4),
                "common_themes": common_themes[:5],
                "description": (
                    f"{len(triples)}/{sum(1 for t in completed if t.get('action_type') == atype)} "
                    f"{atype} actions with high prediction error "
                    f"(avg {avg_err:.2f})"
                ),
                "last_observed": last_obs,
            })

        # Sort by count descending (most frequent failure patterns first)
        patterns.sort(key=lambda p: p["count"], reverse=True)
        self.data["discrepancy_patterns"] = patterns

    def get_discrepancy_patterns(self) -> List[Dict[str, Any]]:
        """Return detected discrepancy patterns.

        Each pattern dict contains:
          - action_type: The action type (shell, write_file, etc.)
          - count: How many high-error triples of this type
          - total_completed: Total completed triples of this type
          - avg_error: Average prediction error for this pattern
          - common_themes: Top keywords in mispredicted descriptions
          - description: Human-readable summary
          - last_observed: ISO timestamp of most recent occurrence

        Returns:
            List of pattern dicts, sorted by frequency (most frequent first).
        """
        return list(self.data.get("discrepancy_patterns", []))

    def get_per_type_accuracy(self) -> Dict[str, Dict[str, Any]]:
        """Return per-action-type prediction error statistics.

        Returns a dict mapping action_type → {
            "count": int,          # How many triples of this type
            "avg_error": float,    # Average prediction error (0.0–1.0)
            "min_error": float,    # Best prediction error
            "max_error": float,    # Worst prediction error
        }

        Useful for calibrating confidence: types with low ``avg_error``
        are well-modelled; types with high ``avg_error`` need
        conservative confidence estimates.
        """
        return dict(self.data.get("per_type_accuracy", {}))

    def adjust_confidence(
        self,
        raw_confidence: float,
        action_type: Optional[str] = None,
    ) -> float:
        """Adjust a raw confidence estimate using historical accuracy.

        When *action_type* is provided, uses per-type historical error
        rates to adjust confidence. Falls back to global avg triple
        error when the type is unknown or not provided.

        The adjustment follows a simple rule:
          - If the type's avg_error is low (<0.25), confidence is
            *increased* toward 1.0 by a proportional amount.
          - If the type's avg_error is high (>0.4), confidence is
            *decreased* toward 0.0.
          - Otherwise, confidence is left largely unchanged.

        When there is no historical data for the type and no global
        data either, returns the raw confidence unchanged (no basis
        for adjustment).

        Args:
            raw_confidence: The raw LLM confidence (0.0–1.0).
            action_type: The action type (e.g. ``"shell"``, ``"write_file"``).

        Returns:
            Adjusted confidence (0.0–1.0), clamped to valid range.
        """
        if not (0.0 <= raw_confidence <= 1.0):
            raw_confidence = max(0.0, min(1.0, raw_confidence))

        # Determine the error to use for adjustment
        per_type = self.get_per_type_accuracy()
        total_triples = self.data.get("prediction_accuracy", {}).get("total_triples", 0)

        if action_type and action_type in per_type:
            avg_err = per_type[action_type]["avg_error"]
            count = per_type[action_type]["count"]
            # Low sample count → conservative adjustment (blend with global)
            if count < 3:
                blend = count / 3.0  # 0 → 1 as samples grow
                global_avg = self.data.get("prediction_accuracy", {}).get(
                    "avg_triple_error", 0.5
                )
                avg_err = avg_err * blend + global_avg * (1.0 - blend)
        elif total_triples > 0:
            # Have global data but no per-type — use global average
            acc = self.data.get("prediction_accuracy", {})
            avg_err = acc.get("avg_triple_error", 0.5)
        else:
            # No data at all — nothing to calibrate with
            return raw_confidence

        # Adjustment factor: scale from error space to confidence space
        # When avg_err is 0.0 → adjustment pulls confidence toward 1.0
        # When avg_err is 1.0 → adjustment pulls confidence toward 0.0
        # The strength of adjustment is proportional to error distance from 0.5
        adjustment_strength = abs(avg_err - 0.5) * 2.0  # 0.0 → 1.0

        # Target confidence: 1.0 for low-error types, 0.0 for high-error
        target = 1.0 - avg_err

        # Blend raw and adjusted
        adjusted = raw_confidence * (1.0 - adjustment_strength) + target * adjustment_strength

        return max(0.0, min(1.0, round(adjusted, 4)))

    def format_calibration_guidance(self) -> str:
        """Generate a calibration context block for the LLM prompt.

        Shows which action types are well-predicted vs poorly-predicted,
        and suggests how to adjust confidence estimates.
        """
        per_type = self.get_per_type_accuracy()
        if not per_type:
            return "  No per-type calibration data yet."

        lines: List[str] = ["  Per-type prediction accuracy:"]
        # Sort by avg_error descending (worst first)
        sorted_types = sorted(per_type.items(), key=lambda x: x[1]["avg_error"], reverse=True)

        for atype, stats in sorted_types:
            err = stats["avg_error"]
            count = stats["count"]
            icon = "✓" if err <= 0.25 else ("△" if err <= 0.4 else "✗")
            lines.append(
                f"    {icon} {atype}: avg_err={err:.2f} "
                f"(n={count}, range={stats['min_error']:.2f}–{stats['max_error']:.2f})"
            )

        # Identify best/worst predicted types
        if sorted_types:
            worst_type, worst_stats = sorted_types[0]
            best_type, best_stats = sorted_types[-1]
            lines.append(
                f"    → Best predicted: {best_type} "
                f"(err={best_stats['avg_error']:.2f}, n={best_stats['count']})"
            )
            lines.append(
                f"    → Worst predicted: {worst_type} "
                f"(err={worst_stats['avg_error']:.2f}, n={worst_stats['count']})"
            )

        return "\n".join(lines)

    # ── Improvement suggestions (bridges Gap 6 → Gap 4) ──────────

    def generate_improvement_suggestions(self) -> List[Dict[str, Any]]:
        """Generate structured improvement suggestions from world model data.

        Analyzes per-type accuracy and discrepancy patterns to produce
        actionable goal proposals the system can pursue. Suggestions include:

          - Action types with elevated prediction error that need calibration
          - Recurring discrepancy patterns that need root-cause analysis
          - Action types with insufficient data for reliable prediction
          - Calibration gaps (confidence vs. accuracy mismatch)

        Returns:
            A list of suggestion dicts, each containing:
              - ``type``: ``calibrate`` | ``investigate`` | ``collect_data`` | ``improve_prediction``
              - ``title``: One-line suggestion title
              - ``description``: What to do and why
              - ``rationale``: Why this matters for self-evolution
              - ``gap_reference``: Which gap this addresses (``"6"`` or ``"4"``)
              - ``priority``: 1–5 (1=highest)
        """
        suggestions: List[Dict[str, Any]] = []
        per_type = self.get_per_type_accuracy()
        patterns = self.get_discrepancy_patterns()
        acc = self.data.get("prediction_accuracy", {})
        triples = self.data.get("action_triples", [])

        # 1. High-error action types → calibrate
        for atype, stats in sorted(per_type.items(),
                                    key=lambda x: x[1]["avg_error"], reverse=True):
            avg_err = stats["avg_error"]
            count = stats["count"]
            if avg_err >= 0.4 and count >= 2:
                suggestions.append({
                    "type": "calibrate",
                    "title": f"Improve prediction calibration for {atype} actions",
                    "description": (
                        f"{atype} actions have avg prediction error {avg_err:.2f} "
                        f"across {count} samples. Calibrate the prediction heuristic "
                        f"to reduce error below 0.3."
                    ),
                    "rationale": (
                        f"Lower {atype} prediction error improves world model accuracy, "
                        f"enabling better action selection and risk assessment."
                    ),
                    "gap_reference": "6",
                    "priority": 2 if avg_err >= 0.7 else 3,
                })

        # 2. Discrepancy patterns → investigate root cause
        for pat in patterns:
            atype = pat.get("action_type", "unknown")
            pcount = pat.get("count", 0)
            ptotal = pat.get("total_completed", 0)
            pavg = pat.get("avg_error", 0)
            themes = pat.get("common_themes", [])
            if pcount >= 2:
                theme_str = f" (keywords: {', '.join(themes[:3])})" if themes else ""
                suggestions.append({
                    "type": "investigate",
                    "title": f"Investigate {atype} prediction failures{theme_str}",
                    "description": (
                        f"{pcount}/{ptotal} {atype} actions have high prediction error "
                        f"(avg {pavg:.2f}). Investigate whether the failure is in "
                        f"the heuristic or the action description quality.{theme_str}"
                    ),
                    "rationale": (
                        "Understanding the root cause of systematic prediction "
                        "failures enables targeted improvements to the world model."
                    ),
                    "gap_reference": "6",
                    "priority": 3,
                })

        # 3. Action types with insufficient data → collect more samples
        for atype, stats in sorted(per_type.items(),
                                    key=lambda x: x[1]["count"]):
            count = stats["count"]
            if count < 3:
                suggestions.append({
                    "type": "collect_data",
                    "title": f"Gather more {atype} action samples for reliable calibration",
                    "description": (
                        f"Only {count} {atype} action{'s' if count != 1 else ''} "
                        f"recorded. Need at least 3 for statistically meaningful "
                        f"per-type calibration."
                    ),
                    "rationale": (
                        "More samples per action type improve confidence adjustment "
                        "accuracy and discrepancy detection."
                    ),
                    "gap_reference": "6",
                    "priority": 4,
                })

        # 4. Calibration gap check — confidence vs actual accuracy
        buckets = acc.get("calibration_buckets", [])
        for i, bucket in enumerate(buckets):
            count = bucket.get("count", 0)
            avg_err = bucket.get("avg_error", 0)
            if count >= 3 and avg_err > 0.3:
                bucket_low = i * 20  # 0, 20, 40, 60, 80
                bucket_high = bucket_low + 20
                suggestions.append({
                    "type": "improve_prediction",
                    "title": (
                        f"Fix overconfidence at {bucket_low}–{bucket_high}% "
                        f"confidence level"
                    ),
                    "description": (
                        f"Predictions at {bucket_low}–{bucket_high}% confidence have "
                        f"avg error {avg_err:.2f} (n={count}), indicating systematic "
                        f"overconfidence. Adjust the confidence threshold or calibration curve."
                    ),
                    "rationale": (
                        "Correcting systematic overconfidence improves the reliability "
                        "of all future predictions and the system's self-awareness."
                    ),
                    "gap_reference": "6",
                    "priority": 2 if avg_err >= 0.5 else 3,
                })

        # Sort by priority (lower number = higher priority)
        suggestions.sort(key=lambda s: s["priority"])
        return suggestions[:8]  # Cap at 8 to avoid overwhelming the LLM

    def format_improvement_context(self) -> str:
        """Format improvement suggestions as a readable context block.

        Returns a string suitable for injecting into the think_daemon
        prompt, with structured suggestions the LLM can act on via
        ``new_goal`` actions. Returns empty string if no suggestions.
        """
        suggestions = self.generate_improvement_suggestions()
        if not suggestions:
            return ""

        parts: List[str] = [
            "## World Model Improvement Suggestions",
            "  (Discrepancy-driven goal proposals from prediction analysis)",
        ]
        for s in suggestions:
            icons = {
                "calibrate": "⚙",
                "investigate": "🔍",
                "collect_data": "📊",
                "improve_prediction": "🎯",
            }
            icon = icons.get(s["type"], "·")
            priority_str = "!" * s["priority"] if s["priority"] <= 3 else "·" * (s["priority"] - 2)
            parts.append(
                f"  {icon} [{priority_str}] {s['title']}"
            )
            parts.append(f"     {s['description']}")
        parts.append(
            "  (Use new_goal with gap_reference='6' or '4' to pursue a suggestion)"
        )
        return "\n".join(parts)

    # ── Proactive action guidance

    def format_action_guidance(
        self,
        action_type: str,
        description: str = "",
    ) -> Optional[str]:
        """Check if a proposed action is risky based on discrepancy patterns.

        Args:
            action_type: The type of action being considered (e.g. ``"shell"``,
                ``"write_file"``, ``"git_commit"``).
            description: The action description for keyword matching.

        Returns:
            A warning string if the action is risky, or None if no concerns.
        """
        per_type = self.get_per_type_accuracy()
        patterns = self.get_discrepancy_patterns()
        parts: List[str] = []

        # 1. Per-type accuracy check
        if action_type in per_type:
            stats = per_type[action_type]
            avg_err = stats["avg_error"]
            count = stats["count"]
            if avg_err >= 0.4 and count >= 2:
                parts.append(
                    f"  ⚠ Type '{action_type}' has elevated prediction error "
                    f"({avg_err:.2f} avg across {count} actions)."
                )

        # 2. Discrepancy pattern check — match action type + description keywords
        if description:
            desc_lower = description.lower()
            for pat in patterns:
                if pat.get("action_type") != action_type:
                    continue
                themes = pat.get("common_themes", [])
                matched_keywords = [kw for kw in themes if kw in desc_lower]
                if matched_keywords:
                    parts.append(
                        f"  ⚠ Keyword match in discrepancy pattern: "
                        f"{', '.join(matched_keywords)} "
                        f"({pat['count']} failures, avg err {pat['avg_error']:.2f})"
                    )

        # 3. Recent action trend — last 3 actions of this type
        recent = [
            t for t in self.data.get("action_triples", [])[-10:]
            if t.get("action_type") == action_type
            and t.get("completed") and t.get("prediction_error") is not None
        ]
        if len(recent) >= 2:
            recent_errors = [t["prediction_error"] for t in recent[-3:]]
            recent_avg = sum(recent_errors) / len(recent_errors)
            if recent_avg >= 0.4:
                parts.append(
                    f"  ⚠ Last {len(recent_errors)} '{action_type}' actions averaged "
                    f"{recent_avg:.2f} prediction error."
                )

        if not parts:
            return None  # No concerns
        return "Action risk assessment:\n" + "\n".join(parts)

    # ── Persistence ───────────────────────────────────────────────

    @staticmethod
    def storage_path() -> Path:
        return get_evolve_dir() / "world_model.json"

    def save(self, path: Optional[Path] = None) -> Path:
        """Persist to disk as JSON (atomic write)."""
        target = path or self.storage_path()
        safe_write_json(target, self.data)
        return target

    def _save(self) -> None:
        """Internal save — convenience wrapper."""
        self.save()

    @classmethod
    def load(cls, path: Optional[Path] = None) -> WorldModel:
        """Load from disk, returning a fresh WorldModel on failure."""
        target = path or cls.storage_path()
        data = safe_read_json(target)
        result = cls(data=data) if isinstance(data, dict) else cls()
        # Recompute computed fields (per-type accuracy) after load
        # to avoid stale cache until the next action is completed.
        result._update_per_type_accuracy()
        return result

    # ── Convenience ───────────────────────────────────────────────

    def __repr__(self) -> str:
        triples = len(self.data.get("action_triples", []))
        preds = len(self.data.get("predictions", []))
        verified = sum(1 for p in self.data.get("predictions", []) if p.get("verified"))
        return (
            f"<WorldModel triples={triples} predictions={preds} "
            f"verified={verified}>"
        )


# ═══════════════════════════════════════════════════════════════════
#  Convenience helpers (module-level — for easy import from daemon)
# ═══════════════════════════════════════════════════════════════════

def load_world_model() -> WorldModel:
    """Load the world model from disk."""
    return WorldModel.load()


def save_world_model(wm: WorldModel) -> None:
    """Save the world model to disk."""
    wm.save()


def format_world_model_context() -> str:
    """Load world model and format context. One-liner for callers."""
    try:
        wm = load_world_model()
        return wm.format_world_model_context()
    except Exception as e:
        logger.warning("Failed to format world model context: %s", e)
        return ""


# ═══════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    """CLI for inspecting the world model."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Hermes Evolved — World Model Inspector"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show world model status summary"
    )
    parser.add_argument(
        "--discrepancies", type=int, nargs="?", const=5, default=0,
        help="Show top N action prediction errors (default: 5)"
    )
    parser.add_argument(
        "--predictions", action="store_true",
        help="List all predictions with verification status"
    )
    parser.add_argument(
        "--unverified", action="store_true",
        help="List unverified predictions"
    )
    args = parser.parse_args()

    wm = load_world_model()

    if args.status or not any([args.discrepancies, args.predictions, args.unverified]):
        print(wm.format_world_model_context())

    if args.discrepancies:
        print("\n=== Top Discrepancies ===")
        for t in wm.analyze_recent_discrepancies(args.discrepancies):
            print(f"  error={t.get('prediction_error', 0):.2f}")
            print(f"    action: {t.get('action_description', '')[:80]}")
            print(f"    expected: {t.get('expected_outcome', '')[:80]}")
            print(f"    actual: {t.get('actual_outcome', '')[:80]}")

    if args.predictions:
        print("\n=== All Predictions ===")
        for p in wm.data.get("predictions", []):
            status = "✓" if p.get("verified") else "○"
            err = p.get("error", "?")
            if isinstance(err, float):
                err_str = f"{err:.2f}"
            else:
                err_str = str(err)
            print(
                f"  {status} [{p.get('confidence', 0):.0%}, {p.get('timeframe', '?')}] "
                f"{p.get('text', '')[:80]} [err={err_str}]"
            )

    if args.unverified:
        print("\n=== Unverified Predictions ===")
        for p in wm.get_unverified_predictions():
            print(
                f"  ○ [{p.get('confidence', 0):.0%}, {p.get('timeframe', '?')}] "
                f"{p.get('text', '')[:100]}"
            )


if __name__ == "__main__":
    main()
