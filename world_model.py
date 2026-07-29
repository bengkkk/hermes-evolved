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

_DEFAULT_WORLD_MODEL: Dict[str, Any] = {
    "version": 3,
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

    # ── Exit-code aware comparison ──
    # Actual output often starts with "exit=N:" — check if exit code
    # correlates with expected success/failure keywords
    exit_match = re.search(r'exit=(\d+)', a_lower)
    if exit_match:
        exit_code = int(exit_match.group(1))
        success_expected = any(kw in e_lower for kw in
            ('success', 'complete', 'create', 'write', 'deploy', 'install',
             'commit', 'push', 'run', 'list', 'show', 'print'))
        if success_expected and exit_code == 0:
            # Expected success and got success — low error
            return 0.15
        elif not success_expected and exit_code != 0:
            # Expected failure and got failure — low error
            return 0.2
        elif success_expected and exit_code != 0:
            # Expected success but got failure — high error
            return 0.85
        # Exit code didn't match expectations — medium-high error
        return 0.6

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
        triple_id = f"act_{now_compact()}"
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
        pred_id = f"pred_{now_compact()}"
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
                return error
        return None

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
            pct = round(correct / max(verified, 1) * 100)
            parts.append(
                f"  Predictions: {total_preds} total, {verified} verified, "
                f"{correct} correct ({pct}% accuracy, avg error: {avg_err:.2f})"
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

        return "\n".join(parts)

    def format_prediction_insight(self) -> str:
        """A one-line summary of prediction accuracy for the prompt header."""
        acc = self.data.get("prediction_accuracy", {})
        total = acc.get("total_predictions", 0)
        verified = acc.get("verified_predictions", 0)
        correct = acc.get("correct_predictions", 0)
        per_type = self.get_per_type_accuracy()
        types_count = len(per_type)
        if verified > 0:
            pct = round(correct / verified * 100)
            base = f"Prediction accuracy: {pct}% ({correct}/{verified} verified, {total} total)"
            if types_count > 0:
                base += f", tracked {types_count} action types for calibration"
            return base
        if total > 0:
            return f"Prediction accuracy: {total} predictions made, none verified yet"
        return "Prediction accuracy: no predictions made yet"

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

        # Also refresh per-type accuracy whenever stats are recalculated
        self._update_per_type_accuracy()

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
        return cls(data=data) if isinstance(data, dict) else cls()

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
