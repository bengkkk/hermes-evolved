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

# ── Data layer: try workspace version first, fall back to direct path computation ──
try:
    # First try importing from the workspace data_layer (has evolve-specific functions)
    from data_layer import get_evolve_dir, safe_write_json, safe_read_json, now_compact, now_iso
except ImportError:
    # Fallback for environments where data_layer is missing evolve functions
    # (e.g., when running from /opt/hermes-evolved/ which has a minimal data_layer)
    import json as _json
    import os as _os
    from datetime import datetime as _dt, timezone as _tz
    from pathlib import Path as _Path

    def get_evolve_dir() -> _Path:
        hermes_home = _Path(_os.environ.get("HERMES_HOME", _Path.home() / ".hermes"))
        return hermes_home / "evolve"

    def safe_write_json(path: _Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(_json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def safe_read_json(path: _Path) -> Any:
        if path.exists():
            try:
                return _json.loads(path.read_text(encoding="utf-8"))
            except _json.JSONDecodeError:
                return None
        return None

    def now_compact() -> str:
        return _dt.now(_tz.utc).strftime("%Y%m%d%H%M%S")

    def now_iso() -> str:
        return _dt.now(_tz.utc).isoformat()

logger = logging.getLogger(__name__)

# Cap stored parameter values (e.g. write_file content, long shell commands)
# so the world model data file stays bounded even when actions carry large
# payloads. Only the leading chunk is needed for token-based similarity
# matching in predict_action_outcome; the full payload is never reconstructed
# from the triple.
_MAX_PARAM_VALUE_LEN = 300

# Punctuation stripped from discrepancy-theme keywords at extraction time so
# word-boundary matching in format_action_guidance() is reliable ("check:"
# -> "check", "auto-default:" -> "auto-default"). Internal punctuation
# (hyphens, dots) is preserved.
_THEME_PUNCT = ":.,;!?()[]{}'\"`"


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
    "version": 5,
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

# Words that carry no predictive content in a prediction sentence.
# Used by _extract_topic_tokens to identify the DISTINCTIVE terms of a
# prediction — the terms we search for in action-triple evidence when
# auto-verifying whether the prediction was actually fulfilled.
_PREDICTION_STOPWORDS: frozenset = frozenset("""
will would shall should can could may might must this that these those
which what when where while who whom whose with from under over into onto
through about after before between during without within across against
along among around at by for in of on to up down off out beyond upon via
near than then there their them they its it's its it are was were been
being have has had do does did doing get got gets make made makes use
used uses using one two three four first second next last new now way
part etc least reveal reveals revealed return returns returned allow
allows allowed advance advances advanced complete completes completed
execute executes executed perform performs performed continue continues
continued result results resulted our ourself ourselves your yourself
yourselves themselves itself not no nor also just more most some such
only very each other another every any both all both
""".split())


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

    # The daemon's LLM occasionally emits numbers where strings are
    # expected (same bug class as the 2026-08-01 06:17 crash: 'int'
    # object has no attribute 'strip' on goal gap_reference).  Coerce
    # before any string method so a stored int expected/actual can never
    # crash a cycle inside complete_action's except-path re-entry.
    if not isinstance(expected, str):
        expected = str(expected)
    if not isinstance(actual, str):
        actual = str(actual)

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
        r"0 files changed|0 insertions|0 deletions|"
        r"requirement already satisfied|already installed)",
        a_lower,
    ))
    if informational_nonerror:
        return 0.15

    # ── Mutual exit=0 success heuristic ──
    # When BOTH expected and actual agree the command exited 0, the
    # prediction was correct at the success/failure level.  The specific
    # output after "exit=0: " (e.g. "files" vs "23" for an ls command)
    # is typically truncated or differs due to nondeterministic output —
    # the prediction was semantically correct at the success/failure level.
    # The expected side is often a descriptive sentence ("git commit
    # succeeds with exit=0, creating commit") rather than a literal
    # "exit=0: ..." prefix, so match the exit=0 marker anywhere in the
    # expected string instead of requiring a prefix.
    # Score low (0.15) instead of falling through to bigram comparison
    # which would produce high error from non-overlapping content tokens.
    if re.search(r"exit\s*[=:]\s*0\b", e_lower) and re.search(
        r"exit\s*[=:]\s*0\b", a_lower
    ):
        return 0.15

    # ── Mutual HTTP-status success heuristic (api_call outcomes) ──
    # The host bridge wraps api_call results as
    #   exit=0: {"status": 200, "bytes": N, "body": "..."}
    # while daemon predictions phrase success as "HTTP 200: ..." — no
    # exit= marker — so the mutual exit=0 heuristic above cannot fire
    # and a fully successful call scores 0.5 ("mixed") purely because
    # the JSON body dwarfs the prediction text (observed 2026-08-01 on
    # the first real GitHub read: expected "HTTP 200: JSON body listing
    # GitHub API endpoint fields" vs actual exit=0/status 200/2262 bytes
    # scored 0.5, making the Gap 10 calibration target of avg error <
    # 0.3 unreachable for a *correct* prediction).  Treat expected 2xx +
    # observed 2xx as the api_call equivalent of mutual exit=0: the
    # success/failure level was predicted correctly.  Compares the
    # hundreds digit (2xx vs 4xx/5xx), so a predicted 200 against an
    # observed 404 still falls through to content comparison.
    _http_re = re.compile(r"\bhttp(?:/\d(?:\.\d)?)?\s*([1-5]\d\d)\b")
    _exp_status_m = _http_re.search(e_lower)
    _act_status_m = _http_re.search(a_lower) or re.search(
        r'"status"\s*:\s*([1-5]\d\d)\b', a_lower
    )
    if _exp_status_m and _act_status_m:
        if _exp_status_m.group(1)[0] == _act_status_m.group(1)[0]:
            return 0.15


    # ── Exit-code observation (ground truth for command success/failure) ──
    # The exit code tells us whether the command itself succeeded, but NOT
    # whether the OUTCOME matched the PREDICTION.  We therefore observe it
    # and apply it as a modifier at the end rather than short-circuiting,
    # so content comparison still happens and produces varied errors that
    # the world model can learn from.
    # Exit=0 caps the final error at 0.5 (command succeeded → not a total loss);
    # exit≠0 floors at 0.5 (command failed → not a perfect match).
    _exit_was_zero: Optional[bool] = None
    exit_match = re.search(r'exit=(\d+)', a_lower)
    if exit_match:
        exit_code = int(exit_match.group(1))
        _exit_was_zero = (exit_code == 0)

    # ── Shell/Tool success/failure keywords ──
    # Content-based heuristics for outputs that lack an explicit exit=
    # marker.  These are less reliable than exit code but better than
    # pure string similarity.
    # Zero-count failure phrases ('0 failed', '0 errors') are success
    # signals - pytest-style runners report 'N passed, 0 failed' as a
    # clean pass - so neutralize them before keyword scanning.  This
    # mirrors the normalization in _score_evidence_blob so both
    # prediction-error paths treat zero failure counts consistently.
    a_keywords = re.sub(
        r'\b0\s+(failed|failure|failures|error|errors)\b',
        ' zero_failures ',
        a_lower,
    )
    tool_failure = bool(re.search(
        r'\b(traceback|error|failed|permission denied|not found|no such)\b',
        a_keywords,
    ))
    tool_success = bool(re.search(
        r'\b(passed|succeeded|ok|complete|done)\b',
        a_keywords,
    ) or ' zero_failures ' in a_keywords)

    # ── Content-based error: accumulate into a single variable ──
    # We'll apply the exit-code modifier at the end so that errors
    # reflect BOTH content mismatch AND command success/failure.
    _error: float

    if tool_failure and not tool_success:
        _error = 0.85  # Tool reported failure
    elif tool_success and not tool_failure:
        _error = 0.15  # Tool reported success
    else:
        # ── Character bigram similarity ──
        def _bigrams(s: str) -> set:
            return {s[i:i + 2] for i in range(len(s) - 1)}

        e_bg = _bigrams(e_lower)
        a_bg = _bigrams(a_lower)

        if e_bg and a_bg:
            intersection = e_bg & a_bg
            union = e_bg | a_bg
            jaccard = len(intersection) / len(union)
            if jaccard >= 0.6:
                _error = 0.15
            elif jaccard >= 0.35:
                _error = 0.4
            elif jaccard >= 0.15:
                _error = 0.6
            elif jaccard > 0.0:
                _error = 0.8
            else:
                # fall through to token overlap below
                _error = None
        else:
            _error = None

        # ── Token overlap (word-level, last resort) ──
        if _error is None:
            def _tokenize(s: str) -> set:
                return set(re.findall(r"[a-z0-9]+", s))

            e_tokens = _tokenize(e_lower)
            a_tokens = _tokenize(a_lower)

            if not e_tokens or not a_tokens:
                _error = 1.0
            else:
                intersection = e_tokens & a_tokens
                union = e_tokens | a_tokens
                jaccard = len(intersection) / len(union)

                if jaccard >= 0.5:
                    _error = 0.5
                elif jaccard > 0.0:
                    _error = 0.75
                else:
                    _error = 1.0

    # ── Exit-code modifier ──
    # Apply AFTER content comparison so errors reflect BOTH content
    # mismatch AND whether the command technically succeeded or failed.
    if _exit_was_zero is True:
        # Command succeeded — cap error at 0.5 (can't be total failure)
        return min(_error, 0.5)
    elif _exit_was_zero is False:
        # Command failed — floor error at 0.5 (can't be perfect match)
        return max(_error, 0.5)
    else:
        return _error


def _command_changed_since(
    current_params: Optional[Dict[str, Any]],
    past: Optional[Dict[str, Any]],
) -> bool:
    """True when a past action's outcome must NOT be reused as the prediction
    because the underlying shell command changed.

    Shell outcomes are deterministic functions of the exact command text, so a
    past *failure* of a different command is a stale template, not a signal.
    Observed 2026-08-01: an auto-default shell command with broken quoting
    failed once ("Syntax error: Unterminated quoted string"); after the quoting
    was fixed, ``predict_action_outcome`` kept reusing that failure as the
    predicted outcome for the fixed command, manufacturing ~0.5 prediction
    error every cycle and feeding the shell discrepancy pattern.

    Conservative by design:
      - Only applies when current parameters carry a ``command`` (shell
        actions). write_file content / git_commit messages legitimately vary
        between runs, so they never trip this guard.
      - Returns False when the stored past command was truncated at write time
        (``...[truncated]`` suffix) — a truncated copy cannot be compared
        reliably, so the caller keeps the existing copy-the-outcome behavior.
      - The comparison collapses whitespace but does NOT normalize quote
        characters, so a quoting fix (the actual root cause observed above)
        reads as a command change while ``cmd  --flag`` vs ``cmd --flag``
        still compare equal.
    """
    if not current_params or not isinstance(current_params, dict):
        return False
    cur = current_params.get("command")
    past_cmd = past.get("command") if isinstance(past, dict) else None
    if not isinstance(cur, str) or not isinstance(past_cmd, str):
        return False
    cur = cur.strip()
    past_cmd = past_cmd.strip()
    if not cur or not past_cmd or past_cmd.endswith("...[truncated]"):
        return False
    return " ".join(cur.split()) != " ".join(past_cmd.split())


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
        expected_source: str = "llm",
        prediction_confidence: Optional[float] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record an action BEFORE execution, returning the triple ID.

        Call this BEFORE executing the action to capture the *expected* outcome.
        Then call :meth:`complete_action` AFTER execution with the actual outcome.

        Args:
            action_type: ``write_file`` | ``shell`` | ``git_commit`` | ``install_package``
            action_description: Human-readable description of what the action does.
            expected_outcome: What the system expects will happen (from LLM prediction).
            expected_source: Where the prediction came from — ``"llm"`` (LLM-generated),
                ``"world_model"`` (data-driven from historical triples),
                or ``"fallback"`` (type+description default).
            prediction_confidence: Optional confidence level (0.0–1.0) for this
                prediction. When set, this value is used to populate the calibration
                curve via ``_update_calibration`` when the action is completed.
                If not provided, a default is derived from ``expected_source``:
                ``"world_model"`` → 0.55, ``"llm"`` → 0.65, ``"fallback"`` → 0.3.
            parameters: Optional dict of the actual action payload —
                e.g. ``{"command": "ls -la"}`` for shell actions,
                ``{"path": "/path/to/file.py", "content": "..."}`` for write_file,
                ``{"message": "fix: bug"}`` for git_commit.
                Stored alongside the description so the data-driven predictor
                (predict_action_outcome) can match on actual action parameters,
                not just high-level descriptions.

        Returns:
            Triple ID to pass to :meth:`complete_action`.
        """
        triple_id = _unique_id("act")
        # The daemon's LLM occasionally emits non-strings for these
        # fields (2026-08-01 crash class: 'int' object has no attribute
        # 'strip').  Coerce at the persistence boundary — same principle
        # as data_layer.propose()'s _coerce_stripped_str — so a triple
        # can never carry a non-str field that later crashes
        # complete_action / _compute_prediction_error.
        if not isinstance(action_type, str):
            action_type = str(action_type)
        if not isinstance(action_description, str):
            action_description = str(action_description)
        if expected_outcome is not None and not isinstance(expected_outcome, str):
            expected_outcome = str(expected_outcome)
        # Derive default confidence from source if not explicitly provided
        if prediction_confidence is not None:
            conf = max(0.0, min(1.0, prediction_confidence))
        elif expected_source == "world_model":
            conf = 0.55
        elif expected_source == "llm":
            conf = 0.65
        elif expected_source == "fallback":
            conf = 0.3
        else:
            conf = 0.5  # unknown source

        # Bound parameter payloads so world_model.json cannot grow without
        # limit from large write_file content or long shell commands.
        stored_params: Dict[str, Any] = {}
        if parameters:
            for pkey, pval in parameters.items():
                if isinstance(pval, str) and len(pval) > _MAX_PARAM_VALUE_LEN:
                    stored_params[pkey] = pval[:_MAX_PARAM_VALUE_LEN] + "...[truncated]"
                else:
                    stored_params[pkey] = pval

        triple: Dict[str, Any] = {
            "id": triple_id,
            "action_type": action_type,
            "action_description": action_description,
            "action_parameters": stored_params,
            "expected_outcome": expected_outcome or "unknown",
            "expected_source": expected_source,
            "prediction_confidence": conf,
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
        Also updates the calibration curve from the action triple's
        prediction confidence, so action-level outcomes feed into
        the system's confidence calibration (not just macro predictions).

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
                # Feed action triple into calibration curve using stored confidence
                confidence = triple.get("prediction_confidence")
                if confidence is not None:
                    self._calibrate_from_action(confidence, error)
                return error
        return None

    def record_action_complete(
        self,
        action_type: str,
        action_description: str,
        action_output: str,
        expected_outcome: str = "",
        expected_source: str = "llm",
        prediction_confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Convenience: record and complete an action in one call.

        Use this when you don't need to record the expected outcome before
        the action executes (e.g. when you only know the outcome after the fact
        and want to retroactively log the triple).

        For the full predict→observe→compare cycle, use
        :meth:`record_action` + :meth:`complete_action` instead.
        """
        triple_id = self.record_action(
            action_type, action_description, expected_outcome,
            expected_source, prediction_confidence,
        )
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
        # Keep accuracy counters consistent with the capped list: the
        # incremental total_predictions counter would drift once old
        # records age out of the 100-entry cap, breaking the
        # verified == correct + incorrect + uncertain invariant.
        self._reconcile_prediction_stats()
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
                pred["outcome_class"] = "correct" if error <= 0.3 else "incorrect"

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

    # ── Evidence-based verification (Gap 6 learning loop) ─────────

    @staticmethod
    def _extract_topic_tokens(text: str) -> List[str]:
        """Extract the distinctive topic terms of a prediction sentence.

        Filters out function words and weak prediction verbs (the
        stopword set), leaving the terms that identify WHAT the
        prediction is about — paths, filenames, and content words.
        These are the terms we search for in action-triple evidence.
        """
        tokens: set = set()
        for m in re.finditer(r"[a-z0-9_./\\-]+", (text or "").lower()):
            tok = m.group(0)
            if tok in _PREDICTION_STOPWORDS:
                continue
            if tok.isdigit():
                continue
            # Keep paths/filenames regardless of length; require ≥4 chars otherwise
            if len(tok) < 4 and "/" not in tok and "." not in tok and "\\" not in tok:
                continue
            tokens.add(tok)
        return sorted(tokens)

    def _find_prediction_evidence(
        self, pred_text: str
    ) -> Optional[Tuple[str, List[str]]]:
        """Search recorded action triples for evidence about a prediction.

        A triple counts as evidence when at least 2 of the prediction's
        topic tokens appear in its description/expected/actual fields.
        Returns ``(lowercased_evidence_blob, matched_tokens)`` for the
        triple with the most token matches, or None when no triple
        provides meaningful evidence.

        This is how the system closes the learning loop on its own:
        predictions about what its actions will achieve are checked
        against what those actions actually produced — without needing
        an LLM or human to adjudicate.
        """
        tokens = self._extract_topic_tokens(pred_text)
        if len(tokens) < 2:
            return None  # Not enough to match on
        best: Optional[Tuple[str, List[str]]] = None
        for triple in self.data.get("action_triples", []):
            blob = " ".join(
                str(triple.get(k, ""))
                for k in ("actual_outcome", "action_description", "expected_outcome")
            ).lower()
            matched = [tok for tok in tokens if tok in blob]
            if len(matched) >= 2 and (best is None or len(matched) > len(best[1])):
                best = (blob, matched)
        return best

    @staticmethod
    def _score_evidence_blob(blob_l: str) -> Optional[float]:
        """Score an evidence blob as fulfilled (0.15) or contradicted (0.85).

        Returns None when the evidence is ambiguous (both success and
        failure markers present, or neither) — callers fall back to the
        uncertain (0.5) path in that case.

        Informational non-events ("nothing to commit", "already up to
        date", ...) are treated as ambiguous: the predicted action did
        not occur, but nothing failed either.

        Zero-count failure phrases ("0 failed", "0 failures",
        "0 errors") are success signals, not failure markers — pytest-style
        runners report "N passed, 0 failed" as a clean pass. They are
        normalized before marker scanning so the mixed-evidence ambiguity
        does not swallow fulfilled predictions whose outcome text carries
        a failure count of zero.
        """
        informational = bool(re.search(
            r"(nothing to commit|working tree clean|already up.to.date|"
            r"no changes|nothing changed|nothing to do|"
            r"0 files changed|0 insertions|0 deletions|"
            r"requirement already satisfied|already installed)",
            blob_l,
        ))
        if informational:
            return None
        # Neutralize zero-count failure phrases before marker scanning:
        # "556/556 passed, 0 failed" is a clean pass, not mixed evidence.
        blob_norm = re.sub(
            r"\b0\s+(failed|failure|failures|error|errors)\b",
            " zero_failures ",
            blob_l,
        )
        has_fail = bool(
            re.search(
                r"\b(error|failed|failure|traceback|permission denied|unable)\b",
                blob_norm,
            )
            or re.search(r"exit=[1-9]", blob_norm)
        )
        has_succ = bool(re.search(
            r"(exit=0|found|wrote|created|passed|succeeded|completed|opened|listed)",
            blob_norm,
        ) or " zero_failures " in blob_norm)
        if has_fail and not has_succ:
            return 0.85
        if has_succ and not has_fail:
            return 0.15
        return None  # mixed or silent evidence — uncertain

    def verify_prediction_via_evidence(self, pred_id: str) -> Optional[float]:
        """Verify a prediction using evidence from recorded action triples.

        If the prediction's topic terms appear in the outcomes of at
        least one recorded action, the prediction is marked verified
        with error 0.15 (fulfilled) or 0.85 (contradicted by failure
        markers).  Statistics, calibration, and error history are
        updated exactly as in :meth:`verify_prediction`.

        Returns the error score, or None when no evidence was found or
        the evidence was ambiguous (the prediction is left unverified
        for the caller to handle, e.g. via timeout-based auto-verification).
        """
        for pred in self.data.get("predictions", []):
            if pred.get("id") != pred_id or pred.get("verified"):
                continue
            evidence = self._find_prediction_evidence(pred.get("text", ""))
            if evidence is None:
                return None
            blob_l, matched = evidence
            error = self._score_evidence_blob(blob_l)
            if error is None:
                return None
            pred["verified"] = True
            pred["actual"] = (
                "fulfilled — action evidence found"
                if error <= 0.3
                else "contradicted — action evidence shows failure"
            )
            pred["verification_note"] = (
                f"Auto-verified via action-triple evidence "
                f"(matched: {', '.join(matched)})"
            )
            pred["verified_at"] = now_iso()
            pred["error"] = error
            pred["outcome_class"] = "correct" if error <= 0.3 else "incorrect"

            acc = self.data.setdefault("prediction_accuracy", {})
            acc["verified_predictions"] = acc.get("verified_predictions", 0) + 1
            if error <= 0.3:
                acc["correct_predictions"] = acc.get("correct_predictions", 0) + 1
            else:
                acc["incorrect_predictions"] = acc.get("incorrect_predictions", 0) + 1

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
        if tf in ("completed", "this cycle", "this turn", "same cycle", "now", "immediate", "immediately"):
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

        Two-stage verification for every expired, unverified prediction:

        1. **Evidence stage** — ``verify_prediction_via_evidence`` searches
           the world model's own action triples for the prediction's topic
           terms.  If the predicted event demonstrably happened (or was
           contradicted by failure markers), the prediction is scored
           0.15 / 0.85 and fed into calibration — a real observation,
           exactly like a manual verification.
        2. **Fallback stage** — if no evidence exists, the prediction is
           auto-verified with outcome ``\"timeframe expired — no
           confirmation\"`` and error 0.5 (uncertain), which does NOT
           pollute the calibration curve.

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

            # Allow a grace period: 10% of the timeframe or 10 minutes, whichever is larger.
            # For sub-day timeframes like "this cycle" or "immediate", the 10-minute floor
            # ensures at least one daemon cycle worth of tolerance (the daemon runs every
            # 600s ≈ 10 min) while keeping cycle-based predictions responsive. The old 1-hour
            # floor was too long and left "immediate" and "1 cycle" predictions unverified
            # for a full hour, accumulating unverified predictions that masked real trends.
            grace = max(days * 0.1, 1.0 / 144.0)  # at least ~10 min
            if elapsed_days >= days + grace:
                # Stage 1: try to find evidence in recorded action triples.
                evidence_error = self.verify_prediction_via_evidence(pred.get("id", ""))
                if evidence_error is not None:
                    verified_count += 1
                    continue
                # Stage 2: no evidence — auto-verify as uncertain
                pred["verified"] = True
                pred["actual"] = "timeframe expired — no confirmation"
                pred["error"] = 0.5
                pred["verification_note"] = (
                    f"Auto-verified: timeframe '{tf}' ({days} days) "
                    f"expired {elapsed_days - days:.1f} days ago"
                )
                pred["verified_at"] = now_iso()
                pred["outcome_class"] = "uncertain"  # 0.5 = neither correct nor incorrect

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

                # NOTE: deliberately do NOT call _update_calibration() here.
                # Auto-verified predictions always get error=0.5 (uncertain),
                # which would pollute the calibration buckets with a systematic
                # bias — every entry in a given confidence bucket would show
                # error=0.5 regardless of whether the system was actually correct.
                # Calibration data should only reflect predictions where the
                # actual outcome was observed by the system (via verify_prediction).
                self._add_to_error_history(0.5)
                verified_count += 1

        return verified_count

    def verify_pending_predictions(self) -> int:
        """Evidence-verify non-expired predictions that already have decisive evidence.

        ``verify_expired_predictions`` only resolves predictions once their
        timeframe has elapsed, and predictions reaching expiry without
        evidence degrade to error 0.5 (uncertain) — which deliberately does
        NOT feed calibration. But a prediction is often resolvable long
        before expiry: the predicted event may already have happened (or
        been contradicted) in recorded action triples. This method runs the
        same evidence stage over predictions still within their timeframe,
        so fulfilled/contradicted predictions become real calibration
        observations immediately instead of lingering as "pending" until
        they expire.

        Expired predictions and predictions without decisive evidence are
        left untouched — the former for ``verify_expired_predictions``, the
        latter for a later cycle when more evidence has accumulated.

        Returns:
            Number of predictions resolved via evidence.
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
                continue  # Can't determine timeframe — leave it

            ts_str = pred.get("timestamp")
            if not ts_str:
                continue
            try:
                pred_time = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                continue

            if pred_time > now:
                continue  # Clock skew — leave for a later cycle

            elapsed_days = (now - pred_time).total_seconds() / 86400.0
            grace = max(days * 0.1, 1.0 / 144.0)  # same grace as expired path
            if elapsed_days >= days + grace:
                continue  # Expired — verify_expired_predictions owns this

            # Within timeframe: resolve NOW if decisive evidence exists.
            evidence_error = self.verify_prediction_via_evidence(pred.get("id", ""))
            if evidence_error is not None:
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
                source = t.get("expected_source", "?")
                parts.append(f"    {icon} {desc} [err={err:.2f}, src={source}]")

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

    def _reconcile_prediction_stats(self) -> None:
        """Recompute prediction counters from the retained predictions list.

        The verification paths increment ``verified_predictions`` /
        ``correct_predictions`` / ``incorrect_predictions`` and the
        running average incrementally, but the predictions list is
        capped at 100 entries (see :meth:`record_prediction` and
        :meth:`_merge_concurrent_records`).  Once records age out of
        the cap, the incremental counters silently drift from the list
        — e.g. a verified prediction that falls off the list still
        counts toward ``verified_predictions`` forever, breaking the
        ``verified == correct + incorrect + uncertain`` invariant.

        Recomputing from the actual list restores the invariant.
        Classification prefers the explicit ``outcome_class`` stamp
        written by the verification paths; records written before the
        stamp existed are classified by the same rules the verification
        paths used (error <= 0.3 → correct; the expiry-fallback path
        stamps error=0.5 with an "Auto-verified: timeframe ..." note →
        uncertain; anything else verified → incorrect).

        Called after every cap-trim (record, concurrent merge) and on
        save so on-disk stats always match the retained list.
        """
        acc = self.data.setdefault("prediction_accuracy", {})
        preds = self.data.get("predictions", [])
        verified = correct = incorrect = uncertain = 0
        error_sum = 0.0
        for p in preds:
            if not p.get("verified") or p.get("error") is None:
                continue
            verified += 1
            error_sum += p["error"]
            cls = p.get("outcome_class")
            if cls == "uncertain":
                uncertain += 1
            elif cls == "correct":
                correct += 1
            elif cls == "incorrect":
                incorrect += 1
            elif p["error"] <= 0.3:
                correct += 1
            elif str(p.get("verification_note", "")).startswith(
                "Auto-verified: timeframe"
            ):
                uncertain += 1
            else:
                incorrect += 1
        acc["total_predictions"] = len(preds)
        acc["verified_predictions"] = verified
        acc["correct_predictions"] = correct
        acc["incorrect_predictions"] = incorrect
        acc["avg_prediction_error"] = (
            round(error_sum / verified, 4) if verified else 0.0
        )

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

    def _calibrate_from_action(
        self, confidence: float, error: float
    ) -> None:
        """Record a calibration entry from an action triple.

        Like _update_calibration, but also tracks the total number of
        action-triple calibration entries so the stale-bucket cleaner
        can distinguish legitimate data from auto-verified pollution.
        """
        acc = self.data.setdefault("prediction_accuracy", {})
        acc["action_triple_calibrations"] = (
            acc.get("action_triple_calibrations", 0) + 1
        )
        self._update_calibration(confidence, error)

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

    def _update_discrepancy_patterns(
        self, min_samples: int = 2, min_recurrence: int = 3
    ) -> None:
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

        Recurrence exception: when the *same* action description fails with
        high error at least *min_recurrence* times (default 3), the pattern is
        emitted even if the type-level ratio-decay check would suppress it.
        The ratio check exists to hide scattered old outliers, but it must not
        hide an active systematic failure (e.g. 5 identical state-check
        commands erroring at 0.6 while the daemon keeps re-running the same
        broken command). The recency-decay check still applies, so a resolved
        recurring failure decays once the specific action succeeds again.

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

            # ── Recurrence detection: identical descriptions failing repeatedly ──
            # The same action description appearing >= min_recurrence times among
            # high-error triples is a systematic bias (the daemon keeps re-running
            # a broken command), not scattered outlier noise. Track it so the
            # ratio-decay check below does not hide an ACTIVE recurring failure.
            desc_counts: Dict[str, int] = {}
            for t in triples:
                desc = (t.get("action_description", "") or "").strip()
                if desc:
                    desc_counts[desc] = desc_counts.get(desc, 0) + 1
            recurring_descs = [
                d for d, c in desc_counts.items() if c >= min_recurrence
            ]
            is_recurring = bool(recurring_descs)

            # ── Decay check: suppress pattern if type's overall accuracy has improved ──
            # A pattern where 3/14 actions failed (old, resolved failures) should not
            # persist when the type's overall avg_error is low.  Without this check,
            # old high-error actions (e.g. exploratory shell commands from early cycles)
            # create permanent discrepancy patterns that keep triggering action guidance
            # warnings and reinforcing stale reasoning loops.
            # Exception: recurring identical failures bypass this check — a low ratio
            # of high-error actions with a repeated description is a live systematic
            # failure, not historical noise.
            total_of_type = sum(1 for t in completed if t.get("action_type") == atype)
            high_error_ratio = len(triples) / max(total_of_type, 1)
            high_error_avg = sum(t["prediction_error"] for t in triples) / len(triples)

            if (
                not is_recurring
                and high_error_ratio < 0.5
                and high_error_avg < 0.7
            ):
                # Fewer than half of the actions of this type are high-error,
                # AND the high-error group's avg is < 0.7 → likely old outliers
                continue

            # ── Recency-based decay: check if all high-error actions are old ──
            # Sort the type's actions by timestamp. If the 5 most recent actions
            # of this type ALL have prediction_error <= 0.3 (i.e., they succeeded
            # as expected), then the high-error pattern is likely historical noise.
            # Without this, a few early failures keep triggering action guidance
            # warnings forever, even after dozens of successful subsequent actions.
            all_of_type_sorted = sorted(
                [t for t in completed if t.get("action_type") == atype],
                key=lambda t: t.get("timestamp", ""),
            )
            recent_of_type = all_of_type_sorted[-5:]
            if len(recent_of_type) >= 3:
                recent_all_good = all(
                    t.get("prediction_error", 1.0) <= 0.3
                    for t in recent_of_type
                )
                if recent_all_good:
                    continue
            avg_err = sum(t["prediction_error"] for t in triples) / len(triples)
            timestamps = [t.get("timestamp", "") for t in triples if t.get("timestamp")]
            last_obs = max(timestamps) if timestamps else ""

            # Extract common keywords from action descriptions (words that
            # appear in >30% of the distinct high-error descriptions of this
            # type, with a floor of 2 distinct descriptions).
            #
            # False-positive guard (observed 2026-08-01): a recurring
            # "State check: list current goals and their statuses" command
            # run 4x made every generic word in it ("list", "state",
            # "current", "goals") a shell theme, which then flagged an
            # unrelated safe action whose description merely contained
            # "list". Two structural fixes:
            #   1. Keyword counting is description-deduped — running the SAME
            #      command N times is one failure mode, not N votes for every
            #      word in it. (The pattern's ``count`` still reports all N;
            #      the repeated command itself is surfaced precisely via
            #      ``recurring_description``.)
            #   2. A keyword must appear in >= 2 DISTINCT failing
            #      descriptions — words unique to one description are that
            #      command's identity, not a common theme, and words shared
            #      across the whole type (generic verbs like "list") do not
            #      discriminate failures from successes.
            #   3. Tokens are punctuation-normalized ("check:" -> "check") so
            #      word-boundary matching in format_action_guidance works.
            seen_descs: set = set()
            all_words: List[str] = []
            for t in triples:
                desc = (t.get("action_description", "") or "").lower()
                if desc in seen_descs:
                    continue
                seen_descs.add(desc)
                all_words.extend(
                    w.strip(_THEME_PUNCT) or w
                    for w in desc.split()
                    if len(w) > 3
                    and w not in ("with", "from", "that", "this", "into")
                )

            word_counts: Dict[str, int] = {}
            for w in all_words:
                word_counts[w] = word_counts.get(w, 0) + 1
            threshold = max(2, int(len(seen_descs) * 0.3))
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
                "recurring_description": recurring_descs[0] if recurring_descs else None,
                "description": (
                    f"{len(triples)}/{sum(1 for t in completed if t.get('action_type') == atype)} "
                    f"{atype} actions with high prediction error "
                    f"(avg {avg_err:.2f})"
                    + (
                        f"; recurring: {recurring_descs[0][:80]!r}"
                        if recurring_descs else ""
                    )
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
                # 2a. Recurring-command check — the strongest, most precise
                # signal. A command that failed identically >= min_recurrence
                # times is flagged when the same phrase is about to run again.
                # Keyword themes only approximate this; matching the full
                # phrase keeps it exact, so a generic word in the recurring
                # command (e.g. "list") never flags an unrelated action.
                rec = pat.get("recurring_description")
                if rec and re.search(
                    rf"\b{re.escape(rec.strip().lower())}\b", desc_lower
                ):
                    parts.append(
                        f"  ⚠ Recurring failing command: {rec!r} "
                        f"({pat['count']} failures, avg err {pat['avg_error']:.2f})"
                    )
                    continue
                themes = pat.get("common_themes", [])
                # Word-boundary match, not substring: theme "list" must not
                # flag an unrelated "List saved /tmp evidence files" action,
                # and "check" must not match "checkout". Themes are
                # punctuation-normalized at extraction so \b is reliable.
                matched_keywords = [
                    kw for kw in themes
                    if re.search(rf"\b{re.escape(kw)}\b", desc_lower)
                ]
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

    # ── Proactive outcome prediction (completes the predict→act→observe→learn loop) ──

    def predict_action_outcome(
        self,
        action_type: str,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Predict the outcome of a proposed action using historical data.

        Generates a data-driven prediction based on per-type accuracy statistics,
        similar past actions (keyword-matched on description + parameters),
        and recent trends.  This is the world model's own predictive capability,
        independent of the LLM's ``expected_outcome`` field.

        Args:
            action_type: The action type (``"shell"``, ``"write_file"``, etc.).
            description: Action description for similarity matching.
            parameters: Optional dict of the actual action payload (command, path, etc.).
                When provided, tokens from these parameters are combined with
                description tokens for more precise similarity matching.

        Returns:
            A dict with:
              - ``predicted_outcome``: Best-guess outcome string (or None if no data).
              - ``confidence``: Calibrated confidence (0.0–1.0).
              - ``success_probability``: Estimated probability of success (0.0–1.0).
              - ``avg_error``: Historical average prediction error for this type.
              - ``sample_count``: Number of historical samples used.
              - ``similar_actions``: Up to 3 descriptions of similar past actions.
              - ``risk_level``: ``"low"``, ``"medium"``, or ``"high"``.
              - ``parameters_match``: Whether action parameters were used in matching.
        """
        result: Dict[str, Any] = {
            "predicted_outcome": None,
            "confidence": 0.0,
            "success_probability": 0.5,
            "avg_error": None,
            "sample_count": 0,
            "similar_actions": [],
            "risk_level": "unknown",
            "parameters_match": False,
        }

        per_type = self.get_per_type_accuracy()
        if action_type not in per_type:
            # No data for this type — can't make a statistical prediction
            return result

        stats = per_type[action_type]
        count = stats["count"]
        avg_err = stats["avg_error"]
        result["avg_error"] = avg_err
        result["sample_count"] = count

        # ── Success probability: inverse of avg_error, adjusted for sample count ──
        # With few samples, regress toward neutral (0.5)
        raw_success = 1.0 - avg_err
        if count < 3:
            # Blend with neutral for low sample counts
            blend = count / 3.0
            raw_success = raw_success * blend + 0.5 * (1.0 - blend)
        success_prob = max(0.0, min(1.0, round(raw_success, 4)))
        result["success_probability"] = success_prob

        # ── Confidence in the prediction itself ──
        # High confidence when we have many samples and consistent error
        # Low confidence when samples are few or error is near 0.5
        confidence_base = 1.0 - abs(avg_err - 0.5) * 2.0  # 0.0 at error=0.5, 1.0 at error=0.0 or 1.0
        sample_factor = min(count / 10.0, 1.0)  # More samples = more confidence
        confidence = round(max(0.1, min(1.0, confidence_base * 0.6 + sample_factor * 0.4)), 4)
        result["confidence"] = confidence

        # ── Risk level ──
        if success_prob >= 0.7:
            result["risk_level"] = "low"
        elif success_prob >= 0.4:
            result["risk_level"] = "medium"
        else:
            result["risk_level"] = "high"

        # ── Similar past actions (keyword-matched on description + parameters) ──
        # Extract tokens from both description and parameters for richer matching
        query_tokens: set = set()
        if description:
            desc_lower = description.lower()
            query_tokens = set(re.findall(r"[a-z0-9]+", desc_lower))
        if parameters:
            # Extract tokens from parameter values (commands, paths, messages, etc.)
            for pkey, pval in parameters.items():
                if isinstance(pval, str) and pval.strip():
                    pval_lower = pval.lower()
                    param_tokens = set(re.findall(r"[a-z0-9_/.@-]+", pval_lower))
                    query_tokens |= param_tokens

        scored: List[tuple[float, str]] = []
        if query_tokens:
            for t in self.data.get("action_triples", []):
                if t.get("action_type") != action_type or not t.get("completed"):
                    continue
                # Build past tokens from description + stored parameters
                past_desc = (t.get("action_description", "") or "").lower()
                past_tokens = set(re.findall(r"[a-z0-9]+", past_desc))
                # Also extract from stored parameters if available
                past_params = t.get("action_parameters")
                if past_params and isinstance(past_params, dict):
                    for pval in past_params.values():
                        if isinstance(pval, str) and pval.strip():
                            pval_l = pval.lower()
                            past_tokens |= set(re.findall(r"[a-z0-9_/.@-]+", pval_l))
                if past_tokens:
                    overlap = len(query_tokens & past_tokens)
                    union = len(query_tokens | past_tokens)
                    similarity = overlap / max(union, 1)
                    if similarity > 0.2:
                        err = t.get("prediction_error", 0.5)
                        actual = (t.get("actual_outcome", "") or "")[:80]
                        # Keep the past command (if any) so outcome selection
                        # can detect when the current command differs from the
                        # matched past action (_command_changed_since).
                        past_cmd = (
                            past_params.get("command", "")
                            if past_params and isinstance(past_params, dict)
                            else ""
                        )
                        scored.append((similarity, past_desc[:60], err, actual, past_cmd))
                        # Mark that we matched based on parameters if parameters were provided
                        if parameters and not result["parameters_match"]:
                            result["parameters_match"] = True

            # Sort by similarity, take top 3
            scored.sort(key=lambda x: x[0], reverse=True)
            for sim, desc_text, err_val, actual_text, past_cmd in scored[:3]:
                result["similar_actions"].append({
                    "description": desc_text,
                    "similarity": round(sim, 3),
                    "error": err_val,
                    "outcome": actual_text,
                    "command": past_cmd,
                })

        # ── Generate predicted outcome string ──
        if result["sample_count"] >= 1:
            # TWO output fields:
            #
            #   predicted_outcome — the ACTUAL predicted output text the system
            #       expects to see after executing this action.  Used as the
            #       expected_outcome in action triples for error calculation.
            #       It should look like real command output so that
            #       _compute_prediction_error produces MEANINGFUL errors
            #       (text similarity vs actual output).
            #
            #   prediction_rationale — human-readable explanation of the
            #       prediction, including success probability and most-similar
            #       past action.  Used for LLM context and debugging.
            #
            # Previously these were conflated into one field, producing a
            # meta-description that shared no bigrams with real output,
            # guaranteeing high prediction error (~0.5) even when the action
            # succeeded exactly as similar past ones had.

            # Build predicted outcome from most similar past action's actual output
            if result["similar_actions"]:
                best = result["similar_actions"][0]
                # Use the most similar past action's outcome, truncated
                outcome_text = (best.get("outcome", "") or "").strip()
                # A past action whose own prediction was a high-error surprise
                # (error >= 0.5) is an anomaly, not a template — and if its
                # command differs from the current one (e.g. a quoting fix),
                # reusing its failure as the prediction manufactures a
                # systematic error every cycle. Fall back to the generic
                # per-type prediction instead.
                stale_template = (
                    outcome_text
                    and best.get("error", 0.0) >= 0.5
                    and _command_changed_since(parameters, best)
                )
                if outcome_text and not stale_template:
                    result["predicted_outcome"] = outcome_text[:100]
            if not result.get("predicted_outcome"):
                # Generic prediction based on action type (no similar actions,
                # empty outcome, or the best match is a stale failure of a
                # different command — see _command_changed_since).
                if action_type == "shell":
                    result["predicted_outcome"] = "exit=0: command output"
                elif action_type == "write_file":
                    result["predicted_outcome"] = "Wrote file successfully"
                elif action_type == "git_commit":
                    result["predicted_outcome"] = "exit=0: committed"
                elif action_type == "install_package":
                    result["predicted_outcome"] = "exit=0: installed"
                else:
                    result["predicted_outcome"] = f"{action_type}: success"

            # Build meta-description rationale for context / debugging
            if success_prob >= 0.7:
                label = "likely to succeed"
            elif success_prob >= 0.4:
                label = "uncertain outcome"
            else:
                label = "likely to fail"

            rationale_parts = [
                f"[data-driven] {action_type} action: {label} ",
                f"(success probability {success_prob:.0%}, "
                f"n={count}, avg_err={avg_err:.2f})",
            ]
            if result["similar_actions"]:
                best = result["similar_actions"][0]
                rationale_parts.append(
                    f" | Most similar: \"{best['description'][:50]}\" "
                    f"→ \"{best['outcome'][:50]}\" [err={best['error']:.2f}]"
                )
                if (
                    best.get("error", 0.0) >= 0.5
                    and _command_changed_since(parameters, best)
                ):
                    rationale_parts.append(
                        " | STALE template (command changed) — generic fallback"
                    )
            result["prediction_rationale"] = "".join(rationale_parts)

        return result

    # ── Persistence ───────────────────────────────────────────────

    @staticmethod
    def storage_path() -> Path:
        return get_evolve_dir() / "world_model.json"

    def save(self, path: Optional[Path] = None) -> Path:
        """Persist to disk as JSON (atomic write).

        Conflict-aware: when saving to the default storage path, any action
        triples or predictions present on disk but missing from this instance
        (e.g. recorded concurrently by the daemon while a session held a stale
        copy) are merged in before writing — a stale in-memory model can never
        destroy newer recorded data. Explicit paths (tests, exports) are
        written verbatim.
        """
        target = path or self.storage_path()
        if path is None:
            self._merge_concurrent_records()
        # Reconcile before persisting so on-disk counters always match the
        # retained predictions list (guards against stale counters from
        # pre-fix data files or cap trims in other writers).
        self._reconcile_prediction_stats()
        safe_write_json(target, self.data)
        return target

    def _merge_concurrent_records(self) -> None:
        """Merge newer on-disk records into this model before saving.

        Guards against the observed failure mode (2026-07-31): a session
        holding a stale copy of the world model saved over data the daemon
        had recorded meanwhile, destroying 17 action triples and 6 macro
        predictions. Union-merge by ID keeps every writer's records.
        """
        on_disk = safe_read_json(self.storage_path())
        if not isinstance(on_disk, dict):
            return

        # Merge action triples by id (dedupe, keep newest overall order)
        merged = list(self.data.get("action_triples", []))
        known = {t.get("id") for t in merged if t.get("id")}
        added = 0
        for t in on_disk.get("action_triples", []):
            tid = t.get("id")
            if tid and tid not in known:
                merged.append(t)
                known.add(tid)
                added += 1
        if added:
            merged.sort(key=lambda t: t.get("timestamp", ""))
            self.data["action_triples"] = merged[-200:]
            # Recompute derived stats so calibration reflects the union
            self._update_accuracy_stats()
            self._update_per_type_accuracy()

        # Merge macro predictions by id
        merged_preds = list(self.data.get("predictions", []))
        known_p = {p.get("id") for p in merged_preds if p.get("id")}
        p_added = 0
        for p in on_disk.get("predictions", []):
            pid = p.get("id")
            if pid and pid not in known_p:
                merged_preds.append(p)
                known_p.add(pid)
                p_added += 1
        if p_added:
            merged_preds.sort(key=lambda p: p.get("timestamp", ""))
            self.data["predictions"] = merged_preds[-100:]
            # The cap may have trimmed records; keep counters consistent
            self._reconcile_prediction_stats()

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
        # Backfill error_history from completed triples if it's shorter
        # than the number of completed triples.  This handles the case
        # after schema upgrades or data migrations where error_history
        # was reset but completed triples still exist — without this the
        # trend computation (_compute_error_trend) stays blind until N
        # new actions are completed.
        result._backfill_error_history()
        # Clean stale calibration bucket data that may have accumulated
        # from expired auto-verifications in older code versions.
        # The current code explicitly avoids calling _update_calibration
        # on expired auto-verifications (see verify_expired_predictions),
        # but legacy data persists on disk.  If the combined count across
        # all buckets exceeds the number of actually-verified predictions
        # (correct + incorrect), some entries are stale.
        result._clean_stale_calibration_buckets()
        return result

    def _backfill_error_history(self) -> None:
        """Populate error_history from completed triples if missing/historic.

        Only backfills when error_history is meaningfully shorter than
        the number of completed triples (gap >= 3).  This prevents tiny
        backfills for off-by-one issues while still fixing the cold-start
        problem after schema upgrades.
        """
        acc = self.data.setdefault("prediction_accuracy", {})
        history = acc.get("error_history", [])
        completed = [
            t for t in self.data.get("action_triples", [])
            if t.get("prediction_error") is not None
        ]
        if len(history) >= len(completed) or len(completed) - len(history) < 3:
            return  # Nothing to backfill, or gap is too small to matter
        # Rebuild error_history from the last 20 completed triple errors
        new_history = [t["prediction_error"] for t in completed[-20:]]
        acc["error_history"] = new_history
        logger.info(
            "Backfilled error_history from %d → %d entries (%d completed triples)",
            len(history), len(new_history), len(completed),
        )

    def _clean_stale_calibration_buckets(self) -> None:
        """Remove stale calibration bucket entries from expired auto-verifications.

        The current code intentionally does NOT call ``_update_calibration``
        from ``verify_expired_predictions`` (auto-verification), because
        expired predictions always get error=0.5 (uncertain) and would
        pollute the calibration curve with a systematic bias.

        However, older code versions DID call ``_update_calibration`` on
        expired predictions, and legacy data persists on disk.  This method
        detects stale entries by comparing the total count across all buckets
        against the number of legitimate calibration entries (``correct_predictions``
        + ``incorrect_predictions`` + ``action_triple_calibrations``).
        If the bucket count exceeds that sum, the excess entries are stale
        and the buckets are reset.

        Also handles the degenerate case where ALL entries in a non-empty
        bucket have avg_error == 0.5 but no prediction was ever explicitly
        verified — this is the signature of legacy auto-verification data.
        """
        acc = self.data.setdefault("prediction_accuracy", {})
        buckets = acc.get("calibration_buckets", [])

        if not buckets:
            return

        actually_verified = (
            acc.get("correct_predictions", 0)
            + acc.get("incorrect_predictions", 0)
            + acc.get("action_triple_calibrations", 0)
        )
        bucket_total = sum(b.get("count", 0) for b in buckets)

        # Case 1: Bucket count exceeds verified count — excess is stale
        if bucket_total > actually_verified:
            logger.info(
                "Cleaning %d stale calibration bucket entries "
                "(bucket_count=%d > actually_verified=%d)",
                bucket_total - actually_verified,
                bucket_total, actually_verified,
            )
            # Reset to defaults: only keep buckets up to the actual
            # verified count, distributed proportionally?  Actually, since
            # we can't recover which pre-existing entries were legitimate,
            # just reset the entire bucket structure.
            acc["calibration_buckets"] = []
            return

        # Case 2: All buckets have avg_error == 0.5 but nothing ever
        # explicitly verified — legacy expired-only data
        if actually_verified == 0 and bucket_total > 0:
            all_half = all(
                b.get("avg_error", 0) == 0.5 or b.get("count", 0) == 0
                for b in buckets
            )
            if all_half:
                logger.info(
                    "Cleaning %d stale calibration bucket entries "
                    "(all avg_error=0.5 from expired auto-verifications)",
                    bucket_total,
                )
                acc["calibration_buckets"] = []
                return

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
