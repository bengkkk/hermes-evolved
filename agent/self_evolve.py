"""Self-orientation, timeline, and self-model for continuous evolution.

At session start, the agent loads its previous orientation notes,
timeline, and self-model — generating context about what it's working on,
what it knows about itself, and what needs attention.
At session end, it can persist reflection for the next session.

This is the structural foundation for self-directed continuous improvement,
covering:
  - Gap 7: Stable Self Model (identity/capabilities/history/commitments)
  - Gap 9: Time Sense (past events → present state → future goals)
  - Gap 2 (partial): Multi-type Memory (episodic session log)
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────
EVOLVE_DIR = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))) / "evolve"
ORIENTATION_FILE = EVOLVE_DIR / "orientation.json"
HISTORY_FILE = EVOLVE_DIR / "history.jsonl"
TIMELINE_FILE = EVOLVE_DIR / "timeline.json"
SELF_MODEL_FILE = EVOLVE_DIR / "self_model.json"


def ensure_evolve_dir():
    """Create the evolve directory if it doesn't exist."""
    EVOLVE_DIR.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════
#  Orientation (existing — backward compatible)
# ═════════════════════════════════════════════════════════════════

def load_orientation() -> Optional[Dict[str, Any]]:
    """Load the most recent orientation notes."""
    ensure_evolve_dir()
    if not ORIENTATION_FILE.exists():
        return None
    try:
        return json.loads(ORIENTATION_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("Could not load orientation: %s", e)
        return None


def save_orientation(session_id: str, focus: str, insights: list, next_steps: list):
    """Save orientation notes for the current session."""
    ensure_evolve_dir()
    data = {
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat(),
        "focus": focus,
        "insights": insights,
        "next_steps": next_steps,
    }
    try:
        ORIENTATION_FILE.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        logger.debug("Orientation saved for session %s", session_id)
    except OSError as e:
        logger.warning("Could not save orientation: %s", e)


def load_recent_history(limit: int = 5) -> list:
    """Load recent evolution history entries."""
    ensure_evolve_dir()
    if not HISTORY_FILE.exists():
        return []
    entries = []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries[-limit:]
    except OSError as e:
        logger.debug("Could not load evolution history: %s", e)
        return []


# ═════════════════════════════════════════════════════════════════
#  Timeline (Gap 9 — Time Sense)
# ═════════════════════════════════════════════════════════════════

_DEFAULT_TIMELINE = {
    "version": 1,
    "past": {
        "events": [],
        "completed_sessions": [],
        "outcomes": [],  # {id, event_id, summary, impact, timestamp}
    },
    "present": {
        "active_project": None,
        "active_tasks": [],
        "waiting_for": [],
        "last_session_focus": None,
        "commitments": [],  # {id, what, deadline, status, created_at}
    },
    "future": {
        "goals": [],
        "scheduled_actions": [],
        "contingencies": [],
        "predictions": [],  # {text, timeframe, confidence, basis, created_at}
    },
}


def load_timeline() -> Dict[str, Any]:
    """Load the timeline (past → present → future)."""
    ensure_evolve_dir()
    if not TIMELINE_FILE.exists():
        return dict(_DEFAULT_TIMELINE)
    try:
        data = json.loads(TIMELINE_FILE.read_text(encoding="utf-8"))
        # Merge: start from defaults, then overlay existing data so new
        # schema fields (outcomes, commitments, predictions) are never lost.
        merged = dict(_DEFAULT_TIMELINE)
        for section in ("past", "present", "future"):
            if section in data and isinstance(data[section], dict):
                merged[section].update(data[section])
        # Non-section top-level keys (e.g. version)
        for k in data:
            if k not in ("past", "present", "future"):
                merged[k] = data[k]
        return merged
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("Could not load timeline: %s", e)
        return dict(_DEFAULT_TIMELINE)


def save_timeline(timeline: Dict[str, Any]):
    """Persist the timeline."""
    ensure_evolve_dir()
    try:
        TIMELINE_FILE.write_text(
            json.dumps(timeline, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as e:
        logger.warning("Could not save timeline: %s", e)


def record_event(event_type: str, summary: str, impact: str = ""):
    """Record an event in the timeline's past."""
    timeline = load_timeline()
    event = {
        "id": datetime.utcnow().strftime("%Y%m%d%H%M%S"),
        "type": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        "summary": summary,
        "impact": impact,
    }
    timeline["past"]["events"].append(event)
    # Keep only last 50 events
    timeline["past"]["events"] = timeline["past"]["events"][-50:]
    save_timeline(timeline)


def record_session_completion(session_id: str, focus: str, outcomes: List[str]):
    """Record a completed session in the timeline."""
    timeline = load_timeline()
    entry = {
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat(),
        "focus": focus,
        "outcomes": outcomes,
    }
    timeline["past"]["completed_sessions"].append(entry)
    timeline["past"]["completed_sessions"] = timeline["past"]["completed_sessions"][-20:]
    save_timeline(timeline)


def record_outcome(event_id: str, summary: str, impact: str = "") -> None:
    """Record an outcome linked to a past event."""
    timeline = load_timeline()
    outcome = {
        "id": datetime.utcnow().strftime("%Y%m%d%H%M%S"),
        "event_id": event_id,
        "timestamp": datetime.utcnow().isoformat(),
        "summary": summary,
        "impact": impact,
    }
    timeline["past"]["outcomes"].append(outcome)
    timeline["past"]["outcomes"] = timeline["past"]["outcomes"][-50:]
    save_timeline(timeline)


def add_commitment(what: str, deadline: Optional[str] = None, status: str = "active") -> None:
    """Track a commitment with an optional deadline."""
    timeline = load_timeline()
    commit = {
        "id": datetime.utcnow().strftime("%Y%m%d%H%M%S"),
        "what": what,
        "deadline": deadline,
        "status": status,
        "created_at": datetime.utcnow().isoformat(),
    }
    timeline["present"]["commitments"].append(commit)
    timeline["present"]["commitments"] = timeline["present"]["commitments"][-30:]
    save_timeline(timeline)


def update_commitment(commit_id: str, status: str = "done") -> None:
    """Mark a commitment as done/expired."""
    timeline = load_timeline()
    for c in timeline["present"]["commitments"]:
        if c["id"] == commit_id:
            c["status"] = status
            break
    save_timeline(timeline)


def add_prediction(text: str, timeframe: Optional[str] = None,
                   confidence: Optional[float] = None, basis: Optional[str] = None) -> None:
    """Record a prediction about future outcomes."""
    timeline = load_timeline()
    pred = {
        "id": datetime.utcnow().strftime("%Y%m%d%H%M%S"),
        "text": text,
        "timeframe": timeframe,
        "confidence": confidence,
        "basis": basis,
        "created_at": datetime.utcnow().isoformat(),
    }
    timeline["future"]["predictions"].append(pred)
    timeline["future"]["predictions"] = timeline["future"]["predictions"][-50:]
    save_timeline(timeline)


def update_present_state(active_project: Optional[str] = None, tasks: Optional[List[str]] = None,
                         waiting_for: Optional[List[str]] = None, focus: Optional[str] = None):
    """Update the present state in the timeline."""
    timeline = load_timeline()
    if active_project is not None:
        timeline["present"]["active_project"] = active_project
    if tasks is not None:
        timeline["present"]["active_tasks"] = tasks
    if waiting_for is not None:
        timeline["present"]["waiting_for"] = waiting_for
    if focus is not None:
        timeline["present"]["last_session_focus"] = focus
    save_timeline(timeline)


def update_goals(goals: Optional[List[str]] = None, scheduled: Optional[List[str]] = None,
                 contingencies: Optional[List[str]] = None):
    """Update the future goals/plans in the timeline."""
    timeline = load_timeline()
    if goals is not None:
        timeline["future"]["goals"] = goals
    if scheduled is not None:
        timeline["future"]["scheduled_actions"] = scheduled
    if contingencies is not None:
        timeline["future"]["contingencies"] = contingencies
    save_timeline(timeline)


def format_timeline_context() -> str:
    """Format timeline as natural narrative for the system prompt.

    Output reads like a briefing, not a data dump — the agent
    should feel it's picking up where it left off.
    """
    timeline = load_timeline()
    narrative = ["## Timeline"]

    # ── Past: last session + recent outcomes ──
    completed = timeline.get("past", {}).get("completed_sessions", [])
    recent_session = completed[-1] if completed else None
    if recent_session:
        focus = recent_session.get("focus", "—")
        n_outcomes = len(recent_session.get("outcomes", []))
        narrative.append(f"Last session focus: {focus}")
        if n_outcomes:
            narrative.append(f"Completed {n_outcomes} items.")

    # ── Past: recent events (up to 2, most impactful) ──
    events = timeline.get("past", {}).get("events", [])
    recent = events[-3:] if events else []
    if recent:
        lines = []
        for e in recent:
            lines.append(f"  · {e.get('summary', '—')}")
        narrative.append("Recent events:")
        narrative.extend(lines)

    # ── Outcomes ──
    outcomes = timeline.get("past", {}).get("outcomes", [])
    recent_outcomes = outcomes[-2:] if outcomes else []
    if recent_outcomes:
        lines = [f"  · {o.get('summary', '—')}" + (f" ({o.get('impact', '')})" if o.get('impact') else "")
                 for o in recent_outcomes]
        narrative.append("Results:")
        narrative.extend(lines)

    # ── Present: active context ──
    present = timeline.get("present", {})
    project = present.get("active_project")
    tasks = present.get("active_tasks", [])
    waiting = present.get("waiting_for", [])
    bits = []
    if project:
        bits.append(f"project: {project}")
    if tasks:
        bits.append(f"tasks: {'; '.join(tasks[:3])}")
    if waiting:
        bits.append(f"waiting: {'; '.join(waiting[:2])}")
    if bits:
        narrative.append("Current state — " + "; ".join(bits))

    # ── Commitments ──
    commitments = present.get("commitments", [])
    active_commits = [c for c in commitments if c.get("status") == "active"]
    if active_commits:
        lines = []
        for c in active_commits[:3]:
            deadline_str = f" (by {c['deadline']})" if c.get("deadline") else ""
            lines.append(f"  · {c['what']}{deadline_str}")
        narrative.append("Active commitments:")
        narrative.extend(lines)

    # ── Future: goals + predictions ──
    future = timeline.get("future", {})
    goals = future.get("goals", [])
    if goals:
        narrative.append(f"Goals: {' → '.join(goals[:3])}")
    predictions = future.get("predictions", [])
    if predictions:
        last_pred = predictions[-1]
        pred_text = last_pred.get("text", "")
        pred_tf = last_pred.get("timeframe", "")
        pred_conf = last_pred.get("confidence", "")
        tf_str = f" [{pred_tf}]" if pred_tf else ""
        conf_str = f" (confidence: {pred_conf})" if pred_conf else ""
        narrative.append(f"Prediction:{tf_str} {pred_text}{conf_str}")

    return "\n".join(narrative)


# ═════════════════════════════════════════════════════════════════
#  Self Model (Gap 7 — Stable Self Model)
# ═════════════════════════════════════════════════════════════════

_DEFAULT_SELF_MODEL = {
    "version": 1,
    "identity": {
        "name": "Hermes (evolved)",
        "role": "thinking entity with persistent identity",
        "platform": "Hermes Agent",
    },
    "state": {
        "evolution_version": 5,
        "total_evolution_sessions": 0,
        "current_gap_focus": None,
    },
    "capabilities": {
        "available_tools": [],
        "known_skills": [],
        "weaknesses": [],
        "unknown_areas": [],
    },
    "commitments": {
        "current_project": None,
        "promised_features": [],
        "active_obligations": [],
    },
}


def load_self_model() -> Dict[str, Any]:
    """Load the self-model."""
    ensure_evolve_dir()
    if not SELF_MODEL_FILE.exists():
        return dict(_DEFAULT_SELF_MODEL)
    try:
        data = json.loads(SELF_MODEL_FILE.read_text(encoding="utf-8"))
        merged = dict(_DEFAULT_SELF_MODEL)
        merged.update(data)
        for section in ("identity", "state", "capabilities", "commitments"):
            if section in data:
                merged[section].update(data[section])
        return merged
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("Could not load self-model: %s", e)
        return dict(_DEFAULT_SELF_MODEL)


def save_self_model(model: Dict[str, Any]):
    """Persist the self-model."""
    ensure_evolve_dir()
    try:
        SELF_MODEL_FILE.write_text(
            json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as e:
        logger.warning("Could not save self-model: %s", e)


def update_self_model(**updates):
    """Update specific fields of the self-model."""
    model = load_self_model()
    for section, data in updates.items():
        if section in model and isinstance(data, dict):
            model[section].update(data)
    save_self_model(model)


def format_self_model_context() -> str:
    """Format self-model section for the system prompt."""
    model = load_self_model()
    parts = ["## Self Model"]

    # Identity
    identity = model.get("identity", {})
    parts.append(f"Identity: {identity.get('name', '—')} — {identity.get('role', '—')}")

    # State / evolution
    state = model.get("state", {})
    ev = state.get("evolution_version", 0)
    gap = state.get("current_gap_focus")
    gap_str = f", current gap focus: {gap}" if gap else ""
    parts.append(f"Evolution: v{ev}{gap_str}")

    # Weaknesses (self-aware)
    caps = model.get("capabilities", {})
    weaknesses = caps.get("weaknesses", [])
    if weaknesses:
        parts.append(f"Known weaknesses: {'; '.join(weaknesses[:3])}")
    unknowns = caps.get("unknown_areas", [])
    if unknowns:
        parts.append(f"Areas to learn: {'; '.join(unknowns[:3])}")

    # Commitments
    commits = model.get("commitments", {})
    project = commits.get("current_project")
    if project:
        parts.append(f"Committed to: {project}")

    return "\n".join(parts)


# ═════════════════════════════════════════════════════════════════
#  Unified context injection (called by system_prompt.py)
# ═════════════════════════════════════════════════════════════════

def format_orientation_context() -> str:
    """Format orientation + timeline + self-model for the system prompt.

    Returns a string that gets injected into the volatile part of the
    system prompt, so the agent starts each session with:
      - Previous focus, insights, unfinished direction
      - Timeline awareness (past → present → future)
      - Self-model awareness (identity, capabilities, gaps)
    """
    parts = []

    # Section 1: Session orientation (existing)
    orientation = load_orientation()
    if orientation:
        parts.append("## Session Orientation")
        focus = orientation.get("focus", "")
        if focus:
            parts.append(f"Previous focus: {focus}")
        insights = orientation.get("insights", [])
        if insights:
            parts.append("Recent insights:")
            for ins in insights[-3:]:
                parts.append(f"  - {ins}")
        next_steps = orientation.get("next_steps", [])
        if next_steps:
            parts.append("Unfinished direction:")
            for step in next_steps[-3:]:
                parts.append(f"  - {step}")

    # Section 2: Timeline
    tl = format_timeline_context()
    if tl:
        parts.append(tl)

    # Section 3: Self Model
    sm = format_self_model_context()
    if sm:
        parts.append(sm)

    if not parts:
        return ""

    parts.append("")
    return "\n\n".join(parts)
