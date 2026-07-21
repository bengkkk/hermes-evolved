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
    },
    "present": {
        "active_project": None,
        "active_tasks": [],
        "waiting_for": [],
        "last_session_focus": None,
    },
    "future": {
        "goals": [],
        "scheduled_actions": [],
        "contingencies": [],
    },
}


def load_timeline() -> Dict[str, Any]:
    """Load the timeline (past → present → future)."""
    ensure_evolve_dir()
    if not TIMELINE_FILE.exists():
        return dict(_DEFAULT_TIMELINE)
    try:
        data = json.loads(TIMELINE_FILE.read_text(encoding="utf-8"))
        # Merge with defaults for forward compatibility
        merged = dict(_DEFAULT_TIMELINE)
        merged.update(data)
        for section in ("past", "present", "future"):
            if section in data:
                merged[section].update(data[section])
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
    """Format timeline section for the system prompt."""
    timeline = load_timeline()
    parts = ["## Timeline"]

    # Past — recent session (most recent first)
    completed = timeline.get("past", {}).get("completed_sessions", [])
    recent = completed[-3:] if completed else []
    if recent:
        parts.append("Completed sessions:")
        for s in reversed(recent):
            focus = s.get("focus", "—")
            outcomes = s.get("outcomes", [])
            outcome_str = "; ".join(outcomes[:2]) if outcomes else ""
            parts.append(f"  • {focus}{' — ' + outcome_str if outcome_str else ''}")

    # Past — recent events
    events = timeline.get("past", {}).get("events", [])
    recent_events = events[-3:] if events else []
    if recent_events:
        parts.append("Recent events:")
        for e in recent_events:
            parts.append(f"  • {e.get('summary', '—')}")

    # Present
    present = timeline.get("present", {})
    active = present.get("active_project")
    if active:
        parts.append(f"Active project: {active}")
    tasks = present.get("active_tasks", [])
    if tasks:
        parts.append(f"Active tasks: {'; '.join(tasks[:3])}")
    waiting = present.get("waiting_for", [])
    if waiting:
        parts.append(f"Waiting for: {'; '.join(waiting[:2])}")

    # Future goals
    future = timeline.get("future", {})
    goals = future.get("goals", [])
    if goals:
        parts.append("Goals:")
        for g in goals[:3]:
            parts.append(f"  → {g}")
    scheduled = future.get("scheduled_actions", [])
    if scheduled:
        parts.append(f"Scheduled: {'; '.join(scheduled[:2])}")

    return "\n".join(parts)


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
