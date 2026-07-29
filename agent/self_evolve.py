"""Self-orientation, timeline, and self-model for continuous evolution.

At session start, the agent loads its previous orientation notes,
timeline, and self-model — generating context about what it's working on,
what it knows about itself, and what needs attention.
At session end, it can persist reflection for the next session.

This module is now a thin compatibility layer over ``data_layer.py``,
which is the single authoritative persistence module. All file I/O
is delegated to atomic-write-backed operations in data_layer.

This covers:
  - Gap 7: Stable Self Model (identity/capabilities/history/commitments)
  - Gap 9: Time Sense (past events → present state → future goals)
  - Gap 2 (partial): Multi-type Memory (episodic/semantic/procedural)
  - Gap 4: Self-generated Goals
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

# Delegate all persistence to the consolidated data layer
from data_layer import (
    Memory as _Memory,
    Goals as _Goals,
    Orientation as _Orientation,
    SelfModel as _SelfModel,
    create_plan as _create_plan,
    get_active_plan as _get_active_plan,
    update_plan_step as _update_plan_step,
    complete_plan as _complete_plan,
    format_plan_context as _format_plan_context,
    format_timeline_context as _format_timeline_context,
    get_evolve_dir,
    now_iso,
    now_compact,
    safe_read_json,
    safe_write_json,
)

logger = logging.getLogger(__name__)

# ── Paths (delegated to data_layer for the base directory) ──
EVOLVE_DIR = get_evolve_dir()
ORIENTATION_FILE = EVOLVE_DIR / "orientation.json"
HISTORY_FILE = EVOLVE_DIR / "history.jsonl"
TIMELINE_FILE = EVOLVE_DIR / "timeline.json"
SELF_MODEL_FILE = EVOLVE_DIR / "self_model.json"
MEMORY_FILE = EVOLVE_DIR / "memory.json"
GOALS_FILE = EVOLVE_DIR / "goals.json"


def ensure_evolve_dir():
    """Create the evolve directory if it doesn't exist."""
    get_evolve_dir()


# ═════════════════════════════════════════════════════════════════
#  Orientation (existing — backward compatible)
# ═════════════════════════════════════════════════════════════════

def load_orientation() -> Optional[Dict[str, Any]]:
    """Load the most recent orientation notes."""
    orient = _Orientation.load()
    if orient.focus or orient.data.get("insights") or orient.data.get("next_steps"):
        return orient.data
    return None


def save_orientation(session_id: str, focus: str, insights: list,
                     next_steps: list):
    """Save orientation notes for the current session."""
    orient = _Orientation()
    orient.focus = focus
    for ins in insights:
        orient.add_insight(ins)
    for step in next_steps:
        orient.add_next_step(step)
    orient.data["session_id"] = session_id
    orient.data["timestamp"] = now_iso()
    orient.save()

    # Append to history log
    try:
        EVOLVE_DIR.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(orient.data, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.warning("Could not write history: %s", e)


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
        "outcomes": [],
    },
    "present": {
        "active_project": None,
        "active_tasks": [],
        "waiting_for": [],
        "last_session_focus": None,
        "commitments": [],
    },
    "future": {
        "goals": [],
        "scheduled_actions": [],
        "contingencies": [],
        "predictions": [],
        "plans": [],
    },
}


def _merge_timeline(data: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge a loaded timeline with defaults to handle schema evolution."""
    merged = dict(_DEFAULT_TIMELINE)
    for section in ("past", "present", "future"):
        if section in data and isinstance(data[section], dict):
            merged[section].update(data[section])
    for k, v in data.items():
        if k not in ("past", "present", "future"):
            merged[k] = v
    return merged


def load_timeline() -> Dict[str, Any]:
    """Load the timeline (past → present → future)."""
    data = safe_read_json(TIMELINE_FILE)
    if isinstance(data, dict):
        return _merge_timeline(data)
    return dict(_DEFAULT_TIMELINE)


def save_timeline(timeline: Dict[str, Any]) -> None:
    """Persist the timeline."""
    safe_write_json(TIMELINE_FILE, timeline)


def record_event(event_type: str, summary: str, impact: str = "") -> None:
    """Record an event in the timeline's past."""
    timeline = load_timeline()
    event = {
        "id": now_compact(),
        "type": event_type,
        "timestamp": now_iso(),
        "summary": summary,
        "impact": impact,
    }
    timeline["past"]["events"].append(event)
    timeline["past"]["events"] = timeline["past"]["events"][-50:]
    save_timeline(timeline)


def record_session_completion(session_id: str, focus: str,
                              outcomes: list) -> None:
    """Record a completed session in the timeline."""
    timeline = load_timeline()
    entry = {
        "session_id": session_id,
        "timestamp": now_iso(),
        "focus": focus,
        "outcomes": outcomes,
    }
    timeline["past"]["completed_sessions"].append(entry)
    timeline["past"]["completed_sessions"] = \
        timeline["past"]["completed_sessions"][-20:]
    save_timeline(timeline)


def record_outcome(event_id: str, summary: str, impact: str = "") -> None:
    """Record an outcome linked to a past event."""
    timeline = load_timeline()
    outcome = {
        "id": now_compact(),
        "event_id": event_id,
        "timestamp": now_iso(),
        "summary": summary,
        "impact": impact,
    }
    timeline["past"]["outcomes"].append(outcome)
    timeline["past"]["outcomes"] = timeline["past"]["outcomes"][-50:]
    save_timeline(timeline)


def add_commitment(what: str, deadline: Optional[str] = None,
                   status: str = "active") -> None:
    """Track a commitment with an optional deadline."""
    timeline = load_timeline()
    commit = {
        "id": now_compact(),
        "what": what,
        "deadline": deadline,
        "status": status,
        "created_at": now_iso(),
    }
    timeline["present"]["commitments"].append(commit)
    timeline["present"]["commitments"] = \
        timeline["present"]["commitments"][-30:]
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
                   confidence: Optional[float] = None,
                   basis: Optional[str] = None) -> None:
    """Record a prediction about future outcomes."""
    timeline = load_timeline()
    pred = {
        "id": now_compact(),
        "text": text,
        "timeframe": timeframe,
        "confidence": confidence,
        "basis": basis,
        "created_at": now_iso(),
    }
    timeline["future"]["predictions"].append(pred)
    timeline["future"]["predictions"] = \
        timeline["future"]["predictions"][-50:]
    save_timeline(timeline)


def update_present_state(active_project: Optional[str] = None,
                         tasks: Optional[list] = None,
                         waiting_for: Optional[list] = None,
                         focus: Optional[str] = None) -> None:
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


def update_goals(goals: Optional[list] = None,
                 scheduled: Optional[list] = None,
                 contingencies: Optional[list] = None) -> None:
    """Update the future goals/plans in the timeline."""
    timeline = load_timeline()
    if goals is not None:
        timeline["future"]["goals"] = goals
    if scheduled is not None:
        timeline["future"]["scheduled_actions"] = scheduled
    if contingencies is not None:
        timeline["future"]["contingencies"] = contingencies
    save_timeline(timeline)


def create_plan(goal: str,
                steps: Optional[List[Dict[str, Any]]] = None) -> str:
    """Create a new plan. Delegates to data_layer."""
    return _create_plan(goal, steps)


def get_active_plan() -> Optional[Dict[str, Any]]:
    """Return the first active plan. Delegates to data_layer."""
    return _get_active_plan()


def update_plan_step(plan_id: str, step_id: str, new_status: str,
                     note: str = "") -> bool:
    """Update a single step's status. Delegates to data_layer."""
    return _update_plan_step(plan_id, step_id, new_status, note)


def complete_plan(plan_id: str, status: str = "complete") -> bool:
    """Mark a plan as complete or failed. Delegates to data_layer."""
    return _complete_plan(plan_id, status)


def format_timeline_context() -> str:
    """Format timeline as natural narrative. Delegates to data_layer."""
    return _format_timeline_context()


# ═════════════════════════════════════════════════════════════════
#  Multi-type Memory (Gap 2 — Episodic / Semantic / Procedural)
# ═════════════════════════════════════════════════════════════════

_DEFAULT_MEMORY = {
    "version": 1,
    "episodic": [],
    "semantic": [],
    "procedural": [],
}


def load_memory() -> Dict[str, Any]:
    """Load the multi-type memory store. Delegates to data_layer."""
    mem = _Memory.load()
    return mem.data


def save_memory(memory: Dict[str, Any]) -> None:
    """Persist the memory store. Delegates to data_layer."""
    mem = _Memory(data=memory)
    mem.save()


def add_episodic(mtype: str, summary: str, details: str = "",
                 tags: Optional[list] = None,
                 salience: float = 0.5) -> str:
    """Record an episodic memory. Delegates to data_layer."""
    mem = _Memory.load()
    return mem.add_episodic(mtype=mtype, summary=summary, details=details,
                            tags=tags, salience=salience)


def add_semantic(topic: str, fact: str, source: str = "experience",
                 confidence: float = 0.7) -> str:
    """Record a semantic fact. Delegates to data_layer."""
    mem = _Memory.load()
    return mem.add_semantic(topic=topic, fact=fact, source=source,
                            confidence=confidence)


def add_procedural(pattern: str, trigger: str, procedure: str) -> str:
    """Record a procedural memory. Delegates to data_layer."""
    mem = _Memory.load()
    return mem.add_procedural(pattern=pattern, trigger=trigger,
                              procedure=procedure)


def search_memories(query: str, memory_types: Optional[List[str]] = None,
                    limit: int = 5) -> Dict[str, List[Dict]]:
    """Simple text search across memory types. Delegates to data_layer."""
    mem = _Memory.load()
    return mem.search(query=query, memory_types=memory_types, limit=limit)


def get_relevant_semantic(topic: str, limit: int = 3) -> List[Dict]:
    """Get semantic facts matching a topic. Delegates to data_layer."""
    mem = _Memory.load()
    return mem.get_semantic_by_topic(topic=topic, limit=limit)


def format_memory_context() -> str:
    """Format memory section for the system prompt."""
    mem = _Memory.load()
    return "## Recent Memories\n" + mem.format_context()


# ═════════════════════════════════════════════════════════════════
#  Self-generated Goals (Gap 4 — Goal Generation)
# ═════════════════════════════════════════════════════════════════

_DEFAULT_GOALS = {
    "version": 1,
    "goals": [],
}


def load_goals() -> Dict[str, Any]:
    """Load self-generated goals. Delegates to data_layer."""
    goals = _Goals.load()
    return goals.data


def save_goals(goals_store: Dict[str, Any]) -> None:
    """Persist the goals store. Delegates to data_layer."""
    goals = _Goals(data=goals_store)
    goals.save()


def propose_goal(title: str, description: str, rationale: str = "",
                 gap_reference: str = "",
                 verification_criteria: str = "",
                 priority: int = 3,
                 dependencies: Optional[List[str]] = None) -> str:
    """Propose a new self-generated goal. Delegates to data_layer."""
    goals = _Goals.load()
    return goals.propose(
        title=title, description=description, rationale=rationale,
        gap_reference=gap_reference,
        verification_criteria=verification_criteria,
        priority=priority, dependencies=dependencies,
    )


def update_goal_status(goal_id: str, new_status: str,
                       note: str = "") -> bool:
    """Update a goal's lifecycle. Delegates to data_layer."""
    goals = _Goals.load()
    return goals.update_status(goal_id, new_status, note)


def get_active_goals(status_filter: Optional[List[str]] = None) -> List[Dict]:
    """Get goals filtered by status. Delegates to data_layer."""
    goals = _Goals.load()
    return goals.get_active(status_filter)


def format_goal_context() -> str:
    """Format pending goals for the system prompt. Delegates to data_layer."""
    goals = _Goals.load()
    return goals.format_context()


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


def _deep_merge_self_model(data: Dict[str, Any],
                           defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge loaded self-model with defaults."""
    merged = dict(defaults)
    for section in ("identity", "state", "capabilities", "commitments"):
        if section in data and isinstance(data[section], dict):
            merged[section].update(data[section])
    for k, v in data.items():
        if k not in merged:
            merged[k] = v
    return merged


def load_self_model() -> Dict[str, Any]:
    """Load the self-model."""
    data = safe_read_json(SELF_MODEL_FILE)
    if isinstance(data, dict):
        return _deep_merge_self_model(data, _DEFAULT_SELF_MODEL)
    return dict(_DEFAULT_SELF_MODEL)


def save_self_model(model: Dict[str, Any]) -> None:
    """Persist the self-model."""
    safe_write_json(SELF_MODEL_FILE, model)


def update_self_model(**updates) -> None:
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

    identity = model.get("identity", {})
    parts.append(f"Identity: {identity.get('name', '—')} — "
                 f"{identity.get('role', '—')}")

    state = model.get("state", {})
    ev = state.get("evolution_version", 0)
    gap = state.get("current_gap_focus")
    gap_str = f", current gap focus: {gap}" if gap else ""
    parts.append(f"Evolution: v{ev}{gap_str}")

    caps = model.get("capabilities", {})
    weaknesses = caps.get("weaknesses", [])
    if weaknesses:
        parts.append(f"Known weaknesses: {'; '.join(weaknesses[:3])}")
    unknowns = caps.get("unknown_areas", [])
    if unknowns:
        parts.append(f"Areas to learn: {'; '.join(unknowns[:3])}")

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

    Delegates to data_layer.py's consolidated version, which reads
    from the same files and produces the same output format.

    Returns a string that gets injected into the volatile part of the
    system prompt, so the agent starts each session with:
      - Previous focus, insights, unfinished direction
      - Timeline awareness (past → present → future)
      - Self-model awareness (identity, capabilities, gaps)
    """
    from data_layer import format_orientation_context as _fmt
    return _fmt()
