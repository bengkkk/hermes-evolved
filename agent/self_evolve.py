"""Self-orientation, timeline, and self-model for continuous evolution.

At session start, the agent loads its previous orientation notes,
timeline, and self-model — generating context about what it's working on,
what it knows about itself, and what needs attention.
At session end, it can persist reflection for the next session.

This module is a thin compatibility layer over ``data_layer.py``,
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
    Goals as _Goals,
    Memory as _Memory,
    Orientation as _Orientation,
    SelfModel as _SelfModel,
    add_commitment as _add_commitment,
    add_prediction as _add_prediction,
    complete_plan as _complete_plan,
    create_plan as _create_plan,
    format_orientation_context as _format_orientation_context,
    format_plan_context as _format_plan_context,
    format_self_model_context as _format_self_model_context,
    format_timeline_context as _format_timeline_context,
    get_active_plan as _get_active_plan,
    get_evolve_dir,
    load_self_model as _load_self_model,
    load_timeline_dict as _load_timeline_dict,
    now_compact,
    now_iso,
    record_event as _record_event,
    record_outcome as _record_outcome,
    record_session_completion as _record_session_completion,
    safe_read_json,
    safe_write_json,
    save_self_model as _save_self_model,
    save_timeline_dict as _save_timeline_dict,
    update_commitment as _update_commitment,
    update_goals as _update_goals,
    update_plan_step as _update_plan_step,
    update_present_state as _update_present_state,
    update_self_model as _update_self_model,
)

logger = logging.getLogger(__name__)

# ── Paths (only paths unique to self_evolve; data_layer owns the canonical ones) ──
EVOLVE_DIR = get_evolve_dir()
ORIENTATION_FILE = EVOLVE_DIR / "orientation.json"
HISTORY_FILE = EVOLVE_DIR / "history.jsonl"


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
#  All functions delegate to data_layer.py
# ═════════════════════════════════════════════════════════════════

def load_timeline() -> Dict[str, Any]:
    """Load the timeline (past → present → future). Delegates to data_layer."""
    return _load_timeline_dict()


def save_timeline(timeline: Dict[str, Any]) -> None:
    """Persist the timeline. Delegates to data_layer."""
    _save_timeline_dict(timeline)


def record_event(event_type: str, summary: str, impact: str = "") -> None:
    """Record an event in the timeline's past. Delegates to data_layer."""
    _record_event(event_type, summary, impact)


def record_session_completion(session_id: str, focus: str,
                              outcomes: list) -> None:
    """Record a completed session in the timeline. Delegates to data_layer."""
    _record_session_completion(session_id, focus, outcomes)


def record_outcome(event_id: str, summary: str, impact: str = "") -> None:
    """Record an outcome linked to a past event. Delegates to data_layer."""
    _record_outcome(event_id, summary, impact)


def add_commitment(what: str, deadline: Optional[str] = None,
                   status: str = "active") -> None:
    """Track a commitment with an optional deadline. Delegates to data_layer."""
    _add_commitment(what, deadline, status)


def update_commitment(commit_id: str, status: str = "done") -> None:
    """Mark a commitment as done/expired. Delegates to data_layer."""
    _update_commitment(commit_id, status)


def add_prediction(text: str, timeframe: Optional[str] = None,
                   confidence: Optional[float] = None,
                   basis: Optional[str] = None) -> None:
    """Record a prediction about future outcomes. Delegates to data_layer."""
    _add_prediction(text, timeframe, confidence, basis)


def update_present_state(active_project: Optional[str] = None,
                         tasks: Optional[list] = None,
                         waiting_for: Optional[list] = None,
                         focus: Optional[str] = None) -> None:
    """Update the present state in the timeline. Delegates to data_layer."""
    _update_present_state(active_project, tasks, waiting_for, focus)


def update_goals(goals: Optional[list] = None,
                 scheduled: Optional[list] = None,
                 contingencies: Optional[list] = None) -> None:
    """Update the future goals/plans in the timeline. Delegates to data_layer."""
    _update_goals(goals, scheduled, contingencies)


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
#  All functions delegate to data_layer.py
# ═════════════════════════════════════════════════════════════════

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
    """Record an episodic memory and persist immediately. Delegates to data_layer."""
    mem = _Memory.load()
    mem_id = mem.add_episodic(mtype=mtype, summary=summary, details=details,
                              tags=tags, salience=salience)
    mem.save()
    return mem_id


def add_semantic(topic: str, fact: str, source: str = "experience",
                 confidence: float = 0.7) -> str:
    """Record a semantic fact and persist immediately. Delegates to data_layer."""
    mem = _Memory.load()
    mem_id = mem.add_semantic(topic=topic, fact=fact, source=source,
                              confidence=confidence)
    mem.save()
    return mem_id


def add_procedural(pattern: str, trigger: str, procedure: str) -> str:
    """Record a procedural memory and persist immediately. Delegates to data_layer."""
    mem = _Memory.load()
    mem_id = mem.add_procedural(pattern=pattern, trigger=trigger,
                                procedure=procedure)
    mem.save()
    return mem_id


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
#  All functions delegate to data_layer.py
# ═════════════════════════════════════════════════════════════════

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
    gid = goals.propose(
        title=title, description=description, rationale=rationale,
        gap_reference=gap_reference,
        verification_criteria=verification_criteria,
        priority=priority, dependencies=dependencies,
    )
    goals.save()
    return gid


def update_goal_status(goal_id: str, new_status: str,
                       note: str = "") -> bool:
    """Update a goal's lifecycle. Delegates to data_layer."""
    goals = _Goals.load()
    result = goals.update_status(goal_id, new_status, note)
    if result:
        goals.save()
    return result


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
#  All functions delegate to data_layer.py
# ═════════════════════════════════════════════════════════════════

def load_self_model() -> Dict[str, Any]:
    """Load the self-model. Delegates to data_layer."""
    return _load_self_model()


def save_self_model(model: Dict[str, Any]) -> None:
    """Persist the self-model. Delegates to data_layer."""
    _save_self_model(model)


def update_self_model(**updates) -> None:
    """Update specific fields of the self-model. Delegates to data_layer."""
    _update_self_model(**updates)


def format_self_model_context() -> str:
    """Format self-model section for the system prompt. Delegates to data_layer."""
    return _format_self_model_context()


# ═════════════════════════════════════════════════════════════════
#  Unified context injection (called by system_prompt.py)
#  Delegates to data_layer.py
# ═════════════════════════════════════════════════════════════════

def format_orientation_context() -> str:
    """Format orientation + timeline + self-model + world model for the system prompt.

    Delegates to data_layer.py's consolidated version, which reads
    from the same files and produces the same output format.  Then
    appends World Model context (prediction accuracy, action triples,
    discrepancy patterns) from ``world_model.py``.

    Returns a string that gets injected into the volatile part of the
    system prompt, so the agent starts each session with:
      - Previous focus, insights, unfinished direction
      - Timeline awareness (past → present → future)
      - Self-model awareness (identity, capabilities, gaps)
      - World Model awareness (prediction accuracy, discrepancies)
    """
    base = _format_orientation_context()
    try:
        from world_model import format_world_model_context as _fmt_wm
        wm_str = _fmt_wm()
        if wm_str:
            return base + "\n\n" + wm_str
    except Exception:
        logger.debug("Could not append world model context", exc_info=True)
    return base
