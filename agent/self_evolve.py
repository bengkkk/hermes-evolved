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
MEMORY_FILE = EVOLVE_DIR / "memory.json"
GOALS_FILE = EVOLVE_DIR / "goals.json"


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
        "plans": [],  # {id, goal, steps[], status, progress, created_at}
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


def create_plan(goal: str, steps: Optional[List[Dict[str, Any]]] = None) -> str:
    """Create a new plan in the future section. Steps get default fields automatically."""
    timeline = load_timeline()
    plan_id = f"plan_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    normalized_steps = []
    for i, s in enumerate(steps or []):
        normalized_steps.append({
            "id": s.get("id", f"step_{i + 1}"),
            "description": s.get("description", ""),
            "verification": s.get("verification", ""),
            "status": s.get("status", "pending"),
            "blocked_by": s.get("blocked_by"),
            "assigned_to": s.get("assigned_to", "user"),
            "completed_at": s.get("completed_at"),
            "note": s.get("note"),
        })
    plan = {
        "id": plan_id,
        "goal": goal,
        "steps": normalized_steps,
        "status": "active",
        "progress": f"0/{len(normalized_steps)} steps" if normalized_steps else "0 steps",
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": None,
    }
    timeline["future"]["plans"].append(plan)
    timeline["future"]["plans"] = timeline["future"]["plans"][-10:]
    save_timeline(timeline)
    return plan_id


def get_active_plan() -> Optional[Dict[str, Any]]:
    """Return the first active plan, if any."""
    timeline = load_timeline()
    for p in timeline.get("future", {}).get("plans", []):
        if p.get("status") == "active":
            return p
    return None


def update_plan_step(plan_id: str, step_id: str, new_status: str, note: str = "") -> bool:
    """Update a single step's status in a plan. Returns True if found."""
    timeline = load_timeline()
    for p in timeline["future"]["plans"]:
        if p["id"] == plan_id:
            for s in p["steps"]:
                if s["id"] == step_id:
                    s["status"] = new_status
                    if note:
                        s["note"] = note
                    if new_status in ("complete", "failed"):
                        s["completed_at"] = datetime.utcnow().isoformat()
                    # Recalculate progress
                    total = len(p["steps"])
                    done = sum(1 for st in p["steps"] if st.get("status") == "complete")
                    p["progress"] = f"{done}/{total} steps"
                    save_timeline(timeline)
                    return True
    return False


def complete_plan(plan_id: str, status: str = "complete") -> bool:
    """Mark a plan as complete or failed."""
    timeline = load_timeline()
    for p in timeline["future"]["plans"]:
        if p["id"] == plan_id:
            p["status"] = status
            p["completed_at"] = datetime.utcnow().isoformat()
            save_timeline(timeline)
            return True
    return False


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

    # ── Future ──
    future = timeline.get("future", {})

    # ── Plan (Gap 3: closed-loop planning) ──
    plans = future.get("plans", [])
    active_plan = next((p for p in plans if p.get("status") == "active"), None)
    if active_plan:
        goal = active_plan.get("goal", "")
        progress = active_plan.get("progress", "")
        steps = active_plan.get("steps", [])
        done_steps = [s for s in steps if s.get("status") == "complete"]
        pending_steps = [s for s in steps if s.get("status") in ("pending", "in_progress")]
        blocked_steps = [s for s in steps if s.get("status") == "blocked"]
        narrative.append(f"Active plan: {goal} ({progress})")
        for s in done_steps[-2:]:
            narrative.append(f"  ✓ {s['description']}")
        for s in blocked_steps[:1]:
            note = f" — {s.get('note', '')}" if s.get('note') else ""
            narrative.append(f"  ⊘ {s['description']} (blocked{note})")
        for s in pending_steps[:2]:
            narrative.append(f"  → {s['description']}")

    # ── Future: goals + predictions ──
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
#  Multi-type Memory (Gap 2 — Episodic / Semantic / Procedural)
# ═════════════════════════════════════════════════════════════════

_DEFAULT_MEMORY = {
    "version": 1,
    "episodic": [],   # experiences, observations, interactions
    "semantic": [],   # facts, knowledge, concepts
    "procedural": [], # how-to patterns, behaviors
}


def load_memory() -> Dict[str, Any]:
    """Load the multi-type memory store."""
    ensure_evolve_dir()
    if not MEMORY_FILE.exists():
        return dict(_DEFAULT_MEMORY)
    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        merged = dict(_DEFAULT_MEMORY)
        merged.update(data)
        for section in ("episodic", "semantic", "procedural"):
            if section in data and isinstance(data[section], list):
                merged[section] = data[section]
        return merged
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("Could not load memory: %s", e)
        return dict(_DEFAULT_MEMORY)


def save_memory(memory: Dict[str, Any]):
    """Persist the memory store."""
    ensure_evolve_dir()
    try:
        MEMORY_FILE.write_text(
            json.dumps(memory, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as e:
        logger.warning("Could not save memory: %s", e)


def add_episodic(mtype: str, summary: str, details: str = "",
                 tags: Optional[List[str]] = None, salience: float = 0.5) -> str:
    """Record an episodic memory (experience/observation/interaction)."""
    memory = load_memory()
    mem_id = f"ep_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    entry = {
        "id": mem_id,
        "timestamp": datetime.utcnow().isoformat(),
        "type": mtype,
        "summary": summary,
        "details": details,
        "salience": salience,
        "tags": tags or [],
    }
    memory["episodic"].append(entry)
    memory["episodic"] = memory["episodic"][-200:]
    save_memory(memory)
    return mem_id


def add_semantic(topic: str, fact: str, source: str = "experience",
                 confidence: float = 0.7) -> str:
    """Record a semantic fact (knowledge/concept learned)."""
    memory = load_memory()
    mem_id = f"sem_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    entry = {
        "id": mem_id,
        "timestamp": datetime.utcnow().isoformat(),
        "topic": topic,
        "fact": fact,
        "source": source,
        "confidence": confidence,
        "last_accessed": datetime.utcnow().isoformat(),
    }
    # Deduplicate: same topic + same fact
    for existing in memory["semantic"]:
        if existing.get("topic") == topic and existing.get("fact") == fact:
            existing["last_accessed"] = datetime.utcnow().isoformat()
            existing["confidence"] = max(existing.get("confidence", 0), confidence)
            save_memory(memory)
            return existing["id"]
    memory["semantic"].append(entry)
    memory["semantic"] = memory["semantic"][-500:]
    save_memory(memory)
    return mem_id


def add_procedural(pattern: str, trigger: str, procedure: str) -> str:
    """Record a procedural memory (how-to pattern)."""
    memory = load_memory()
    mem_id = f"pro_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    entry = {
        "id": mem_id,
        "timestamp": datetime.utcnow().isoformat(),
        "pattern": pattern,
        "trigger": trigger,
        "procedure": procedure,
        "success_count": 1,
        "failure_count": 0,
    }
    for existing in memory["procedural"]:
        if existing.get("pattern") == pattern:
            existing["success_count"] += 1
            save_memory(memory)
            return existing["id"]
    memory["procedural"].append(entry)
    memory["procedural"] = memory["procedural"][-100:]
    save_memory(memory)
    return mem_id


def search_memories(query: str, memory_types: Optional[List[str]] = None,
                    limit: int = 5) -> Dict[str, List[Dict]]:
    """Simple text search across memory types."""
    memory = load_memory()
    q = query.lower()
    types = memory_types or ["episodic", "semantic", "procedural"]
    results = {}
    for mt in types:
        matches = []
        for entry in memory.get(mt, []):
            text = json.dumps(entry).lower()
            if q in text:
                matches.append(entry)
        results[mt] = matches[:limit]
    return results


def get_relevant_semantic(topic: str, limit: int = 3) -> List[Dict]:
    """Get semantic facts matching a topic."""
    memory = load_memory()
    tl = topic.lower()
    matches = []
    for entry in memory.get("semantic", []):
        if tl in entry.get("topic", "").lower() or tl in entry.get("fact", "").lower():
            matches.append(entry)
    return matches[-limit:]


def format_memory_context() -> str:
    """Format memory section for the system prompt."""
    memory = load_memory()
    parts = ["## Recent Memories"]

    episodic = memory.get("episodic", [])
    recent_ep = episodic[-3:] if episodic else []
    if recent_ep:
        parts.append("Recent experiences:")
        for e in recent_ep:
            sal = "★" if e.get("salience", 0) >= 0.8 else "·"
            parts.append(f"  {sal} {e['summary'][:80]}")

    semantic = memory.get("semantic", [])
    recent_sem = semantic[-3:] if semantic else []
    if recent_sem:
        parts.append("Knowledge gained:")
        for s in recent_sem:
            conf = f" ({s.get('confidence', 0):.0%})" if s.get('confidence') else ""
            parts.append(f"  · {s['topic']}: {s['fact'][:100]}{conf}")

    procedural = memory.get("procedural", [])
    recent_pro = procedural[-2:] if procedural else []
    if recent_pro:
        parts.append("Learned patterns:")
        for p in recent_pro:
            sc = p.get("success_count", 0)
            fc = p.get("failure_count", 0)
            parts.append(f"  · {p['pattern']} (✓{sc} ✗{fc})")

    if not (recent_ep or recent_sem or recent_pro):
        parts.append("  (no recent memories yet)")

    return "\n".join(parts)


# ═════════════════════════════════════════════════════════════════
#  Self-generated Goals (Gap 4 — Goal Generation)
# ═════════════════════════════════════════════════════════════════

_DEFAULT_GOALS = {
    "version": 1,
    "goals": [],
}


def load_goals() -> Dict[str, Any]:
    """Load self-generated goals."""
    ensure_evolve_dir()
    if not GOALS_FILE.exists():
        return dict(_DEFAULT_GOALS)
    try:
        data = json.loads(GOALS_FILE.read_text(encoding="utf-8"))
        merged = dict(_DEFAULT_GOALS)
        merged.update(data)
        if "goals" in data and isinstance(data["goals"], list):
            merged["goals"] = data["goals"]
        return merged
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("Could not load goals: %s", e)
        return dict(_DEFAULT_GOALS)


def save_goals(goals_store: Dict[str, Any]):
    """Persist the goals store."""
    ensure_evolve_dir()
    try:
        GOALS_FILE.write_text(
            json.dumps(goals_store, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as e:
        logger.warning("Could not save goals: %s", e)


def propose_goal(title: str, description: str, rationale: str,
                 gap_reference: str = "", verification_criteria: str = "",
                 priority: int = 3, dependencies: Optional[List[str]] = None) -> str:
    """Propose a new self-generated goal."""
    goals_store = load_goals()
    goal_id = f"goal_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    goal = {
        "id": goal_id,
        "title": title,
        "description": description,
        "rationale": rationale,
        "priority": priority,
        "status": "proposed",
        "gap_reference": gap_reference,
        "dependencies": dependencies or [],
        "estimated_effort": "",
        "verification_criteria": verification_criteria,
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "notes": "",
    }
    goals_store["goals"].append(goal)
    goals_store["goals"] = goals_store["goals"][-100:]
    save_goals(goals_store)
    return goal_id


def update_goal_status(goal_id: str, new_status: str, note: str = "") -> bool:
    """Update a goal's lifecycle. Valid: active|in_progress|completed|abandoned."""
    goals_store = load_goals()
    for g in goals_store["goals"]:
        if g["id"] == goal_id:
            g["status"] = new_status
            if new_status in ("completed", "abandoned"):
                g["completed_at"] = datetime.utcnow().isoformat()
            if note:
                g["notes"] = note
            save_goals(goals_store)
            return True
    return False


def get_active_goals(status_filter: Optional[List[str]] = None) -> List[Dict]:
    """Get goals filtered by status. Default: proposed + active + in_progress."""
    goals_store = load_goals()
    statuses = status_filter or ["proposed", "active", "in_progress"]
    filtered = [g for g in goals_store["goals"] if g.get("status") in statuses]
    filtered.sort(key=lambda g: (g.get("priority", 5), g.get("created_at", "")))
    return filtered


def format_goal_context() -> str:
    """Format pending goals for the system prompt."""
    goals_store = load_goals()
    pending = get_active_goals()
    parts = ["## Self-generated Goals"]
    if not pending:
        parts.append("  (no self-generated goals yet)")
        return "\n".join(parts)

    for g in pending:
        sym = {"proposed": "◇", "active": "○", "in_progress": "◎"}.get(g.get("status", ""), "·")
        deps = f" [depends: {', '.join(g['dependencies'][:3])}]" if g.get("dependencies") else ""
        gap = f" [{g['gap_reference']}]" if g.get("gap_reference") else ""
        parts.append(f"  {sym} P{g.get('priority', 3)} — {g['title']}{deps}{gap}")
        parts.append(f"      {g.get('description', '')[:80]}")

    active_count = sum(1 for g in pending if g.get("status") in ("active", "in_progress"))
    proposed_count = sum(1 for g in pending if g.get("status") == "proposed")
    parts.append(f"  ({active_count} active, {proposed_count} proposed)")
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

    # Section 3: Multi-type Memory (Gap 2)
    mem = format_memory_context()
    if mem:
        parts.append(mem)

    # Section 4: Self-generated Goals (Gap 4)
    gl = format_goal_context()
    if gl:
        parts.append(gl)

    # Section 5: Self Model
    sm = format_self_model_context()
    if sm:
        parts.append(sm)

    if not parts:
        return ""

    parts.append("")
    return "\n\n".join(parts)
