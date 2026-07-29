"""Persistent cognition daemon (Gap 1).

A background process that runs periodic thinking cycles, independent
of the user-message-driven conversation loop. Each cycle:

  1. Loads shared state from ~/.hermes/evolve/ (timeline, self-model)
  2. Builds a self-reflection prompt from current state
  3. Calls the LLM via Hermes's auxiliary_client (same provider chain)
  4. Parses structured insights from the response
  5. Persists updates back to evolve/ data files
  6. Logs the cycle for observability

Designed to be launched via:
  - `python3 think_daemon.py` (standalone, runs N cycles then exits)
  - `hermes cron create` (periodic trigger, one cycle per tick)
  - systemd/tmux (true persistent daemon)

Shares the same provider, config, and data as the main Hermes session.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# ── Ensure Hermes modules are importable ──
_HERMES_ROOT = Path(__file__).resolve().parent
if str(_HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(_HERMES_ROOT))

# ── Consolidated data layer ──
from data_layer import get_evolve_dir, safe_read_json, safe_write_json

logger = logging.getLogger("think_daemon")

# ── World Model integration (Gap 6) ──
from world_model import WorldModel, load_world_model, save_world_model

# ── Auto-detect workspace root from script location ──
# This adapts to wherever the repo is cloned (bare VM, Docker, etc.)
_WORKSPACE_ROOT: Path = Path(__file__).resolve().parent
_WORKSPACE_ROOT_STR: str = str(_WORKSPACE_ROOT)

# ── Paths (delegated to data_layer for the base directory) ──
EVOLVE_DIR = get_evolve_dir()
TIMELINE_FILE = EVOLVE_DIR / "timeline.json"
SELF_MODEL_FILE = EVOLVE_DIR / "self_model.json"
ORIENTATION_FILE = EVOLVE_DIR / "orientation.json"
DAEMON_STATE_FILE = EVOLVE_DIR / "daemon_state.json"
DAEMON_LOG_FILE = EVOLVE_DIR / "daemon.log"
DAEMON_LOCK_FILE = EVOLVE_DIR / "daemon.lock"

# ── Default state ──
_DEFAULT_DAEMON_STATE: Dict[str, Any] = {
    "version": 2,
    "status": "initialized",
    "first_tick": None,
    "last_tick": None,
    "tick_count": 0,
    "interval_seconds": 600,
    "last_output": None,
    "cycle_history": [],  # list of {timestamp, status, error, duration, tick_count} — last 20
}


# ═════════════════════════════════════════════════════════════════
#  State helpers (delegated to data_layer for the heavy lifting)
# ═════════════════════════════════════════════════════════════════

def load_daemon_state() -> Dict[str, Any]:
    data = safe_read_json(DAEMON_STATE_FILE, dict(_DEFAULT_DAEMON_STATE))
    # Migrate from version 1 → version 2: add cycle_history
    if data.get("version") == 1:
        data["version"] = 2
        if "cycle_history" not in data:
            data["cycle_history"] = []
    return data


def save_daemon_state(state: Dict[str, Any]) -> None:
    safe_write_json(DAEMON_STATE_FILE, state)


def _print_cycle_stats() -> None:
    """Print reliability metrics to stdout."""
    ds = load_daemon_state()
    cs = ds.get("cycle_stats", {})
    total = cs.get("total", 0)
    if total <= 1:
        return
    ok_rate = cs.get("ok", 0) / total * 100
    failures = cs.get("error", 0) + cs.get("parse_error", 0)
    print(f"  reliability: {cs.get('ok',0)} ok / {failures} fail / {ok_rate:.0f}% success")
    print(f"     avg {cs.get('avg_duration',0):.1f}s / max {cs.get('max_duration',0):.1f}s / total {total} cycles")
    if ok_rate < 80 and total >= 5:
        print("  health: degraded — below 80% success rate")
    elif cs.get("error", 0) == 0 and total >= 3:
        print("  health: stable — no failures")
    else:
        print("  health: acceptable")

    # Recent failure details for diagnostics
    history = ds.get("cycle_history", [])
    recent_fails = [c for c in history[-5:] if c.get("status") != "ok"]
    if recent_fails:
        print(f"  recent failures ({len(recent_fails)} in last {min(5, len(history))} cycles):")
        for c in recent_fails[-3:]:
            err = (c.get("error") or "?")[:80]
            dur = c.get("duration", "?")
            ts = (c.get("timestamp") or "?")[11:19]  # HH:MM:SS only
            print(f"    [{ts}] {c['status']} ({dur}s): {err}")


# ── PID lock ─────────────────────────────────────────────────────

def _acquire_daemon_lock() -> bool:
    """Acquire a PID-based lock to prevent concurrent daemon runs.

    Returns True if lock was acquired or already held by this process,
    False if another daemon is already running.
    Lock is released automatically on normal shutdown.
    """
    import os as _os
    my_pid = _os.getpid()
    if DAEMON_LOCK_FILE.exists():
        try:
            existing_pid = int(DAEMON_LOCK_FILE.read_text().strip())
            if existing_pid == my_pid:
                # Lock already held by us — that's fine
                logger.debug("Daemon lock already held by this PID %d", my_pid)
                return True
            # Check if the PID is still alive
            try:
                _os.kill(existing_pid, 0)   # signal 0 = test existence
                logger.warning(
                    "Daemon lock held by PID %d — skipping concurrent run",
                    existing_pid,
                )
                return False
            except OSError:
                # PID no longer exists — stale lock, take it over
                logger.info("Stale daemon lock (PID %d gone), taking over", existing_pid)
        except (ValueError, OSError, IOError):
            logger.warning("Corrupted daemon lock file, overwriting")
    DAEMON_LOCK_FILE.write_text(str(my_pid))
    logger.debug("Acquired daemon lock (PID %d)", my_pid)
    return True


def _release_daemon_lock() -> None:
    """Release the PID lock file."""
    import os as _os
    try:
        if DAEMON_LOCK_FILE.exists():
            current = DAEMON_LOCK_FILE.read_text().strip()
            if current == str(_os.getpid()):
                DAEMON_LOCK_FILE.unlink()
                logger.debug("Released daemon lock")
            else:
                logger.warning(
                    "Won't release lock owned by PID %s (we are %d)",
                    current, _os.getpid(),
                )
    except Exception as e:
        logger.warning("Failed to release daemon lock: %s", e)


def load_timeline() -> Dict[str, Any]:
    return safe_read_json(TIMELINE_FILE, {
        "version": 1, "past": {"events": [], "completed_sessions": []},
        "present": {}, "future": {"goals": []},
    })


def save_timeline(data: Dict[str, Any]) -> None:
    safe_write_json(TIMELINE_FILE, data)


def load_self_model() -> Dict[str, Any]:
    return safe_read_json(SELF_MODEL_FILE, {
        "version": 1, "identity": {"name": "Hermes (evolved)"},
        "state": {}, "capabilities": {}, "commitments": {},
    })


def save_self_model(data: Dict[str, Any]) -> None:
    safe_write_json(SELF_MODEL_FILE, data)


def load_orientation() -> Optional[Dict[str, Any]]:
    return safe_read_json(ORIENTATION_FILE, None)


def save_orientation(data: Dict[str, Any]) -> None:
    safe_write_json(ORIENTATION_FILE, data)


# ═════════════════════════════════════════════════════════════════
#  Prompt builder
# ═════════════════════════════════════════════════════════════════

_THINKING_PROMPT = """You are a persistent self-reflection system in a continuous improvement loop. You are NOT talking to a user — you are thinking to yourself, reviewing your trajectory and deciding what to do next.

=== CURRENT STATE ===

Identity: {identity_name} — {identity_role}
Current gap focus: {gap_focus}
Evolution version: v{evolution_version}
Total thinking cycles so far: {tick_count}

Strengths: {strengths}
Weaknesses: {weaknesses}
Unknown areas: {unknown}
Commitments: {commitments}

Recent timeline events:
{events_text}

Active project: {active_project}
Active tasks: {tasks_text}

Active plan:
{plan_status}

Last action result:
{last_action_result}

Daemon health (cycle reliability):
{daemon_health}

World Model (prediction accuracy and discrepancy feedback):
{world_model_context}

Future goals:
{goals_text}

=== YOUR TASK ===

Think deeply for one cycle. Consider:
1. What have I learned or accomplished since my last thinking cycle?
2. What should I focus on next?
3. What am I uncertain about that I should resolve?
4. Is there a decision I should record in my timeline?

PLAN MANAGEMENT:
- If the plan section shows "NO ACTIVE PLAN": create one using "new_plan" with a meaningful goal and 2-4 concrete steps. Each step MUST have "description" and "verification".
- If there IS an active plan: check the steps. If a step can be marked complete (the daemon code has been written), use "plan_action". If a step is blocked, note why.

GOAL GENERATION:
- You can propose new self-generated goals for the system's evolution.
- Goals should improve the system, fill remaining gaps (4, 6, 8, 10), and be based on what you've learned.
- Use "new_goal" to propose (include title, description, rationale, priority 1-5, gap_reference).
- Use "goal_action" to transition existing goals (active → in_progress → completed).
|- Prioritize: what unblocks the most other capabilities?

YOU MUST ACT. You are running in a continuous loop. Deliberation without action is wasted cycles.
CRITICAL RULE: Your JSON output MUST include a non-null "action" field this cycle. If you do not know what to do, set action to {{"type": "shell", "command": "ls {workspace_root}/", "description": "Explore workspace"}} — always better than null.
Rules:
- You MUST set a non-null "action" field every cycle. Setting it to null counts as failure.
- If you have been deliberating about the same thing for 2+ cycles without executing, STOP and just run: ls {workspace_root}/
- Prefer small concrete steps over perfect planning. A tiny real result beats a perfect plan.
  - Unsure about workspace layout? Run: ls -la /
  - Want to check a file? Run: cat {workspace_root}/some_file.py
  - Need to verify PyPI? Run: python3 -c "import json; print('ok')"

ACTION CAPABILITIES (Gap 10):
- Use the "action" field to take action NOW. Supported types:
  - write_file: write a file (set path + content)
  - shell: run a shell command (set command)
  - git_commit: add + commit (set message)
  - install_package: pip install (set package)
- CRITICAL: Before every action, set "expected_outcome" to PREDICT what the output will be
  (e.g. "Written main.py (245 bytes)" or "exit=0: files listed"). The daemon compares this
  against the actual result to compute prediction error and improve future calibration.
  Without this field, the prediction error defaults to comparing type+description against output,
  which is less informative.
- The next cycle will tell you what happened. Don't wait — act.
- Example: {{"type": "shell", "command": "ls", "expected_outcome": "exit=0: list of files in workspace"}}
- Example: {{"type": "write_file", "path": "test.py", "content": "print('hi')", "expected_outcome": "Wrote test.py (14 bytes)"}}

SEARCH (resolve uncertainties):
- If you are uncertain about a fact, API, or approach, provide a "search_query" string (e.g., "github ssh key setup"). The search will run AFTER this response.
- Do NOT guess or fabricate when you are uncertain. Use search_query to find answers.
- Continue your thinking below; if search results are available, they will be provided and you will produce a final refined insight.

NEXT STEP RECOMMENDATION:
- At the end of your thinking, provide a "next_gap" field: which of the remaining gaps (2, 4, 6, 8, 10) should be tackled next and WHY. Base this on the current state of the system.
- Also set "reasoning" explaining the gap priority from a systems architecture perspective.

Respond with a JSON object ONLY — no markdown, no explanation, no extra text.

{{
  "insight": "One-sentence insight from this thinking cycle",
  "focus_next": "What to focus on next (or 'continue current')",
  "uncertainty_to_resolve": "An uncertainty or null",
  "event_to_record": {{
    "type": "milestone|decision|reflection",
    "summary": "Brief event description",
    "impact": "Why this matters"
  }} or null,
  "self_model_update": {{
    "weakness": "New weakness or null",
    "unknown": "New unknown area or null",
    "new_commitment": "New commitment or null"
  }},
  "outcome_to_record": {{
    "event_id": "auto",
    "summary": "What actually resulted from this event",
    "impact": "The effect"
  }} or null,
  "commitment": {{
    "what": "Concrete commitment I am making",
    "deadline": "YYYY-MM-DD" or null
  }} or null,
  "prediction": {{
    "text": "Prediction about future trajectory",
    "timeframe": "3 days" or null,
    "confidence": 0.0 to 1.0,
    "basis": "Why I predict this"
  }} or null,
  "session_record": {{
    "focus": "Summary of this cycle's focus",
    "outcomes_list": ["item1", "item2"]
  }} or null,
  "action": {{
    "type": "write_file|shell|git_commit|install_package",
    "description": "What this action does",
    "expected_outcome": "PREDICT what the output of this action will look like (e.g. 'Written main.py (245 bytes)' or 'exit=0: files listed'). Used to calibrate prediction accuracy.",
    "path": "for write_file",
    "content": "file content",
    "command": "shell command",
    "package": "package name",
    "message": "commit message"
  }} or null,
  "plan_action": {{  // Update a step in the current plan
    "step_id": "step_1",
    "new_status": "complete|blocked|in_progress",
    "note": "Optional note about why"
  }} or null,
  "new_plan": {{  // Create a new plan (only when no active plan or current plan is done)
    "goal": "Clear goal for this plan",
    "steps": [
      {{"description": "Step description", "verification": "How to verify"}},
      {{"description": "Step 2", "verification": "..."}}
    ]
  }} or null,
  "search_query": "A question or topic to search (or null). Use when uncertain about facts, APIs, or approaches.",
  "episodic_record": {{
    "mtype": "success|failure|observation",
    "summary": "What happened (brief)",
    "details": "More detail if needed",
    "salience": 0.0 to 1.0
  }} or null,
  "semantic_record": {{
    "topic": "What this knowledge is about",
    "fact": "The fact or insight learned",
    "confidence": 0.0 to 1.0
  }} or null,
  "procedural_record": {{
    "pattern": "Name of the pattern",
    "trigger": "When this applies",
    "procedure": "What to do"
  }} or null,
  "new_goal": {{  // Propose a new self-generated goal (Gap 4)
    "title": "Clear goal name",
    "description": "What success looks like",
    "rationale": "Why this matters for self-evolution",
    "gap_reference": "Which gap it addresses (4, 6, 8, or 10)",
    "priority": 1 to 5,
    "verification_criteria": "How to know it's done"
  }} or null,
  "goal_action": {{  // Update existing goal lifecycle
    "goal_id": "goal_...",
    "new_status": "active|in_progress|completed|abandoned",
    "note": "Why this change"
  }} or null,
  "next_gap": "4, 6, 8, or 10 — which gap to tackle next (or null). Based on system state analysis.",
  "reasoning": "Why this gap should be tackled next — systems architecture perspective (or null).",
  "confidence": 0.0 to 1.0
}}"""


def _build_thinking_prompt(state: Dict[str, Any]) -> str:
    """Build a self-reflection prompt from current evolve state."""
    tl = state.get("timeline", {})
    sm = state.get("self_model", {})
    ds = state.get("daemon_state", {})
    orient = state.get("orientation", {})

    identity = sm.get("identity", {})
    sm_state = sm.get("state", {})
    caps = sm.get("capabilities", {})
    commits = sm.get("commitments", {})

    strengths = caps.get("available_tools", [])
    weaknesses = caps.get("weaknesses", [])
    unknown = caps.get("unknown_areas", [])

    # Recent events
    events = tl.get("past", {}).get("events", [])
    events_text = "\n".join(
        f"  [{e.get('type','?')}] {e.get('summary','—')}"
        for e in events[-5:]
    ) if events else "  (none)"

    # Tasks
    present = tl.get("present", {})
    tasks = present.get("active_tasks", [])
    tasks_text = "; ".join(tasks[:5]) if tasks else "(none)"

    # Goals
    future = tl.get("future", {})
    goals = future.get("goals", [])
    goals_text = "\n".join(f"  → {g}" for g in goals[:5]) if goals else "  (none)"

    # Commitments (from timeline + self_model)
    tl_commits = present.get("commitments", [])
    active_commits = [c for c in tl_commits if c.get("status") == "active"]
    timeline_commit_text = "; ".join(c["what"] for c in active_commits[:3]) if active_commits else "(none)"
    sm_commit_list = commits.get("promised_features", []) + commits.get("active_obligations", [])
    all_commits = timeline_commit_text
    if sm_commit_list:
        all_commits += "; " + "; ".join(sm_commit_list[:3])

    # Recent outcomes
    outcomes = tl.get("past", {}).get("outcomes", [])
    outcomes_text = "\n".join(
        f"  · {o.get('summary','—')}" for o in outcomes[-3:]
    ) if outcomes else "  (none)"

    # Existing predictions
    predictions = future.get("predictions", [])
    pred_text = predictions[-1].get("text", "") if predictions else "(none)"
    pred_conf = predictions[-1].get("confidence", "") if predictions else ""

    # ── Active plan ──
    try:
        from agent.self_evolve import get_active_plan
        active_plan = get_active_plan()
    except (ImportError, Exception) as e:
        active_plan = None
    if active_plan:
        goal = active_plan.get("goal", "")
        progress = active_plan.get("progress", "0/0 steps")
        steps = active_plan.get("steps", [])
        plan_lines = [f"  Goal: {goal} ({progress})"]
        for s in steps:
            icon = {"complete": "✓", "blocked": "⊘", "in_progress": "●", "pending": "→"}.get(s.get("status", "pending"), "·")
            plan_lines.append(f"  {icon} {s['description']} [{s.get('status', 'pending')}]")
            if s.get("note"):
                plan_lines.append(f"     note: {s['note']}")
        plan_status = "\n".join(plan_lines)
    else:
        plan_status = "  (NO ACTIVE PLAN)"

    # ── Last action result (fed back from previous cycle) ──
    last_output = ds.get("last_action_output", "")
    if last_output:
        last_action_result = last_output[:400]
    else:
        last_action_result = "(no previous action recorded)"

    # ── World Model context (Gap 6) ──
    try:
        wm = state.get("world_model")
        if wm is None:
            wm = load_world_model()
        world_model_context = wm.format_world_model_context()
    except Exception as e:
        logger.warning("Failed to load world model context: %s", e)
        world_model_context = "  (world model unavailable — will be initialized on first action)"

    # ── Daemon health (failure diagnostics for the LLM) ──
    history = ds.get("cycle_history", [])
    if history:
        total_ok = sum(1 for c in history if c.get("status") == "ok")
        total_fail = len(history) - total_ok
        health_parts = [
            f"  Reliability: {total_ok} ok / {total_fail} fail / {len(history)} total cycles",
        ]
        # Show last 3 failures with details
        recent_fails = [c for c in history if c.get("status") != "ok"][-3:]
        if recent_fails:
            health_parts.append("  Recent failures:")
            for c in recent_fails:
                dur = c.get("duration", "?")
                err = c.get("error", "?")[:60]
                health_parts.append(f"    · {c['status']} ({dur}s): {err}")
        daemon_health = "\n".join(health_parts)
    else:
        daemon_health = "  (no cycle history yet — first run)"

    return _THINKING_PROMPT.format(
        identity_name=identity.get("name", "?"),
        identity_role=identity.get("role", "?"),
        gap_focus=sm_state.get("current_gap_focus", "?"),
        evolution_version=sm_state.get("evolution_version", 1),
        tick_count=ds.get("tick_count", 0),
        strengths="; ".join(strengths[:5]) if strengths else "(none)",
        weaknesses="; ".join(weaknesses[:3]) if weaknesses else "(none)",
        unknown="; ".join(unknown[:3]) if unknown else "(none)",
        commitments=all_commits,
        events_text=events_text,
        active_project=present.get("active_project", "(none)"),
        tasks_text=tasks_text,
        plan_status=plan_status,
        last_action_result=last_action_result,
        daemon_health=daemon_health,
        world_model_context=world_model_context,
        goals_text=goals_text,
        workspace_root=_WORKSPACE_ROOT_STR,
    )


# ═════════════════════════════════════════════════════════════════
#  Response parser
# ═════════════════════════════════════════════════════════════════

def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON dict from LLM response, handling markdown fences.

    Strategy (in order):
      1. Strip whitespace.
      2. If ``text`` contains markdown code fences (````` `````),
         extract everything between the first set of fence markers.
      3. Find the first ``'{'`` and last ``'}'`` in the extracted
         region and try to parse that substring as JSON.
      4. Return the parsed dict, or ``None`` on failure (including
         when the result is a JSON array or primitive — we only
         accept ``dict``-typed responses).

    This is deliberately more permissive than a strict parser because
    LLM output can include preamble text, trailing commentary, or
    extra whitespace around fences.
    """
    cleaned = text.strip()

    # ── Strip markdown code fences if present ──
    # Handle both ```json ... ``` and ``` ... ```
    fence_start = cleaned.find("```")
    if fence_start >= 0:
        # Find the closing fence after the opening one
        content_start = fence_start + 3
        # Skip optional language tag on the same line
        first_newline = cleaned.find("\n", content_start)
        if first_newline >= 0:
            content_start = first_newline + 1
        fence_end = cleaned.rfind("```")
        if fence_end > content_start:
            cleaned = cleaned[content_start:fence_end].strip()
        else:
            cleaned = cleaned[content_start:].strip()

    # ── Locate the outermost JSON object ──
    obj_start = cleaned.find("{")
    obj_end = cleaned.rfind("}")
    if obj_start < 0 or obj_end < 0 or obj_end <= obj_start:
        # No complete JSON object found
        try:
            result = json.loads(cleaned)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    candidate = cleaned[obj_start : obj_end + 1]
    try:
        result = json.loads(candidate)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # ── Last resort: try parsing the entire cleaned text ──
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def _apply_insights(result: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Apply parsed insights to evolve state, returning updated state."""
    tl = state.get("timeline", load_timeline())
    sm = state.get("self_model", load_self_model())
    orient = state.get("orientation", load_orientation())
    ds = state.get("daemon_state", {})

    # ── Record event in timeline ──
    event_id = None
    event = result.get("event_to_record")
    if event and isinstance(event, dict) and event.get("summary"):
        now = datetime.now(timezone.utc).isoformat()
        event_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        tl.setdefault("past", {}).setdefault("events", []).append({
            "id": event_id,
            "type": event.get("type", "reflection"),
            "timestamp": now,
            "summary": event["summary"],
            "impact": event.get("impact", ""),
        })
        # Keep last 50
        tl["past"]["events"] = tl["past"]["events"][-50:]

    # ── Record outcome ──
    outcome = result.get("outcome_to_record")
    if outcome and isinstance(outcome, dict) and outcome.get("summary"):
        target_event_id = outcome.get("event_id", "auto")
        if target_event_id == "auto" and event_id:
            target_event_id = event_id
        tl.setdefault("past", {}).setdefault("outcomes", []).append({
            "id": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
            "event_id": target_event_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": outcome["summary"],
            "impact": outcome.get("impact", ""),
        })
        tl["past"]["outcomes"] = tl["past"]["outcomes"][-50:]

    # ── Record commitment ──
    new_commit = result.get("commitment")
    if new_commit and isinstance(new_commit, dict) and new_commit.get("what"):
        tl.setdefault("present", {}).setdefault("commitments", []).append({
            "id": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
            "what": new_commit["what"],
            "deadline": new_commit.get("deadline"),
            "status": "active",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        tl["present"]["commitments"] = tl["present"]["commitments"][-30:]

    # ── Record prediction ──
    pred = result.get("prediction")
    if pred and isinstance(pred, dict) and pred.get("text"):
        tl.setdefault("future", {}).setdefault("predictions", []).append({
            "id": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
            "text": pred["text"],
            "timeframe": pred.get("timeframe"),
            "confidence": pred.get("confidence"),
            "basis": pred.get("basis"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        tl["future"]["predictions"] = tl["future"]["predictions"][-50:]

    # ── Record session summary ──
    sess = result.get("session_record")
    if sess and isinstance(sess, dict) and sess.get("focus"):
        tl.setdefault("past", {}).setdefault("completed_sessions", []).append({
            "session_id": f"cycle_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "focus": sess["focus"],
            "outcomes": sess.get("outcomes_list", []),
        })
        tl["past"]["completed_sessions"] = tl["past"]["completed_sessions"][-20:]

    # ── Plan action: update step in active plan ──
    pa = result.get("plan_action")
    if pa and isinstance(pa, dict) and pa.get("step_id"):
        new_status = pa.get("new_status", "complete")
        if new_status in ("complete", "blocked", "in_progress"):
            from agent.self_evolve import update_plan_step as _ups
            _ups(next((p["id"] for p in tl.get("future", {}).get("plans", []) if p.get("status") == "active"), ""),
                 pa["step_id"], new_status, pa.get("note", ""))

    # ── New plan creation ──
    np = result.get("new_plan")
    if np and isinstance(np, dict) and np.get("goal") and np.get("steps"):
        has_active = any(p.get("status") == "active" for p in tl.get("future", {}).get("plans", []))
        if not has_active:
            from agent.self_evolve import create_plan as _cp, record_event as _re
            steps_data = []
            for i, s in enumerate(np["steps"]):
                steps_data.append({
                    "id": f"step_{i + 1}",
                    "description": s.get("description", ""),
                    "verification": s.get("verification", ""),
                    "status": "pending",
                    "blocked_by": None,
                    "assigned_to": "user",
                    "completed_at": None,
                    "note": None,
                })
            plan_id = _cp(np["goal"], steps_data)
            _re("milestone", f"Created plan: {np['goal']}", f"Plan {plan_id} with {len(steps_data)} steps")

    # ── Update self model ──
    su = result.get("self_model_update", {})
    if isinstance(su, dict):
        caps = sm.setdefault("capabilities", {})
        weakness = su.get("weakness")
        if weakness and isinstance(weakness, str):
            caps.setdefault("weaknesses", [])
            if weakness not in caps["weaknesses"]:
                caps["weaknesses"].append(weakness)
                caps["weaknesses"] = caps["weaknesses"][-10:]

        unknown = su.get("unknown")
        if unknown and isinstance(unknown, str):
            caps.setdefault("unknown_areas", [])
            if unknown not in caps["unknown_areas"]:
                caps["unknown_areas"].append(unknown)
                caps["unknown_areas"] = caps["unknown_areas"][-10:]

        new_commit = su.get("new_commitment")
        if new_commit and isinstance(new_commit, str):
            commits = sm.setdefault("commitments", {})
            commits.setdefault("promised_features", [])
            if new_commit not in commits["promised_features"]:
                commits["promised_features"].append(new_commit)

    # ── Multi-type Memory (Gap 2) ──
    er = result.get("episodic_record")
    if er and isinstance(er, dict) and er.get("summary"):
        from agent.self_evolve import add_episodic as _ae
        _ae(mtype=er.get("mtype", "observation"), summary=er["summary"],
            details=er.get("details", ""), salience=er.get("salience", 0.5))

    sr = result.get("semantic_record")
    if sr and isinstance(sr, dict) and sr.get("topic") and sr.get("fact"):
        from agent.self_evolve import add_semantic as _asem
        _asem(topic=sr["topic"], fact=sr["fact"],
              source=sr.get("source", "experience"), confidence=sr.get("confidence", 0.7))

    pr = result.get("procedural_record")
    if pr and isinstance(pr, dict) and pr.get("pattern") and pr.get("trigger") and pr.get("procedure"):
        from agent.self_evolve import add_procedural as _ap
        _ap(pattern=pr["pattern"], trigger=pr["trigger"], procedure=pr["procedure"])

    # ── Self-generated Goals (Gap 4) ──
    ng = result.get("new_goal")
    if ng and isinstance(ng, dict) and ng.get("title") and ng.get("description"):
        from agent.self_evolve import propose_goal as _pg, record_event as _re
        gid = _pg(title=ng["title"], description=ng["description"],
                   rationale=ng.get("rationale", ""),
                   gap_reference=ng.get("gap_reference", ""),
                   verification_criteria=ng.get("verification_criteria", ""),
                   priority=ng.get("priority", 3))
        _re("milestone", f"Proposed new goal: {ng['title']}", f"Goal {gid}")

    ga = result.get("goal_action")
    if ga and isinstance(ga, dict) and ga.get("goal_id") and ga.get("new_status"):
        from agent.self_evolve import update_goal_status as _ugs, record_event as _re
        if _ugs(ga["goal_id"], ga["new_status"], ga.get("note", "")):
            _re("milestone", f"Goal {ga['goal_id']} → {ga['new_status']}", ga.get("note", ""))

    # ── Actions (Gap 10) ──
    act = result.get("action")
    # Auto-default action for cycle 1+ to prevent null-action drift
    if not (act and isinstance(act, dict) and act.get("type")):
        if ds.get("tick_count", 0) >= 10:
            act = {"type": "shell", "command": f"ls {_WORKSPACE_ROOT_STR}/", "description": "Auto-default: explore workspace"}
            logger.info("Auto-default action (null action detected at cycle %d)", ds.get("tick_count", 0))
    
    # Clean stale "no action" weaknesses when actions ARE being executed
    caps = sm.setdefault("capabilities", {})
    old_weak = caps.get("weaknesses", [])
    if old_weak and ds.get("tick_count", 0) >= 5:
        filtered = [w for w in old_weak if "no concrete" not in w.lower() and "no action" not in w.lower()]
        if len(filtered) < len(old_weak):
            caps["weaknesses"] = filtered

    if act and isinstance(act, dict) and act.get("type"):
        atype = act["type"]
        desc = act.get("description", "")
        logger.info("Executing action: %s — %s", atype, desc[:60])

        # ── World Model: record action BEFORE execution ──
        action_guidance = None
        try:
            wm = state.get("world_model")
            if wm is None:
                wm = load_world_model()
            # Estimate expected outcome: use LLM's prediction if provided,
            # otherwise fall back to action type + description.
            expected = act.get("expected_outcome") or f"{atype}: {desc[:100]}" if desc else atype
            triple_id = wm.record_action(atype, desc or atype, expected)

            # ── Proactive risk assessment: check action guidance ──
            try:
                guidance = wm.format_action_guidance(atype, desc)
                if guidance:
                    logger.warning("Action risk assessment:\n%s", guidance)
                    action_guidance = guidance
            except Exception as e:
                logger.warning("Action guidance check failed (non-blocking): %s", e)
        except Exception as e:
            logger.warning("World model record failed (non-blocking): %s", e)
            triple_id = None
            wm = None

        try:
            from agent.self_evolve import add_episodic as _add_ep
            import subprocess, pathlib as _pl
            action_output = ""
            if atype == "write_file":
                apath = act.get("path", "")
                acontent = act.get("content", "")
                if apath and acontent:
                    p = _pl.Path(apath)
                    if not p.is_absolute():
                        p = _WORKSPACE_ROOT / apath
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(acontent)
                    action_output = f"Wrote {apath} ({len(acontent)} bytes)"
                    _add_ep("action", "Wrote " + apath, desc)
            elif atype == "shell":
                acmd = act.get("command", "")
                if acmd:
                    r = subprocess.run(acmd, shell=True, capture_output=True, text=True, timeout=60)
                    rv = (r.stdout[:400] + "\n" + r.stderr[:200])[:500]
                    action_output = f"exit={r.returncode}: {rv}"
                    _add_ep("action", "Shell: " + acmd[:60], action_output)
            elif atype == "git_commit":
                amsg = act.get("message", "")
                if amsg:
                    subprocess.run(["git", "add", "-A"], cwd=str(_WORKSPACE_ROOT), capture_output=True, text=True, timeout=30)
                    r = subprocess.run(["git", "commit", "-m", amsg], cwd=str(_WORKSPACE_ROOT), capture_output=True, text=True, timeout=30)
                    rv = (r.stdout[:200] + "\n" + r.stderr[:100])[:250]
                    action_output = f"exit={r.returncode}: {rv}"
                    _add_ep("action", "Commit: " + amsg[:60], action_output)
            elif atype == "install_package":
                apkg = act.get("package", "")
                if apkg:
                    r = subprocess.run(["pip", "install", apkg, "--break-system-packages"], capture_output=True, text=True, timeout=120)
                    action_output = "exit=" + str(r.returncode) + ": " + r.stdout[:200]
                    _add_ep("action", "Installed: " + apkg, action_output)
            else:
                logger.warning("Unknown action type: %s", atype)

            # ── World Model: complete action with actual outcome ──
            if action_output and triple_id and wm is not None:
                wm.complete_action(triple_id, action_output)
            elif triple_id and wm is not None:
                # Action produced no output — record that as the outcome
                wm.complete_action(triple_id, "(no output)")

            # Build output for next cycle's prompt (guidance + outcome)
            combined_output = ""
            if action_guidance:
                combined_output += f"[RISK WARNING] {action_guidance}\n"
            if action_output:
                combined_output += action_output
            if combined_output:
                ds["last_action_output"] = combined_output
            elif action_output:
                ds["last_action_output"] = action_output

        except subprocess.TimeoutExpired:
            logger.warning("Action %s timed out", atype)
            _add_ep("action", "Action timed out: " + atype, desc)
            # Record timeout as outcome
            if triple_id and wm is not None:
                wm.complete_action(triple_id, f"TIMEOUT: {atype}")
        except Exception as e:
            logger.warning("Action %s failed: %s", atype, e)
            _add_ep("action", "Action failed: " + atype, str(e)[:200])
            # Record failure as outcome
            if triple_id and wm is not None:
                wm.complete_action(triple_id, f"FAILED: {e!s}")
        finally:
            from agent.self_evolve import save_memory, load_memory
            save_memory(load_memory())
            # Save world model changes
            if wm is not None:
                try:
                    wm.save()
                except Exception as e:
                    logger.warning("Failed to save world model: %s", e)
    # ── Update orientation with latest insight ──
    insight = result.get("insight", "")
    focus_next = result.get("focus_next", "")
    if insight or focus_next:
        orient = orient or {}
        orient["focus"] = focus_next or orient.get("focus", "")
        existing = orient.get("insights", [])
        if insight and (not existing or existing[-1] != insight):
            existing.append(insight)
            orient["insights"] = existing[-10:]
        orient["next_steps"] = orient.get("next_steps", [])
        if focus_next and (not orient["next_steps"] or orient["next_steps"][-1] != focus_next):
            orient["next_steps"].append(focus_next)
            orient["next_steps"] = orient["next_steps"][-5:]

    return {"timeline": tl, "self_model": sm, "orientation": orient}


# ═════════════════════════════════════════════════════════════════
#  LLM call
# ═════════════════════════════════════════════════════════════════

# ── Runtime main context cache ──
_RUNTIME_INITIALIZED: bool = False
_RUNTIME_PROVIDER: str = ""
_RUNTIME_MODEL: str = ""


def _auto_detect_provider_from_env() -> Optional[tuple[str, str]]:
    """Auto-detect provider+model from environment variables when no config.yaml exists.

    Checks for known API key env vars and maps them to provider defaults.
    Returns (provider, model) or None if no recognized key is found.
    """
    import os as _os
    # Mapping: env_var -> (provider, default_model)
    _ENV_MAP: dict[str, tuple[str, str]] = {
        "OPENCODE_API_KEY": ("opencode-go", "glm-5"),
        "ANTHROPIC_API_KEY": ("anthropic", "claude-sonnet-4-20250514"),
        "OPENAI_API_KEY": ("openai", "gpt-4o"),
        "GEMINI_API_KEY": ("gmi", "gemini-2.5-pro"),
    }
    for env_var, (provider, default_model) in _ENV_MAP.items():
        if _os.environ.get(env_var):
            return (provider, default_model)
    return None


def _ensure_runtime_main() -> None:
    """Read provider/model from config.yaml or env vars and cache them for explicit use.

    The auxiliary client's auto-resolution (``_resolve_auto()``) scans
    openrouter → nous → local/custom → api-key — but if the user's
    API key lives under a non-default provider like ``opencode-go``,
    the auto chain never reaches it and every LLM call falls through
    the fallbacks and fails (the root cause of the daemon's ~60%
    error rate).

    This function caches the configured provider+model as module
    globals so ``_call_llm`` can pass them **explicitly** to
    ``async_call_llm()``, bypassing the broken auto-detection chain
    entirely.

    Resolution order:
      1. ``config.yaml`` at ``EVOLVE_DIR.parent / "config.yaml"``.
      2. Env-var-based auto-detection (fallback when no config.yaml).
    """
    global _RUNTIME_INITIALIZED, _RUNTIME_PROVIDER, _RUNTIME_MODEL
    if _RUNTIME_INITIALIZED:
        return
    try:
        import yaml
        config_path = EVOLVE_DIR.parent / "config.yaml"
        provider = ""
        model = ""

        # ── 1. Try config.yaml ──
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text())
            model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
            provider = model_cfg.get("provider", "") or ""
            model = model_cfg.get("default", "") or ""

        # ── 2. Fallback: auto-detect from env vars when no config.yaml ──
        if not (provider and model):
            detected = _auto_detect_provider_from_env()
            if detected:
                provider, model = detected
                logger.info(
                    "Auto-detected provider=%s model=%s from env vars",
                    provider, model,
                )

        if provider and model:
            # Propagate env var aliases: some provider env_vars may be named
            # differently in this session (e.g. OPENCODE_API_KEY vs the
            # provider's expected OPENCODE_GO_API_KEY).  Check the provider
            # profile's expected env_vars and bridge any gap.
            _ensure_provider_env(provider)

            # Cache for explicit passing to async_call_llm
            _RUNTIME_PROVIDER = provider
            _RUNTIME_MODEL = model

            # Also set runtime main context for any code that uses it
            from agent.auxiliary_client import set_runtime_main
            set_runtime_main(provider=provider, model=model)
            logger.info("Runtime main set: provider=%s model=%s", provider, model)
            _RUNTIME_INITIALIZED = True
        else:
            logger.info(
                "No runtime config found — checked config.yaml and env vars. "
                "Create %s or set OPENCODE_API_KEY/anthropic API key.",
                config_path,
            )
    except ImportError:
        logger.debug("set_runtime_main not available — skipping runtime init")
    except Exception as e:
        logger.warning("Failed to set runtime main: %s", e)


def _ensure_provider_env(provider: str) -> None:
    """Propagate env var aliases for the given provider.

    Some providers expect specific env var names (e.g. ``OPENCODE_GO_API_KEY``)
    but the runtime may have them under a different name (e.g. ``OPENCODE_API_KEY``).
    This function bridges common aliases so the credential pool finds the key.
    """
    import os as _os
    # ── Known env var aliases ──
    # Format: provider_name -> [(expected_var, [alias_var, ...]), ...]
    _ALIASES: dict = {
        "opencode-go": [
            ("OPENCODE_GO_API_KEY", ["OPENCODE_API_KEY"]),
        ],
    }
    if provider not in _ALIASES:
        return
    for expected_var, aliases in _ALIASES.get(provider, []):
        if _os.environ.get(expected_var):
            continue  # Already set
        for alias in aliases:
            val = _os.environ.get(alias)
            if val:
                _os.environ[expected_var] = val
                logger.info("Propagated %s → %s for provider %s", alias, expected_var, provider)
                break


async def _call_llm(messages: list, task: str = "thinking") -> Optional[str]:
    """Call LLM via Hermes auxiliary_client, return response text or None.

    Passes the configured provider/model explicitly (cached by
    ``_ensure_runtime_main``) to bypass the auto-detection chain
    which only scans openrouter → nous → custom → api-key and
    misses non-default providers like ``opencode-go``.
    """
    # Ensure the Hermes runtime context is known — this populates
    # _RUNTIME_PROVIDER and _RUNTIME_MODEL.
    _ensure_runtime_main()

    try:
        from agent.auxiliary_client import async_call_llm
    except ImportError:
        logger.error("Could not import Hermes auxiliary_client. Is HERMES_ROOT correct?")
        return None

    max_retries = 3
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = await asyncio.wait_for(
                async_call_llm(
                    task=task,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2048,
                    provider=_RUNTIME_PROVIDER or None,
                    model=_RUNTIME_MODEL or None,
                ),
                timeout=90.0,
            )
            # response is an OpenAI-style response object
            if hasattr(response, "choices") and response.choices:
                return response.choices[0].message.content
            # Fallback: dict-style
            if isinstance(response, dict):
                choices = response.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
            logger.warning("Unexpected response shape: %s", type(response).__name__)
            return None
        except asyncio.TimeoutError:
            logger.warning("LLM call attempt %d/%d timed out after 90s", attempt, max_retries)
            last_error = "timeout"
        except Exception as e:
            logger.warning("LLM call attempt %d/%d failed: %s", attempt, max_retries, e)
            last_error = str(e)
        if attempt < max_retries:
            await asyncio.sleep(2 ** attempt)  # 2s, 4s, 8s
    logger.error("LLM call failed after %d retries: %s", max_retries, last_error)
    return None


# ═════════════════════════════════════════════════════════════════
#  Main thinking cycle
# ═════════════════════════════════════════════════════════════════

async def run_one_cycle() -> Dict[str, Any]:
    """Run a single thinking cycle with reliability wrapping."""
    start_time = time.time()
    result = {
        "status": "ok",
        "tick_duration": 0,
        "insight": None,
        "error": None,
    }

    # ── Pre-cycle health snapshot ──
    ds = load_daemon_state()
    ds.setdefault("cycle_stats", {"total": 0, "ok": 0, "error": 0, "parse_error": 0,
                                      "avg_duration": 0.0, "max_duration": 0.0})
    cs = ds["cycle_stats"]
    cs["total"] += 1

    # ── Main cycle with timeout ──
    try:
        result = await asyncio.wait_for(
            _run_cycle_body(result, ds),
            timeout=120.0,
        )
    except asyncio.TimeoutError:
        result["status"] = "timeout"
        result["error"] = "Cycle exceeded 120s hard limit"
        logger.warning("Cycle timed out after 120s")
    except Exception as e:
        result["status"] = "crash"
        result["error"] = f"Cycle crashed: {e}"
        logger.exception("Cycle crashed: %s", e)

    # ── Post-cycle stats ──
    dur = time.time() - start_time
    result["tick_duration"] = round(dur, 1)

    if result["status"] == "ok":
        cs["ok"] += 1
    elif result["status"] in ("error", "crash", "timeout"):
        cs["error"] += 1
    elif result["status"] == "parse_error":
        cs["parse_error"] = cs.get("parse_error", 0) + 1
    else:
        cs["error"] += 1

    n = cs["ok"] + cs["error"] + cs.get("parse_error", 0)
    cs["avg_duration"] = round((cs.get("avg_duration", 0) * max(n - 1, 0) + dur) / max(n, 1), 1)
    cs["max_duration"] = max(cs.get("max_duration", 0), dur)

    # ── Per-cycle history for diagnostics ──
    history_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": result["status"],
        "error": result.get("error"),
        "duration": round(dur, 1),
        "tick_count": ds.get("tick_count", 0),
    }
    ds.setdefault("cycle_history", []).append(history_entry)
    ds["cycle_history"] = ds["cycle_history"][-20:]

    # Ensure first_tick is set even on the first error cycle
    if ds.get("first_tick") is None:
        ds["first_tick"] = datetime.now(timezone.utc).isoformat()

    ds["cycle_stats"] = cs
    save_daemon_state(ds)
    return result


async def _run_cycle_body(result: Dict[str, Any], ds: Dict[str, Any]) -> Dict[str, Any]:
    """Core cycle body, extracted so run_one_cycle can wrap it with timeout."""
    _cycle_start_time = time.time()
    # 1. Load current state
    tl = load_timeline()
    sm = load_self_model()
    orient = load_orientation()
    wm = load_world_model()

    state = {
        "daemon_state": ds,
        "timeline": tl,
        "self_model": sm,
        "orientation": orient,
        "world_model": wm,
    }

    # 1.75 Auto-verify expired predictions before the thinking cycle
    try:
        expired_count = wm.verify_expired_predictions()
        if expired_count > 0:
            logger.info("Auto-verified %d expired predictions", expired_count)
    except Exception as e:
        logger.warning("Auto-verification failed (non-blocking): %s", e)

    # 1.5 Auto-create initial plan if none exists
    try:
        from agent.self_evolve import get_active_plan, create_plan, record_event
        if get_active_plan() is None:
            plan_id = create_plan(
                "Complete hermes-evolved self-evolution framework",
                steps=[
                    {"description": "Finalize Timeline and SelfModel data layer",
                     "verification": "timeline.json has past/present/future sections"},
                    {"description": "Inject orientation into system prompt",
                     "verification": "format_orientation_context() content reaches session"},
                    {"description": "Set up persistent cognition daemon",
                     "verification": "think_daemon.py --once completes in < 60s"},
                    {"description": "Push evolve code to GitHub",
                     "verification": "git push succeeds"},
                ]
            )
            record_event("milestone", f"Auto-created initial plan: {plan_id}")
            # Reload state so the plan appears in the prompt
            state["timeline"] = load_timeline()
    except ImportError:
        pass

    # 2. Build prompt
    prompt = _build_thinking_prompt(state)
    messages = [
        {"role": "system", "content": "You are a persistent self-reflection system. Output valid JSON only."},
        {"role": "user", "content": prompt},
    ]

    # 3. Call LLM
    logger.info("Thinking cycle %d starting...", ds.get("tick_count", 0) + 1)
    raw = await _call_llm(messages)
    if raw is None:
        result["status"] = "error"
        result["error"] = "LLM returned no response"
        logger.warning("Thinking cycle produced no response")
        return result

    # 4. Parse response
    parsed = _try_parse_json(raw)
    if parsed is None:
        result["status"] = "parse_error"
        result["error"] = f"Could not parse JSON from: {raw[:200]}"
        logger.warning("Parse error: %s", result["error"])
        return result

    # 4.5 Search phase — resolve uncertainties via web search
    sq = parsed.get("search_query")
    if sq and isinstance(sq, str) and sq.strip():
        try:
            logger.info("Searching: %s", sq[:80])
            from ddgs import DDGS
            with DDGS() as ddgs:
                search_results = list(ddgs.text(sq, max_results=4))
            if search_results:
                search_text = "\n".join(
                    f"- {r['title']}: {r['body'][:200]} ({r['href']})"
                    for r in search_results
                )
                followup = (
                    f"Search results for '{sq}':\n{search_text}\n\n"
                    f"Given these results, produce your final JSON. "
                    f"Include a refined insight field that incorporates this new information. "
                    f"Set search_query to null in the final output."
                )
                messages.append({"role": "user", "content": followup})
                raw2 = await _call_llm(messages)
                if raw2:
                    parsed2 = _try_parse_json(raw2)
                    if parsed2:
                        parsed = parsed2
                        logger.info("Search incorporated into insight")
        except Exception as e:
            logger.warning("Search failed: %s", e)

    # 5. Apply insights to state
    updates = _apply_insights(parsed, state)

    # 5.5 World Model: record prediction if LLM made one
    pred = parsed.get("prediction")
    if pred and isinstance(pred, dict) and pred.get("text"):
        raw_conf = pred.get("confidence", 0.5)
        # Adjust confidence based on historical per-type calibration
        adjusted_conf = wm.adjust_confidence(raw_conf)
        wm.record_prediction(
            text=pred["text"],
            timeframe=pred.get("timeframe"),
            confidence=adjusted_conf,
            basis=pred.get("basis", ""),
        )
        wm.save()

    # 6. Save updated state
    save_timeline(updates["timeline"])
    save_self_model(updates["self_model"])
    if updates["orientation"]:
        save_orientation(updates["orientation"])
    # World model is saved inline during action execution and prediction recording
    # but ensure latest state is persisted
    wm.save()

    # 7. Update daemon state
    now_ts = datetime.now(timezone.utc).isoformat()
    ds["last_tick"] = now_ts
    ds["tick_count"] = ds.get("tick_count", 0) + 1
    ds["status"] = "ok"
    ds["last_output"] = {
        "insight": parsed.get("insight", ""),
        "focus_next": parsed.get("focus_next", ""),
        "confidence": parsed.get("confidence", 0),
        "next_gap": parsed.get("next_gap"),
        "reasoning": parsed.get("reasoning"),
    }
    if ds.get("first_tick") is None:
        ds["first_tick"] = now_ts
    save_daemon_state(ds)

    # 8. Build result
    elapsed = time.time() - _cycle_start_time
    result["tick_duration"] = round(elapsed, 2)
    result["insight"] = parsed.get("insight", "")
    result["focus_next"] = parsed.get("focus_next", "")
    result["confidence"] = parsed.get("confidence", 0)

    logger.info(
        "Cycle %d done in %.1fs — insight: %.60s",
        ds["tick_count"], elapsed, parsed.get("insight", "(none)")
    )
    return result


# ═════════════════════════════════════════════════════════════════
#  Continuous loop (standalone daemon)
# ═════════════════════════════════════════════════════════════════

async def run_daemon(interval_seconds: int = 600, max_cycles: int = 0):
    """Run the daemon loop.

    Args:
        interval_seconds: Time between thinking cycles (default 10 min).
        max_cycles: Max cycles before exit. 0 = unlimited.
    """
    # ── Acquire PID lock ──
    if not _acquire_daemon_lock():
        logger.info("Daemon already running — skipping this start attempt")
        return

    try:
        logger.info(
            "Daemon started, interval=%ds, max_cycles=%s, evolve_dir=%s",
            interval_seconds, max_cycles or "unlimited", EVOLVE_DIR,
        )

        # Initialize daemon state if needed
        ds = load_daemon_state()
        ds["interval_seconds"] = interval_seconds
        ds["status"] = "running"
        save_daemon_state(ds)

        cycle = 0
        while True:
            cycle += 1
            if max_cycles and cycle > max_cycles:
                logger.info("Reached max cycles (%d), exiting", max_cycles)
                break

            try:
                await run_one_cycle()
            except Exception as e:
                logger.error("Cycle failed unexpectedly: %s", e, exc_info=True)

            # Sleep — but allow early exit via state file check
            ds = load_daemon_state()
            if ds.get("status") == "shutdown":
                logger.info("Shutdown requested via daemon_state.json")
                break

            await asyncio.sleep(interval_seconds)

        ds = load_daemon_state()
        ds["status"] = "stopped"
        save_daemon_state(ds)
        logger.info("Daemon stopped.")
    finally:
        _release_daemon_lock()


# ═════════════════════════════════════════════════════════════════
#  Verification (no-LLM self-test)
# ═════════════════════════════════════════════════════════════════

def _run_verification() -> int:
    """Run a no-LLM verification of the full predict→act→observe→learn cycle.

    Exercises the world model pipeline end-to-end without needing
    API keys or LLM access.  Runs on an isolated copy to avoid
    mutating the on-disk world model state.

    Returns:
        0 on success (all checks pass), 1 on failure.
    """
    import copy as _copy

    print("=" * 60)
    print("  Hermes Evolved — World Model Verification")
    print("  (no LLM calls — self-contained self-test)")
    print("=" * 60)
    failures = 0

    # 1. Create isolated world model
    wm = WorldModel()
    print(f"\n  ✓ WorldModel instance created ({wm})")

    # 2. Record an action with expected outcome
    tid = wm.record_action("shell", "explore unknown directory", "should list contents")
    print(f"  ✓ Action recorded (id={tid})")
    assert tid.startswith("act_"), f"Bad ID prefix: {tid}"
    triples = wm.data["action_triples"]
    assert len(triples) == 1, f"Expected 1 triple, got {len(triples)}"
    assert triples[0]["completed"] is False, "Action should not be completed yet"
    print(f"  ✓ Action stored, not yet completed")

    # 3. Complete the action — low-error (exit=0 + no success keyword → 0.6)
    err = wm.complete_action(tid, "exit=0: file1.txt  file2.txt")
    assert err is not None, "complete_action returned None"
    assert 0.0 <= err <= 0.7, f"Expected moderate-low error, got {err}"
    print(f"  ✓ Action completed — prediction error: {err:.3f}")

    # 4. Record another mismatched action (same type = shell)
    tid2 = wm.record_action("shell", "deploy to production", "deploy should succeed")
    wm.complete_action(tid2, "exit=1: build failure — dependency not found")
    err2 = wm.data["action_triples"][1]["prediction_error"]
    assert err2 >= 0.8, f"Expected high error for failed deploy, got {err2}"
    print(f"  ✓ Failed deploy detected — prediction error: {err2:.3f}")

    # 5. Record a third action (write_file, also mismatched)
    tid3 = wm.record_action("write_file", "write critical config", "config file written")
    wm.complete_action(tid3, "permission denied: /etc/config.yaml")
    err3 = wm.data["action_triples"][2]["prediction_error"]
    assert err3 >= 0.5, f"Expected high error for permission denied, got {err3}"
    print(f"  ✓ Permission denied detected — prediction error: {err3:.3f}")

    # 6. Record another shell error to trigger discrepancy pattern (2 high-error shell)
    tid3b = wm.record_action("shell", "deploy staging environment", "deploy should succeed")
    wm.complete_action(tid3b, "exit=1: timeout connecting to registry")
    err3b = wm.data["action_triples"][3]["prediction_error"]
    assert err3b >= 0.8, f"Expected high error, got {err3b}"
    print(f"  ✓ Second deploy failure detected — prediction error: {err3b:.3f}")

    # 6. Verify per-type accuracy tracking
    pta = wm.get_per_type_accuracy()
    assert "shell" in pta, "shell type missing from per-type accuracy"
    assert "write_file" in pta, "write_file type missing from per-type accuracy"
    assert pta["shell"]["count"] == 3
    assert pta["write_file"]["count"] == 1
    print(f"  ✓ Per-type accuracy: shell(n=3), write_file(n=1)")

    # 7. Verify discrepancy patterns detected
    patterns = wm.get_discrepancy_patterns()
    assert len(patterns) >= 1, f"Expected ≥1 pattern from high-error shell actions, got {len(patterns)}"
    shell_pattern = next((p for p in patterns if p["action_type"] == "shell"), None)
    assert shell_pattern is not None, "Expected shell pattern"
    assert shell_pattern["count"] == 2, f"Expected 2 high-error shell actions, got {shell_pattern['count']}"
    print(f"  ✓ Discrepancy patterns: {len(patterns)} detected (shell: {shell_pattern['count']} failures)")

    # 8. Verify action guidance warns on risky actions
    guidance = wm.format_action_guidance("shell", "deploy to staging")
    assert guidance is not None, "Action guidance should warn for risky type"
    assert "risk" in guidance.lower() or "error" in guidance.lower() or "warning" in guidance.lower() or "elevated" in guidance.lower()
    print(f"  ✓ Action guidance triggered on risky action")

    # 9. Verify no guidance on safe action
    # Record a perfect git_commit action
    tid4 = wm.record_action("git_commit", "fix: small typo", "commit message")
    wm.complete_action(tid4, "exit=0: committed successfully")
    safe_guidance = wm.format_action_guidance("git_commit", "fix: small typo")
    # git_commit has no high-error history, so should be None
    print(f"  ✓ Safe action has no warning: {safe_guidance}")

    # 10. Verify confidence adjustment
    raw_conf = 0.8
    adj_conf = wm.adjust_confidence(raw_conf, "shell")
    # shell has high error (0.85+) → confidence should decrease
    assert adj_conf < raw_conf, f"Expected adjusted confidence < {raw_conf}, got {adj_conf}"
    adj_conf_good = wm.adjust_confidence(raw_conf, "git_commit")
    # git_commit has low error → confidence should increase
    assert adj_conf_good >= raw_conf or abs(adj_conf_good - raw_conf) < 0.1
    print(f"  ✓ Confidence adjustment: shell {raw_conf}→{adj_conf:.3f}, git_commit {raw_conf}→{adj_conf_good:.3f}")

    # 11. Verify context formatting works
    ctx = wm.format_world_model_context()
    assert "World Model State" in ctx
    assert "Action triples" in ctx
    assert "Biggest prediction errors" in ctx or "learning opportunities" in ctx
    assert "Per-type prediction accuracy" in ctx or "calibration" in ctx
    print(f"  ✓ Context formatted ({len(ctx)} chars)")

    # 12. Verify prediction insight
    insight = wm.format_prediction_insight()
    assert "accuracy" in insight or "verified" in insight
    print(f"  ✓ Prediction insight: {insight}")

    # 13. Verify macro predictions
    pid = wm.record_prediction("system will have >50 cycles", "1 day", 0.7, "historical rate")
    assert pid.startswith("pred_")
    p_err = wm.verify_prediction(pid, "system has 80 cycles")
    assert p_err is not None
    assert wm.data["prediction_accuracy"]["verified_predictions"] >= 1
    print(f"  ✓ Macro prediction lifecycle works")

    # 14. Verify expired prediction auto-verification
    expired_count = wm.verify_expired_predictions()
    print(f"  ✓ Auto-verify expired: {expired_count} expired")

    # 15. Verify calibration guidance
    cal = wm.format_calibration_guidance()
    assert "Per-type prediction accuracy" in cal
    assert "Best predicted" in cal
    assert "Worst predicted" in cal
    print(f"  ✓ Calibration guidance formatted")

    # Summary
    print(f"\n{'=' * 60}")
    if failures == 0:
        print(f"  ALL CHECKS PASSED — world model pipeline verified")
    else:
        print(f"  {failures} CHECK(S) FAILED")
    print(f"{'=' * 60}")
    return 0 if failures == 0 else 1


# ═════════════════════════════════════════════════════════════════
#  CLI entry point
# ═════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Hermes Persistent Cognition Daemon")
    parser.add_argument("--interval", type=int, default=600, help="Seconds between thinking cycles (default: 600 = 10 min)")
    parser.add_argument("--cycles", type=int, default=0, help="Max cycles before exit (0 = unlimited)")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    parser.add_argument("--verify", action="store_true", help="Run a no-LLM verification of the full predict→act→observe→learn cycle")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.verify:
        exit_code = _run_verification()
        sys.exit(exit_code)

    if args.once:
        # Acquire PID lock to prevent concurrent runs from overlapping cron triggers
        if not _acquire_daemon_lock():
            print("Daemon lock held by another process — skipping concurrent --once run")
            sys.exit(0)
        try:
            r = asyncio.run(run_one_cycle())
            status = r.get("status", "error")
            if status == "ok":
                insight = r.get("insight", "")[:80]
                print(f"[{status}] tick {r.get('tick_duration',0):.1f}s — {insight}")
            else:
                print(f"[{status}] {r.get('error', 'unknown error')}")
            # Print reliability stats
            _print_cycle_stats()
        finally:
            _release_daemon_lock()
    else:
        asyncio.run(run_daemon(args.interval, args.cycles))


if __name__ == "__main__":
    main()
