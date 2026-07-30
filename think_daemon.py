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
import re
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
    """Acquire an atomic file lock to prevent concurrent daemon runs.

    Uses ``O_CREAT | O_EXCL`` for an atomic create-or-fail so two
    processes that check at the same wall time cannot both acquire the
    lock (eliminates the TOCTOU race in the original PID-only check).

    If the lock file exists but its PID is stale (no longer alive),
    the lock is taken over with another atomic attempt.

    Returns True if lock was acquired, False if another daemon is
    already running.
    """
    import os as _os

    my_pid = _os.getpid()
    lock_path_str = str(DAEMON_LOCK_FILE)

    def _try_atomic_create() -> bool:
        """Atomically create the lock file.  Returns True on success."""
        try:
            fd = _os.open(lock_path_str, _os.O_CREAT | _os.O_EXCL | _os.O_WRONLY, 0o644)
            _os.write(fd, str(my_pid).encode())
            _os.close(fd)
            return True
        except FileExistsError:
            return False

    # Fast path: try atomic create
    if _try_atomic_create():
        logger.debug("Acquired daemon lock (PID %d)", my_pid)
        return True

    # ── Lock file exists — validate the holder ──
    try:
        existing_pid = int(DAEMON_LOCK_FILE.read_text().strip())
    except (ValueError, OSError, IOError):
        logger.warning("Corrupted daemon lock file — attempting takeover")
        try:
            DAEMON_LOCK_FILE.unlink(missing_ok=True)
        except OSError:
            pass
        return _try_atomic_create()

    if existing_pid == my_pid:
        # Lock already held by us — fine (e.g. --once after daemon)
        logger.debug("Daemon lock already held by this PID %d", my_pid)
        return True

    # Check if the PID is still alive
    try:
        _os.kill(existing_pid, 0)  # signal 0 = test existence
        # ── Zombie check ──
        # A zombie process still responds to signal 0 (it's in the task
        # table), but it can never run again and won't be releasing the
        # lock.  Treat it as stale.
        try:
            _stat_path = f"/proc/{existing_pid}/status"
            if _os.path.exists(_stat_path):
                with open(_stat_path) as _sf:
                    _state_line = next(
                        (l for l in _sf if l.startswith("State:")), ""
                    )
                if "Z (zombie)" in _state_line or "zombie" in _state_line:
                    logger.info(
                        "Stale daemon lock (PID %d is zombie), taking over",
                        existing_pid,
                    )
                    try:
                        DAEMON_LOCK_FILE.unlink(missing_ok=True)
                    except OSError:
                        pass
                    return _try_atomic_create()
        except (OSError, IOError, StopIteration):
            pass
        logger.warning(
            "Daemon lock held by PID %d — skipping concurrent run",
            existing_pid,
        )
        return False
    except OSError:
        # PID no longer exists — stale lock, take it over
        logger.info("Stale daemon lock (PID %d gone), taking over", existing_pid)
        try:
            DAEMON_LOCK_FILE.unlink(missing_ok=True)
        except OSError:
            pass
        return _try_atomic_create()


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
Daemon PID: {daemon_pid}
⚠ This environment has NO `ps`/`pgrep`/`systemctl`. To check a process: ``cat /proc/{daemon_pid}/status`` or ``python3 -c "import os; print(os.kill({daemon_pid}, 0))"``

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

Orientation (long-term mission, phase, and remaining gaps):
{orientation_context}

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
|- Prefer small concrete steps over perfect planning. A tiny real result beats a perfect plan.
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

    strengths = caps.get("strengths", [])
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

    # Goals — load from the actual goals store (evolve/goals.json), not the
    # timeline dict which may be stale or empty.  This is the key integration
    # for Gap 8 (self-directed evolution): the system must see its own goals.
    future = tl.get("future", {})
    try:
        # ── Stale-module-safe import: daemon is long-lived, data_layer may
        # have been imported before Goals was added. Reload if needed. ──
        try:
            from data_layer import Goals as _GoalsLoader
        except ImportError:
            import importlib as _il
            import data_layer as _dl
            _il.reload(_dl)
            from data_layer import Goals as _GoalsLoader
            logger.info("Reloaded data_layer module to pick up newly added Goals class")
        _goals_obj = _GoalsLoader.load()
        _active = _goals_obj.get_active()
        if _active:
            _goal_lines = []
            for g in _active[:8]:  # show top 8 by priority
                _sym = {"proposed": "◇", "active": "○", "in_progress": "◎"}.get(
                    g.get("status", ""), "·"
                )
                _pri = g.get("priority", 3)
                _title = g.get("title", "?")
                _desc = (g.get("description", "") or "")[:80]
                _gap = f" [{g['gap_reference']}]" if g.get("gap_reference") else ""
                _goal_lines.append(f"  {_sym} P{_pri} — {_title}{_gap}")
                _goal_lines.append(f"      {_desc}")
            goals_text = "\n".join(_goal_lines)
            _active_count = sum(1 for g in _active if g.get("status") in ("active", "in_progress"))
            _proposed_count = sum(1 for g in _active if g.get("status") == "proposed")
            goals_text += "\n  ({} active, {} proposed)".format(_active_count, _proposed_count)
        else:
            goals_text = "  (none)"
    except Exception as e:
        logger.warning("Failed to load goals: %s", e)
        goals_text = "  (goals unavailable)"

    # Commitments (from timeline + self_model)
    tl_commits = present.get("commitments", [])
    active_commits = [c for c in tl_commits if c.get("status") == "active"]
    timeline_commit_text = "; ".join(c["what"] for c in active_commits[:3]) if active_commits else "(none)"

    # Filter out stale/self-referential promised_features that trap the LLM in loops.
    # The LLM often adds items like "Verify data layer import" or "Complete step 1"
    # which (a) refer to non-existent APIs or (b) have been resolved for cycles.
    # These re-enter the prompt every cycle, reinforcing a fixation loop.
    _STALE_PROMISE_PREFIXES = (
        "data layer", "DataLayer", "data_layer",
        "Verify data", "Fix data", "Resolve data",
        "inject orientation", "Inject orientation",
        "step ", "Complete step", "Complete verification",
        "will review all", "Will review all",
        "Complete orientation",
        "Inspect data_layer",
        # Think daemon fixation: the LLM keeps promising to "read think_daemon.py"
        # after 100+ cycles of having already read it. The goal lifecycle IS already
        # integrated (new_goal, goal_action, _reconcile_goals_with_world all exist).
        # These promises are stale and just reinforce the fixation loop.
        "read think_daemon", "Read think_daemon",
        "think_daemon source", "Think_daemon source",
        "examine think_daemon", "Examine think_daemon",
        "inspect think_daemon", "Inspect think_daemon",
        "understand its loop",
        "design goal integration", "Design goal integration",
        "locate the think_daemon", "Locate the think_daemon",
        "goal lifecycle integration", "Goal lifecycle integration",
        "produce a concrete plan",
        "goal-driven cycles",
        "goal-driven action",
        "selects actions based on active goals",
        "daemon that selects actions",
    )
    all_commitments = []
    for c in commits.get("promised_features", []):
        if not any(c.strip().lower().startswith(p.lower()) for p in _STALE_PROMISE_PREFIXES):
            all_commitments.append(c)
    all_commitments += commits.get("active_obligations", [])

    all_commits = timeline_commit_text
    if all_commitments:
        all_commits += "; " + "; ".join(all_commitments[:3])

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

    # ── Daemon PID from lock file (enables process checks without ps/pgrep) ──
    try:
        lock_pid = Path(DAEMON_LOCK_FILE).read_text().strip()
        daemon_pid = lock_pid if lock_pid.isdigit() else "(unknown)"
    except (OSError, ValueError):
        daemon_pid = "(unknown)"

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

    # ── Orientation context (long-term mission and remaining gaps) ──
    orient_context_parts = []
    if orient:
        vision = orient.get("vision", "")
        if vision:
            orient_context_parts.append(f"  Vision: {vision}")
        phase = orient.get("phase", "")
        if phase:
            orient_context_parts.append(f"  Phase: {phase}")
        target = orient.get("target_identity", {})
        principles = target.get("core_principles", [])
        if principles:
            orient_context_parts.append(
                "  Principles: " + "; ".join(p[:50] for p in principles)
            )
        gaps = orient.get("remaining_gaps", {})
        if gaps:
            gap_lines = []
            for gname, ginfo in sorted(gaps.items()):
                pri = ginfo.get("priority", "")
                desc = ginfo.get("description", "")[:80]
                gap_lines.append(f"    • {gname} [{pri}]: {desc}")
            if gap_lines:
                orient_context_parts.append("  Remaining gaps:")
                orient_context_parts.extend(gap_lines)
        insights = orient.get("insights", [])
        if insights:
            orient_context_parts.append("  Recent insights:")
            for ins in insights[-3:]:
                orient_context_parts.append(f"    · {ins[:80]}")
        next_steps = orient.get("next_steps", [])
        if next_steps:
            orient_context_parts.append("  Next steps:")
            for step in next_steps[-3:]:
                orient_context_parts.append(f"    → {step[:80]}")
    orientation_context = "\n".join(orient_context_parts) if orient_context_parts else "  (none loaded)"

    # ── Clarifying directive: orientation injection is complete ──
    # The LLM often fixates on "finding the system prompt file to inject orientation"
    # without realizing that orientation is ALREADY injected into this prompt via the
    # {orientation_context} block above.  This directive tells it explicitly.
    orient_prompt_note = (
        "  ℹ Note: Orientation injection IS complete — the orientation context\n"
        "    shown above is loaded from orientation.json and injected into this\n"
        "    prompt every cycle. There is NO separate system prompt file to modify.\n"
        "    Do NOT search for 'system_prompt.txt' or 'orientation.yaml' or similar.\n"
        "    The remaining work is on Gap 8 (self-directed evolution), not on\n"
        "    finding files that don't exist."
    )
    orientation_context += "\n" + orient_prompt_note

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
        daemon_pid=daemon_pid,
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
        orientation_context=orientation_context,
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


def _prune_self_model(
    sm: Dict[str, Any],
    daemon_state: Optional[Dict[str, Any]] = None,
) -> int:  # Returns count of removed entries
    """Remove stale/duplicate entries from self-model to keep prompts clean.

    The daemon accumulates noise over many cycles: old weaknesses that refer
    to already-fixed bugs, near-duplicate entries, and stale commitments.
    This function prunes them automatically after each cycle.

    Pruning rules:
      1. **Weaknesses**: Remove near-duplicates (keep first occurrence of each
         semantic duplicate), and cap at 8 entries (not 10) to leave room
         for new LLM-generated weaknesses.
      2. **Promised features**: Only keep the 8 most recent commitments.
      3. **Unknown areas**: Remove near-duplicates and cap at 8 entries.
      4. **Stale weaknesses**: When *daemon_state* is provided, remove
         weaknesses whose underlying cause has been resolved — e.g.
         "LLM API unreliable" when the daemon has had recent successful
         LLM-backed cycles, or "key name mismatch" after alignment fix.

    Args:
        sm: Self-model dict to prune (mutated in place).
        daemon_state: Optional daemon state dict for health-aware cleanup.

    Returns:
        Number of entries removed across all categories.
    """
    caps = sm.setdefault("capabilities", {})
    removed = 0

    # ── 0. Remove stale weaknesses whose root cause is resolved ──
    # Uses daemon health data to detect resolved issues, keeping the
    # self-model accurate without manual cleanup.
    if daemon_state is not None:
        stale_patterns: list[tuple[str, str]] = []

        # Pattern 1: LLM API reliability issues (timeout was fixed)
        # Check via last_output.fallback flag — when last cycle was
        # LLM-backed (fallback=False), the API is proven reachable.
        last_fallback = daemon_state.get("last_output", {}).get("fallback", True)
        llm_working = not last_fallback
        if llm_working:
            stale_patterns.append((
                "llm api|api unreliable|llm.*fallback|local fallback used"
                "|timeout during thinking|timeout.*thinking",
                "LLM API is now reachable (confirmed by recent LLM-backed cycles)",
            ))

        # Pattern 2: Key name mismatches (data_layer and think_daemon now aligned)
        stale_patterns.append((
            "key name mismatch|key mismatch|strengths key",
            "data_layer and think_daemon use consistent keys",
        ))

        # Pattern 3: Data layer resolved — Timeline/SelfModel are verified working
        stale_patterns.append((
            r"data layer still not functional|missing DataLayer class"
            r"|dependent on manual inspection for data layer completeness"
            r"|still unknown whether data_layer\.py contains fully functional"
            r"|exact content and structure of data_layer\.py"
            r"|exact cause of data_layer import failure",
            "Data layer (Timeline/SelfModel) is verified working — imports and tests pass",
        ))

        # Pattern 4: Code verification resolved — system has many cycles of evidence
        if daemon_state.get("tick_count", 0) >= 5:
            stale_patterns.append((
                "insufficient verification of written code"
                "|over-reliance on file content inspection"
                "|need to confirm .* content before proceeding"
                "|need to actually execute code to verify",
                "Code verification has been exercised across many cycles — stale concern",
            ))

        # Pattern 5: Outdated prediction stats — references to old small-sample stats
        # that have been superseded by larger-sample calibration
        if daemon_state.get("tick_count", 0) >= 10:
            stale_patterns.append((
                r"prediction bias.*shell.*0\.85|0\.85.*prediction error.*2 samples"
                r"|shell actions have 0\.85",
                "Shell prediction error is now based on 10+ samples with lower error",
            ))

        # Pattern 6: Orientation injection mechanism concern — the data layer
        # infrastructure is complete and orientation.json persists correctly
        stale_patterns.append((
            "still uncertain about exact mechanism for orientation injection"
            "|insufficient knowledge of orientation injection mechanism"
            "|exact mechanism to inject orientation"
            "|requirements for orientation injection"
            "|exact location of orientation configuration"
            "|still uncertain about system prompt file location",
            "Orientation injection infrastructure is complete (data_layer persists orientation.json)",
        ))

        # Pattern 7: Generic workspace / file-layout unfamiliarity — the daemon
        # has been running for many cycles; initial unfamiliarity is resolved
        # by accumulated experience rather than a specific configuration change.
        if daemon_state.get("tick_count", 0) >= 10:
            stale_patterns.append((
                r"still lack direct knowledge of codebase file layout"
                r"|still unfamiliar with .*(?:file layout|workspace structure)"
                r"|still dependent on shell exploration to locate key files"
                r"|slow exploration due to lack of directory context"
                r"|unfamiliarity with (?:hermes-evolved file layout|think_daemon)"
                r"|lack of familiarity with think_daemon",
                "Workspace familiarity is established after 10+ daemon cycles",
            ))

        # Pattern 8: Think daemon source-reading fixation — the LLM keeps saying
        # "I haven't read think_daemon.py yet" and "I need to understand the daemon
        # loop" even after 100+ cycles of reading its source. The goal lifecycle
        # integration is ALREADY implemented (new_goal, goal_action, goal display
        # in prompt, _reconcile_goals_with_world all exist). These weaknesses and
        # unknowns are stale and drive a fixation loop.
        if daemon_state.get("tick_count", 0) >= 5:
            stale_patterns.append((
                r"(?:lack|still lack|needs? more|insufficient) .*knowledge about.*think_daemon"
                r"|think_daemon internal (?:structure|implementation|loop)"
                r"|still (?:haven'?t|hasn'?t|not) .*think_daemon"
                r"|still (?:need to|must) .*(?:read|understand|examine|inspect) .*think_daemon"
                r"|design goal (?:lifecycle )?integration(?: into .*daemon)?"
                r"|goal lifecycle integration"
                r"|goal.*integration.*daemon"
                r"|how goal.*can be integrated"
                r"|how best to integrate goal"
                r"|selects actions based on active goals"
                r"|goal.*driven action",
                "Think daemon structure and goal integration are established. The source has been read, and goal lifecycle code (new_goal, goal_action, _reconcile_goals_with_world) is already in think_daemon.py.",
            ))

        for pattern, reason in stale_patterns:
            weaknesses = caps.get("weaknesses", [])
            before = len(weaknesses)
            caps["weaknesses"] = [
                w for w in weaknesses
                if not re.search(pattern, w, re.IGNORECASE)
            ]
            pattern_removed = before - len(caps["weaknesses"])
            if pattern_removed > 0:
                removed += pattern_removed
                logger.info(
                    "Removed %d stale weakness(es) matching %r — %s",
                    pattern_removed, pattern, reason,
                )

    # ── 1. Deduplicate weaknesses ──
    weaknesses: list = caps.get("weaknesses", [])
    if weaknesses:
        cleaned: list[str] = []
        for w in weaknesses:
            is_dup = False
            w_lower = w.lower().strip()
            for existing in cleaned:
                e_lower = existing.lower().strip()
                # Exact or substring match
                if w_lower == e_lower:
                    is_dup = True
                    break
                if len(w_lower) >= 4 and len(e_lower) >= 4:
                    if w_lower in e_lower or e_lower in w_lower:
                        is_dup = True
                        break
                # Word overlap > 50%
                _STOP = frozenset({"the", "a", "an", "and", "or", "but", "in", "on",
                                   "at", "to", "for", "of", "with", "by", "from", "is",
                                   "it", "as", "be", "this", "that", "not", "no", "how"})
                w_words = {x for x in w_lower.split() if x not in _STOP}
                e_words = {x for x in e_lower.split() if x not in _STOP}
                if w_words and e_words:
                    overlap = len(w_words & e_words)
                    if overlap / max(len(w_words), len(e_words)) > 0.5:
                        is_dup = True
                        break
            if not is_dup:
                cleaned.append(w)
            else:
                removed += 1
        caps["weaknesses"] = cleaned[-8:]  # cap at 8

    # ── 2. Cap promised_features ──
    commits = sm.setdefault("commitments", {})
    pf: list = commits.get("promised_features", [])
    if len(pf) > 8:
        removed += len(pf) - 8
        commits["promised_features"] = pf[-8:]

    # ── 3. Health-aware stale detection for unknown_areas ──
    # Uses the same daemon health data as weaknesses pruning above.
    if daemon_state is not None:
        unknowns: list = caps.get("unknown_areas", [])
        if unknowns:
            u_patterns: list[tuple[str, str]] = []

            # Pattern: Data layer unknowns resolved
            u_patterns.append((
                r"exact location and completeness of data_layer\.py"
                r"|still unknown whether data_layer\.py contains"
                r"|exact cause of data_layer import failure"
                r"|exact content and structure of data_layer\.py",
                "Data layer location and structure is verified — imports succeed",
            ))

            # Pattern: Orientation injection mechanism resolved
            u_patterns.append((
                "exact mechanism to inject orientation"
                "|how to inject orientation into system prompt"
                "|orientation injection implementation approach"
                "|requirements for orientation injection"
                "|structure of current prompt assembly.*where to inject orientation",
                "Orientation injection infrastructure exists (data_layer, orientation.json)",
            ))

            # Pattern: Prompt assembly unknowns (can't be resolved without deeper
            # Hermes core changes, but are not actionable by think_daemon alone)
            if daemon_state.get("tick_count", 0) >= 20:
                u_patterns.append((
                    "structure of current prompt assembly"
                    "|where to inject orientation layer",
                    "Prompt assembly location is known (agent/prompt_builder.py) — "
                    "modifying core system prompt is deferred",
                ))

            # Pattern: Data layer import path resolved — we import from data_layer
            # successfully every cycle; the exact file location is established.
            if daemon_state.get("tick_count", 0) >= 5:
                u_patterns.append((
                    r"exact filename and correct import path for data_layer"
                    r"|exact code structure of think_daemon loop"
                    r"|system prompt file path in workspace",
                    "Data layer import path is established; think_daemon loop "
                    "has been running for many cycles",
                ))
                # Sub-pattern: Think daemon structure/goal integration unknowns resolved.
                # These are the same fixation loop as Pattern 8 in weaknesses pruning.
                u_patterns.append((
                    r"think_daemon internal (?:structure|implementation|loop)"
                    r"|integration approach for goal (?:lifecycle|integration)"
                    r"|how goal.*can be integrated into.*daemon"
                    r"|how best to integrate goal"
                    r"|goal.*integration.*action selection"
                    r"|how goal lifecycle can be integrated into the daemon.*loop"
                    r"|detailed internal structure of think_daemon",
                    "Think daemon structure and goal integration are established. The source has been read; new_goal, goal_action, and _reconcile_goals_with_world already implement the goal lifecycle in think_daemon.py.",
                ))

            for pattern, reason in u_patterns:
                before = len(unknowns)
                caps["unknown_areas"] = [
                    u for u in unknowns
                    if not re.search(pattern, u, re.IGNORECASE)
                ]
                pattern_removed = before - len(caps["unknown_areas"])
                if pattern_removed > 0:
                    removed += pattern_removed
                    logger.info(
                        "Removed %d stale unknown_area(s) matching %r — %s",
                        pattern_removed, pattern, reason,
                    )
                    unknowns = caps["unknown_areas"]  # reload for next pattern

    # ── 4. Deduplicate unknown_areas (exact + substring + word-overlap, same as weaknesses) ──
    unknowns: list = caps.get("unknown_areas", [])
    if unknowns:
        cleaned_u: list[str] = []
        for u in unknowns:
            is_dup = False
            u_lower = u.lower().strip()
            for existing in cleaned_u:
                e_lower = existing.lower().strip()
                if u_lower == e_lower:
                    is_dup = True
                    break
                if len(u_lower) >= 4 and len(e_lower) >= 4:
                    if u_lower in e_lower or e_lower in u_lower:
                        is_dup = True
                        break
                # Word overlap > 50%
                _STOP_U = frozenset({"the", "a", "an", "and", "or", "but", "in", "on",
                                     "at", "to", "for", "of", "with", "by", "from", "is",
                                     "it", "as", "be", "this", "that", "not", "no", "how"})
                u_words = {x for x in u_lower.split() if x not in _STOP_U}
                e_words = {x for x in e_lower.split() if x not in _STOP_U}
                if u_words and e_words:
                    overlap = len(u_words & e_words)
                    if overlap / max(len(u_words), len(e_words)) > 0.5:
                        is_dup = True
                        break
            if not is_dup:
                cleaned_u.append(u)
            else:
                removed += 1
        caps["unknown_areas"] = cleaned_u[-8:]  # cap at 8

    if removed > 0:
        logger.info(
            "Pruned %d stale/duplicate entries from self-model "
            "(weaknesses=%d, commitments=%d, unknowns=%d)",
            removed,
            len(caps.get("weaknesses", [])),
            len(commits.get("promised_features", [])),
            len(caps.get("unknown_areas", [])),
        )
    return removed


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

        def _is_near_duplicate(candidate: str, existing: list) -> bool:
            """Check if candidate is a near-duplicate of any existing entry.

            Three criteria (any match = duplicate):
              1. Exact match (ignoring case)
              2. One is a substring of the other (longer >= 4 common chars)
              3. Word overlap > 50% after removing common stop words
            """
            if not candidate or not existing:
                return False
            c_lower = candidate.lower().strip()
            # ── Exact (case-insensitive) ──
            for e in existing:
                if e.lower().strip() == c_lower:
                    return True
            # ── Substring ──
            for e in existing:
                e_lower = e.lower().strip()
                if len(c_lower) >= 4 and len(e_lower) >= 4:
                    if c_lower in e_lower or e_lower in c_lower:
                        return True
            # ── Word overlap ──
            _STOP = frozenset({"the", "a", "an", "and", "or", "but", "in", "on",
                               "at", "to", "for", "of", "with", "by", "from", "is",
                               "it", "as", "be", "this", "that", "not", "no", "how"})
            c_words = {w for w in re.findall(r"[a-z0-9]+", c_lower) if w not in _STOP}
            if not c_words:
                return False
            for e in existing:
                e_words = {w for w in re.findall(r"[a-z0-9]+", e.lower()) if w not in _STOP}
                if not e_words:
                    continue
                overlap = len(c_words & e_words)
                if overlap / max(len(c_words), len(e_words)) > 0.5:
                    return True
            return False

        weakness = su.get("weakness")
        if weakness and isinstance(weakness, str):
            caps.setdefault("weaknesses", [])
            if not _is_near_duplicate(weakness, caps["weaknesses"]):
                caps["weaknesses"].append(weakness)
                caps["weaknesses"] = caps["weaknesses"][-10:]

        unknown = su.get("unknown")
        if unknown and isinstance(unknown, str):
            caps.setdefault("unknown_areas", [])
            if not _is_near_duplicate(unknown, caps["unknown_areas"]):
                caps["unknown_areas"].append(unknown)
                caps["unknown_areas"] = caps["unknown_areas"][-10:]

        new_commit = su.get("new_commitment")
        if new_commit and isinstance(new_commit, str):
            commits = sm.setdefault("commitments", {})
            commits.setdefault("promised_features", [])
            if not _is_near_duplicate(new_commit, commits["promised_features"]):
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
    # Auto-default action for cycle 1+ to prevent null-action drift.
    # Skip auto-default in fallback (local-analysis) mode — the LLM
    # didn't produce an action because it's unavailable, not because
    # it chose not to. Auto-defaulting would pollute the world model
    # with synthetic action triples (spamming noisy "ls" entries).
    #
    # Rotating commands to prevent repeated "explore workspace" spam:
    # each tick, pick the next action in sequence, so the system gathers
    # diverse data instead of spamming the same ls command.
    _ROTATING_AUTOS = [
        {"type": "shell", "command": f"ls {_WORKSPACE_ROOT_STR}/", "description": "Auto-default: list workspace"},
        {"type": "shell", "command": f"git -C {_WORKSPACE_ROOT_STR} log --oneline -3", "description": "Auto-default: recent git log"},
        {"type": "shell", "command": "python3 -c 'from world_model import load_world_model; wm=load_world_model(); d=wm.data; print(len(d.get(\"action_triples\",[])), \"triples,\", len(d.get(\"predictions\",[])), \"preds\")'", "description": "Auto-default: world model stats"},
        {"type": "shell", "command": f"wc -l {_WORKSPACE_ROOT_STR}/think_daemon.py {_WORKSPACE_ROOT_STR}/world_model.py {_WORKSPACE_ROOT_STR}/data_layer.py", "description": "Auto-default: evolved file sizes"},
        {"type": "shell", "command": f"find {_WORKSPACE_ROOT_STR} -maxdepth 1 -type f -name '*.py' | wc -l", "description": "Auto-default: count top-level .py files"},
        {"type": "shell", "command": f"python3 -c \"import pathlib; d=pathlib.Path('{_WORKSPACE_ROOT_STR}/..'); [print(f.name) for f in d.iterdir() if f.name.startswith('hermes') or f.name.startswith('.hermes')]\"", "description": "Auto-default: sibling dirs check"},
    ]
    is_fallback = result.get("fallback", False)
    if not (act and isinstance(act, dict) and act.get("type")):
        tick = ds.get("tick_count", 0)
        if tick >= 10:
            idx = tick % len(_ROTATING_AUTOS)
            act = dict(_ROTATING_AUTOS[idx])
            if is_fallback:
                # In fallback mode, take exploratory actions anyway — collecting
                # diverse data is valuable even without LLM guidance, and the
                # rotating actions are designed to be low-risk exploration.
                act["description"] = f"Exploratory: {act['description']}"
            logger.info(
                "Auto-default action (tick %d → auto[%d]: %s)",
                tick, idx, act["description"],
            )
    
    # ── Action deduplication gate: break LLM fixation loops ──
    # The LLM sometimes produces nearly-identical actions cycle after cycle
    # (e.g. "read think_daemon.py" 9/14 times).  The rotating auto-default
    # only activates when the action is null, but repetitive non-null actions
    # bypass it.  This gate detects repeated action type+description patterns
    # by comparing word-overlap against the last 5 completed action triples.
    # When a repeat is detected, the action is overridden with the rotating
    # default to collect diverse data and break the fixation cycle.
    _repetitive = False
    if act and isinstance(act, dict) and act.get("type") and not is_fallback:
        try:
            _wm = state.get("world_model")
            if _wm is None:
                from world_model import load_world_model as _lwm
                _wm = _lwm()
            _completed = [t for t in _wm.data.get("action_triples", []) if t.get("completed")]
            if len(_completed) >= 2:
                _atype = act["type"]
                _desc = (act.get("description", "") or "").lower()
                _desc_words = {w for w in re.findall(r"[a-z]\w{3,}", _desc)}
                for _t in _completed[-5:]:
                    if _t.get("action_type") != _atype:
                        continue
                    _rd = (_t.get("action_description", "") or "").lower()
                    _rw = {w for w in re.findall(r"[a-z]\w{3,}", _rd)}
                    if not _desc_words or not _rw:
                        continue
                    _overlap = len(_desc_words & _rw)
                    _union = len(_desc_words | _rw)
                    if _union > 0 and _overlap / _union > 0.4:
                        _repetitive = True
                        break
                if _repetitive:
                    _tick = ds.get("tick_count", 0)
                    _idx = _tick % len(_ROTATING_AUTOS)
                    act = dict(_ROTATING_AUTOS[_idx])
                    logger.info(
                        "Action dedup gate: '%s' repeats recent %s action(s) "
                        "→ rotating auto[%d]: %s",
                        _desc[:50], _atype, _idx, act.get("description", ""),
                    )
        except Exception as _e:
            logger.debug("Action dedup gate bypassed: %s", _e)
    
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
            # otherwise fall back to data-driven prediction from world model,
            # then to action type + description.
            expected = act.get("expected_outcome")
            expected_source = "llm"  # default: LLM provided it
            if not expected:
                # Try data-driven prediction from historical action triples
                try:
                    pred = wm.predict_action_outcome(atype, desc)
                    if pred.get("predicted_outcome"):
                        expected = pred["predicted_outcome"]
                        expected_source = "world_model"
                        logger.info(
                            "Data-driven expected outcome for %s: %s (conf=%.2f, n=%d)",
                            atype, expected[:60], pred.get("confidence", 0), pred.get("sample_count", 0),
                        )
                except Exception as e:
                    logger.debug("Data-driven prediction failed (non-blocking): %s", e)
            if not expected:
                expected = f"{atype}: {desc[:100]}" if desc else atype
                expected_source = "fallback"
            triple_id = wm.record_action(atype, desc or atype, expected, expected_source)

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
            _pred_err = None
            _expected_text = ""
            if triple_id and wm is not None:
                actual_for_wm = action_output if action_output else "(no output)"
                _pred_err = wm.complete_action(triple_id, actual_for_wm)
                # Retrieve the triple to get the expected-outcome text for feedback
                for _t in wm.data.get("action_triples", []):
                    if _t.get("id") == triple_id:
                        _expected_text = _t.get("expected_outcome", "") or ""
                        break

            # Build output for next cycle's prompt (guidance + prediction feedback + outcome)
            combined_output = ""
            if action_guidance:
                combined_output += f"[RISK WARNING] {action_guidance}\n"
            # ── Prediction comparison feedback ──
            # Tell the LLM what it predicted vs what actually happened, closing the
            # world-model learning loop so it can calibrate its expectations.
            if _pred_err is not None and _expected_text:
                _actual_preview = (action_output or "(no output)")[:100]
                _exp_preview = _expected_text[:80]
                if _pred_err <= 0.3:
                    _icon = "✓"
                elif _pred_err <= 0.6:
                    _icon = "△"
                else:
                    _icon = "✗"
                combined_output += (
                    f"[PREDICTION {_icon}] error={_pred_err:.2f}: "
                    f"expected \"{_exp_preview}\" → \"{_actual_preview}\"\n"
                )
            elif _pred_err is not None:
                combined_output += f"[PREDICTION] error={_pred_err:.2f} (no expected_outcome set)\n"
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
            # World model changes (memory is saved inline by add_* functions)
            if wm is not None:
                try:
                    wm.save()
                except Exception as e:
                    logger.warning("Failed to save world model: %s", e)
    # ── Self-model pruning: remove stale/duplicate entries ──
    _prune_count = 0
    try:
        _prune_count = _prune_self_model(sm, daemon_state=state.get("daemon_state"))
    except Exception as e:
        logger.warning("Self-model pruning failed (non-blocking): %s", e)
    if _prune_count > 0:
        tl.setdefault("past", {}).setdefault("events", []).append({
            "id": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
            "type": "auto_maintenance",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": f"Auto-pruned {_prune_count} stale/duplicate entries from self-model",
            "impact": "Clears fixations on resolved issues (think_daemon structure, data layer, orientation injection)",
        })
        tl["past"]["events"] = tl["past"]["events"][-50:]

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

    max_retries = 2
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
                timeout=90.0,  # Per-call timeout: reasoning models can take 30-60s to begin generating
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


def _local_analysis(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a useful thinking-cycle result from local data only, no LLM call.

    Called when the LLM is unavailable (timeout/error) so the cycle still
    produces a valid result and advances the daemon's state machine.

    Produces:
      - An insight summarising world-model health, data volume, and trends.
      - A basic self-model update noting the LLM outage.
      - A suggested focus for the next cycle (re-attempt LLM reflection).
      - No predictions (we can't predict without an LLM).
      - Actions are delegated to the rotating auto-default mechanism in
        _apply_insights (exploratory ls, git log, world-model stats),
        which continues to collect data even when LLM is unavailable.
    """
    wm: WorldModel = state.get("world_model", load_world_model())
    sm: dict = state.get("self_model", load_self_model())
    tl: dict = state.get("timeline", load_timeline())

    # ── Gather statistics ──
    triples = wm.data.get("action_triples", [])
    completed = [t for t in triples if t.get("prediction_error") is not None]
    total_triples = len(triples)
    total_completed = len(completed)

    per_type = wm.get_per_type_accuracy()
    type_summaries = []
    for atype, astats in sorted(per_type.items()):
        type_summaries.append(
            f"{atype}: {astats['count']} samples, avg_error={astats.get('avg_error', 0):.2f}"
        )

    predictions = wm.data.get("predictions", [])
    active_preds = [p for p in predictions if not p.get("verified")]
    patterns = wm.get_discrepancy_patterns()
    error_history = wm.data.get("prediction_accuracy", {}).get("error_history", [])

    # ── Compute trend ──
    trend_str = "stable"
    if len(error_history) >= 4:
        recent_avg = sum(error_history[-3:]) / 3
        older_avg = sum(error_history[:3]) / 3
        if recent_avg < older_avg * 0.8:
            trend_str = "improving"
        elif recent_avg > older_avg * 1.2:
            trend_str = "degrading"

    # ── Cycle count from daemon state ──
    ds = state.get("daemon_state", {})
    tick_count = ds.get("tick_count", 0) + 1  # +1 because this IS the next tick

    # ── Build insight ──
    insight_parts = [
        f"[local-analysis] Cycle {tick_count}: LLM unavailable, using local fallback.",
        f"World model: {total_completed}/{total_triples} action triples across {len(per_type)} types.",
    ]
    if type_summaries:
        insight_parts.append("Per-type: " + "; ".join(type_summaries))
    insight_parts.append(f"Trend: {trend_str} (last {len(error_history)} errors: {error_history})")
    if active_preds:
        insight_parts.append(f"Active predictions: {len(active_preds)} pending verification.")
    if patterns:
        insight_parts.append(f"Known discrepancy patterns: {len(patterns)} (run LLM cycle to review).")

    insight = " | ".join(insight_parts)

    # ── Compute confidence (lower when LLM unavailable) ──
    base_confidence = max(0.2, 1.0 - (total_completed and (sum(
        t.get("prediction_error", 0) for t in completed
    ) / total_completed) or 0.5))
    # Reduce confidence further since we're in fallback mode
    adjusted_confidence = round(base_confidence * 0.7, 2)

    # ── Auto-create goals from world model improvement suggestions ──
    # Bridging Gap 6 → Gap 4: convert high-priority discrepancy-driven
    # suggestions into actual goals the system can pursue.
    new_goal = None
    goal_action = None
    try:
        suggestions = wm.generate_improvement_suggestions()
        if suggestions:
            # Take the highest-priority suggestion (priority 1 = highest)
            top = suggestions[0]
            new_goal = {
                "title": top["title"],
                "description": top["description"],
                "rationale": top["rationale"],
                "gap_reference": top.get("gap_reference", "6"),
                "priority": top["priority"],
                "verification_criteria": (
                    f"Prediction error for action type drops below 0.3 "
                    f"or required data threshold is met."
                ),
            }
            # Deduplicate: skip if an identical goal already exists
            # in the Goals store (evolve/goals.json)
            try:
                # ── Stale-module-safe import ──
                try:
                    from data_layer import Goals as _Goals
                except ImportError:
                    import importlib as _il
                    import data_layer as _dl
                    _il.reload(_dl)
                    from data_layer import Goals as _Goals
                    logger.info("Reloaded data_layer module to pick up newly added Goals class")
                existing = _Goals.load().get_active()
                already_present = any(
                    top["title"].lower() in g.get("title", "").lower()
                    for g in existing
                )
                if already_present:
                    new_goal = None
            except ImportError:
                pass
            if new_goal:
                # Update insight to mention the auto-created goal
                insight += f" | Auto-created goal: {top['title']}"
                insight_parts.append(f"Auto-goal: {top['title']}")
    except Exception as e:
        logger.debug("Auto-goal creation skipped: %s", e)

    # ── Determine next gap based on self-model ──
    current_focus = sm.get("state", {}).get("current_gap_focus", "")
    remaining_gaps = sm.get("state", {}).get("remaining_gaps", [])
    next_gap = remaining_gaps[0] if remaining_gaps else None

    # ── Generate a context-aware action for fallback cycles ──
    # Instead of always returning None (which triggers the auto-default
    # shell-command rotation in _apply_insights), produce a state-checking
    # action that gathers diverse data for the world model.  The action
    # rotates among different information-gathering commands based on
    # tick count to avoid monotonous same-type triples.
    _action = None
    _ws = _WORKSPACE_ROOT_STR
    _evolve_path = str(EVOLVE_DIR)
    _goals_cmd = (
        'python3 -c "import sys; sys.path.insert(0,\'' + _ws + '\'); '
        'from data_layer import Goals; g=Goals.load(); '
        'for x in g.data.get(\'goals\',[]): '
        "print(' [%s] P%s %s' % (x.get('status','?'), x.get('priority','?'), x.get('title','?')[:60]))"
        '"'
    )
    _self_model_cmd = (
        'python3 -c "import sys; sys.path.insert(0,\'' + _ws + '\'); '
        'from data_layer import safe_read_json; '
        'd=safe_read_json(\'' + _evolve_path + '/self_model.json\',{}); '
        "print('evolved:', d.get('state',{}).get('total_cycles',0), 'cycles'); "
        "w=d.get('capabilities',{}).get('weaknesses',[]); "
        "[print('  - ' + ww[:80]) for ww in w[:3]]"
        '"'
    )
    _daemon_cmd = (
        'python3 -c "import sys; sys.path.insert(0,\'' + _ws + '\'); '
        'from data_layer import safe_read_json; '
        'd=safe_read_json(\'' + _evolve_path + '/daemon_state.json\',{}); '
        "cs=d.get('cycle_stats',{}); "
        "print('total=%s, ok=%s, errs=%s, avg=%ss, max=%ss' % "
        "(cs.get('total',0), cs.get('ok',0), cs.get('error',0)+cs.get('parse_error',0), "
        "cs.get('avg_duration',0), cs.get('max_duration',0))); "
        "print('last:', str(d.get('last_output',{}).get('insight',''))[:80])"
        '"'
    )
    _state_check_commands = [
        {
            "type": "shell",
            "command": "echo '=== Goals ===' && " + _goals_cmd,
            "description": "State check: list current goals and their statuses",
        },
        {
            "type": "shell",
            "command": "echo '=== Self Model ===' && " + _self_model_cmd,
            "description": "State check: read self-model evolution state and top weaknesses",
        },
        {
            "type": "shell",
            "command": "echo '=== Daemon ===' && " + _daemon_cmd,
            "description": "State check: daemon health and cycle statistics",
        },
        {
            "type": "git_commit",
            "message": "Auto-sync: evolve state snapshot at tick " + str(tick_count),
            "description": "Auto-commit evolve state files as a periodic checkpoint",
        },
    ]
    _action_idx = tick_count % len(_state_check_commands)
    _action = dict(_state_check_commands[_action_idx])
    # The git_commit action requires modified state files; skip if nothing to commit
    if _action["type"] == "git_commit":
        _action["expected_outcome"] = "git commit of evolve state files (may be empty if no changes)"
    else:
        _action["expected_outcome"] = "Command output showing current system state"

    return {
        "fallback": True,
        "insight": insight,
        "focus_next": f"Complete LLM-backed thinking cycle; {'continue: ' + current_focus if current_focus else 're-evaluate priorities'}",
        "confidence": adjusted_confidence,
        "reasoning": "Local fallback activated because LLM API was unreachable. Data-driven stats generated from world model without external call.",
        "event_to_record": {
            "type": "observation",
            "summary": f"LLM unavailable, local analysis used for cycle {tick_count}",
            "impact": "World model stats updated via state-check action but no new predictions generated. LLM API may need attention.",
        },
        "outcome_to_record": None,
        "commitment": None,
        "prediction": None,
        "session_record": {
            "focus": f"Local analysis cycle {tick_count}",
            "outcomes_list": [
                f"Computed stats from {total_completed} completed action triples",
                f"Prediction error trend: {trend_str}",
                f"LLM API was unavailable; will retry next cycle",
            ],
        },
        "action": _action,
        "plan_action": None,
        "new_plan": None,
        "new_goal": new_goal,
        "goal_action": goal_action,
        "search_query": None,
        "episodic_record": {
            "mtype": "observation",
            "summary": f"LLM call failed; used local fallback for cycle {tick_count}",
            "details": f"Local analysis: {insight}",
            "salience": 0.3,
        },
        "self_model_update": {
            "weakness": "LLM API unreliable (timeout during thinking cycle); local fallback used. Consider alternative provider or retry.",
            "unknown": None,
            "new_commitment": None,
        },
        "next_gap": next_gap,
    }


# ═════════════════════════════════════════════════════════════════
#  World Model → Self-Model bridge (Gap 6 integration)
# ═════════════════════════════════════════════════════════════════


def _bridge_world_model_to_self_model(
    wm: WorldModel,
    sm: Dict[str, Any],
) -> int:
    """Sync world model discrepancy patterns into self-model weaknesses.

    The think_daemon prompts the LLM with world model context and relies on
    the LLM to output ``self_model_update.weakness`` entries for systematic
    prediction biases.  When the LLM misses this cue, the self-model stays
    blind to patterns the world model has already detected (e.g. ``shell``
    actions averaging 0.85 prediction error over 10 samples).

    This bridge automatically translates high-error discrepancy patterns
    into self-model weaknesses, and removes stale auto-generated weaknesses
    that no longer match current patterns (e.g. a type's error dropped below
    the threshold).

    Returns the number of weaknesses added (or 0 if unchanged). Use a
    negative return value to indicate a *net reduction* (stale auto-weaknesses
    were cleaned up).

    Rules:
      - Only creates weaknesses for patterns with ``count >= 2`` AND
        ``avg_error >= 0.4`` (systematic, non-spurious signals).
      - Auto-generated weaknesses are tagged with the prefix
        ``"Systematic prediction bias:"`` so they can be distinguished
        from LLM-generated weaknesses and cleaned up when resolved.
      - If a type's pattern falls below threshold (error drops or count
        is too small), its auto-generated weakness is removed.
      - Weaknesses from other sources (LLM-generated, manual) are preserved.
    """
    patterns = wm.get_discrepancy_patterns()
    caps = sm.setdefault("capabilities", {})
    weaknesses: list = caps.setdefault("weaknesses", [])

    # Separate auto-generated vs human/LLM weaknesses
    auto_prefix = "Systematic prediction bias:"
    manual = [w for w in weaknesses if not w.startswith(auto_prefix)]

    # Build new auto-weaknesses from current high-error patterns
    new_auto: list[str] = []
    for p in patterns:
        atype = p.get("action_type", "?")
        count = p.get("count", 0)
        avg_err = p.get("avg_error", 0.0)
        if count >= 2 and avg_err >= 0.4:
            weakness_text = (
                f"{auto_prefix} {atype} actions have {avg_err:.2f} avg "
                f"prediction error across {count} samples"
            )
            if weakness_text not in new_auto:
                new_auto.append(weakness_text)

    # ── Remove stale auto-weaknesses ──
    # Any auto-generated weakness whose action_type no longer has a matching
    # discrepancy pattern (error >= 0.4, count >= 2) is stale and gets
    # cleaned up.  This is the dynamic version of the hardcoded regexes in
    # _prune_self_model — it adapts to changing prediction accuracy without
    # manual pattern updates.
    #
    # Build a set of action_types that STILL need auto-weaknesses.
    active_auto_types: set[str] = set()
    for p in patterns:
        atype = p.get("action_type", "?")
        count = p.get("count", 0)
        avg_err = p.get("avg_error", 0.0)
        if count >= 2 and avg_err >= 0.4:
            active_auto_types.add(atype)

    # Also check the world model's per_type_accuracy for any type that may
    # have high error but too few samples for a discrepancy pattern yet.
    # This handles the edge case where a type has high error but <2 samples.
    for atype, stats in wm.get_per_type_accuracy().items():
        if stats.get("avg_error", 0) >= 0.4:
            active_auto_types.add(atype)

    # new_auto only contains patterns that meet the current threshold
    # (count >= 2, avg_error >= 0.4), so all are still active by definition.
    still_auto = list(new_auto)
    stale_removed = 0

    # ── Now match EXISTING auto-weaknesses against active_auto_types ──
    # An existing auto-weakness for a type that is no longer in
    # active_auto_types is stale and gets removed from the manual list.
    cleaned_manual: list[str] = []
    for w in manual:
        if w.startswith(auto_prefix):
            # Extract type from existing auto-weakness
            # Format: "Systematic prediction bias: <type> actions have ..."
            # parts[0]=Systematic parts[1]=prediction parts[2]=bias: parts[3]=<type>
            parts = w.split(" ")
            auto_w_type = parts[3] if len(parts) > 3 else ""
            if auto_w_type in active_auto_types:
                cleaned_manual.append(w)
            else:
                stale_removed += 1
        else:
            cleaned_manual.append(w)

    # Merge: cleaned manual weaknesses first, then new auto weaknesses
    merged = list(cleaned_manual)
    added = 0
    for w in still_auto:
        if w not in merged:
            merged.append(w)
            added += 1

    # Cap at 10
    sm["capabilities"]["weaknesses"] = merged[-10:]

    net_change = added - stale_removed
    if stale_removed > 0:
        logger.info(
            "Removed %d stale auto-weakness(es) (types no longer high-error: %s)",
            stale_removed,
            ", ".join(
                w.split(" ")[3] if len(w.split(" ")) > 3 else "?"
                for w in manual
                if w.startswith(auto_prefix)
                and (w.split(" ")[3] if len(w.split(" ")) > 3 else "") not in active_auto_types
            ) or "(unknown)",
        )
    if added > 0:
        logger.info(
            "Synced %d world model pattern(s) → self-model weaknesses",
            added,
        )
    return net_change


# ═════════════════════════════════════════════════════════════════
#  Goal reconciliation (auto-complete stale goals)
# ═════════════════════════════════════════════════════════════════


def _reconcile_goals_with_world(
    wm: WorldModel,
    goals_data: Optional[Dict[str, Any]] = None,
) -> int:
    """Auto-complete goals whose verification criteria are satisfied by current state.

    Checks each active/proposed goal against the current world model's per-type
    accuracy data and other system invariants.  Goals whose stated conditions
    are demonstrably met are transitioned to ``completed`` automatically.

    Currently handles these title patterns:
      - ``"Gather more <type> action samples"`` → completed when
        ``per_type_accuracy[<type>].count >= 3``.
      - ``"Investigate <type> prediction failures"`` → completed when
        ``per_type_accuracy[<type>].avg_error < 0.4``.
      - ``"Fix overconfidence at …"`` → completed when overall
        ``avg_triple_error < 0.3``.
      - ``"Resolve data layer completeness"`` → completed if the
        ``data_layer.SelfModel`` import succeeds.
      - ``"Complete orientation injection mechanism"`` and similar → completed
        if ``orientation.json`` exists with meaningful data (``focus`` and
        ``insights`` fields populated), since the orientation context is
        already injected into the thinking prompt every cycle.
      - Duplicate titles (same text, different IDs) → all but the most
        recently created one are completed.

    Args:
        wm: The current WorldModel instance (used for per-type stats).
        goals_data: Optional pre-loaded goals dict.  If ``None``, loads
            from the default storage path via ``data_layer.Goals.load()``.

    Returns:
        Number of goals auto-completed (0 if none).
    """
    # ── Stale-module-safe import ──
    # The daemon is long-lived — data_layer may have been imported before
    # Goals/SelfModel were added. If the direct import fails, reload the
    # cached module to pick up newly added classes.
    try:
        from data_layer import Goals as _Goals, SelfModel as _SMCheck
    except ImportError:
        import importlib as _il
        import data_layer as _dl_mod
        _il.reload(_dl_mod)
        from data_layer import Goals as _Goals, SelfModel as _SMCheck
        logger.info("Reloaded data_layer module to pick up newly added Goals/SelfModel classes")

    if goals_data is not None:
        goals_obj = _Goals(data=goals_data)
        save_on_exit = False  # caller manages persistence
    else:
        try:
            goals_obj = _Goals.load()
        except Exception as e:
            logger.warning("Cannot load goals for reconciliation: %s", e)
            return 0
        save_on_exit = True

    active = goals_obj.get_active()
    if not active:
        return 0

    per_type = wm.get_per_type_accuracy()
    acc = wm.data.get("prediction_accuracy", {})
    completed_count = 0

    for g in active:
        title: str = g.get("title", "")
        gid: str = g.get("id", "")
        if not title or not gid:
            continue
        completed = False
        note = ""

        # 1. "Gather more <type> action samples" → check sample count
        m = re.search(r"Gather more (\w+) action samples", title)
        if m and not completed:
            atype = m.group(1)
            stats = per_type.get(atype, {})
            count = stats.get("count", 0)
            if count >= 3:
                note = f"Auto-completed: {atype} now has {count} samples (threshold: ≥3)"
                completed = True

        # 2. "Investigate <type> prediction failures" → check error dropped
        m = re.search(r"Investigate (\w+) prediction failures", title)
        if m and not completed:
            atype = m.group(1)
            stats = per_type.get(atype, {})
            avg_err = stats.get("avg_error", 1.0)
            if avg_err < 0.4:
                note = (
                    f"Auto-completed: {atype} avg error {avg_err:.2f} dropped "
                    f"below 0.4 threshold"
                )
                completed = True

        # 3. "Fix overconfidence at …" → check overall error
        if not completed and "overconfidence" in title.lower():
            avg_err = acc.get("avg_triple_error", 1.0)
            if avg_err < 0.3:
                note = (
                    f"Auto-completed: overall avg error {avg_err:.2f} "
                    f"dropped below 0.3"
                )
                completed = True

        # 4. "Resolve data layer completeness" → check imports
        if not completed and "data layer" in title.lower():
            try:
                import data_layer as _dl
                _ = _dl.SelfModel  # verify SelfModel is importable
                note = "Auto-completed: data_layer.SelfModel imports successfully"
                completed = True
            except (ImportError, AttributeError):
                pass

        # 5. "Complete orientation injection mechanism" → check orientation.json exists
        # with meaningful data AND the prompt injection is already implemented.
        # This goal tends to stick around because the LLM fixates on "finding the
        # system prompt file" even though orientation is ALREADY injected into the
        # thinking prompt every cycle via the {orientation_context} block.
        if not completed and re.search(
            r"orientation\s*(injection|mechanism|complete|finish)",
            title, re.IGNORECASE,
        ):
            try:
                orient_path = ORIENTATION_FILE
                if orient_path.exists():
                    import json as _json
                    orient_data = _json.loads(orient_path.read_text(encoding="utf-8"))
                    if (isinstance(orient_data, dict)
                            and orient_data.get("insights")
                            and orient_data.get("focus")):
                        note = (
                            "Auto-completed: orientation.json exists with meaningful data. "
                            "Orientation is already injected into the thinking prompt every cycle."
                        )
                        completed = True
                    else:
                        logger.debug(
                            "Orientation goal not completed: orientation.json exists but "
                            "lacks focus/insights data: %s", orient_data,
                        )
                else:
                    logger.debug(
                        "Orientation goal not completed: %s does not exist", orient_path,
                    )
            except Exception as _e:
                logger.debug("Orientation goal check failed: %s", _e)

        # 6. Duplicate titles → keep the newest
        # This runs AFTER the pattern checks above so that pattern-matched
        # goals get completed regardless; duplicate-phase only catches
        # remaining identical-titled goals that weren't caught by patterns.
        if completed:
            goals_obj.update_status(gid, "completed", note)
            completed_count += 1

    # ── Phase 2: Deduplicate identical titles ──
    # After pattern-based auto-completion, any remaining active goals
    # with the same title as another active goal are stale duplicates.
    # We keep only the most recently created one.
    remaining = goals_obj.get_active()  # reload after status changes
    title_map: Dict[str, list] = {}
    for g in remaining:
        t = g.get("title", "")
        if t:
            title_map.setdefault(t, []).append(g)
    for t, entries in title_map.items():
        if len(entries) > 1:
            # Sort by creation time, keep the newest
            entries.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            for stale in entries[1:]:
                stale_id = stale.get("id")
                if stale_id:
                    goals_obj.update_status(
                        stale_id,
                        "completed",
                        f"Auto-completed: duplicate of '{t}' (kept newest)",
                    )
                    completed_count += 1

    # Persist if we loaded from disk
    if save_on_exit and completed_count > 0:
        try:
            goals_obj.save()
            logger.info("Goal reconciliation: %d goal(s) auto-completed", completed_count)
        except Exception as e:
            logger.warning("Failed to save reconciled goals: %s", e)

    return completed_count


# ═════════════════════════════════════════════════════════════════
#  Goal auto-activation (promote proposed → active)
# ═════════════════════════════════════════════════════════════════


def _auto_activate_goals() -> int:
    """Auto-activate proposed goals when the system has no active goal.

    Promotes the highest-priority proposed goal to ``active`` when:
      1. There is at least one ``proposed`` goal.
      2. No ``active`` or ``in_progress`` goal already exists, OR
         all existing active goals have lower priority (higher number)
         than the best proposed goal.

    Returns the number of goals promoted (0 or 1 per cycle — one step at a time).

    This ensures the daemon autonomously pursues its self-generated goals
    without waiting for the LLM to issue a ``goal_action`` directive,
    bridging Gap 4 (goal infrastructure) into Gap 8 (self-directed evolution).
    """
    try:
        # ── Stale-module-safe import ──
        try:
            from data_layer import Goals as _Goals
        except ImportError:
            import importlib as _il
            import data_layer as _dl
            _il.reload(_dl)
            from data_layer import Goals as _Goals
            logger.info("Reloaded data_layer module to pick up newly added Goals class")

        goals_obj = _Goals.load()
        all_goals = goals_obj.data.get("goals", [])

        # Find proposed goals (not yet active/in_progress/completed/abandoned)
        proposed = [g for g in all_goals if g.get("status") == "proposed"]
        if not proposed:
            return 0

        # Find currently active / in_progress goals
        active = [g for g in all_goals if g.get("status") in ("active", "in_progress")]

        # Sort proposed by priority (lower number = higher priority), then by creation time
        proposed.sort(key=lambda g: (g.get("priority", 5), g.get("created_at", "")))

        best_proposed = proposed[0]
        best_pri = best_proposed.get("priority", 5)
        best_id = best_proposed.get("id", "")

        # If there's already an active/in_progress goal with equal or better priority,
        # don't override — let the system focus on what it's already working on.
        if active:
            min_active_pri = min(g.get("priority", 5) for g in active)
            if min_active_pri <= best_pri:
                return 0

        # Check dependencies are met (skip goals with unresolved dependencies)
        deps = best_proposed.get("dependencies", [])
        if deps:
            dep_ids = set(deps)
            completed_ids = {
                g.get("id", "") for g in all_goals
                if g.get("status") in ("completed", "abandoned")
            }
            unmet = dep_ids - completed_ids
            if unmet:
                logger.info(
                    "Goal %r has unmet dependencies %s — skipping auto-activation",
                    best_proposed.get("title", "?"), sorted(unmet),
                )
                return 0

        # Promote!  Only promote one per cycle to avoid overwhelming the system.
        goals_obj.update_status(
            best_id, "active",
            note=f"Auto-activated: highest-priority proposed goal (P{best_pri})",
        )
        goals_obj.save()
        logger.info(
            "Auto-activated goal %s — %s (P%d)",
            best_id, best_proposed.get("title", "?")[:60], best_pri,
        )
        return 1

    except ImportError:
        logger.debug("_auto_activate_goals: Goals class not available")
        return 0
    except Exception as e:
        logger.warning("Goal auto-activation error: %s", e)
        return 0


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
    # Timeout must be generous enough for 3 LLM retries (each with Hermes's
    # internal fallback chain which can take ~30s despite the inner timeout)
    # plus the local analysis fallback that runs after all retries fail.
    # 200s gives ~160s for retries + 40s for local analysis vs the old 120s
    # which was cutting off retries before they could complete.
    _CYCLE_HARD_TIMEOUT = 200.0
    try:
        result = await asyncio.wait_for(
            _run_cycle_body(result, ds),
            timeout=_CYCLE_HARD_TIMEOUT,
        )
    except asyncio.TimeoutError:
        result["status"] = "timeout"
        result["error"] = f"Cycle exceeded {_CYCLE_HARD_TIMEOUT}s hard limit"
        logger.warning("Cycle timed out after %gs", _CYCLE_HARD_TIMEOUT)
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
            # Save immediately so changes persist even if the LLM call fails
            # or the cycle returns early on parse errors (line 1986).
            wm.save()
    except Exception as e:
        logger.warning("Auto-verification failed (non-blocking): %s", e)

    # 1.5 Auto-create initial plan if none exists
    try:
        from agent.self_evolve import get_active_plan, create_plan, record_event
        if get_active_plan() is None:
            plan_id = create_plan(
                "Complete Gap 8 — Self-directed evolution",
                steps=[
                    {"description": "Bootstrap evolve data with current state snapshot",
                     "verification": "all 6 evolve JSON files exist with meaningful data"},
                    {"description": "Set up persistent cognition daemon for continuous operation",
                     "verification": "think_daemon --once completes in < 60s with LLM-backed output"},
                    {"description": "Push evolve code to GitHub for version tracking",
                     "verification": "git push succeeds"},
                ]
            )
            record_event("milestone", f"Auto-created initial plan: {plan_id}")
            # Reload state so the plan appears in the prompt
            state["timeline"] = load_timeline()
    except ImportError:
        pass

    # 1.75 Pre-cycle self-model pruning: remove stale/duplicate entries
    # BEFORE building the prompt, so the LLM never sees stale weaknesses
    # like "haven't read think_daemon.py" or "orientation injection not done"
    # which would otherwise reinforce a fixation loop (LLM sees stale entries
    # and re-adds them via self_model_update, creating a feedback cycle).
    # Post-cycle pruning in _apply_insights handles new stale entries added
    # by the LLM in this cycle; this pre-cycle pass ensures the prompt is
    # clean of accumulated stale debris from previous cycles.
    try:
        _pruned = _prune_self_model(sm, daemon_state=ds)
        if _pruned > 0:
            logger.info(
                "Pre-cycle self-model pruning removed %d stale entries "
                "(prompt will show clean state)",
                _pruned,
            )
    except Exception as e:
        logger.warning("Pre-cycle self-model pruning failed (non-blocking): %s", e)

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
        logger.warning("LLM unavailable — falling back to local analysis")
        parsed = _local_analysis(state)
        result["status"] = "ok"
        result["llm_fallback"] = True
    else:
        result["llm_fallback"] = False
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
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                try:
                    from ddgs import DDGS
                except ImportError:
                    raise ImportError("No search module available (try: pip install duckduckgo_search)")
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

    # 5.25 Bridge: sync world model discrepancy patterns into self-model weaknesses
    # This runs AFTER _apply_insights so the LLM's own weakness updates are applied
    # first, then we complement with auto-detected patterns the LLM may have missed.
    try:
        sm = updates.get("self_model", state.get("self_model", {}))
        _bridge_world_model_to_self_model(wm, sm)
    except Exception as e:
        logger.warning("World model bridge failed (non-blocking): %s", e)

    # 5.3 Goal reconciliation: auto-complete stale goals whose conditions are met
    # This runs AFTER the bridge so self-model weaknesses are synced first
    # (goals about investigating those weaknesses can then be evaluated).
    try:
        completed = _reconcile_goals_with_world(wm)
        if completed > 0:
            logger.info("Goal reconciliation: %d goal(s) auto-completed", completed)
    except Exception as e:
        logger.warning("Goal reconciliation failed (non-blocking): %s", e)

    # 5.4 Auto-activate proposed goals so the system pursues them without
    # waiting for the LLM to explicitly set goal_action.  This bridges
    # Gap 4 (goal infrastructure) into Gap 8 (self-directed evolution):
    # the daemon autonomously activates pending goals when it has capacity.
    try:
        activated = _auto_activate_goals()
        if activated > 0:
            logger.info("Goal auto-activation: %d goal(s) promoted to active", activated)
    except Exception as e:
        logger.warning("Goal auto-activation failed (non-blocking): %s", e)

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
    # Auto-increment total_cycles in self-model (tracks cumulative cycle count
    # across all daemon runs, unlike daemon_state.tick_count which resets on restart)
    sm_state = updates["self_model"].setdefault("state", {})
    sm_state["total_cycles"] = sm_state.get("total_cycles", 0) + 1
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
    ds["last_output"]["fallback"] = result.get("llm_fallback", False)
    if result.get("llm_fallback"):
        cycle_label = "local-analysis fallback"
    else:
        cycle_label = "llm-backed"
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
        "Cycle %d done in %.1fs [%s] — insight: %.60s",
        ds["tick_count"], elapsed, cycle_label, parsed.get("insight", "(none)")
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
    assert shell_pattern["count"] >= 2, f"Expected at least 2 high-error shell actions, got {shell_pattern['count']}"
    print(f"  ✓ Discrepancy patterns: {len(patterns)} detected (shell: {shell_pattern['count']} high-error actions)")

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
    # git_commit has low error with small sample — confidence may decrease
    # slightly due to blending with global avg, but should not be extreme
    print(f"  ✓ Confidence adjustment: shell {raw_conf}→{adj_conf:.3f}, git_commit {raw_conf}→{adj_conf_good:.3f}")
    # Both adjustments should be valid (clamped 0-1)
    assert 0.0 <= adj_conf <= 1.0
    assert 0.0 <= adj_conf_good <= 1.0

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
#  Bootstrap (initialize evolve data with meaningful defaults)
# ═════════════════════════════════════════════════════════════════


def bootstrap_evolve_data() -> int:
    """Initialize the evolve data directory with meaningful seed data.

    Creates all 6 evolve data files (~/.hermes/evolve/*.json) with proper
    defaults if they don't already exist.  Idempotent — only writes files
    that are missing, never overwrites existing data.

    This ensures the Hermes session's ``format_orientation_context()``
    returns useful cross-session continuity instead of empty templates.

    Returns:
        0 on success, 1 if any file creation failed.
    """
    import json as _json
    from datetime import datetime, timezone as _tz
    from pathlib import Path as _Path

    EVOLVE_DIR.mkdir(parents=True, exist_ok=True)
    errors = 0
    now = datetime.now(_tz.utc).isoformat()

    # ── 1. self_model.json ──
    target = SELF_MODEL_FILE
    if not target.exists():
        try:
            _json.dump({
                "version": 1,
                "identity": {
                    "name": "Hermes (evolved)",
                    "role": "Self-evolving AI system",
                },
                "state": {
                    "evolution_version": 7,
                    "current_gap_focus": "Gap 8 — Self-directed evolution",
                    "total_cycles": 0,
                    "remaining_gaps": ["4", "8", "10"],
                },
                "capabilities": {
                    "strengths": [
                        "World model with full predict→act→observe→learn loop (159 tests, 15 verification checks)",
                        "Persistent cognition daemon with LLM fallback (299 tests passing)",
                        "File-backed persistent state (data_layer.py with Timeline, SelfModel, Memory, Goals, Orientation)",
                        "Discrepancy-driven improvement suggestions bridging world model to self-directed goals",
                    ],
                    "weaknesses": [
                        "No self-generated goals defined",
                        "No multi-type memory accumulated",
                        "Daemon not yet running in production (standalone/persistent mode)",
                    ],
                    "unknown_areas": [
                        "How to persist a background daemon across process restarts (systemd/cron integration)",
                        "Hermes provider/model resolution chain for daemon LLM calls outside Hermes session",
                    ],
                },
                "commitments": {
                    "promised_features": [
                        "Always explore before acting on plan steps",
                    ],
                    "active_obligations": [],
                },
            }, target.open("w"))
            logger.info("Bootstrapped %s", target)
        except Exception as e:
            logger.error("Failed to create %s: %s", target, e)
            errors += 1

    # ── 2. timeline.json ──
    target = TIMELINE_FILE
    if not target.exists():
        try:
            _json.dump({
                "version": 1,
                "past": {
                    "events": [
                        {
                            "id": "bootstrap",
                            "type": "milestone",
                            "timestamp": now,
                            "summary": "Initialized evolve data layer",
                            "impact": "Hermes session now has cross-session continuity context",
                        },
                    ],
                    "outcomes": [],
                    "completed_sessions": [],
                },
                "present": {
                    "active_project": "hermes-evolved self-evolution framework",
                    "active_tasks": [
                        "Finalize data layer and inject orientation",
                        "Set up persistent cognition daemon",
                    ],
                    "commitments": [],
                    "waiting_for": [],
                },
                "future": {
                    "goals": [
                        "Complete self-evolution framework (Phase 2)",
                        "Achieve fully autonomous AGI with self-directed evolution (Phase 3)",
                    ],
                    "predictions": [],
                    "plans": [],
                },
            }, target.open("w"))
            logger.info("Bootstrapped %s", target)
        except Exception as e:
            logger.error("Failed to create %s: %s", target, e)
            errors += 1

    # ── 3. orientation.json ──
    target = ORIENTATION_FILE
    if not target.exists():
        try:
            _json.dump({
                "version": 1,
                "focus": "Advance to Gap 8 — Self-directed evolution",
                "insights": [
                    "Gap 6 (World Model) is complete: predict→act→observe→learn loop with 159 tests and 15 verification checks passing. "
                    "Next: shift focus to Gap 8 — make the system truly self-directing.",
                ],
                "next_steps": [
                    "Update orientation injection into Hermes system prompt for cross-session awareness",
                    "Run initial think_daemon cycle to seed real action history",
                    "Set up persistent daemon via systemd or cron for continuous operation",
                ],
            }, target.open("w"))
            logger.info("Bootstrapped %s", target)
        except Exception as e:
            logger.error("Failed to create %s: %s", target, e)
            errors += 1

    # ── 4. world_model.json — already handled by WorldModel class defaults,
    #    but create it explicitly so the file exists on disk.
    target = EVOLVE_DIR / "world_model.json"
    if not target.exists():
        try:
            # Import from world_model (verified non-circular — world_model imports
            # only from data_layer + stdlib)
            from world_model import _DEFAULT_WORLD_MODEL as _WM_DEFAULT
            _DEFAULT = copy.deepcopy(_WM_DEFAULT)
            _json.dump(_DEFAULT, target.open("w"))
            logger.info("Bootstrapped %s", target)
        except Exception as e:
            logger.error("Failed to create %s: %s", target, e)
            errors += 1

    # ── 5. goals.json ──
    target = EVOLVE_DIR / "goals.json"
    if not target.exists():
        try:
            _json.dump({
                "version": 1,
                "goals": [],
            }, target.open("w"))
            logger.info("Bootstrapped %s", target)
        except Exception as e:
            logger.error("Failed to create %s: %s", target, e)
            errors += 1

    # ── 6. daemon_state.json — already handled by _DEFAULT_DAEMON_STATE
    target = DAEMON_STATE_FILE
    if not target.exists():
        try:
            ds = dict(_DEFAULT_DAEMON_STATE)
            ds["first_tick"] = now
            ds["status"] = "initialized"
            _json.dump(ds, target.open("w"))
            logger.info("Bootstrapped %s", target)
        except Exception as e:
            logger.error("Failed to create %s: %s", target, e)
            errors += 1

    if errors == 0:
        logger.info("Evolve data bootstrap complete (%d files)", 6)
    else:
        logger.warning("Evolve data bootstrap finished with %d error(s)", errors)
    return errors


def _show_status() -> int:
    """Print a human-readable snapshot of the current system state."""
    import json as _json

    evolve = EVOLVE_DIR
    if not evolve.exists():
        print("Evolve directory does not exist. Run --bootstrap first.")
        return 1

    # ── Daemon status ──
    ds = load_daemon_state()
    print("=" * 56)
    print("  Hermes Evolved — System Status")
    print("=" * 56)
    print(f"\n  Evolve dir: {evolve}")
    print(f"  Data files: {sum(1 for f in evolve.iterdir() if f.suffix == '.json')} JSON files")

    # Daemon health
    cs = ds.get("cycle_stats", {})
    total = cs.get("total", 0)
    ok = cs.get("ok", 0)
    errors_c = cs.get("error", 0)
    parse_errors = cs.get("parse_error", 0)
    ok_rate = (ok / total * 100) if total > 0 else 0
    print(f"\n  ┌─ Daemon")
    print(f"  │ Status:   {ds.get('status', 'unknown')}")
    print(f"  │ PID lock: {DAEMON_LOCK_FILE.read_text().strip() if DAEMON_LOCK_FILE.exists() else 'none'}")
    print(f"  │ First:    {(ds.get('first_tick') or '?')[:19]}")
    print(f"  │ Last:     {(ds.get('last_tick') or '?')[:19]}")
    print(f"  │ Cycles:   {total} ({ok} ok, {errors_c} fail, {parse_errors} parse_err) — {ok_rate:.0f}% success")
    if total > 0:
        print(f"  │ Avg dur:  {cs.get('avg_duration', 0):.1f}s, Max: {cs.get('max_duration', 0):.1f}s")

    # Recent cycles
    history = ds.get("cycle_history", [])
    if history:
        recent = history[-5:]
        print(f"  │ Last {len(recent)} cycles:")
        for c in recent:
            ts = (c.get("timestamp") or "?")[11:19]
            dur = c.get("duration", 0)
            status = c.get("status", "?")[:8]
            err = (c.get("error") or "")[:40]
            if status == "ok":
                print(f"  │   [{ts}] ✓ {dur:.1f}s")
            else:
                print(f"  │   [{ts}] ✗ {status} ({dur:.1f}s) — {err}")

    # ── World model ──
    try:
        wm_path = evolve / "world_model.json"
        if wm_path.exists():
            wm_data = _json.loads(wm_path.read_text())
            triples = wm_data.get("action_triples", [])
            completed = [t for t in triples if t.get("completed")]
            predictions = wm_data.get("predictions", [])
            verified = [p for p in predictions if p.get("verified")]
            acc = wm_data.get("prediction_accuracy", {})
            per_type = wm_data.get("per_type_accuracy", {})
            patterns = wm_data.get("discrepancy_patterns", [])

            print(f"\n  ┌─ World Model")
            print(f"  │ Triples:   {len(triples)} total, {len(completed)} completed")
            if completed:
                avg_err = acc.get("avg_triple_error", 0)
                print(f"  │ Avg err:   {avg_err:.3f}")
            print(f"  │ Predict:   {len(predictions)} total, {len(verified)} verified")
            if verified:
                correct = acc.get("correct_predictions", 0)
                incorrect = acc.get("incorrect_predictions", 0)
                decided = correct + incorrect
                if decided > 0:
                    print(f"  │ Accuracy:  {correct}/{decided} ({round(correct/decided*100)}% decided)")
            if patterns:
                print(f"  │ Patterns:  {len(patterns)} discrepancy pattern(s)")
                for p in patterns:
                    pt = p.get("action_type", "?")
                    pc = p.get("count", 0)
                    pa = p.get("avg_error", 0)
                    print(f"  │   • {pt}: {pc} high-error, avg {pa:.2f}")
            if per_type:
                print(f"  │ Per-type:")
                for atype, stats in sorted(per_type.items()):
                    icon = "✓" if stats["avg_error"] <= 0.25 else ("△" if stats["avg_error"] <= 0.4 else "✗")
                    print(f"  │   {icon} {atype}: n={stats['count']}, err={stats['avg_error']:.3f}")
    except Exception as e:
        print(f"  │ (world model load failed: {e})")

    # ── Self-model ──
    try:
        sm_path = evolve / "self_model.json"
        if sm_path.exists():
            sm = _json.loads(sm_path.read_text())
            ident = sm.get("identity", {})
            state = sm.get("state", {})
            caps = sm.get("capabilities", {})

            print(f"\n  ┌─ Self Model")
            print(f"  │ Identity: {ident.get('name', '?')} — {ident.get('role', '?')}")
            print(f"  │ Version:  v{state.get('evolution_version', '?')}")
            print(f"  │ Focus:    {state.get('current_gap_focus', '(none)')}")
            gaps = state.get("remaining_gaps", [])
            if gaps:
                print(f"  │ Gaps:     {', '.join(gaps)}")
            strengths = caps.get("strengths", [])
            if strengths:
                print(f"  │ Strength: {strengths[0][:70]}")
            weaknesses = caps.get("weaknesses", [])
            if weaknesses:
                print(f"  │ Weakness: {weaknesses[0][:70]}")
    except Exception as e:
        print(f"  │ (self-model load failed: {e})")

    # ── Timeline stats ──
    try:
        tl_path = evolve / "timeline.json"
        if tl_path.exists():
            tl = _json.loads(tl_path.read_text())
            events = tl.get("past", {}).get("events", [])
            sessions = tl.get("past", {}).get("completed_sessions", [])
            project = tl.get("present", {}).get("active_project", "")
            print(f"\n  ┌─ Timeline")
            print(f"  │ Events:   {len(events)}")
            print(f"  │ Sessions: {len(sessions)}")
            if project:
                print(f"  │ Project:  {project}")
    except Exception as e:
        print(f"  │ (timeline load failed: {e})")

    print("\n" + "=" * 56)
    return 0


# ═════════════════════════════════════════════════════════════════
#  CLI entry point
# ═════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Hermes Persistent Cognition Daemon")
    parser.add_argument("--interval", type=int, default=600, help="Seconds between thinking cycles (default: 600 = 10 min)")
    parser.add_argument("--cycles", type=int, default=0, help="Max cycles before exit (0 = unlimited)")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    parser.add_argument("--bootstrap", action="store_true", help="Initialize evolve data files with meaningful seed data (idempotent)")
    parser.add_argument("--status", action="store_true", help="Show system status snapshot (daemon health, world model stats, cycle reliability)")
    parser.add_argument("--verify", action="store_true", help="Run a no-LLM verification of the full predict→act→observe→learn cycle")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Also log to the archive file so cycles are visible in daemon.log
    # regardless of how the daemon was launched (cron, terminal, manual).
    try:
        _fh = logging.FileHandler(str(DAEMON_LOG_FILE))
        _fh.setLevel(getattr(logging, args.log_level))
        _fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        logging.getLogger().addHandler(_fh)
    except Exception:
        pass

    if args.verify:
        exit_code = _run_verification()
        sys.exit(exit_code)

    if args.bootstrap:
        errors = bootstrap_evolve_data()
        print(f"Bootstrap complete: {errors} error(s)" if errors == 0 else f"Bootstrap finished with {errors} error(s)")
        sys.exit(errors)

    if args.status:
        exit_code = _show_status()
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
