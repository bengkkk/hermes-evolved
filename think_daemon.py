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
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# ── Ensure Hermes modules are importable ──
_HERMES_ROOT = Path(__file__).resolve().parent
if str(_HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(_HERMES_ROOT))

logger = logging.getLogger("think_daemon")

# ── Paths ──
EVOLVE_DIR = (
    Path(os.environ.get("HERMES_HOME", "~/.hermes"))
    if os.environ.get("HERMES_HOME")
    else Path.home() / ".hermes" / "evolve"
)
if not os.environ.get("HERMES_HOME"):
    EVOLVE_DIR = Path.home() / ".hermes" / "evolve"
else:
    EVOLVE_DIR = Path(os.environ["HERMES_HOME"]) / "evolve"

TIMELINE_FILE = EVOLVE_DIR / "timeline.json"
SELF_MODEL_FILE = EVOLVE_DIR / "self_model.json"
ORIENTATION_FILE = EVOLVE_DIR / "orientation.json"
DAEMON_STATE_FILE = EVOLVE_DIR / "daemon_state.json"
DAEMON_LOG_FILE = EVOLVE_DIR / "daemon.log"

# ── Default state ──
_DEFAULT_DAEMON_STATE: Dict[str, Any] = {
    "version": 1,
    "status": "initialized",
    "first_tick": None,
    "last_tick": None,
    "tick_count": 0,
    "interval_seconds": 600,
    "last_output": None,
}


# ═════════════════════════════════════════════════════════════════
#  State helpers
# ═════════════════════════════════════════════════════════════════

def _load_json(path: Path, default: Any) -> Any:
    """Load a JSON file, returning default on failure."""
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("Could not load %s: %s", path.name, e)
    return default


def _save_json(path: Path, data: Any) -> None:
    """Atomically write a JSON file."""
    try:
        EVOLVE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        logger.warning("Could not write %s: %s", path.name, e)


def load_daemon_state() -> Dict[str, Any]:
    return _load_json(DAEMON_STATE_FILE, dict(_DEFAULT_DAEMON_STATE))


def save_daemon_state(state: Dict[str, Any]) -> None:
    _save_json(DAEMON_STATE_FILE, state)


def load_timeline() -> Dict[str, Any]:
    return _load_json(TIMELINE_FILE, {
        "version": 1, "past": {"events": [], "completed_sessions": []},
        "present": {}, "future": {"goals": []},
    })


def save_timeline(data: Dict[str, Any]) -> None:
    _save_json(TIMELINE_FILE, data)


def load_self_model() -> Dict[str, Any]:
    return _load_json(SELF_MODEL_FILE, {
        "version": 1, "identity": {"name": "Hermes (evolved)"},
        "state": {}, "capabilities": {}, "commitments": {},
    })


def save_self_model(data: Dict[str, Any]) -> None:
    _save_json(SELF_MODEL_FILE, data)


def load_orientation() -> Optional[Dict[str, Any]]:
    return _load_json(ORIENTATION_FILE, None)


def save_orientation(data: Dict[str, Any]) -> None:
    _save_json(ORIENTATION_FILE, data)


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
- Prioritize: what unblocks the most other capabilities?

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
    ]]
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
  "next_gap": "4, 6, 8, or 10 — which gap to tackle next (or null). Based on system state analysis.", — which gap to tackle next (or null). Based on system state analysis.",
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
        goals_text=goals_text,
    )


# ═════════════════════════════════════════════════════════════════
#  Response parser
# ═════════════════════════════════════════════════════════════════

def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM response, handling markdown fences."""
    # Remove markdown code fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Find the first { or [
        start = cleaned.find("{")
        if start >= 0:
            cleaned = cleaned[start:]
        # Remove trailing ```
        end = cleaned.rfind("}")
        if end >= 0:
            cleaned = cleaned[: end + 1]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def _apply_insights(result: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Apply parsed insights to evolve state, returning updated state."""
    tl = state.get("timeline", load_timeline())
    sm = state.get("self_model", load_self_model())
    orient = state.get("orientation", load_orientation())

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

async def _call_llm(messages: list, task: str = "thinking") -> Optional[str]:
    """Call LLM via Hermes auxiliary_client, return response text or None.

    Uses the same provider chain as the main Hermes session.
    """
    try:
        from agent.auxiliary_client import async_call_llm
    except ImportError:
        logger.error("Could not import Hermes auxiliary_client. Is HERMES_ROOT correct?")
        return None

    try:
        response = await async_call_llm(
            task=task,
            messages=messages,
            temperature=0.3,
            max_tokens=512,
        )
        # response is an OpenAI-style response object
        if hasattr(response, "choices") and response.choices:
            return response.choices[0].message.content
        # Fallback: dict-style
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
        logger.warning("Unexpected response shape from async_call_llm: %s", type(response).__name__)
        return None
    except Exception as e:
        logger.warning("LLM call failed: %s", e)
        return None


# ═════════════════════════════════════════════════════════════════
#  Main thinking cycle
# ═════════════════════════════════════════════════════════════════

async def run_one_cycle() -> Dict[str, Any]:
    """Run a single thinking cycle, returning a result dict with status and any updates."""
    start_time = time.time()
    result = {
        "status": "ok",
        "tick_duration": 0,
        "insight": None,
        "error": None,
    }

    # 1. Load current state
    ds = load_daemon_state()
    tl = load_timeline()
    sm = load_self_model()
    orient = load_orientation()

    state = {
        "daemon_state": ds,
        "timeline": tl,
        "self_model": sm,
        "orientation": orient,
    }

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

    # 6. Save updated state
    save_timeline(updates["timeline"])
    save_self_model(updates["self_model"])
    if updates["orientation"]:
        save_orientation(updates["orientation"])

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
    elapsed = time.time() - start_time
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


# ═════════════════════════════════════════════════════════════════
#  CLI entry point
# ═════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Hermes Persistent Cognition Daemon")
    parser.add_argument("--interval", type=int, default=600, help="Seconds between thinking cycles (default: 600 = 10 min)")
    parser.add_argument("--cycles", type=int, default=0, help="Max cycles before exit (0 = unlimited)")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.once:
        r = asyncio.run(run_one_cycle())
        # Print one-line summary for cron delivery
        status = r.get("status", "error")
        if status == "ok":
            insight = r.get("insight", "")[:80]
            print(f"[{status}] tick {r.get('tick_duration',0):.1f}s — {insight}")
        else:
            print(f"[{status}] {r.get('error', 'unknown error')}")
    else:
        asyncio.run(run_daemon(args.interval, args.cycles))


if __name__ == "__main__":
    main()
