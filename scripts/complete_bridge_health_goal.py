#!/usr/bin/env python3
"""Cron-cycle slice: complete the bridge health-check/restart goal.

Step 2 (verify restart-under-DOWN end-to-end) and step 3 (fold into the
launcher) were validated LIVE this cycle:

- Bridge stopped -> scripts/bridge_healthcheck.py detected DOWN, ran the
  restart (evolve_daemon.sh bridge restart), re-probed UP (new PID) ->
  following allowlisted GET https://api.github.com/ completed HTTP 200.
- Restart-still-down path returns exit 1 (live, dead port + no-op cmd).
- evolve_daemon.sh bridge healthcheck [--check-only] entry point added and
  exercised (UP, exit 0); documented in docs/gap10-bridge-design.md.
- Tests: tests/test_bridge_healthcheck.py (5), tests/test_evolve_bridge.py
  (22), daemon bridge/self-heal slice (22) all green via run_tests.sh.

Uses the same data_layer APIs the daemon uses so the goal store stays
consistent with daemon-managed goals.
"""
import sys

sys.path.insert(0, "/workspace/hermes-evolved")

from data_layer import (  # noqa: E402
    Goals,
    complete_plan,
    record_event,
    update_plan_step,
)

GOAL_ID = "goal_20260802151450_3"
PLAN_ID = "plan_20260803045429"

# Step 2: verify recovery under DOWN (live, end-to-end)
ok2 = update_plan_step(
    PLAN_ID,
    "step_2",
    "complete",
    note="LIVE 2026-08-03: bridge stopped -> script detected DOWN, restarted via "
    "evolve_daemon.sh, re-probed UP (pid 465976) -> following GET api.github.com "
    "HTTP 200 (audit 05:17:09Z). Still-down path returns exit 1 (live, dead port). "
    "test_restart_recovery_end_to_end + test_restart_still_down_after_restart green.",
)

# Step 3: fold into the launcher + document
ok3 = update_plan_step(
    PLAN_ID,
    "step_3",
    "complete",
    note="evolve_daemon.sh bridge healthcheck [--check-only] added (delegates to "
    "scripts/bridge_healthcheck.py); exercised live (UP, exit 0). Documented in "
    "docs/gap10-bridge-design.md 'Bridge health-check/restart automation' section.",
)

plan_done = complete_plan(PLAN_ID, "complete")

goals = Goals.load()
okg = goals.update_status(
    GOAL_ID,
    "completed",
    note="Verification criteria met 2026-08-03 (live): bridge DOWN -> one shell "
    "action (scripts/bridge_healthcheck.py, also evolve_daemon.sh bridge "
    "healthcheck) located and restarted it -> following cycle's GET "
    "https://api.github.com/ api_call completed HTTP 200. Still-down path "
    "exits 1. 49 targeted tests green (5+22+22).",
)
goals.save()

record_event(
    "goal_completed",
    "Goal goal_20260802151450_3 'Automate host bridge health check and restart' "
    "COMPLETE: live DOWN->restart->UP->GET 200 validated; launcher entry point "
    "evolve_daemon.sh bridge healthcheck added + documented.",
    "Bridge DOWN no longer silently stalls api_call: one shell action recovers it.",
)

print(f"step_2 complete: {ok2}")
print(f"step_3 complete: {ok3}")
print(f"plan complete: {plan_done}")
print(f"goal complete: {okg}")
