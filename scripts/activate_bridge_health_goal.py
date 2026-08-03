#!/usr/bin/env python3
"""Cron-cycle slice: activate the bridge health-check/restart goal and
create a bounded plan for it, marking the committed script step complete.

Uses the same data_layer APIs the daemon uses (Goals.update_status,
create_plan, update_plan_step) so the goal store stays consistent.
"""
import sys

sys.path.insert(0, "/workspace/hermes-evolved")

from data_layer import Goals, create_plan, update_plan_step  # noqa: E402

GOAL_ID = "goal_20260802151450_3"

goals = Goals.load()
ok = goals.update_status(
    GOAL_ID,
    "in_progress",
    note="Activated by cron cycle 2026-08-03: standalone health-check/restart "
    "script committed (scripts/bridge_healthcheck.py); daemon in-process "
    "self-heal already landed; remaining steps verify restart-under-DOWN "
    "behavior end-to-end and fold the script into the launcher.",
)
goals.save()
print(f"goal {GOAL_ID} -> in_progress: {ok}")

plan_id = create_plan(
    "Automate host bridge health check and restart",
    steps=[
        {
            "id": "step_1",
            "description": "Commit standalone bridge health-check/restart script "
            "(scripts/bridge_healthcheck.py) with probe -> restart -> re-probe and "
            "deterministic exit codes",
            "verification": "script exists in repo; tests/test_bridge_healthcheck.py "
            "passes (5 tests); live --check-only reports UP with exit 0",
            "status": "pending",
            "assigned_to": "daemon",
        },
        {
            "id": "step_2",
            "description": "Verify the script recovers a DOWN bridge end-to-end "
            "(fake-bridge restart test passes; audit + re-probe path exercised)",
            "verification": "test_restart_recovery_end_to_end green; restart-still-down "
            "path returns exit 1",
            "status": "pending",
            "assigned_to": "daemon",
        },
        {
            "id": "step_3",
            "description": "Fold the script into the launcher so a DOWN bridge is "
            "recovered before daemon start and on demand via evolve_daemon.sh",
            "verification": "evolve_daemon.sh exposes a health-check entry point that "
            "calls scripts/bridge_healthcheck.py; documented in gap10-bridge-design.md",
            "status": "pending",
            "assigned_to": "daemon",
        },
    ],
)
print(f"plan {plan_id} created")

# The script IS committed in this cycle — mark step 1 complete now.
ok1 = update_plan_step(
    plan_id,
    "step_1",
    "complete",
    note="scripts/bridge_healthcheck.py committed (cron cycle 2026-08-03); "
    "5 tests green; live check-only probe returned UP (HTTP 200, pid 459020), exit 0",
)
print(f"step_1 complete: {ok1}")
