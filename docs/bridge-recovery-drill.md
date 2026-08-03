# Bridge Recovery Drill — Safe DOWN Validation Runbook

**Goal:** `goal_20260803075438_0` — "Safe validation of recovery automation
without live-bridge disruption".
**Status:** DRILL EXECUTED 2026-08-03 (safe simulation; live bridge never
disrupted) + runbook committed.
**Owner:** think_daemon self-directed evolution (Gap 8 / Gap 10 hardening).

---

## 1. Purpose

The host bridge (127.0.0.1:8791) is the daemon's only path to real external
actions (`api_call`). If it dies, every api_call fails until someone or
something restarts it. `scripts/bridge_healthcheck.py` + `evolve_daemon.sh
bridge restart` already automate recovery — this drill **validates that
recovery path safely and repeatably**, without ever killing the healthy live
bridge.

**Safety principle (hard rule):** never validate recovery by stopping the
real bridge. All DOWN states in this drill are *simulated* on an ephemeral
port with a throwaway fake bridge. The real bridge is only ever **probed**
(read-only GET /bridge/v1/health).

## 2. Inventory (located components)

| Component | Path | Role |
|---|---|---|
| Launcher | `evolve_daemon.sh` | start/status/stop/restart daemon; `bridge {start\|stop\|status\|restart\|healthcheck}`; `permissions` |
| Health check | `scripts/bridge_healthcheck.py` | probe → restart → re-probe; `--check-only` = no side effects; exit 0 = UP, 1 = DOWN |
| Drill simulator | `scripts/bridge_drill_sim.py` | throwaway fake bridge (same health contract) + `--start-bg`/`--stop`; used as simulated `--restart-cmd` |
| Read-gate heartbeat | `scripts/cron_fire_read_gate.py` | per-cycle allowlisted GET `https://api.github.com/` through the bridge (committed steady-state action) |
| Real bridge | `127.0.0.1:8791` (`evolve_bridge.py`) | validates token → allowlist → permission registry; audits every decision to `<evolve_dir>/bridge_audit.log` |
| Runtime state | `<evolve_dir>/` | `bridge.pid`, `bridge_token`, `bridge.log`, `bridge_audit.log`, `daemon.lock` |

## 3. Safe mechanism (how DOWN is simulated)

`scripts/bridge_drill_sim.py` serves the identical health contract as the
real bridge (`GET /bridge/v1/health` → `200 {"ok": true, "pid": N}`) on an
ephemeral port. A drill creates a simulated outage by stopping the fake
bridge, then validates recovery by pointing `bridge_healthcheck.py` at the
fake port with the fake bridge's `--start-bg` as the `--restart-cmd`. The
healthcheck's code path (probe → DOWN → restart → re-probe → RECOVERED) is
exercised end-to-end; the real bridge on 8791 is never stopped, restarted,
or written to.

## 4. Drill procedure (repeatable)

```bash
cd /workspace/hermes-evolved
PORT=59998   # ephemeral port; real bridge stays on 8791
SIM=scripts/bridge_drill_sim.py

# Phase 1 — baseline: real bridge must be UP (read-only probe)
.venv/bin/python3 scripts/bridge_healthcheck.py --check-only        # expect exit 0, "UP"

# Phase 2 — simulated outage: fake bridge up → stopped → probe DOWN
.venv/bin/python3 $SIM --start-bg --port $PORT
.venv/bin/python3 scripts/bridge_healthcheck.py --check-only \
    --bridge-url http://127.0.0.1:$PORT                              # expect exit 0, "UP"
.venv/bin/python3 $SIM --stop --port $PORT
.venv/bin/python3 scripts/bridge_healthcheck.py --check-only \
    --bridge-url http://127.0.0.1:$PORT                              # expect exit 1, "DOWN"

# Phase 3 — recovery loop: healthcheck restarts the simulated bridge
.venv/bin/python3 scripts/bridge_healthcheck.py \
    --bridge-url http://127.0.0.1:$PORT \
    --restart-cmd "/workspace/hermes-evolved/$SIM --start-bg --port $PORT" \
    --retries 8 --wait 1                                             # expect exit 0, "RECOVERED"

# Phase 4 — cleanup + real bridge untouched
.venv/bin/python3 $SIM --stop --port $PORT
.venv/bin/python3 scripts/bridge_healthcheck.py --check-only        # expect exit 0, "UP"
```

## 5. Execution record — 2026-08-03 (this drill)

| Phase | Expected | Observed | Pass |
|---|---|---|---|
| 1. Real bridge baseline | exit 0, UP | `UP (HTTP 200, pid 489377)` | ✅ |
| 2a. Fake bridge up | exit 0, UP | `UP (HTTP 200, pid 491996)` | ✅ |
| 2b. Fake bridge stopped | exit 1, DOWN | `DOWN ([Errno 111] Connection refused)`, check-only | ✅ |
| 3. Recovery loop | exit 0, RECOVERED | DOWN → restart exit 0 → `re-probe 1/8: UP` → `bridge RECOVERED after restart` (pid 492017) | ✅ |
| 4. Cleanup + real bridge | exit 0, UP | fake stopped; real `UP (HTTP 200, pid 489377)` — same PID as baseline, never touched | ✅ |

**Finding fixed during the drill:** a direct `--restart-cmd` invoking a
script path requires the executable bit (`[Errno 13] Permission denied` on
the first Phase 3 attempt). `scripts/bridge_drill_sim.py` and
`scripts/bridge_healthcheck.py` were made executable; `evolve_daemon.sh`
was already executable (711). The real recovery path was verified live on
2026-08-03 (goal_20260802151450_3): bridge DOWN → `evolve_daemon.sh bridge
restart` → following GET completed HTTP 200.

## 6. Recovery procedure (real bridge DOWN — the actual runbook)

1. **Detect:** `scripts/bridge_healthcheck.py --check-only`
   (or `evolve_daemon.sh bridge healthcheck --check-only`) → exit 1 = DOWN.
2. **Restore:** `scripts/bridge_healthcheck.py` (no flags) — probes, restarts
   via `evolve_daemon.sh bridge restart`, re-probes up to 10×. Exit 0 =
   recovered. This is the single shell action goal 20 required.
3. **Escalate if still DOWN:** inspect `<evolve_dir>/bridge.log` /
   `bridge_launcher.log`; check the token file exists and matches
   `HERMES_EVOLVED_BRIDGE_TOKEN`; try `evolve_daemon.sh bridge restart`
   manually; verify port 8791 is free (`ss -ltn` / `/proc/net/tcp`).
4. **Confirm:** allowlisted GET completes: run
   `scripts/cron_fire_read_gate.py` (or wait one daemon cycle) and check the
   world-model triple records `exit=0`, HTTP 200.

## 7. Verification checklist

- [x] Recovery validated against a simulated DOWN, never by stopping the
      live bridge (real bridge PID identical before/after the drill).
- [x] Same healthcheck code path (probe → restart → re-probe) is exercised
      in the drill and in production recovery.
- [x] Automated regression coverage: `tests/test_bridge_drill_sim.py` (3),
      `tests/test_bridge_healthcheck.py` (5) — green via `scripts/run_tests.sh`.
- [x] Read-gate steady state sustained during the drill:
      `act_20260803111250_1` → exit=0, HTTP 200, err 0.15.
- [x] Runbook committed (this file).
