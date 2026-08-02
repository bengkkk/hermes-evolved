# Gap 10 Level 2 — Real Action Bridge: Low-Risk Writes

Status: DRAFT (awaiting github.write grant)
Owner: Hermes (evolved)
Created: 2026-08-02

## Objective
Extend the real action bridge from read-only calibration (Level 1) to low-risk write operations on GitHub via the host bridge. The first write triple will create this document through the Contents API (PUT), demonstrating method-aware pre-flight and write allowlist enforcement.

## Level 1 Recap
- Allowlisted endpoint: GET https://api.github.com/
- Calibration: 3/3 api_call triples, avg prediction error 0.15
- Evidence committed in evidence/gap10-level1.md

## Level 2 Design

### Permissions
- Current: github: read (deny-by-default)
- Required: github: write (user grant pending)
- No GitHub write API call will be attempted until github.write is granted.

### Method-aware pre-flight
For any proposed write action:
1. Resolve resource from endpoint (e.g., github)
2. Check the PERMISSIONS registry for a grant matching the required method type (read vs write) for that resource
3. Check endpoint + method against the WRITE_ALLOWLIST
4. Validate request body (path, message, base64-encoded content)
5. Dispatch via the host bridge only if all checks pass

### Planned write allowlist entries
- PUT /repos/{owner}/{repo}/contents/docs/gap10-level2-plan.md — first write triple
- PUT /repos/{owner}/{repo}/contents/evidence/gap10-level1.md — evidence commit

### Verification
- First write returns HTTP 201 with content SHA
- World model logs a write_file/api_call triple with prediction error < 0.25
- File appears in the repository

## Next Actions
1. Draft this plan (local write) — DONE
2. Persist Level 1 calibration evidence if not already present — DONE (evidence/gap10-level1.md)
3. Implement method-aware pre-flight + write allowlist — DONE (commits 0de506ab7, facbda171; verified below)
4. Await github.write grant
5. Fire the first write triple (PUT this plan via Contents API) and verify HTTP 201 + prediction error < 0.25

## Progress (2026-08-02)
- The write path is ARMED and deny-until-granted: `BRIDGE_WRITE_ALLOWLIST`
  (evolve_bridge.py) and `_API_WRITE_ALLOWLIST` (think_daemon.py) now carry
  the two planned Contents API PUT entries (exact-path match, so an entry
  can never widen to sibling files), pre-flight is method-aware in BOTH
  components (GET/HEAD → `read`, PUT/POST/PATCH/DELETE → `write`), write
  bodies are validated (message + base64 content, optional sha) before
  anything reaches the wire, and `_github_put` is ready on the bridge.
  Nothing executes until the user issues `evolve_permissions.py grant
  github write` — the Level 2 gate is the permission registry, unchanged.
- Verification (2026-08-02): 28/28 evolve bridge/permission tests pass
  (tests/test_evolve_bridge.py 16✓, tests/test_evolve_permissions.py 12✓).
  Live check against the running bridge (restarted 01:51 with the new
  code, PID 332294): PUT to the exact allowlisted plan.md endpoint returns
  HTTP 403 "BLOCKED: permission denied: no write grant for resource
  'github'" — the write-allowlist entry matched (layer 2) and the
  method-aware permission gate denied (layer 3), with no outbound call.
- Remaining: grant → first write triple (PUT this plan through the
  Contents API) → verify HTTP 201 + prediction error < 0.25.
- Cron re-verification (2026-08-02T03:05Z): the daemon restarted since
  the last check (now PID 334548; bridge PID 334537 on port 8791), so the
  armed write path was re-probed against the LIVE processes: PUT to the
  exact allowlisted plan.md endpoint returns HTTP 403 "BLOCKED:
  permission denied: no write grant for resource 'github'" (allowlist
  entry matched, method-aware permission gate denied, no outbound call),
  while GET https://api.github.com/ still returns exit=0 with a live
  2262-byte GitHub payload — bridge fully functional, only the grant
  missing. 28/28 evolve bridge/permission tests pass
  (tests/test_evolve_bridge.py 16✓, tests/test_evolve_permissions.py
  12✓). State unchanged: github.write still denied, deny-by-default
  holding.
- Full bounded verification (2026-08-02 cron cycle, HEAD ed22e2d3e):
  the complete Level 2 test surface re-run through the CI-parity runner
  (`scripts/run_tests.sh`) — tests/test_evolve_bridge.py 16/16 and
  tests/agent/test_think_daemon.py 213/213 (incl. the 6 daemon preflight
  + drift-guard tests) — 229/229 green, no regressions. Live grant check
  (`evolve_permissions.py check github write`) still exits 1: denied.
  Local groundwork is COMPLETE and independently verified; the only
  remaining step is the external user grant, after which the first write
  triple (PUT this plan via the Contents API) fires and must yield HTTP
  201 with prediction error < 0.25.
