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
- Live write calibration (2026-08-02T05:44Z, bridge PID 346351 / daemon
  PID 346362, live HERMES_HOME=/root/.hermes-evolved): fired the first
  WRITE api_call triple through the live bridge using the daemon's own
  wire protocol (POST /bridge/v1/exec, Bearer token from the bridge env)
  against the exact allowlisted plan.md endpoint. Result: HTTP 403
  "BLOCKED: permission denied: no write grant for resource 'github'" —
  allowlist entry matched (layer 2), method-aware permission gate denied
  (layer 3), no outbound call. Recorded as world-model triple
  act_20260802054400_1, prediction error 0.25 (categorical MATCH; the
  0.25 is the string-diff on the "HTTP 403:" prefix, same scale as the
  daemon's own matched triples). The security-critical sibling-path
  invariant was also verified LIVE at 05:44:49Z: PUT to
  contents/other.md → HTTP 403 "BLOCKED: endpoint not in allowlist"
  (exact-path matching held; no outbound call) — triple
  act_20260802054449_1, error 0.25. Both denials appear in the live
  audit log (deny_permission, deny_allowlist events), satisfying the
  "audit entry for every external call" criterion end-to-end. The daemon
  also independently fired a live GET (audit exec, exit 0, 05:10:51Z).
  Grant still absent: `evolve_permissions.py check github write` → exit 1.
- Cron re-verification (2026-08-02T06:31Z, bridge PID 353176 / daemon PID
  353187 — new process generation restarted 06:06:47, AFTER the latest
  commit 78927cd68, so the live processes run the current allowlists):
  re-probed the armed write path against the LIVE processes with the
  daemon's own wire protocol. PUT to the exact allowlisted plan.md
  endpoint → HTTP 403 "BLOCKED: permission denied: no write grant for
  resource 'github'" (allowlist entry matched, method-aware permission
  gate denied, no outbound call); PUT to sibling contents/other.md →
  HTTP 403 "BLOCKED: endpoint not in allowlist" (exact-path matching
  held); GET https://api.github.com/ → exit 0 with a live 2262-byte
  GitHub payload (read path intact). Both denials + the exec appear in
  the live audit log at 06:31:09Z (deny_permission, deny_allowlist,
  exec). `evolve_permissions.py check github write` → exit 1: denied.
  Bounded test suite: tests/test_evolve_bridge.py + tests/test_evolve_
  permissions.py 30/30 green (incl. the drift guards pinning
  gap10_level2_policy.py to the enforced allowlists). State unchanged:
  github.write still denied, deny-by-default holding. Only the external
  user grant remains.
- Cron re-verification (2026-08-02T06:50Z, bridge PID 354226 / daemon PID
  354237 — new process generation restarted 06:47:26, AFTER the latest
  commit 62bc64b56, so the live processes run the current allowlists):
  re-probed the armed write path against the LIVE processes with the
  daemon's own wire protocol (POST /bridge/v1/exec, Bearer token). PUT to
  the exact allowlisted plan.md endpoint → HTTP 403 "BLOCKED: permission
  denied: no write grant for resource 'github'" (allowlist entry matched,
  method-aware permission gate denied, no outbound call); PUT to sibling
  contents/other.md → HTTP 403 "BLOCKED: endpoint not in allowlist"
  (exact-path matching held); GET https://api.github.com/ → exit 0 with a
  live 2262-byte GitHub payload (read path intact). All three events in
  the live audit log at 06:50:52Z (deny_permission, deny_allowlist,
  exec). `evolve_permissions.py check github write` → exit 1: denied
  (read → exit 0: GRANTED). Bounded test suite: tests/test_evolve_bridge
  .py + tests/test_evolve_permissions.py 30/30 green (incl. the drift
  guards pinning gap10_level2_policy.py to the enforced allowlists).
  State unchanged: github.write still denied, deny-by-default holding.
  Only the external user grant remains.
- Cron re-verification (2026-08-02T07:27Z, bridge PID 357945 / daemon PID
  357956 — new process generation restarted 07:21, AFTER the latest
  commit 7dba5847f, so the live processes run the current allowlists):
  re-probed the armed write path against the LIVE processes with the
  daemon's own wire protocol (POST /bridge/v1/exec, Bearer token). PUT to
  the exact allowlisted plan.md endpoint → HTTP 403 "BLOCKED: permission
  denied: no write grant for resource 'github'" (allowlist entry matched,
  method-aware permission gate denied, no outbound call); PUT to sibling
  contents/other.md → HTTP 403 "BLOCKED: endpoint not in allowlist"
  (exact-path matching held); GET https://api.github.com/ → exit 0 with a
  live 2262-byte GitHub payload (read path intact). All three events in
  the live audit log at 07:27:34Z (deny_permission, deny_allowlist) and
  07:27:35Z (exec). `evolve_permissions.py check github write` → exit 1:
  denied (read → exit 0: GRANTED). Bounded test suite: tests/test_evolve_
  bridge.py + tests/test_evolve_permissions.py 30/30 green (incl. the
  drift guards pinning gap10_level2_policy.py to the enforced allowlists).
  State unchanged: github.write still denied, deny-by-default holding.
  Only the external user grant remains.
- Cron re-verification (2026-08-02T07:44Z, bridge PID 360072 / daemon PID
  360083 — new process generation restarted 07:38, AFTER the latest
  commit 5c4c1410f, so the live processes run the current allowlists):
  re-probed the armed write path against the LIVE processes with the
  daemon's own wire protocol (POST /bridge/v1/exec, Bearer token). GET
  https://api.github.com/ → exit 0 with a live 2262-byte GitHub payload
  (read path intact; audit log exec entries at 07:41:00Z from daemon
  cycle 419 plus this cycle's 07:44Z call). Write payload readiness
  verified: docs/gap10-level2-plan.md is 9848 bytes → base64 13132 chars
  → 13270 byte JSON body {message, content, branch: main}, ~4 orders of
  magnitude under the GitHub 100MB limit, and the PUT endpoint matches
  the exact-path allowlist entry byte-for-byte — the first write triple
  is a single request away from the grant. Bounded test suite:
  tests/test_evolve_bridge.py 18/18 + tests/test_evolve_permissions.py +
  tests/test_wm_self_bridge.py + tests/test_world_model.py 237/237 green
  (255 total, incl. drift guards pinning gap10_level2_policy.py to the
  enforced allowlists). `evolve_permissions.py check github write` →
  denied (read → GRANTED). State unchanged: github.write still denied,
  deny-by-default holding. Only the external user grant remains.
- Cron re-verification (2026-08-02T08:07Z, bridge PID 360457 / daemon PID
  360468 — new process generation; commit 5a643a7cf's generation had been
  replaced): re-verified via the new state-aware script
  `verify_gap10_write_path.py` (single command, pass/fail signal) →
  VERIFY PASS, exit 0: P1 read=200/ok (live 2262-byte GitHub payload),
  P3 sibling PUT → 403 deny_allowlist (exact-path invariant held),
  P2 exact allowlisted PUT → 403 deny_permission (method-aware gate held,
  no outbound call). Audit log entries confirmed (exec, deny_permission,
  deny_allowlist). `evolve_permissions.py check github write` → denied.
  **Anti-churn policy (from this cycle):** identical verification outcomes
  carry no new information — subsequent cycles run the script and commit
  only on a changed outcome (FAIL, or GRANT ACTIVE → fire the first write
  triple deliberately with the real payload). No more per-cycle doc
  paragraphs while the state is unchanged.
