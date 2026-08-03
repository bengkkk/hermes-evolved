# Gap 10 — Real Action Bridge: Design

**Status:** Level 1 complete (calibration 2026-08-02, 3/3 read-only
api_call triples, avg prediction error 0.15 < 0.3 bar) and Level 2
COMPLETE (2026-08-02 — github.write granted; both allowlisted PUTs fired
and verified, see docs/gap10-level2-plan.md). Level 3 (valued actions:
Polymarket) is SCOPED — see docs/gap10-level3-scope.md; pending user
review of the proposed allowlist + permissions.
**Owner:** think_daemon self-directed evolution
**Target:** Turn the daemon from a sandboxed local actor into an economic actor
with real-world permissions, keeping every action verifiable through the
existing world-model predict→execute→compare loop.

---

## 1. Goal

The daemon currently has exactly 4 action types — `write_file`, `shell`,
`git_commit`, `install_package` — all container-local. Gap 10 means **real
action permissions**: email, calendar, bank, ERP/CRM, cloud, code repositories,
browsers, robots, payment systems. The endpoint state is *an economic actor*,
not an assistant.

## 2. Current state (baseline)

- World model: calibrated — 155+ action triples, per-type avg error 0.15–0.24,
  decreasing. This is the **trust substrate**: any new external action plugs
  into the same predict→execute→compare loop automatically.
- Self-model: has capabilities/unknowns/commitments, and now a **permissions
  field** (Level 1 step 1, 2026-08-01): deny-by-default per-resource registry
  (`{resource: {read, write, act, cap}}`) with a declared-but-zero-grant
  `github` entry, grant/revoke/check API, schema validator
  (`SelfModel.validate_permissions()`), and a compact grants summary in the
  state snapshot injected into the thinking prompt. Still no `api_call` action
  type and no host bridge to consume it.
- Host toolchain: skills exist (google-workspace, notion, airtable, xurl,
  polymarket, computer-use, openhue) but CLIs are not installed.
- No HTTP bridge daemon→host, no credential vault, no allowlist.

## 3. Architecture (3 new components)

1. **Permission registry** — extend `self_model.json` with a `permissions`
   field: per-resource capability matrix (`{resource: {read: bool, write:
   bool, act: bool, cap: null|amount}}`). This is the single source of truth
   for what the daemon may do. Every new action type MUST declare its
   permission entry; pre-flight validation rejects actions with no entry.

2. **`api_call` action type** in think_daemon.py —
   `{"type": "api_call", "endpoint": "...", "method": "GET|POST|...",
   "body": {...}, "expected_outcome": "..."}`. Executed by the host bridge
   (credentials live on the host, never in the container).

3. **Host gateway bridge** — a small HTTP service on the host:
   `POST /bridge/v1/exec` validates the request against the permission
   registry + endpoint allowlist, executes via the installed CLI/skill,
   returns a structured result `{exit: 0, output: {...}}`. Auth: shared
   token from env. Transport: localhost TCP.

### Mandatory safety layers (every level)

- Endpoint + method allowlist
- Per-action permission check (registry)
- World-model prediction recorded BEFORE execution (existing loop)
- Read-only first; write actions require explicit user grant
- Money/value actions (Level 3+): explicit grant + per-transaction cap +
  human confirmation webhook
- Full audit trail via timeline events (existing)

## 4. Levels

| Level | Scope | Actions | Gate |
|---|---|---|---|
| 0 | now | 4 local sandboxed types | — |
| 1 | read-only external | Gmail read, calendar read, GitHub read | allowlist + permission check |
| 2 | low-risk writes | Telegram/Discord send, GitHub issue, Notion page | explicit user grant |
| 3 | valued actions | Polymarket orders, payments (future) | grant + cap + confirm |
| 4 | economic actor | CRM/ERP ops, cloud changes, robots | long-horizon |

## 5. First milestone (Level 1)

1. ✅ Add `permissions` to self-model default + validator + tests
   (DONE 2026-08-01 — `SelfModel.permissions` in data_layer.py, deny-by-default
   `github` entry, `check_permission`/`grant_permission`/`revoke_permission`,
   `validate_permissions()` schema validator, snapshot summary, 8 tests in
   `tests/test_data_layer.py`)
2. ✅ Add `api_call` action type to daemon with pre-flight validation
   (DONE 2026-08-01 — `_API_CALL_ALLOWLIST` + `_validate_api_call` in
   think_daemon.py: deny-by-default two-layer check = endpoint allowlist
   (method + host + path prefix, currently GET api.github.com → `github`)
   AND permission-registry read grant (`sm.check_permission` semantics);
   unauthorized endpoints rejected with a clear `BLOCKED:` error and NO
   execution. `_execute_api_call` POSTs to the host bridge
   (`HERMES_EVOLVED_BRIDGE_URL`, default localhost:8791) and records honest
   connection-refused triples until step 3 lands. 11 tests in
   `TestApiCallPreflightValidation`; 175 daemon tests + 53 adjacent pass.)
3. ✅ Stand up host bridge (DONE 2026-08-01 — `evolve_bridge.py`:
   FastAPI + uvicorn, binds 127.0.0.1:8791, shared Bearer token
   (`HERMES_EVOLVED_BRIDGE_TOKEN` or `<evolve_dir>/bridge_token`,
   generated by `evolve_daemon.sh bridge start`), deny-by-default
   three-layer validation (auth → endpoint allowlist mirroring the
   daemon's `_API_CALL_ALLOWLIST` → permission registry via
   `SelfModel.check_permission`), credential access limited to
   `~/.git-credentials`, outbound GET only via httpx, and every
   decision (executed AND blocked) appended to
   `<evolve_dir>/bridge_audit.log`. `evolve_daemon.sh` gains
   `bridge {start|stop|status|restart}` and exports the shared token
   into the daemon launch. Live `github.read` grant issued via
   `SelfModel.grant_permission`; E2E verified: real
   `GET api.github.com/repos/NousResearch/hermes-agent` → `exit=0`
   (200, 6864 bytes). 12 tests in `tests/test_evolve_bridge.py`,
   incl. deny-never-executes and allowlist-drift guards.)
4. Calibrate: record 3+ read-only api_call triples, confirm prediction error
   drops below 0.3 for the new type (DONE 2026-08-02 — 3/3 triples, avg err
   0.15; see CALIBRATION COMPLETE below. Background: the daemon's cycles
   can now reach the bridge; the deny-by-default grant gate is
   open for read-only GitHub GETs. 2026-08-01: root-caused why 0 triples
   had been recorded despite the open grant — the ACTION CAPABILITIES
   prompt still carried the pre-grant static text "until a grant exists …
   do not spam api_call attempts", actively suppressing the only action
   type that could produce the triples. `_api_call_capability_text` now
   renders the block dynamically from the permission registry + allowlist:
   open gate → ">>> GATE OPEN <<<" + ready-to-use endpoint listed;
   closed gate → discourage text kept. Live prompt verified; 653 tests.)
   FIRST LIVE TRIPLE 2026-08-01 19:06:55Z: the daemon itself fired
   `GET https://api.github.com/` through the bridge and recorded
   `act_20260801190655_2` (expected "HTTP 200: JSON body listing GitHub
   API endpoint fields" → actual `exit=0: {"status": 200, "bytes": 2262,
   "body": "{\"current_user_url\":...}"}`). The bridge is proven
   end-to-end from inside the daemon's own loop: pre-flight allowlist +
   permission gate passed, execution delegated to the host, structured
   outcome returned, triple persisted automatically — no explicit
   evidence write_file was needed. CALIBRATION FIX: the raw-string error
   scorer gave this fully-correct prediction 0.5 ("mixed") because the
   mutual exit=0 heuristic requires an `exit=` marker on the expected
   side, while api_call predictions phrase success as "HTTP 200".
   `_compute_prediction_error` now has a mutual HTTP-status heuristic
   (expected 2xx + observed 2xx → 0.15; hundreds-digit class match so a
   predicted 200 vs observed 404 still falls through). Recomputed error
   for the stored triple: 0.5 → 0.15. 211 world-model + daemon tests pass.)
   CALIBRATION COMPLETE 2026-08-02 00:21:40Z: 3/3 api_call triples recorded
   and verified on disk — `act_20260801190655_2` (err 0.15),
   `act_20260802000058_2` (err 0.15), `act_20260802002140_1` (err 0.15,
   fired 00:21:40Z by the cron cycle to reach the threshold). All three are
   real HTTP 200 GETs of `api.github.com/` through the host bridge, all
   with prediction error 0.15 < the 0.3 milestone bar. Level 1 calibration
   step is COMPLETE. Guidance state closed at the same moment (orientation
   orientation focus/next_steps updated) so the daemon does not keep re-firing sample
   #3. Level 2 (low-risk writes — Telegram/Discord send, GitHub issue,
   Notion page) is PLANNED in docs/gap10-level2-plan.md: the grant CLI
   (evolve_permissions.py + `evolve_daemon.sh permissions grant <res> <act>`)
   is the explicit-grant channel, and write endpoints stay deny-until-granted
   in both allowlists.)
5. ✅ Commit + update self-model state (DONE 2026-08-01 — live triple
   validated, scoring fixed, unknown_areas resolved: api_call IS wired into
   the action loop and gated by allowlist + permission registry; outcomes
   ARE auto-recorded as world-model triples)

**Recommendation for first bridge endpoint: `gh` (GitHub).** Reason: a
classic PAT is already stored in `~/.git-credentials` on the host — zero new
credential setup, immediately testable read-only (`gh api repos/...`), and
code repositories are one of the highest-value Gap 10 resources.

### Read-gate resilience during LLM outages (2026-08-03)

Observed drift: the committed per-cycle read-gate GET (`api.github.com/`)
went silent during the 6-cycle LLM outage (ticks 491-496, all git_commit
auto-sync). Root cause: the fallback action selector
(`_select_state_check_action`) only knew `shell`/`git_commit`/`write_file`
— api_call had no slot in the fallback rotation, so the standing
commitment "fire one allowlisted GET per cycle" could only be honored by
the LLM path, which was down.

Fix (think_daemon.py):
- `_state_check_commands` gains an `api_call` read-gate candidate
  (allowlisted `GET https://api.github.com/`, same endpoint the LLM path
  uses).
- `_select_state_check_action` gains a read-gate priority: when
  `read_gate_granted` (permission registry grants github.read) AND the
  gate is due (`_read_gate_is_due`: no api_call triple yet, or the most
  recent is older than `_READ_GATE_DUE_SECONDS` = 1500s ≈ 1.7 cycles),
  the api_call candidate wins outright — fallback cycles keep the read
  gate alive during outages.
- Without the grant the api_call candidate is removed entirely, so the
  pre-Gap-10 rotation (5 slots) is preserved exactly and no BLOCKED
  triple spam can occur.
- New tests: `TestReadGatePriority` in tests/test_daemon_local_analysis.py
  (fresh WM → GET wins; stale triple → GET re-fires; recent triple →
  defers to least-sampled diversity; no grant → never api_call; end-to-end
  via `_local_analysis`). 298 daemon + bridge tests pass.

### Bridge health-check/restart automation (2026-08-03)

Standalone recovery path for when the host bridge dies while the daemon is
mid-cycle, asleep, or absent (the in-process `_bridge_ensure_running` only
covers daemon-alive windows). Goal
`goal_20260802151450_3` — "Automate host bridge health check and restart".

Components:

- `scripts/bridge_healthcheck.py` — stdlib-only probe → restart → re-probe
  with deterministic exit codes (0 = UP at end, 1 = DOWN). Probe is
  `GET {BRIDGE_URL}/bridge/v1/health`; restart reuses the launcher's own
  path (`evolve_daemon.sh bridge restart`) so token/PID/log handling stays
  consistent. `--check-only` probes without side effects.
- `evolve_daemon.sh bridge healthcheck [--check-only]` — launcher entry
  point that delegates to the script; makes recovery reachable on demand
  (cron, daemon startup, or a human shell) as one shell action.

Verified end-to-end 2026-08-03 (live, not just fake-bridge tests): bridge
stopped → `scripts/bridge_healthcheck.py` detected DOWN, ran the restart,
re-probed UP (new PID) → following allowlisted GET `https://api.github.com/`
completed HTTP 200. Still-down path (restart can't recover) exits 1. Tests:
`tests/test_bridge_healthcheck.py` (5), `tests/test_evolve_bridge.py` (22),
daemon bridge/self-heal slice (22) — all green.

## 6. Verification criteria

- `api_call` passes pre-flight for allowlisted endpoints; unauthorized
  endpoints rejected with a clear error and NO execution
- World model records external-action triples with avg error < 0.3
- Existing test suite still passes (401 tests); daemon cycles stay healthy
- Audit trail per external call: world-model triple + host-side
  `bridge_audit.log` entry (executed AND blocked decisions; verified for
  the first live call 2026-08-01 19:06:55Z). Note: the daemon timeline is
  LLM-curated milestones (via `event_to_record`), NOT a per-action log —
  per-call audit lives in the triple + bridge log, not the timeline.

## 7. Open questions

- Credential storage: host `.env` files vs vault (start: host env files;
  daemon never sees secrets)
- Bridge transport: localhost TCP vs docker socket mount (start: localhost +
  token)
- First CLI to install: `gh` (recommended, zero setup) vs `gws` (Gmail/Calendar)
