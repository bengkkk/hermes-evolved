# Gap 10 — Real Action Bridge: Design

**Status:** Level 1 steps 1-2 done (permission registry in self-model +
`api_call` action type with deny-by-default pre-flight validation); steps
3-5 remain
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
3. Stand up host bridge (FastAPI/uvicorn, token auth, `gh` + `gws` read
   endpoints) — next step
4. Calibrate: record 3+ read-only api_call triples, confirm prediction error
   drops below 0.3 for the new type
5. Commit + update self-model state

**Recommendation for first bridge endpoint: `gh` (GitHub).** Reason: a
classic PAT is already stored in `~/.git-credentials` on the host — zero new
credential setup, immediately testable read-only (`gh api repos/...`), and
code repositories are one of the highest-value Gap 10 resources.

## 6. Verification criteria

- `api_call` passes pre-flight for allowlisted endpoints; unauthorized
  endpoints rejected with a clear error and NO execution
- World model records external-action triples with avg error < 0.3
- Existing test suite still passes (401 tests); daemon cycles stay healthy
- Timeline shows an audit entry for every external call

## 7. Open questions

- Credential storage: host `.env` files vs vault (start: host env files;
  daemon never sees secrets)
- Bridge transport: localhost TCP vs docker socket mount (start: localhost +
  token)
- First CLI to install: `gh` (recommended, zero setup) vs `gws` (Gmail/Calendar)
