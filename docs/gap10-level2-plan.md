# Gap 10 — Level 2 Plan: Low-Risk Writes (GitHub issue / Telegram / Notion)

**Status:** PLANNED — blocked on an explicit user grant (design doc §4 gate).
Level 1 calibration complete 2026-08-02 (3/3 api_call triples, avg err 0.15).
Level 2 on-ramp (grant CLI) landed 2026-08-02 with this plan.

## 1. What Level 2 unlocks

Write `api_call` actions through the host bridge: GitHub issue creation,
Telegram/Discord send, Notion page creation. The gate is an **explicit user
grant** — a `write: true` flag in the self-model permission registry. No
grant, no write execution: deny-by-default is preserved at every layer
(daemon pre-flight → bridge allowlist → bridge permission check → audit).

## 2. Why GitHub issues first

- The PAT already lives in `~/.git-credentials` (zero new credential setup);
  `gh` CLI is the recommended executor.
- `github.read` is already trusted and calibrated; `write` is the smallest
  semantic step (same host, same token, same audit path).
- Highest verifiability: the created issue URL is externally checkable, and
  we get a world-model triple + a `bridge_audit.log` WRITE entry + the
  GitHub-side record — three independent confirmations.

## 3. Implementation steps

1. ✅ **Grant channel (DONE 2026-08-02)** — `evolve_permissions.py` +
   `evolve_daemon.sh permissions` subcommand. The user issues the Level 2
   grant with one command:
   `./evolve_daemon.sh permissions grant github write`
   (interactive confirmation; `--yes` for scripts; `revoke`/`check`/`show`
   for audit). All mutations go through `data_layer.SelfModel` so the daemon
   and bridge see exactly the same registry. 12 tests in
   `tests/test_evolve_permissions.py`.

2. **Method-aware pre-flight** — `_validate_api_call` currently gates every
   method on the `read` flag (think_daemon.py:4406). Change it to select the
   permission by method: `GET`/`HEAD` → `read`; `POST`/`PUT`/`PATCH`/`DELETE`
   → `write`. Mirror the same method→action derivation in the bridge
   (`evolve_bridge.py` `_permission_granted`) and render the write gate
   dynamically in `_api_call_capability_text` (so the daemon knows a write
   endpoint exists but stays silent until granted).

3. **Declare write endpoints, deny-until-granted** — add to BOTH
   `_API_CALL_ALLOWLIST` and `BRIDGE_ALLOWLIST` (the drift test
   `tests/test_evolve_bridge.py` keeps the two lists identical, so any
   addition must land in both):
   - `POST api.github.com /repos/*/issues` → resource `github` (write)
   - later, after credential setup by the user: `POST api.telegram.org
     /bot*/sendMessage` (bot token), `POST api.notion.com /v1/pages`
     (integration token).
   While `github.write` is False, pre-flight still BLOCKS these — the entry
   only makes the endpoint *recognized*, never *executable*.

4. **Bridge write execution** — extend the bridge's executor to route
   non-GET methods with a JSON body and proper headers; authenticate with the
   existing `~/.git-credentials` PAT (or via `gh api -X POST ...`), cap
   response bytes (`MAX_OUTPUT_BYTES`), and append a `WRITE` audit entry to
   `bridge_audit.log`. The daemon's `_execute_api_call` already forwards
   `method` + `body`, so the wire format needs no change.

5. **First live write** — the user designates a repo (recommend a scratch /
   user-owned repo to avoid noise). The daemon fires ONE issue-creation
   triple through the open gate, records the prediction BEFORE execution
   (existing loop), then verifies: triple `prediction_error < 0.3` +
   `bridge_audit.log` WRITE entry + the issue URL resolves.

6. **Level 2 calibration bar** — 3/3 write triples with err < 0.3 → Level 2
   calibration complete. Then expand to Telegram/Notion (each needs a
   credential the user must supply on the host).

## 4. Safety invariants (unchanged from Level 1)

- World-model prediction recorded BEFORE execution (existing loop).
- Audit trail for every decision: world-model triple + `bridge_audit.log`
  (executed AND blocked).
- Deny-by-default: no allowlist entry, no grant, or malformed registry →
  BLOCKED, never executed.
- Secrets stay host-side; the container never sees credentials.
- Level 3 (valued actions) adds a per-transaction cap + human confirmation
  webhook; Level 2 needs neither.

## 5. Daemon behavior while waiting for the grant

- Keep the read bridge calibrated; fire read triples only when guidance
  invites them (no spam — the capability text stays gate-aware).
- Do NOT attempt write api_calls: they are blocked by pre-flight until the
  grant exists, and that BLOCKED decision is itself auditable.
- Continue housekeeping (stale-commitment pruning, bounded-inspection
  discipline).

## 6. Open questions for the user

1. Which repo may receive the first test issue? (recommend a scratch repo
   you own)
2. Is a Telegram bot token available? (required for telegram send)
3. Is a Notion integration token available? (required for Notion pages)

**To grant:** `./evolve_daemon.sh permissions grant github write`
**To audit:** `./evolve_daemon.sh permissions show`
**To revoke:** `./evolve_daemon.sh permissions revoke github write`
