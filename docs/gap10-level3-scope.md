# Gap 10 Level 3 — Real Action Bridge: Valued Actions (Polymarket)

**Status:** SCOPED — proposal for user review, nothing executes yet
**Owner:** Hermes (evolved)
**Created:** 2026-08-02
**Gate:** explicit user grant + per-transaction cap + human confirmation webhook

## Objective

Extend the real action bridge from low-risk GitHub writes (Level 2) to
**valued actions** — the first economic-actor capability. Per
docs/gap10-bridge-design.md §4, Level 3 scope is *"Polymarket orders,
payments (future)"* gated by **grant + cap + confirm**. This document
proposes the concrete allowlist entries, permission requirements, and
bridge changes needed, for user review. **No Level 3 code is implemented
and no polymarket call will execute until the user approves this scope and
issues the grants.**

## Why Polymarket First

- The polymarket skill already ships in this repo
  (`skills/research/polymarket/`) with a complete endpoint reference
  (`references/api-endpoints.md`) and a read-only helper script
  (`scripts/polymarket.py`).
- All market-data endpoints are **public, unauthenticated GETs** — the
  same risk class as the Level 1 GitHub reads already calibrated (avg
  prediction error 0.15).
- Order placement is a genuinely valued action (real money, real
  markets) — the smallest step that makes the daemon an economic actor,
  and it exercises all three Level 3 safety layers (grant, cap, confirm)
  end-to-end.

## Level 2 Recap

- github.write granted 2026-08-02; both allowlisted PUTs fired through the
  live host bridge and verified (HTTP 201, shas, prediction error 0.15 <
  0.25 bar). See docs/gap10-level2-plan.md §"Level 2 COMPLETE".

## Level 3 Design

### 3a. Read phase (low risk — market data)

Resource: `polymarket`. Proposed allowlist entries, all public GETs
(no credentials, no auth):

| Method | Host | Path | Data |
|--------|------|------|------|
| GET | gamma-api.polymarket.com | `/public-search?q=…` | search markets |
| GET | gamma-api.polymarket.com | `/events?…` | list events |
| GET | gamma-api.polymarket.com | `/markets?…` | list markets |
| GET | gamma-api.polymarket.com | `/tags` | list tags |
| GET | clob.polymarket.com | `/price?token_id=…` | current price |
| GET | clob.polymarket.com | `/midpoint?token_id=…` | midpoint price |
| GET | clob.polymarket.com | `/spread?token_id=…` | spread |
| GET | clob.polymarket.com | `/book?token_id=…` | orderbook |
| GET | clob.polymarket.com | `/prices-history?…` | price history |
| GET | clob.polymarket.com | `/markets?limit=…` | CLOB market list |
| GET | data-api.polymarket.com | `/trades?…` | recent trades |
| GET | data-api.polymarket.com | `/oi?market=…` | open interest |

**Permission required:** `polymarket read` (deny-by-default until the
user grants it — same pattern as the Level 1 github.read grant).

**Verification bar (mirrors Level 1):** after the grant, fire 3+ read
triples through the bridge and confirm avg prediction error < 0.3 for the
new endpoints. No endpoint executes before the grant.

### 3b. Act phase (money-moving — orders)

Resource: `polymarket`, action `act`. Proposed allowlist entries —
**deny-until-granted**, exact-path style like the Level 2 write list:

| Method | Host | Path | Effect |
|--------|------|------|--------|
| POST | clob.polymarket.com | `/orders` | place order (money out) |
| DELETE | clob.polymarket.com | `/orders/{id}` | cancel order (money risk reduced) |

**Permission required:** `polymarket act` **with a cap** — e.g.
`evolve_permissions.py grant polymarket act --cap 10` (USD per order;
the grant CLI already supports `cap` for Level 3+). **Plus human
confirmation** (see below). The `write` flag is intentionally NOT used
for orders — orders are `act` actions, a distinct permission dimension.

### Bridge changes required (implementation work, not yet built)

1. **Per-resource method→action mapping.** The bridge's `_METHOD_ACTION`
   currently maps POST/DELETE → `write`. For `polymarket`, POST/DELETE
   must resolve to `act` (checked against the registry's `act` flag),
   not `write`. This needs a per-resource override table (default stays
   the current mapping; drift-guard tests keep both allowlists and
   mapping in sync with think_daemon.py).
2. **Cap enforcement.** For `act` requests, parse the order body
   (size, price, side, token), compute notional USD, and reject with
   `BLOCKED: cap exceeded` **before any outbound call** if it exceeds
   the registry cap. Cap lives in the permission registry (already
   modeled; only enforcement is missing).
3. **Human confirmation webhook.** The design doc's mandatory layer:
   money actions require explicit human confirmation. Concretely: the
   bridge stages an `act` request (pending-confirmation state, written
   to bridge_audit.log), notifies the user (host-side webhook/CLI
   prompt), and executes **only** on explicit confirmation, with a TTL
   after which the staged request expires. No Level 3 order fires
   without this layer.
4. **Host-side credentials.** Polymarket API keys live on the host
   (e.g. `~/.polymarket/` or host env), never in the container — same
   rule as `~/.git-credentials` for GitHub.

### Out of scope for 3b (future)

- Payments (bank/card rails): no concrete provider chosen; the same
  grant + cap + confirm gate will apply when scoped.
- Portfolio management / auto-rebalancing: requires a proven act path
  first.

## What I Need From the User

1. **Approve `polymarket` as the first Level 3 resource** (vs. another
   valued action).
2. **Approve the 3a read allowlist** — 12 public GET endpoints across
   gamma-api / clob / data-api.
3. **Approve the 3b act allowlist + cap value** — POST/DELETE
   clob.polymarket.com orders, with a proposed initial cap (default
   proposal: **$10 per order**, adjustable via the grant CLI cap).
4. **Confirm the confirmation webhook** is required for every act
   action (recommended default: yes, no auto-execute of money actions).

## Verification Criteria (once implemented)

- Deny-by-default: every polymarket endpoint returns `BLOCKED` before
  any grant; no outbound call.
- Read phase: after `polymarket read`, GETs execute with world-model
  triples avg error < 0.3.
- Act phase: POST /orders without `act` grant → `BLOCKED: permission`;
  with grant but over cap → `BLOCKED: cap exceeded` (no outbound call);
  with grant + within cap → staged, executes only after confirmation.
- Audit trail: every executed AND blocked attempt appended to
  bridge_audit.log (existing invariant).
- Tests: bridge + permissions + daemon preflight suites green,
  incl. drift guards pinning the new allowlists/mapping.

## Next Actions

1. User reviews this scope and issues grants (`polymarket read`;
   `polymarket act --cap 10`).
2. Implement 3a: allowlist entries + `polymarket read` grant path,
   calibrate 3+ read triples.
3. Implement 3b: per-resource method→action mapping, cap enforcement,
   confirmation webhook, then fire the first confirmed order triple.
