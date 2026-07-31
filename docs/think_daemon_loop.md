# think_daemon.py — Action-Selection Loop (Reference)

Persistent-cognition daemon for the Hermes Evolved Phase 2 world-model loop.
This document describes the action-selection loop as implemented at
commit `be67d80a5` (Jul 2026). It is the counterpart to the daemon-reliability
work: the loop below is what the PID-lock re-verification protects.

## Process model

```
main()
 ├─ --verify   → _run_verification()      (no-LLM self-test of predict→act→observe→learn)
 ├─ --bootstrap → bootstrap_evolve_data()
 ├─ --status   → _show_status()
 ├─ --once     → acquire lock → run_one_cycle() → release lock
 └─ (default)  → run_daemon(interval, max_cycles)   ← persistent loop
```

`run_daemon()` acquires the PID lock once at startup, then loops:
**re-verify lock ownership → run_one_cycle() → check shutdown flag → sleep(interval)**.

Since `be67d80a5`, the lock is re-verified at the **top of every cycle**:
if another live process owns `daemon.lock`, the daemon exits instead of
running duplicate cycles. This closes the failure mode where a daemon that
started before the lock file existed (or lost the lock to a newer instance)
kept cycling forever alongside the lock holder.

## One cycle (`run_one_cycle` → `_run_cycle_body`)

Each cycle is wrapped in a 200 s hard timeout and reliability accounting
(`cycle_stats`: total/ok/error/parse_error/avg_duration/max_duration +
last-20 `cycle_history`). The cycle body:

1. **Load state** — timeline, self-model, orientation, world model.
2. **Auto-verify expired predictions** — `world_model.verify_expired_predictions()`
   compares old predictions against what actually happened, learning from
   discrepancies (non-blocking on failure).
3. **Auto-create initial plan** if none exists (Gap 8 bootstrap; best-effort).
4. **Pre-cycle self-model pruning** — remove stale/duplicate weaknesses
   *before* building the prompt so the LLM never re-fixates on stale entries.
5. **Build prompt** (`_build_thinking_prompt`) — JSON-instructing system
   prompt + state snapshot.
6. **Call LLM** (`_call_llm`) — retries with provider fallback; on total
   failure, `_local_analysis()` produces a data-driven fallback insight
   (world-model stats, rotating exploratory action) and the cycle continues.
   Consecutive-fallback count is tracked; on recovery a note is injected.
7. **Parse JSON** (`_try_parse_json`) — parse_error is recorded, cycle ends.
8. **Search phase (optional)** — if the LLM emitted `search_query`,
   DuckDuckGo results are appended and the LLM is re-prompted once.
9. **Apply insights** (`_apply_insights`) — timeline/self-model/orientation
   updates from the parsed JSON.
10. **Bridge world model → self-model** — auto-detect discrepancy patterns
    and surface them as self-model weaknesses (post-LLM, so the LLM's own
    updates win first).
11. **Goal reconciliation** — auto-complete goals whose conditions are met.
12. **Auto-activate proposed goals** — promote pending goals to active
    (Gap 4 → Gap 8 bridge).
13. **Record prediction** — if the LLM made one, confidence is calibrated
    against per-type historical error (`world_model.adjust_confidence`).
14. **Persist state** — timeline, self-model (+`total_cycles`), orientation,
    world model, daemon_state.

## Action selection and execution (inside `_apply_insights`)

The action to execute comes from the LLM's JSON (`action: {type, ...}`),
subject to two overrides and one gate, in order:

1. **Null-action fallback** — if no action or `tick_count >= 10`, the
   rotating auto-default list (`_ROTATING_AUTOS`: shell exploration,
   write_file state snapshots, git_commit) is used. `expected_outcome` is
   cleared so the world model's data-driven predictor fills it in (hardcoded
   generic expectations caused systematic ~0.5 prediction error).
2. **Action dedup gate** — if the LLM's action is a near-duplicate (≥40%
   word overlap of description) of any of the last 5 completed triples of
   the same type, it is replaced with the rotating default. Breaks fixation
   loops (e.g. "read think_daemon.py" 9/14 cycles).
3. **Execution** — for each action, BEFORE executing:
   - `world_model.predict_action_outcome(type, desc, parameters)` computes a
     data-driven expected outcome. The data-driven prediction is preferred
     when the LLM gave no expected outcome, or echoed meta-context
     (`[data-driven]` prefix), or when its confidence is low.
   - The triple `{action_type, action_description, expected_outcome,
     expected_source}` is recorded.
   - The action is executed (shell command / write_file / git_commit /
     install_package / send_message), the actual outcome is captured, the
     triple is marked complete, prediction error is computed, and the world
     model is saved.

## Reliability properties (as of be67d80a5)

- **Single instance** — atomic `O_CREAT|O_EXCL` PID lock, stale/zombie-PID
  takeover, and a per-cycle ownership re-check. A live owner's lock is never
  clobbered (regression-tested).
- **No-LLM degradation** — local-analysis fallback keeps the loop alive and
  still collects world-model data when the LLM is unreachable.
- **Hard timeout** — 200 s per cycle; crashes/timeouts are counted, not fatal.
- **Graceful stop** — `daemon_state.status == "shutdown"` checked after each
  cycle; `scripts/start_daemon.sh --stop` sets it and SIGTERMs the holder.
- **Fixation protection** — pre-cycle pruning + action dedup gate.
