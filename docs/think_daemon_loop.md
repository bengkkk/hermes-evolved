# think_daemon.py — Action-Selection Loop (Reference)

Persistent-cognition daemon for the Hermes Evolved Phase 2 world-model loop.
This document describes the action-selection loop as implemented at commit
`d6b741b3d` + the extended-outage retry policy (0-attempt probe skip).
It is the counterpart to the daemon-reliability work: the loop below is what
the PID-lock re-verification protects.

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
**re-verify lock ownership → code-drift check → run_one_cycle() → check
shutdown flag → sleep(interval)**.

The lock is re-verified at the **top of every cycle**: if another live
process owns `daemon.lock`, the daemon exits instead of running duplicate
cycles. This closes the failure mode where a daemon that started before the
lock file existed (or lost the lock to a newer instance) kept cycling
forever alongside the lock holder.

**Code-drift detection**: `_mark_startup` records the repo HEAD at start
(`startup_head` in daemon_state.json). Every cycle, `_check_code_drift`
compares it against the current HEAD and logs a warning + records a
`code_drift` block when the repo has moved on — a long-lived daemon runs
OLD code after a fix is committed unless restarted, and the marker makes
that staleness visible instead of silent.

## One cycle (`run_one_cycle` → `_run_cycle_body`)

Each cycle is wrapped in a 200 s hard timeout and reliability accounting
(`cycle_stats`: total/ok/error/parse_error/avg_duration/max_duration +
last-20 `cycle_history`). The cycle body:

1. **Load state** — timeline, self-model, orientation, world model.
2. **Auto-verify expired predictions** — `world_model.verify_expired_predictions()`
   first checks each expired prediction against the action-triple record
   (`verify_prediction_via_evidence`: topic-token match on actual outcomes);
   predictions the system's own actions demonstrably fulfilled (or
   contradicted) are scored 0.15 / 0.85 and fed into calibration. Only
   when no evidence exists does it fall back to "timeframe expired — no
   confirmation" (error 0.5, calibration-neutral). `verify_pending_predictions()`
   additionally resolves non-expired predictions as soon as decisive
   action-triple evidence exists.
3. **Auto-create initial plan** if none exists (Gap 8 bootstrap; guarded
   against re-planning goals that already have active/complete plans).
4. **Pre-cycle self-model pruning** — remove stale/duplicate weaknesses
   *before* building the prompt so the LLM never re-fixates on stale entries.
5. **Build prompt** (`_build_thinking_prompt`) — JSON-instructing system
   prompt + state snapshot.
6. **Call LLM** (`_call_llm`) with an **adaptive retry budget** chosen once
   per cycle by `_set_llm_retry_policy(consecutive_fallback_cycles)`:

   | Outage depth | Budget            | Rationale                          |
   |--------------|-------------------|------------------------------------|
   | 0 (healthy)  | 2 attempts × 90 s | full budget (pre-adaptive behavior)|
   | 1            | 1 attempt × 60 s  | warm outage                        |
   | 2            | 1 attempt × 45 s  | deep outage, one probe             |
   | ≥3, skip     | **0 attempts**    | extended outage: straight to local analysis, no dead time |
   | ≥3, probe    | 1 attempt × 30 s  | every 4th cycle, so recovery is detected within 4 cycles |

   Each failed probe costs ~45–60 s of dead time (auxiliary client's
   internal retry + fallback stages), so during a multi-cycle outage probes
   would consume most of the 200 s budget with no chance of a different
   outcome. The 0-attempt tier turns those cycles into pure data
   collection; the periodic probe prevents permanent blindness after the
   provider recovers. On total failure, `_local_analysis()` produces a
   data-driven fallback insight and the cycle continues. The
   consecutive-fallback count is tracked; on recovery a note is injected.
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

## Fallback analysis (`_local_analysis`, no LLM)

Produces a data-driven result that keeps the state machine advancing:

- **Insight** from world-model stats: completed triples, per-type sample
  counts + avg error, prediction-error trend (delegated to the world
  model's canonical split-half trend computation), active predictions,
  known discrepancy patterns.
- **Auto-goal creation**: the top world-model improvement suggestion is
  converted into a goals.json entry (deduplicated against existing goals),
  bridging Gap 6 → Gap 4 during outages too.
- **Confidence**: base confidence from average triple error, scaled ×0.7
  because the LLM is unavailable.
- **Action**: a state-check action chosen by `_select_state_check_action` —
  data-driven selection that prefers the **least-sampled action type**
  (shell / git_commit / write_file) so fallback cycles diversify the world
  model's training data instead of monotonically re-sampling the type the
  blind rotation lands on most often. Falls back to plain tick rotation
  when no calibration data exists yet. Each candidate carries a specific
  `expected_outcome` for prediction feedback.

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
   - Shell commands pass `_validate_shell_command` pre-flight (unbalanced
     quotes; embedded `python3 -c` code that does not compile). Defective
     commands are blocked WITHOUT spawning a subprocess, recording an
     explicit `exit=-1: PRE-FLIGHT BLOCKED: <reason>` outcome instead of a
     cryptic shell error — the world model learns from a genuine generation
     defect, not a fake world-model miss.
   - The triple `{action_type, action_description, expected_outcome,
     expected_source}` is recorded.
   - The action is executed (shell command / write_file / git_commit /
     install_package / send_message), the actual outcome is captured, the
     triple is marked complete, prediction error is computed, and the world
     model is saved.

## Reliability properties

- **Single instance** — atomic `O_CREAT|O_EXCL` PID lock, stale/zombie-PID
  takeover, and a per-cycle ownership re-check. A live owner's lock is never
  clobbered (regression-tested).
- **No-LLM degradation** — local-analysis fallback keeps the loop alive and
  still collects world-model data when the LLM is unreachable; the adaptive
  retry policy stops burning cycle budget on a dead endpoint during
  extended outages while a periodic probe preserves recovery detection.
- **Hard timeout** — 200 s per cycle; crashes/timeouts are counted, not fatal.
- **Graceful stop** — `daemon_state.status == "shutdown"` checked after each
  cycle; `evolve_daemon.sh stop` SIGTERMs the holder and clears the lock.
- **Fixation protection** — pre-cycle pruning + action dedup gate.
- **Code-drift visibility** — stale daemon processes surface via the
  `code_drift` marker + launcher `status`; `evolve_daemon.sh restart`
  loads the latest logic.
