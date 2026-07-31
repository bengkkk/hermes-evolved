# think_daemon.py — Action-Selection Loop (Reference)

Persistent-cognition daemon for the Hermes Evolved Phase 2 world-model loop.
This document describes the action-selection loop as implemented at commit
`d6b741b3d` + the extended-outage retry policy (0-attempt probe skip).
Line numbers and cycle steps are verified against HEAD `d8edfb4cd`
(2026-07-31 19:57; per-cycle goal progress feedback in the thinking prompt).
It is the counterpart to the daemon-reliability work: the loop below is what
the PID-lock re-verification protects.

> Line-number drift note (2026-07-31): d8edfb4cd added 52 lines to
> think_daemon.py (`_format_goal_evidence` + goal rendering in
> `_build_thinking_prompt`), so every think_daemon.py reference below is
> ~52 lines past the 68bc9c8d1 numbers. The navigation map in the
> "think_daemon.py core-loop navigation map" section was re-verified by
> direct read at HEAD d8edfb4cd; the world-model.py map is unaffected
> (world_model.py was not touched by that commit).

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

## Goal integration (design map)

The daemon treats evolve/goals.json as a persistent store it writes to but
never deletes from. Four touchpoints connect the cycle to the goal system:

1. **Plan auto-create (cycle step 3)** — if no active plan exists and no
   active/complete plan targets "Complete Gap 8 — Self-directed evolution",
   `agent.self_evolve.create_plan` seeds one (think_daemon.py:3247-3282).
   Placeholder steps (bare "Step A" descriptions, or "V"/"n/a"
   verification) are rejected at creation by `_is_placeholder_step`
   (line 212), so a plan with no actionable content can never re-enter the
   prompt as an active plan and drive a deliberation-fixation loop.
2. **Goal reconciliation (cycle step 11)** — `_reconcile_goals_with_world`
   (line 2755) auto-completes goals whose verification conditions are met
   by world-model evidence, so finished work is retired without an LLM.
3. **Goal auto-activation (cycle step 12)** — `_auto_activate_goals`
   (line 2954) promotes proposed goals to active when capacity exists,
   closing the Gap 4 → Gap 8 loop without waiting for the LLM to set
   `goal_action` in its JSON.
4. **Outage-path goal creation** — `_local_analysis` (line 2341) converts
   the top world-model improvement suggestion into a goals.json entry
   (deduplicated against existing active goals), so even LLM-outage cycles
   keep the goal store evolving.

Design invariant: the LLM influences goals only through structured JSON
fields; every write path (create/activate/complete) also has a no-LLM
equivalent, so goal progress never depends on provider availability.
Per-cycle goal progress feedback (implemented 2026-07-31, HEAD after
555370f0d): each goal shown in the thinking prompt now carries its
`verification_criteria` (if set) plus a world-model evidence line.
`_format_goal_evidence` mirrors the reconciler's title patterns
("Gather more <type> action samples" → sample count, "Investigate
<type> prediction failures" → per-type avg error, "Fix overconfidence"
→ overall avg triple error), so LLM cycles see the same data
`_reconcile_goals_with_world` auto-completes on and can report progress
explicitly instead of only learning of completion after the fact.

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

## World-model API surface (world_model.py, navigation map)

The daemon consumes the `WorldModel` class (line 332). Methods referenced
from `think_daemon.py`, with line numbers as of HEAD 68bc9c8d1:

| Method | Line | Role in the loop |
|---|---|---|
| `_compute_prediction_error(expected, actual)` | 139 | Canonical error metric (0 = exact match … 1 = unrelated) |
| `record_action(type, desc, expected, source, …)` | 366 | Opens an action triple (predict step) |
| `complete_action(triple_id, actual)` | 444 | Closes the triple, computes error (called from daemon line 1932; timeout/exception close paths at 1973/1979) |
| `record_action_complete(...)` | 482 | Alternate close entry point (verify step) |
| `record_prediction(text, timeframe, confidence, basis)` | 512 | Macro-prediction log |
| `verify_prediction(pred_id)` | 552 | Score a single prediction against evidence |
| `verify_prediction_via_evidence(pred_id)` | 705 | Topic-token match against actual outcomes |
| `verify_expired_predictions()` | 820 | Expired + evidence-resolved (cycle step 2) |
| `verify_pending_predictions()` | 921 | Resolve early when decisive evidence exists |
| `format_world_model_context()` | 1003 | Prompt context builder |
| `format_prediction_insight()` | 1121 | Calibration insight for prompts |
| `adjust_confidence(raw, atype)` | 1512 | Per-type historical-error calibration (cycle step 13) |
| `format_calibration_guidance()` | 1580 | Guidance when a type is poorly calibrated |
| `format_improvement_context()` | 1745 | Top improvement suggestions |
| `format_action_guidance(type, desc)` | 1780 | Pre-execution risk assessment (daemon logs as `[RISK WARNING]`) |
| `predict_action_outcome(type, desc, params)` | 1846 | Data-driven expected outcome (daemon line 1813) |
| `save(path)` / `load(path)` | 2047 / 2117 | Persistence |

Module helpers: `load_world_model()` (2246), `save_world_model()` (2251),
`format_world_model_context()` (2256).

## think_daemon.py core-loop navigation map

Direct-read verification (2026-07-31, HEAD `d8edfb4cd`) of the daemon's own
structure — the counterpart to the world-model map above. This fulfils the
standing commitment to read and outline the core loop (lines 200–400 turned
out to be infrastructure, not the loop itself: placeholder-step guard,
state helpers, code-drift detection, PID lock; the loop lives in
`_run_cycle_body` / `run_daemon`):

| Function | Line | Role |
|---|---|---|
| `_is_placeholder_step(step)` | 212 | Rejects plan steps with no actionable content (empty/“Step A” desc, “n/a” verification) |
| `load_daemon_state` / `save_daemon_state` | 240 / 250 | v1→v2 migration (cycle_history); atomic JSON persistence |
| `_print_cycle_stats()` | 254 | Reliability summary (`--once` / status output) |
| `_git_head()` | 294 | Repo HEAD short hash for drift detection |
| `_check_code_drift(ds)` | 309 | Warns + records `code_drift` block when repo moved past `startup_head` |
| `_acquire_daemon_lock()` | 341 | Atomic `O_CREAT|O_EXCL` PID lock; stale/zombie-PID takeover; per-cycle ownership re-verify |
| `_release_daemon_lock()` | 431 | Lock release (also on shutdown) |
| `_format_goal_evidence(goal, wm)` | 672 | Goal progress line: verification criteria + world-model evidence (new in d8edfb4cd) |
| `_build_thinking_prompt(state)` | 706 | JSON-instructing prompt: state snapshot, goals w/ evidence, calibration guidance |
| `_try_parse_json(raw)` | 1002 | Lenient JSON extraction from LLM output |
| `_prune_self_model(sm, daemon_state)` | 1069 | Pre-/post-cycle stale-weakness pruning (fixation protection) |
| `_apply_insights(result, state)` | 1515 | Action selection + execution: dedup gate, predict → record → execute → complete → feedback |
| `_select_state_check_action(state)` | 2273 | Fallback-cycle action picker (least-sampled action type) |
| `_local_analysis(state)` | 2341 | No-LLM fallback: data-driven insight + auto-goal creation + state-check action |
| `_bridge_world_model_to_self_model(wm, sm)` | 2620 | Discrepancy patterns → self-model weaknesses |
| `_reconcile_goals_with_world(wm)` | 2755 | Auto-complete goals whose criteria are met by world-model evidence |
| `_auto_activate_goals()` | 2954 | Promote proposed goals → active (Gap 4 → Gap 8 bridge) |
| `_validate_shell_command(cmd)` | 3124 | Pre-flight shell validation (unbalanced quotes, compile check) |
| `_execute_shell_action(cmd, timeout)` | 3190 | Runs shell action, returns `exit=<code>: <out>` canonical outcome |
| `_run_cycle_body(result, ds)` | 3211 | **The core cycle** (steps 1–8 in “One cycle” above) |
| `run_one_cycle()` | 3045 | Timeout wrapper around `_run_cycle_body` |
| `_mark_startup(ds, interval, head)` | 3504 | Stamp `startup_head`, clear stale `code_drift` block |
| `run_daemon(interval, max_cycles)` | 3519 | **Persistent loop**: re-verify lock → drift check → cycle → shutdown check → sleep |
| `_run_verification()` | 3594 | No-LLM self-test of the predict→act→observe→learn cycle |
| `bootstrap_evolve_data()` | 3742 | Seed evolve JSON files (idempotent) |
| `_show_status()` | 3920 | `--status` snapshot |
| `main()` | 4057 | argparse dispatch: `--verify` / `--bootstrap` / `--status` / `--once` / daemon |

Action-execution call sites inside `_apply_insights` (current lines):
`predict_action_outcome` 1865, `record_action` 1909, `complete_action` 1984,
`[PREDICTION]` feedback 2008/2012, timeout close 2025, exception close 2031.

## Prediction feedback (closing the loop)

After each executed action the daemon (lines 2004–2012; timeout/exception
close paths at 2025/2031) appends a
`[PREDICTION ✓/△/✗] error=<n>: expected "<…>" → "<…>"` line to the
combined output that feeds the next cycle's prompt — the LLM sees its own
prediction vs. the actual outcome with an error icon (≤0.3 ✓, ≤0.6 △,
else ✗), so calibration happens at the source, not only in the stored
triples. Risk warnings from `format_action_guidance` are prepended the
same way.

## Verified gap candidates (for future cycles)

- **Cron `--once` vs 200 s timeout**: a `--once` cycle launched *from a
  cron job* can be interrupted at the 3-minute cron hard limit (200 s
  cycle + startup/teardown > 180 s). The persistent daemon
  (`evolve_daemon.sh`, interval 900) is unaffected; only cron-launched
  single cycles hit this.
- **LLM outage depth**: `consecutive_fallback_cycles` reached 7 on
  2026-07-31 (19:26) — the 0-attempt tier + every-4th-cycle probe is
  active by design; recovery detection is expected within 4 cycles of the
  provider returning.
- **Daemon code drift (actioned this cycle)**: the daemon started at
  18:56:30 on HEAD 89692b9f5, before the git_commit staging fix
  (68bc9c8d1, 19:17) landed; `_check_code_drift` flagged it at 19:26.
  Restarted via `evolve_daemon.sh restart` (~19:36) so the running daemon
  stages only evolve-owned paths.
- **Daemon code drift (actioned this cycle, 2nd)**: the daemon restarted at
  ~19:37 on HEAD 555370f0d, before the goal-progress feedback
  (d8edfb4cd, 19:57) landed; `_check_code_drift` flagged it at 20:10.
  Restarted again via `evolve_daemon.sh restart` (~20:15) so the running
  daemon renders per-goal verification criteria + world-model evidence in
  its thinking prompt.
- **think_daemon.py core-loop read (RESOLVED this cycle)**: the standing
  commitment to read lines 200–400 and outline the core loop is fulfilled —
  see the navigation map above. The committed range was infrastructure
  (placeholder guard / state helpers / drift detection / PID lock); the
  loop proper is `_run_cycle_body` (3211) + `run_daemon` (3519), now
  mapped line-by-line at HEAD d8edfb4cd.

## Resolved gaps

- **`git add -A` breadth** (fixed 2026-07-31): the `git_commit` executor
  staged *everything* uncommitted in the workspace repo, so any stray
  change in the hermes-agent tree (build artifacts, website edits,
  half-finished work) was swept into the daemon's auto-sync commit under
  a misleading message. The executor now stages only `_EVOLVE_TRACKED_PATHS`
  (the evolve-owned files), filtered to paths that exist; deletions are
  intentionally not auto-staged. Regression-tested in
  `tests/test_daemon_local_analysis.py`.
