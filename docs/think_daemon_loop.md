# think_daemon.py — Action-Selection Loop (Reference)

Persistent-cognition daemon for the Hermes Evolved Phase 2 world-model loop.
This document describes the action-selection loop as implemented at commit
`d6b741b3d` + the extended-outage retry policy (0-attempt probe skip).
Line numbers and cycle steps are verified against HEAD `feb075dde`
(2026-07-31 22:28; evidence-based self-model resolution for documented
subjects) and re-verified at HEAD `d6afa6faa` (2026-08-01; the
`_action_params_from_act` predict/record parity refactor).
It is the counterpart to the daemon-reliability work: the loop below is what
the PID-lock re-verification protects.

> Line-number drift note (2026-07-31, re-verified at feb075dde): the five
> commits between d8edfb4cd and feb075dde (docs map afc02489d, drift
> auto-restart 48b3ce5cd, probe-cap raise c25ce6646, --once budget
> 83618253f, evidence-based resolution feb075dde) added 267 lines to
> think_daemon.py (+267/−22 overall), so every think_daemon.py reference
> below is ~61–139 lines past the d8edfb4cd numbers. The navigation map in
> the "think_daemon.py core-loop navigation map" section was re-verified by
> direct AST read at HEAD feb075dde; the world-model.py map is unaffected
> (world_model.py is unchanged across those commits — every table line
> number re-confirmed at HEAD).
>
> Second drift (2026-08-01, re-verified at d6afa6faa): the
> `_action_params_from_act` refactor (+28/−25, net +3) inserted a shared
> parameter extractor at line 1743 and shrank `_apply_insights`, so
> `_apply_insights` moved 1743 → 1767 and everything from
> `_auto_detect_provider_from_env` (2311) onward is +3. The world-model.py
> map is unaffected (world_model.py untouched by that commit).
>
> Third drift (2026-08-01, startup-outage-counter reset): `_mark_startup`
> now resets an inherited `consecutive_fallback_cycles >= 3` to the probe
> tier (4) so a fresh daemon re-tests LLM health on its first cycle
> instead of inheriting the dead process's extended-outage blindness.
> The function body grew, so `run_daemon` 3750 → 3770, `_run_verification`
> 3832 → 3852, `bootstrap_evolve_data` 3980 → 4000, `_show_status`
> 4158 → 4178, `main` 4295 → 4315 (+20 past `_mark_startup`). The
> world-model.py map is unaffected.

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
   | ≥3, probe    | 1 attempt × 90 s  | every 4th cycle, so recovery is detected within 4 cycles. Cap raised 30 s → 90 s (2026-07-31): the auxiliary client's internal transport timeout (~30 s) + one in-client retry must fit inside it; a 30 s cap failed probes on endpoints that were merely slow (healthy opencode-go latencies observed at 31 s) |

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
   `agent.self_evolve.create_plan` seeds one (think_daemon.py:3489-3513).
   Placeholder steps (bare "Step A" descriptions, or "V"/"n/a"
   verification) are rejected at creation by `_is_placeholder_step`
   (line 273), so a plan with no actionable content can never re-enter the
   prompt as an active plan and drive a deliberation-fixation loop.
2. **Goal reconciliation (cycle step 11)** — `_reconcile_goals_with_world`
   (line 2986) auto-completes goals whose verification conditions are met
   by world-model evidence, so finished work is retired without an LLM.
3. **Goal auto-activation (cycle step 12)** — `_auto_activate_goals`
   (line 3185) promotes proposed goals to active when capacity exists,
   closing the Gap 4 → Gap 8 loop without waiting for the LLM to set
   `goal_action` in its JSON.
4. **Outage-path goal creation** — `_local_analysis` (line 2572) converts
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
- **Evidence-based self-model resolution** — `_prune_self_model` also clears
  weaknesses/unknowns/commitments about *documented subjects* (source file
  exists + reference doc exists — e.g. think_daemon.py + this doc) regardless
  of phrasing, and `_apply_insights` refuses to re-add them. This replaced the
  regex-whack-a-mole that let new phrasings of the resolved think_daemon.py
  structure read survive for many cycles.
- **Code-drift visibility + auto-restart** — stale daemon processes surface
  via the `code_drift` marker + launcher `status`; `_schedule_drift_restart`
  (48b3ce5cd) additionally schedules a detached restart onto fresh code 5 s
  after drift is detected. First observed in the wild 2026-07-31 22:28: the
  83618253f → feb075dde drift auto-restarted the daemon, and
  `daemon_state.startup_head` now tracks feb075dde with `code_drift: null`.

## World-model API surface (world_model.py, navigation map)

The daemon consumes the `WorldModel` class (line 332). Methods referenced
from `think_daemon.py`, with line numbers as of HEAD 68bc9c8d1:

| Method | Line | Role in the loop |
|---|---|---|
| `_compute_prediction_error(expected, actual)` | 139 | Canonical error metric (0 = exact match … 1 = unrelated) |
| `record_action(type, desc, expected, source, …)` | 366 | Opens an action triple (predict step) |
| `complete_action(triple_id, actual)` | 444 | Closes the triple, computes error (called from daemon line 2215; timeout/exception close paths at 2256/2262) |
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
| `predict_action_outcome(type, desc, params)` | 1846 | Data-driven expected outcome (daemon line 2106, params via `_action_params_from_act`) |
| `save(path)` / `load(path)` | 2047 / 2117 | Persistence |

Module helpers: `load_world_model()` (2246), `save_world_model()` (2251),
`format_world_model_context()` (2256).

## think_daemon.py core-loop navigation map

Direct-read verification (2026-08-01, HEAD `d6afa6faa`) of the daemon's own
structure — the counterpart to the world-model map above. This fulfils the
standing commitment to read and outline the core loop (lines 200–400 turned
out to be infrastructure, not the loop itself: placeholder-step guard,
state helpers, code-drift detection, PID lock; the loop lives in
`_run_cycle_body` / `run_daemon`):

| Function | Line | Role |
|---|---|---|
| `_is_placeholder_step(step)` | 273 | Rejects plan steps with no actionable content (empty/“Step A” desc, “n/a” verification) |
| `load_daemon_state` / `save_daemon_state` | 301 / 311 | v1→v2 migration (cycle_history); atomic JSON persistence |
| `_print_cycle_stats()` | 315 | Reliability summary (`--once` / status output) |
| `_git_head()` | 355 | Repo HEAD short hash for drift detection |
| `_check_code_drift(ds)` | 370 | Warns + records `code_drift` block when repo moved past `startup_head` |
| `_schedule_drift_restart(ds)` | 415 | Drift auto-restart (48b3ce5cd): schedules detached `evolve_daemon.sh restart` in 5 s, exits this process |
| `_acquire_daemon_lock()` | 480 | Atomic `O_CREAT|O_EXCL` PID lock; stale/zombie-PID takeover; per-cycle ownership re-verify |
| `_release_daemon_lock()` | 570 | Lock release (also on shutdown) |
| `_format_goal_evidence(goal, wm)` | 811 | Goal progress line: verification criteria + world-model evidence (new in d8edfb4cd) |
| `_build_thinking_prompt(state)` | 845 | JSON-instructing prompt: state snapshot, goals w/ evidence, calibration guidance |
| `_try_parse_json(raw)` | 1141 | Lenient JSON extraction from LLM output |
| `_subject_is_resolved(text)` | 1242 | Evidence-based resolution check (feb075dde): subject is documented (source file + reference doc exist) |
| `_prune_self_model(sm, daemon_state)` | 1264 | Pre-/post-cycle stale-weakness pruning + documented-subject resolution (fixation protection) |
| `_action_params_from_act(act, atype)` | 1743 | Shared parameter extractor for predict/record parity (new in d6afa6faa) |
| `_apply_insights(result, state)` | 1767 | Action selection + execution: dedup gate, predict → record → execute → complete → feedback |
| `_auto_detect_provider_from_env` / `_ensure_runtime_main` / `_ensure_provider_env` | 2314 / 2334 / 2407 | Runtime provider bootstrap (opencode-go main, key propagation) |
| `_call_llm(messages, task)` | 2435 | Async LLM call via auxiliary client under the adaptive retry budget |
| `_select_state_check_action(state)` | 2504 | Fallback-cycle action picker (least-sampled action type) |
| `_local_analysis(state)` | 2572 | No-LLM fallback: data-driven insight + auto-goal creation + state-check action |
| `_bridge_world_model_to_self_model(wm, sm)` | 2851 | Discrepancy patterns → self-model weaknesses |
| `_reconcile_goals_with_world(wm)` | 2986 | Auto-complete goals whose criteria are met by world-model evidence |
| `_auto_activate_goals()` | 3185 | Promote proposed goals → active (Gap 4 → Gap 8 bridge) |
| `run_one_cycle()` | 3276 | Timeout wrapper around `_run_cycle_body` |
| `_validate_shell_command(cmd)` | 3355 | Pre-flight shell validation (unbalanced quotes, compile check) |
| `_execute_shell_action(cmd, timeout)` | 3421 | Runs shell action, returns `exit=<code>: <out>` canonical outcome |
| `_run_cycle_body(result, ds)` | 3442 | **The core cycle** (steps 1–8 in “One cycle” above) |
| `_mark_startup(ds, interval, head)` | 3735 | Stamp `startup_head`, clear stale `code_drift` block, reset inherited extended-outage counter to probe tier |
| `run_daemon(interval, max_cycles)` | 3770 | **Persistent loop**: re-verify lock → drift check → cycle → shutdown check → sleep |
| `_run_verification()` | 3852 | No-LLM self-test of the predict→act→observe→learn cycle |
| `bootstrap_evolve_data()` | 4000 | Seed evolve JSON files (idempotent) |
| `_show_status()` | 4178 | `--status` snapshot |
| `main()` | 4315 | argparse dispatch: `--verify` / `--bootstrap` / `--status` / `--once` / daemon |

Action-execution call sites inside `_apply_insights` (current lines):
`_action_params_from_act` 2105 (predict path) / 2139 (record path),
`predict_action_outcome` 2106, `record_action` 2140, `format_action_guidance`
2146, `complete_action` 2215, `[PREDICTION]` feedback 2239/2243, timeout
close 2256, exception close 2262.

## Prediction feedback (closing the loop)

After each executed action the daemon (lines 2236–2240; timeout/exception
close paths at 2253/2259) appends a
`[PREDICTION ✓/△/✗] error=<n>: expected "<…>" → "<…>"` line to the
combined output that feeds the next cycle's prompt — the LLM sees its own
prediction vs. the actual outcome with an error icon (≤0.3 ✓, ≤0.6 △,
else ✗), so calibration happens at the source, not only in the stored
triples. Risk warnings from `format_action_guidance` are prepended the
same way.

## Verified gap candidates (for future cycles)

- **LLM outage depth**: `consecutive_fallback_cycles` reached 3 at tick 296
  (2026-07-31 23:02) — the ≥3 extended-outage tier (0-attempt skip +
  every-4th-cycle 90 s probe) engages at tick 297 by design; recovery
  detection is expected within 4 cycles of the provider returning.
  **Observed through tick 305 (2026-08-01 01:19):** the outage deepened to
  11 consecutive fallback cycles — the longest continuous outage recorded.
  The periodic probes (fired when the counter hits a multiple of 4) at
  ticks 298 and 302 both failed, confirming the tier is working as
  designed (cycle budget preserved for local analysis + action execution
  instead of dead LLM time) and that the provider had not yet recovered.
  The next probe fires at tick 306 (counter 12). Ticks 297/301/303-305
  were pure local-analysis + action-execution cycles per daemon_state/
  last_output; ticks 298 and 302 each burned one bounded 90 s probe.
- **Startup outage-counter inheritance (FIXED 2026-08-01)**: a restarted
  daemon inherited `consecutive_fallback_cycles` from the dead process,
  so after the 01:04 drift-restart (which inherited counter=10) ticks
  304-305 skipped LLM probes even though a direct `_call_llm` probe from
  the workspace venv returned PONG in 2.9 s at 01:22 — the provider had
  recovered but the fresh process stayed blind until the counter would
  reach 12. `_mark_startup` now resets an inherited counter ≥ 3 to the
  probe tier (4), so the first cycle after any restart re-tests LLM
  health with one bounded 90 s attempt; a still-down endpoint costs only
  that single attempt before the skip tier re-engages (counter → 5).
  Regression-tested in `TestCodeDriftDetection`.
- **Daemon code drift (auto-restart working, observed 22:28)**: the daemon
  drifted 83618253f → feb075dde and `_schedule_drift_restart` auto-restarted
  it; `daemon_state.startup_head = feb075dde`, `code_drift = null` as of
  tick 296. No manual restart needed — the 48b3ce5cd mechanism is proven in
  the wild.
- **think_daemon.py core-loop read (RE-VERIFIED 2026-08-01)**: the standing
  commitment to read and outline the core loop is fulfilled again at HEAD
  d6afa6faa — see the navigation map above. The `_action_params_from_act`
  refactor (+28/−25, net +3) added a shared parameter extractor at line 1743
  (`_apply_insights` moved 1743 → 1767) and shifted everything from
  `_auto_detect_provider_from_env` (2311) onward by +3; all numbers above
  were re-read by AST at d6afa6faa, and the world-model.py map was
  re-confirmed unchanged (world_model.py untouched by that commit).

## Resolved gaps

- **Cron `--once` vs 200 s timeout** (fixed 2026-07-31): a `--once` cycle
  launched *from a cron job* could be interrupted at the 3-minute cron
  hard limit (200 s cycle + startup/teardown > 180 s). `main()` now
  applies a wall-clock budget (`--budget`, default 170 s) to `--once`
  runs via `_apply_cycle_budget()`; `_llm_retry_policy()` clamps its
  retry budget so the worst-case LLM phase fits the remaining wall clock
  (healthy 2×90 s shrinks to a single 90 s attempt), leaving room for
  local analysis + action execution before the kill. The persistent
  daemon passes no budget and is byte-for-byte unaffected. Regression
  tests in `tests/test_daemon_local_analysis.py`
  (`TestLlmRetryPolicyBudget`).
- **`git add -A` breadth** (fixed 2026-07-31): the `git_commit` executor
  staged *everything* uncommitted in the workspace repo, so any stray
  change in the hermes-agent tree (build artifacts, website edits,
  half-finished work) was swept into the daemon's auto-sync commit under
  a misleading message. The executor now stages only `_EVOLVE_TRACKED_PATHS`
  (the evolve-owned files), filtered to paths that exist; deletions are
  intentionally not auto-staged. Regression-tested in
  `tests/test_daemon_local_analysis.py`.
