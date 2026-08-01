# think_daemon.py — Core Loop & World Model Integration Outline

*Verified against the code at HEAD `d3ec3a61f` (+ failure-feedback fix, this cycle).*
*File: 4428 lines (~206 KB). Module-level rotating auto-default actions begin ~line 51.*

This outline maps the daemon's continuous loop and its prediction integration
(action → expected_outcome → actual_outcome → prediction error → feedback), so
future cycles can navigate the file without re-deriving its structure.

---

## 1. Entry & Process Model

```
main() (4352)
 └─ subcommand dispatch: run | once | verify | bootstrap | status
     ├─ run    → run_daemon(interval_seconds, max_cycles)   # continuous loop
     ├─ once   → run_one_cycle()                            # single cycle
     ├─ verify → _run_verification()                        # no-LLM self-test
     ├─ bootstrap → bootstrap_evolve_data()
     └─ status → _show_status()
```

## 2. Continuous Loop — `run_daemon(interval_seconds=600, max_cycles=0)` (3807)

```
_acquire_daemon_lock() (501)          # PID lock; skip if another daemon runs
_mark_startup(ds, interval, head) (3772)
save_daemon_state(ds)
while True:
    _acquire_daemon_lock()            # re-verify ownership every cycle
                                      #   (prevents two interleaved daemons)
    if max_cycles and cycle > max_cycles: break
    _check_code_drift(ds) (391)       # repo moved past loaded code?
    └─ _schedule_drift_restart(ds) (436)  # spawn detached restart, then break
    await run_one_cycle()             # one thinking cycle (see §3)
    if ds.status == "shutdown": break
    await asyncio.sleep(interval_seconds)
finally:
    _release_daemon_lock() (591)
```

## 3. One Cycle — `run_one_cycle()` (3313)

```
pre-cycle health snapshot: ds["cycle_stats"] (total/ok/error/parse_error/avg/max)
result = await asyncio.wait_for(
    _run_cycle_body(result, ds), timeout=_CYCLE_HARD_TIMEOUT=200.0)
  ├─ TimeoutError → status="timeout"
  └─ Exception    → status="crash"
post-cycle: duration stats, cycle_history (last 20), save_daemon_state
```

## 4. Cycle Body — `_run_cycle_body(result, ds)` (3479)

| Step | What | Where |
|------|------|-------|
| 1    | Load state (timeline, self_model, orientation, world_model) | 3482-3494 |
| 1.75 | Auto-verify predictions: `wm.verify_expired_predictions()` (820), `wm.verify_pending_predictions()` (921) → save | 3496-3513 |
| 1.5  | Auto-create initial Gap-8 plan if none (guarded against duplicate active/complete plans) | 3515-3550 |
| 1.75 | Pre-cycle `_prune_self_model` (1285) — prompt shows clean state | 3552-3569 |
| 2    | `_build_thinking_prompt(state)` (866) — includes world-model context (677-678, 1043-1049) **and last action's prediction feedback** (1035-1037) | 3571-3576 |
| 3    | `_call_llm(messages)` (2460) — adaptive retry budget `_set_llm_retry_policy` (264); `None` → `_local_analysis` (2609) fallback + `consecutive_fallback_cycles` tracking + recovery note | 3578-3624 |
| 4    | `_try_parse_json(raw)` (1162); `parse_error` → early return | 3626-3632 |
| 4.5  | Search phase: DDGS web search → follow-up LLM call, re-parse | 3634-3667 |
| 5    | **`_apply_insights(parsed, state)` (1788) — the predict→act→observe→learn core (see §5)** | 3669-3670 |
| 5.25 | `_bridge_world_model_to_self_model(wm, sm)` (2888) — discrepancy patterns → self-model weaknesses | 3672-3679 |
| 5.3  | `_reconcile_goals_with_world(wm)` (3023) — auto-complete stale goals | 3681-3689 |
| 5.4  | `_auto_activate_goals()` (3222) — promote proposed goals to active | 3691-3700 |
| 5.5  | Record LLM prediction: `wm.adjust_confidence` (1512) + `wm.record_prediction` (512) | 3702-3714 |
| 6    | Save state: timeline, self_model (total_cycles++), orientation, wm | 3716-3727 |
| 7    | daemon_state: last_tick, tick_count++, last_output (+fallback flag, recovery note) | 3729-3752 |
| 8    | Build result (insight, focus_next, confidence, duration) | 3754-3765 |

## 5. Prediction Integration — `_apply_insights` action path (2106-2310)

The world-model learning loop inside each action execution:

```
record BEFORE execution (2165):
  wm.record_action(atype, desc, expected, expected_source,
                   prediction_confidence, parameters)
  expected = act["expected_outcome"]          # LLM-provided
  OR data-driven: wm.predict_action_outcome(1846)
     via _action_params_from_act (1764)       # same params for predict & record
  source: "llm" | "world_model" | "fallback"

execute (2186-2233):
  write_file    → p.write_text(content)
  shell         → _execute_shell_action (3458)
                   └─ pre-flight _validate_shell_command (3392)
                      (quote balance + embedded python3 -c compile check)
  git_commit    → git add (evolve-tracked paths only) + commit
  install_package → pip install

observe AFTER (2238-2245):
  _pred_err = wm.complete_action(triple_id, actual_outcome)

feedback (2247-2274):  # closes the loop for the NEXT prompt
  _build_prediction_feedback_line (1788)  # helper [NEW this cycle]
  → "[PREDICTION ✓/△/✗] error=X: expected \"…\" → \"…\""
  → ds["last_action_output"] → shown to LLM next cycle (1035-1037)

failure paths (2276-2308):   # [NEW this cycle — previously silent]
  except subprocess.TimeoutExpired → complete_action(triple_id, "TIMEOUT: <type>")
                                     + prediction feedback line
  except Exception                → complete_action(triple_id, "FAILED: <err>")
                                     + prediction feedback line
  finally → wm.save()

calibration (inside world_model.complete_action → _update_*):
  _update_calibration (1288), _update_per_type_accuracy (1322),
  _update_discrepancy_patterns (1366), _update_accuracy_stats (1201),
  _add_to_error_history (1273)
```

## 6. LLM Call — `_call_llm(messages, task="thinking")` (2460)

- Resolves provider/model via `_ensure_runtime_main` (2359); passes explicitly
  to `agent.auxiliary_client.async_call_llm` (bypasses auto-detect).
- Per-attempt timeout passed INTO the auxiliary client (not just outer
  `wait_for`) — fixes "outage blindness" for slow-but-healthy endpoints.
- Retries with exponential backoff (2s, 4s, 8s); retry budget shrinks during
  extended outages (`_clamp_retry_budget` 166, `_llm_retry_policy` 188).

## 7. Local-Analysis Fallback — `_local_analysis(state)` (2609)

Runs when the LLM is unavailable. No standalone LLM predictions, but actions
still go through the same world-model data-driven prediction path
(`_select_state_check_action` 2541) so triples/errors keep accumulating.

## 8. Verification

- `_run_verification()` (3889) — no-LLM self-test of the full
  predict→act→observe→learn cycle on an isolated world-model copy.
- `scripts/verify_loop_map.py` — AST navigation-map verification.
- Tests: `tests/agent/test_think_daemon.py`,
  `tests/test_world_model.py`, `tests/agent/test_world_model.py`,
  `tests/test_daemon_local_analysis.py`, `tests/test_wm_self_bridge.py`
  (482 passed before this cycle's change; re-run after).

## Change Log

- 2026-08-01: Added `_build_prediction_feedback_line` helper; wired prediction
  feedback into the failure/timeout action paths (previously only successful
  actions closed the prediction loop). Added
  `test_failed_action_sets_prediction_feedback` +
  `test_timed_out_action_sets_prediction_feedback`.
