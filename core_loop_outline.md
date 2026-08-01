# think_daemon.py — Core Loop & World Model Integration Outline

*Line numbers re-verified against live code root 2026-08-01 (4835 lines).*
*File: 4835 lines (~225 KB). Module-level rotating auto-default actions begin ~line 64.*
*Status: living document — update as the daemon evolves.*

This outline maps the daemon's continuous loop and its prediction integration
(action → expected_outcome → actual_outcome → prediction error → feedback), so
future cycles can navigate the file without re-deriving its structure.

---

## 1. Entry & Process Model

```
main() (4775)
 └─ subcommand dispatch: run | once | verify | bootstrap | status
     ├─ run    → run_daemon(interval_seconds, max_cycles)   # continuous loop
     ├─ once   → run_one_cycle()                            # single cycle
     ├─ verify → _run_verification()                        # no-LLM self-test
     ├─ bootstrap → bootstrap_evolve_data()
     └─ status → _show_status()
```

## 2. Continuous Loop — `run_daemon(interval_seconds=600, max_cycles=0)` (4230)

```
_acquire_daemon_lock() (501)          # PID lock; skip if another daemon runs
_mark_startup(ds, interval, head)
save_daemon_state(ds)
while True:
    _acquire_daemon_lock()            # re-verify ownership every cycle
                                      #   (prevents two interleaved daemons)
    if max_cycles and cycle > max_cycles: break
    _check_code_drift(ds)             # repo moved past loaded code?
    └─ _schedule_drift_restart(ds)    # spawn detached restart, then break
    await run_one_cycle()             # one thinking cycle (see §3)
    if ds.status == "shutdown": break
    await asyncio.sleep(interval_seconds)
finally:
    _release_daemon_lock()
```

## 3. One Cycle — `run_one_cycle()` (3680)

```
pre-cycle health snapshot: ds["cycle_stats"] (total/ok/error/parse_error/avg/max)
result = await asyncio.wait_for(
    _run_cycle_body(result, ds), timeout=_CYCLE_HARD_TIMEOUT=200.0)
  ├─ TimeoutError → status="timeout"
  └─ Exception    → status="crash"
post-cycle: duration stats, cycle_history (last 20), save_daemon_state
```

## 4. Cycle Body — `_run_cycle_body(result, ds)` (3846)

Step offsets below are relative to the function start (3846) — add the offset
to get the current line (e.g. offset 105 → 3846+105 = 3951).

| Step | What happens | Offset |
|------|--------------|--------|
| 1    | Load state: timeline, self_model, orientation, world_model into `state` | 0 |
| 1.75 | Auto-verify predictions: `wm.verify_expired_predictions()` + `wm.verify_pending_predictions()`, `wm.save()` immediately if any | 33 |
| 1.5  | Auto-create initial plan if none (guarded against duplicate active/complete plans) | 45 |
| 1.75 | Pre-cycle `_prune_self_model` (1431) — prompt shows clean state | 79 |
| 2    | `_build_thinking_prompt(state)` (944) — includes world-model context and last action's prediction feedback | 100 |
| 3    | `_call_llm(messages)` (2724) — adaptive retry budget `_set_llm_retry_policy` (299); `None` → `_local_analysis` (2927) fallback + `consecutive_fallback_cycles` tracking + recovery note | 105 |
| 3.25 | Record the LLM-call outcome as an `llm_call` world-model triple (guarded on the `_fresh` marker; params carry tier/attempts/backoff telemetry) | 125 |
| 4    | `_try_parse_json(raw)` (1261); `parse_error` → early return | 200 |
| 4.5  | Search phase: DDGS web search → follow-up LLM call, re-parse | 208 |
| 4.75 | `_coerce_llm_response_fields(parsed)` (1992) — type-guard all LLM fields | 245 |
| 5    | **`_apply_insights(parsed, state)` (2030) — the predict→act→observe→learn core (see §5)** | 248 |
| 5.25 | `_bridge_world_model_to_self_model(wm, sm)` (3206) — discrepancy patterns → self-model weaknesses | 250 |
| 5.3  | `_reconcile_goals_with_world(wm)` (3341) — auto-complete stale goals | 259 |
| 5.4  | `_auto_activate_goals()` (3570) — promote proposed goals to active | 269 |
| 5.5  | Record LLM prediction: `wm.adjust_confidence` + `wm.record_prediction`; `wm.save()` | 280 |
| 6    | Save state: timeline, self_model (total_cycles++), orientation, wm | 294 |
| 7    | daemon_state: last_tick, tick_count++, last_output (+fallback flag, recovery note); `save_daemon_state` | 307 |
| 8    | Build result (insight, focus_next, confidence, duration) | 332 |

## 5. Prediction Integration — `_apply_insights` action path (2030)

The world-model learning loop inside each action execution:

```
record BEFORE execution:
  wm.record_action(atype, desc, expected, expected_source,
                   prediction_confidence, parameters)
  expected = act["expected_outcome"]          # LLM-provided
  OR data-driven: wm.predict_action_outcome
     via _action_params_from_act              # same params for predict & record
  source: "llm" | "world_model" | "fallback"

execute:
  write_file    → p.write_text(content)
  shell         → _execute_shell_action (3825)
                   └─ pre-flight _validate_shell_command (3759)
                      (quote balance + embedded python3 -c compile check)
  git_commit    → git add (evolve-tracked paths only) + commit
  install_package → pip install

observe AFTER:
  _pred_err = wm.complete_action(triple_id, actual_outcome)

feedback:  # closes the loop for the NEXT prompt
  _build_prediction_feedback_line (1955)
  → "[PREDICTION ✓/△/✗] error=X: expected \"…\" → \"…\""
  → ds["last_action_output"] → shown to LLM next cycle

failure paths:  # previously silent — now they close the loop too
  except subprocess.TimeoutExpired → complete_action(triple_id, "TIMEOUT: <type>")
                                     + prediction feedback line
  except Exception                → complete_action(triple_id, "FAILED: <err>")
                                     + prediction feedback line
  finally → wm.save()

calibration (inside world_model.complete_action):
  _update_calibration, _update_per_type_accuracy, _update_discrepancy_patterns,
  _update_accuracy_stats, _add_to_error_history
```

## 6. LLM Call — `_call_llm(messages, task="thinking")` (2724)

- Resolves provider/model via `_ensure_runtime_main` (2623); passes explicitly
  to `agent.auxiliary_client.async_call_llm` (bypasses auto-detect).
- Per-attempt timeout passed INTO the auxiliary client (not just outer
  `wait_for`) — fixes "outage blindness" for slow-but-healthy endpoints.
- Retries with exponential backoff (2s, 4s, 8s); retry budget shrinks during
  extended outages (`_clamp_retry_budget` 181, `_llm_retry_policy` 237,
  `_llm_retry_tier` + `_LLM_RETRY_BUDGETS` 201-234 — single source of truth
  for tier names and budgets).
- **Retry-path telemetry (P3, 2026-08-01):** every call writes
  `_last_llm_call_stats` (policy_tier, max_retries, per_attempt_timeout,
  attempts_used, backoff_slept_s, duration_s, success, skipped, last_error).
  The cycle body persists this as the `llm_call` world-model triple's
  parameters, so retry/skip decisions feed prediction calibration.

## 7. Local-Analysis Fallback — `_local_analysis(state)` (2927)

Runs when the LLM is unavailable. No standalone LLM predictions, but actions
still go through the same world-model data-driven prediction path
(`_select_state_check_action` 2859) so triples/errors keep accumulating.

## 8. Verification

- `_run_verification()` (4312) — no-LLM self-test of the full
  predict→act→observe→learn cycle on an isolated world-model copy.
- `scripts/verify_loop_map.py` — AST navigation-map verification.
- Tests: `tests/agent/test_think_daemon.py`,
  `tests/test_world_model.py`, `tests/agent/test_world_model.py`,
  `tests/test_daemon_local_analysis.py`, `tests/test_wm_self_bridge.py`
  (520 passed after the 2026-08-01 telemetry change).

## Change Log

- 2026-08-01: Restored the detailed outline (last cycle's rewrite had slimmed
  it to 30 lines) with fresh line numbers for the 4835-line file. Added the
  P3 retry-path telemetry section (`_llm_retry_tier`/`_LLM_RETRY_BUDGETS` +
  `_last_llm_call_stats` backoff/tier fields).
- 2026-08-01 (previous cycle): Verified structural markers against function
  bodies; added `_build_prediction_feedback_line`; wired prediction feedback
  into failure/timeout action paths.
