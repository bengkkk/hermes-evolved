#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# evolve_daemon.sh — start/status/stop for the persistent cognition daemon
# (think_daemon.py, Gap 1 / Gap 8 "continuous operation").
#
# Why this exists: the daemon's own PID lock (daemon.lock, O_EXCL atomic
# create) prevents two daemons from running, but it does NOT make the daemon
# durable. If launched as a plain background child of a shell, the daemon
# dies when that shell exits. This launcher detaches with setsid(1) so the
# daemon survives the launching shell, and provides a reproducible restart
# path for a systemd-less environment (no ps/pgrep/systemctl available —
# process checks go through /proc).
#
# Usage:
#   ./evolve_daemon.sh start [--interval SECONDS]   # default interval 900
#   ./evolve_daemon.sh restart [--interval SECONDS] # stop + start (loads new code)
#   ./evolve_daemon.sh status                       # report current state + code drift
#   ./evolve_daemon.sh stop                         # SIGTERM the running daemon
#   ./evolve_daemon.sh permissions ...              # grant/revoke/check external-action
#                                                   # permissions (Gap 10 Level 2 on-ramp)
#                                                   # e.g. `permissions grant github write`
#
# Env: HERMES_HOME (default: $HOME/.hermes-evolved) selects the evolve dir.
# ─────────────────────────────────────────────────────────────────────────────

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes-evolved}"
EVOLVE_DIR="$HERMES_HOME/evolve"
LOCK_FILE="$EVOLVE_DIR/daemon.lock"
LAUNCHER_LOG="$EVOLVE_DIR/daemon_launcher.log"
BRIDGE_PID_FILE="$EVOLVE_DIR/bridge.pid"
BRIDGE_LOG="$EVOLVE_DIR/bridge.log"
BRIDGE_TOKEN_FILE="$EVOLVE_DIR/bridge_token"
BRIDGE_PORT="${HERMES_EVOLVED_BRIDGE_PORT:-8791}"
INTERVAL=900

# Shared bridge token: one file, chmod 600, used by BOTH the daemon (which
# forwards it as Bearer auth) and the bridge (which validates it). Generated
# on first use; persists across restarts so daemon/bridge stay in sync.
_ensure_bridge_token() {
    if [ ! -f "$BRIDGE_TOKEN_FILE" ]; then
        .venv/bin/python3 - "$BRIDGE_TOKEN_FILE" <<'PY' 2>/dev/null || true
import secrets, sys
with open(sys.argv[1], "w") as f:
    f.write(secrets.token_hex(32) + "\n")
PY
        chmod 600 "$BRIDGE_TOKEN_FILE" 2>/dev/null || true
    fi
}

_pid_alive() {
    local pid="$1"
    [ -d "/proc/$pid" ] || return 1
    # A zombie can never run again — treat as dead so we can take over.
    if [ -f "/proc/$pid/status" ]; then
        if grep -q "Z (zombie)" "/proc/$pid/status" 2>/dev/null; then
            return 1
        fi
    fi
    return 0
}

_locked_pid() {
    [ -f "$LOCK_FILE" ] || { echo ""; return; }
    cat "$LOCK_FILE" 2>/dev/null | tr -d '[:space:]'
}

_drift_line() {
    # Surface code drift from daemon_state.json (written by
    # _check_code_drift in think_daemon.py): the daemon is running code
    # from an older commit than the repo.  Empty when there is no drift.
    local state_file="$EVOLVE_DIR/daemon_state.json"
    [ -f "$state_file" ] || { echo ""; return; }
    .venv/bin/python3 - "$state_file" <<'PY' 2>/dev/null
import json, sys
try:
    with open(sys.argv[1]) as f:
        ds = json.load(f)
except Exception:
    sys.exit(0)
cd = ds.get("code_drift") or {}
if cd.get("startup_head") and cd.get("startup_head") != cd.get("current_head"):
    print("  DRIFT: daemon running %s, repo at %s (detected %s)" % (
        cd["startup_head"], cd["current_head"], (cd.get("detected_at") or "?")[:19]))
    print("  hint: ./evolve_daemon.sh restart to load latest logic")
PY
}

cmd_status() {
    local pid
    pid="$(_locked_pid)"
    if [ -n "$pid" ] && _pid_alive "$pid"; then
        local started state drift
        started="$(stat -c '%y' "/proc/$pid" 2>/dev/null | cut -d. -f1)"
        state="$(awk '/^State:/{print $2 " " $3}' "/proc/$pid/status" 2>/dev/null)"
        echo "RUNNING: PID $pid (state: $state, started: $started)"
        echo "  cmd: $(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)"
        echo "  log: $EVOLVE_DIR/daemon.log (tail for cycle activity)"
        drift="$(_drift_line)"
        [ -n "$drift" ] && echo "$drift"
        return 0
    fi
    if [ -n "$pid" ]; then
        echo "STALE: lock file holds dead PID $pid — daemon not running"
    else
        echo "STOPPED: no lock file — daemon not running"
    fi
    return 1
}

cmd_start() {
    local pid
    pid="$(_locked_pid)"
    if [ -n "$pid" ] && _pid_alive "$pid"; then
        echo "already running (PID $pid) — refusing to start a second daemon"
        cmd_status
        return 0
    fi
    if [ -n "$pid" ]; then
        echo "clearing stale lock (dead PID $pid)"
        rm -f "$LOCK_FILE"
    fi
    mkdir -p "$EVOLVE_DIR"
    # Ensure the shared bridge token exists BEFORE launching the daemon so
    # the daemon inherits it (HERMES_EVOLVED_BRIDGE_TOKEN) and can
    # authenticate against the bridge once it is started.
    _ensure_bridge_token
    export HERMES_EVOLVED_BRIDGE_TOKEN
    HERMES_EVOLVED_BRIDGE_TOKEN="$(cat "$BRIDGE_TOKEN_FILE" 2>/dev/null || true)"
    # Detach: new session (setsid), ignore SIGHUP (nohup), no stdin.
    # The daemon writes its own daemon.log via logging.FileHandler; this
    # launcher log captures stray stdout/stderr (startup tracebacks etc.).
    setsid nohup .venv/bin/python3 think_daemon.py \
        --interval "$INTERVAL" --log-level INFO \
        </dev/null >>"$LAUNCHER_LOG" 2>&1 &
    local new_pid=$!
    echo "started daemon (PID $new_pid, interval ${INTERVAL}s)"
    echo "  launcher log: $LAUNCHER_LOG"
    sleep 2
    if _pid_alive "$new_pid"; then
        echo "  OK: process alive after 2s; state: $(awk '/^State:/{print $2}' /proc/$new_pid/status 2>/dev/null)"
    else
        echo "  WARNING: process not alive after 2s — check launcher log"
        tail -5 "$LAUNCHER_LOG" 2>/dev/null
    fi
}

cmd_stop() {
    local pid
    pid="$(_locked_pid)"
    if [ -z "$pid" ] || ! _pid_alive "$pid"; then
        echo "no running daemon to stop"
        [ -n "$pid" ] && rm -f "$LOCK_FILE"
        return 1
    fi
    echo "sending SIGTERM to PID $pid"
    kill -TERM "$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null
    for _ in $(seq 1 10); do
        _pid_alive "$pid" || break
        sleep 1
    done
    if _pid_alive "$pid"; then
        echo "  WARNING: still alive after 10s — sending SIGKILL"
        kill -9 "$pid" 2>/dev/null
    else
        echo "  stopped"
    fi
    rm -f "$LOCK_FILE"
}

cmd_restart() {
    echo "=== restarting evolve daemon (daemon + bridge) ==="
    cmd_stop || true
    cmd_bridge_stop || true
    cmd_bridge_start
    cmd_start
}

# ── host bridge (Gap 10 step 3) ──────────────────────────────────────────
_bridge_pid() {
    [ -f "$BRIDGE_PID_FILE" ] || { echo ""; return; }
    cat "$BRIDGE_PID_FILE" 2>/dev/null | tr -d '[:space:]'
}

cmd_bridge_start() {
    local pid port="${BRIDGE_PORT}"
    pid="$(_bridge_pid)"
    if [ -n "$pid" ] && _pid_alive "$pid"; then
        echo "bridge already running (PID $pid) on port $port"
        return 0
    fi
    _ensure_bridge_token
    export HERMES_EVOLVED_BRIDGE_TOKEN
    HERMES_EVOLVED_BRIDGE_TOKEN="$(cat "$BRIDGE_TOKEN_FILE" 2>/dev/null || true)"
    setsid nohup .venv/bin/python3 evolve_bridge.py --port "$port" \
        </dev/null >>"$BRIDGE_LOG" 2>&1 &
    local new_pid=$!
    echo "$new_pid" > "$BRIDGE_PID_FILE"
    echo "started host bridge (PID $new_pid, port $port, token file $BRIDGE_TOKEN_FILE)"
    echo "  bridge log: $BRIDGE_LOG"
    sleep 2
    if _pid_alive "$new_pid"; then
        echo "  OK: process alive after 2s"
    else
        echo "  WARNING: bridge not alive after 2s — check log"
        tail -5 "$BRIDGE_LOG" 2>/dev/null
    fi
}

cmd_bridge_stop() {
    local pid
    pid="$(_bridge_pid)"
    if [ -z "$pid" ] || ! _pid_alive "$pid"; then
        echo "no running bridge to stop"
        rm -f "$BRIDGE_PID_FILE"
        return 1
    fi
    echo "sending SIGTERM to bridge PID $pid"
    kill -TERM "$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null
    for _ in $(seq 1 10); do
        _pid_alive "$pid" || break
        sleep 1
    done
    _pid_alive "$pid" && kill -9 "$pid" 2>/dev/null
    rm -f "$BRIDGE_PID_FILE"
    echo "  bridge stopped"
}

cmd_bridge_status() {
    local pid started
    pid="$(_bridge_pid)"
    if [ -n "$pid" ] && _pid_alive "$pid"; then
        started="$(stat -c '%y' "/proc/$pid" 2>/dev/null | cut -d. -f1)"
        echo "BRIDGE RUNNING: PID $pid (port $BRIDGE_PORT)"
        echo "  log: $BRIDGE_LOG"
        [ -n "$started" ] && echo "  started: $started (compare against repo HEAD: $(git -C "$SCRIPT_DIR" log -1 --format=%ci 2>/dev/null))"
        return 0
    fi
    echo "bridge STOPPED"
    return 1
}

case "${1:-status}" in
    start)
        # optional --interval N
        if [ "${2:-}" = "--interval" ]; then
            INTERVAL="${3:-900}"
        fi
        cmd_start
        ;;
    restart)
        # optional --interval N (same handling as start)
        if [ "${2:-}" = "--interval" ]; then
            INTERVAL="${3:-900}"
        fi
        cmd_restart
        ;;
    status)
        cmd_status
        ;;
    stop)
        cmd_stop
        ;;
    bridge)
        case "${2:-status}" in
            start)
                if [ "${3:-}" = "--port" ]; then
                    BRIDGE_PORT="${4:-$BRIDGE_PORT}"
                fi
                cmd_bridge_start
                ;;
            stop)    cmd_bridge_stop ;;
            status)  cmd_bridge_status ;;
            restart)
                cmd_bridge_stop || true
                cmd_bridge_start
                ;;
            *) echo "usage: $0 bridge {start [--port N] | stop | status | restart}" >&2; exit 2 ;;
        esac
        ;;
    permissions)
        # Gap 10 Level 2 on-ramp — explicit user grant channel. Everything
        # runs through data_layer.SelfModel so the daemon and bridge see
        # exactly the same registry. Deny-by-default is never bypassed.
        shift
        .venv/bin/python3 evolve_permissions.py "$@"
        ;;
    *)
        echo "usage: $0 {start [--interval SECONDS] | restart [--interval SECONDS] | status | stop | bridge {start [--port N] | stop | status | restart} | permissions {show | grant | revoke | check}}" >&2
        exit 2
        ;;
esac
