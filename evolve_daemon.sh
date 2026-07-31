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
#   ./evolve_daemon.sh status                       # report current state
#   ./evolve_daemon.sh stop                         # SIGTERM the running daemon
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
INTERVAL=900

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

cmd_status() {
    local pid
    pid="$(_locked_pid)"
    if [ -n "$pid" ] && _pid_alive "$pid"; then
        local started state
        started="$(stat -c '%y' "/proc/$pid" 2>/dev/null | cut -d. -f1)"
        state="$(awk '/^State:/{print $2 " " $3}' "/proc/$pid/status" 2>/dev/null)"
        echo "RUNNING: PID $pid (state: $state, started: $started)"
        echo "  cmd: $(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)"
        echo "  log: $EVOLVE_DIR/daemon.log (tail for cycle activity)"
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

case "${1:-status}" in
    start)
        # optional --interval N
        if [ "${2:-}" = "--interval" ]; then
            INTERVAL="${3:-900}"
        fi
        cmd_start
        ;;
    status)
        cmd_status
        ;;
    stop)
        cmd_stop
        ;;
    *)
        echo "usage: $0 {start [--interval SECONDS] | status | stop}" >&2
        exit 2
        ;;
esac
