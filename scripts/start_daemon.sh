#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# Start persistent think_daemon as a background process.
#
# Usage:
#   ./scripts/start_daemon.sh                    # starts daemon (900s interval, unlimited cycles)
#   ./scripts/start_daemon.sh --interval 300     # 5 min interval
#   ./scripts/start_daemon.sh --once             # run one cycle and exit
#   ./scripts/start_daemon.sh --status           # check if running
#   ./scripts/start_daemon.sh --stop             # graceful stop via state file
#   ./scripts/start_daemon.sh --restart          # stop + start
#
# Designed to be called from cron, startup scripts, or manually.
# Uses PID lock to prevent duplicate daemons.
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EVOLVE_DIR="${HERMES_HOME:-$HOME/.hermes}/evolve"
DAEMON_LOG="$EVOLVE_DIR/daemon.log"
DAEMON_LOCK="$EVOLVE_DIR/daemon.lock"
DAEMON_STATE="$EVOLVE_DIR/daemon_state.json"
PYTHON="${PYTHON:-python3}"

mkdir -p "$EVOLVE_DIR"

# ── Help / no-arg defaults ──────────────────────────────────────────
INTERVAL=900       # 15 minutes between thinking cycles
MODE="loop"        # loop | once | status | stop | restart

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval) INTERVAL="$2"; shift 2 ;;
        --once) MODE="once"; shift ;;
        --status) MODE="status"; shift ;;
        --stop|--shutdown) MODE="stop"; shift ;;
        --restart) MODE="restart"; shift ;;
        --log-level) LOG_LEVEL="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# ── Status check ────────────────────────────────────────────────────
if [[ "$MODE" == "status" ]]; then
    if [[ -f "$DAEMON_LOCK" ]]; then
        PID=$(cat "$DAEMON_LOCK")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Daemon RUNNING (PID $PID)"
            exit 0
        else
            echo "Daemon STOPPED (stale lock PID $PID)"
            exit 1
        fi
    else
        echo "Daemon NOT RUNNING"
        exit 1
    fi
fi

# ── Graceful stop ────────────────────────────────────────────────────
if [[ "$MODE" == "stop" ]]; then
    echo "Requesting daemon shutdown..."
    if command -v python3 &>/dev/null; then
        python3 -c "
import json, sys
path = '$DAEMON_STATE'
try:
    with open(path) as f: data = json.load(f)
    data['status'] = 'shutdown'
    with open(path, 'w') as f: json.dump(data, f, indent=2)
    print('Shutdown requested in daemon_state.json')
except Exception as e:
    print(f'Could not request shutdown: {e}')
    sys.exit(1)
"
    fi
    # Also try to kill the PID directly
    if [[ -f "$DAEMON_LOCK" ]]; then
        PID=$(cat "$DAEMON_LOCK")
        kill "$PID" 2>/dev/null && echo "Sent SIGTERM to PID $PID" || echo "PID $PID not found (already dead)"
        rm -f "$DAEMON_LOCK"
    fi
    exit 0
fi

# ── Run one cycle ────────────────────────────────────────────────────
if [[ "$MODE" == "once" ]]; then
    cd "$REPO_ROOT"
    exec $PYTHON think_daemon.py --once --log-level "$LOG_LEVEL"
fi

# ── Continuous loop ──────────────────────────────────────────────────
if [[ "$MODE" == "loop" ]]; then
    # Check if already running
    if [[ -f "$DAEMON_LOCK" ]]; then
        PID=$(cat "$DAEMON_LOCK")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Daemon already running (PID $PID) — not starting another"
            exit 0
        fi
        echo "Removing stale lock from PID $PID"
        rm -f "$DAEMON_LOCK"
    fi

    echo "Starting persistent think_daemon (interval=${INTERVAL}s)..."
    cd "$REPO_ROOT"
    nohup $PYTHON think_daemon.py \
        --interval "$INTERVAL" \
        --log-level "$LOG_LEVEL" \
        >> "$DAEMON_LOG" 2>&1 &
    
    DAEMON_PID=$!
    echo "Daemon started (PID $DAEMON_PID)"
    echo "Log: $DAEMON_LOG"
    
    # Wait a moment and verify it started
    sleep 2
    if kill -0 "$DAEMON_PID" 2>/dev/null; then
        echo "Daemon confirmed running"
        exit 0
    else
        echo "WARNING: Daemon exited immediately — check $DAEMON_LOG"
        exit 1
    fi
fi

# ── Restart ──────────────────────────────────────────────────────────
if [[ "$MODE" == "restart" ]]; then
    # shellcheck disable=SC2068
    $0 --stop || true
    sleep 1
    # shellcheck disable=SC2068
    exec $0 --interval "$INTERVAL"
fi
