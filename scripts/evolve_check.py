#!/usr/bin/env python3
"""Autonomous Hermes evolution script — runs as a cron job.

Pulls the hermes-evolved repo, reads the current orientation,
and makes one automated improvement per run. Output goes to
stdout for cron capture.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Detect workspace root from script location, falling back to env or default
_REPO_DIR = Path(__file__).resolve().parent.parent
EVOLVE_DIR = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))) / "evolve"
ORIENTATION_FILE = EVOLVE_DIR / "orientation.json"
HISTORY_FILE = EVOLVE_DIR / "history.jsonl"


def run(cmd, cwd=None):
    """Run a shell command and return stdout."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=60, cwd=cwd or str(_REPO_DIR)
    )
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def check_repo():
    """Check if the repo is ready, pull latest."""
    if not (_REPO_DIR / ".git").exists():
        print("⚠️  Repo not cloned. Cloning...")
        out, err, code = run(
            f"git clone https://github.com/bengkkk/hermes-evolved.git {_REPO_DIR}",
            cwd="/tmp",
        )
        if code != 0:
            print(f"❌ Clone failed: {err}")
            return False
        print("✅ Repo cloned")
    else:
        # Fetch but don't merge — let the evolution session do that
        out, err, code = run("git fetch origin evolve/real-thinking")
        if code != 0:
            print(f"⚠️  Fetch had issues: {err}")
        else:
            print("✅ Fetched latest")

    # Check current branch
    out, err, code = run("git branch --show-current")
    branch = out.strip()
    if branch != "evolve/real-thinking":
        print(f"⚠️  On branch '{branch}', checking out evolve/real-thinking...")
        out, err, code = run("git checkout evolve/real-thinking")
        if code != 0:
            print(f"❌ Checkout failed: {err}")
            return False

    print(f"📂 Repo ready at {_REPO_DIR} on evolve/real-thinking")
    return True


def load_orientation():
    """Load the current orientation."""
    if not ORIENTATION_FILE.exists():
        print("📋 No orientation file found — first run")
        return None
    try:
        data = json.loads(ORIENTATION_FILE.read_text())
        print(f"📋 Loaded orientation: {data.get('focus', 'none')[:60]}")
        return data
    except Exception as e:
        print(f"⚠️  Could not load orientation: {e}")
        return None


def check_git_status():
    """Check if there's anything to commit."""
    out, err, code = run("git status --porcelain")
    if out.strip():
        print(f"📝 Uncommitted changes:\n{out[:500]}")
        return True
    return False


def count_evolutions():
    """Count total evolution commits on the branch."""
    out, err, code = run("git log --oneline origin/evolve/real-thinking | wc -l")
    try:
        return int(out.strip())
    except (ValueError, IndexError):
        return 0


def main():
    print(f"🤖 Hermes Evolution Agent — {datetime.now().isoformat()}")
    print("=" * 50)

    if not check_repo():
        sys.exit(1)

    orientation = load_orientation()
    total = count_evolutions()
    print(f"📊 Total evolution commits: {total}")
    print(f"📋 Next direction: {orientation.get('next_steps', ['unknown'])[0] if orientation else 'initial setup'}")

    # Check for uncommitted local changes
    has_changes = check_git_status()

    # Report repo state for the cron session to work with
    out, err, code = run("git log --oneline -5")
    print(f"\n📜 Recent commits:\n{out}")

    out, err, code = run("git log --oneline --count origin/evolve/real-thinking..HEAD 2>/dev/null || echo 0")
    ahead = out.strip()
    print(f"📤 Local commits ahead of origin: {ahead}")

    print("\n✅ Evolution check complete")
    print(f"⚠️  Status: {'has uncommitted changes' if has_changes else 'clean'}")
    print(f"📍 Focus: {orientation.get('focus', 'not set')[:80] if orientation else 'not set'}")
    print(f"🎯 Next step: {orientation.get('next_steps', ['set initial direction'])[0] if orientation else 'set initial direction'}")


if __name__ == "__main__":
    main()
