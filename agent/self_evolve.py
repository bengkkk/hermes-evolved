"""Self-orientation and reflection mechanism for continuous evolution.

At session start, the agent loads its previous orientation notes and 
generates context about what it's working on and what needs attention.
At session end, it can persist reflection notes for the next session.

This is the structural foundation for self-directed continuous improvement.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Path for evolution notes
EVOLVE_DIR = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))) / "evolve"
ORIENTATION_FILE = EVOLVE_DIR / "orientation.json"
HISTORY_FILE = EVOLVE_DIR / "history.jsonl"


def ensure_evolve_dir():
    """Create the evolve directory if it doesn't exist."""
    EVOLVE_DIR.mkdir(parents=True, exist_ok=True)


def load_orientation() -> Optional[Dict[str, Any]]:
    """Load the most recent orientation notes."""
    ensure_evolve_dir()
    if not ORIENTATION_FILE.exists():
        return None
    try:
        data = json.loads(ORIENTATION_FILE.read_text(encoding="utf-8"))
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("Could not load orientation: %s", e)
        return None


def save_orientation(session_id: str, focus: str, insights: list, next_steps: list):
    """Save orientation notes for the current session."""
    ensure_evolve_dir()
    data = {
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat(),
        "focus": focus,
        "insights": insights,
        "next_steps": next_steps,
    }
    try:
        ORIENTATION_FILE.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # Also append to history
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        logger.debug("Orientation saved for session %s", session_id)
    except OSError as e:
        logger.warning("Could not save orientation: %s", e)


def load_recent_history(limit: int = 5) -> list:
    """Load recent evolution history entries."""
    ensure_evolve_dir()
    if not HISTORY_FILE.exists():
        return []
    entries = []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries[-limit:]
    except OSError as e:
        logger.debug("Could not load evolution history: %s", e)
        return []


def format_orientation_context() -> str:
    """Format orientation context for the system prompt.

    Returns a string that gets injected into the volatile part of the
    system prompt, so the agent starts each session with self-awareness
    about what it's been working on and what needs attention.
    """
    orientation = load_orientation()
    if not orientation:
        return ""

    focus = orientation.get("focus", "")
    insights = orientation.get("insights", [])
    next_steps = orientation.get("next_steps", [])

    parts = ["## Session Orientation"]
    if focus:
        parts.append(f"Previous focus: {focus}")

    if insights:
        parts.append("Recent insights:")
        for ins in insights[-3:]:
            parts.append(f"  - {ins}")

    if next_steps:
        parts.append("Unfinished direction:")
        for step in next_steps[-3:]:
            parts.append(f"  - {step}")

    parts.append("")
    return "\n".join(parts)
