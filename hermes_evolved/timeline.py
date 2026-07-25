#!/usr/bin/env python3
"""Timeline data model for Hermes evolved self-evolution."""
from datetime import datetime
import json

class Timeline:
    """Simple append-only ledger of thinking events."""
    def __init__(self, path: str = "/tmp/hermes-evolved/timeline.json"):
        self.path = path
        self.events = []

    def record(self, event_type: str, summary: str, impact: str = "") -> dict:
        """Record a timeline event and append to file."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "summary": summary,
            "impact": impact
        }
        self.events.append(event)
        self._flush()
        return event

    def _flush(self):
        with open(self.path, "a") as f:
            f.write(json.dumps(self.events[-1]) + "\n")

    def load(self):
        try:
            with open(self.path, "r") as f:
                self.events = [json.loads(line) for line in f]
        except FileNotFoundError:
            self.events = []
