"""Timeline and SelfModel data layer for self-evolution persistence."""
import json
import os
from datetime import datetime

DATA_DIR = "data"
TIMELINE_FILE = "timeline.json"
SELF_MODEL_FILE = "self_model.json"


class SelfModel:
    """Represents the system's self-model: strengths, weaknesses, etc."""
    def __init__(self, data=None):
        if data is None:
            self.data = {
                "strengths": [],
                "weaknesses": [],
                "unknowns": [],
                "commitments": [],
                "version": "1.0"
            }
        else:
            self.data = data

    def save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        path = os.path.join(DATA_DIR, SELF_MODEL_FILE)
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2)
        return path

    @classmethod
    def load(cls):
        path = os.path.join(DATA_DIR, SELF_MODEL_FILE)
        if not os.path.exists(path):
            return cls()
        with open(path, "r") as f:
            data = json.load(f)
        return cls(data)


class Timeline:
    """Records events in the system's evolution timeline."""
    def __init__(self):
        self.events = []

    def add_event(self, event_type, summary, impact=""):
        """Add a new event record."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "summary": summary,
            "impact": impact
        }
        self.events.append(event)
        return event

    def save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        path = os.path.join(DATA_DIR, TIMELINE_FILE)
        with open(path, "w") as f:
            json.dump(self.events, f, indent=2)
        return path

    @classmethod
    def load(cls):
        path = os.path.join(DATA_DIR, TIMELINE_FILE)
        if not os.path.exists(path):
            return cls()
        with open(path, "r") as f:
            events = json.load(f)
        t = cls()
        t.events = events
        return t
