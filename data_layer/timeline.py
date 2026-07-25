import json
from datetime import datetime
from pathlib import Path

class Timeline:
    def __init__(self, storage_path: str = "/tmp/hermes-evolved/data/timeline.json"):
        self.path = Path(storage_path)
        self.events = []
        if self.path.exists():
            with open(self.path) as f:
                self.events = json.load(f)

    def record_event(self, event_type: str, summary: str, impact: str = ""):
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "summary": summary,
            "impact": impact
        }
        self.events.append(event)
        self._save()
        return event

    def get_recent(self, n: int = 10):
        return self.events[-n:]

    def get_by_type(self, event_type: str):
        return [e for e in self.events if e["type"] == event_type]

    def get_all(self):
        return self.events

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.events, f, indent=2)
