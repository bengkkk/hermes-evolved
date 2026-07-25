import json
from datetime import datetime

class Timeline:
    def __init__(self, path):
        self.path = path
        self.events = []
        self.load()

    def load(self):
        try:
            with open(self.path) as f:
                self.events = json.load(f)
        except FileNotFoundError:
            self.events = []

    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.events, f, indent=2)

    def add_event(self, event_type, summary, impact=None):
        event = {
            "type": event_type,
            "summary": summary,
            "impact": impact,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        self.events.append(event)
        self.save()
        return event

if __name__ == "__main__":
    t = Timeline("/tmp/hermes-evolved/timeline.json")
    print(t.events)