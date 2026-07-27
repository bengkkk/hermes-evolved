import json
import os
from datetime import datetime

DATA_DIR = "/tmp/hermes-evolved"

class DataLayer:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.timeline_file = os.path.join(DATA_DIR, "timeline.json")
        self.self_model_file = os.path.join(DATA_DIR, "self_model.json")

    def append_timeline_event(self, event):
        events = self._load_file(self.timeline_file, [])
        event["timestamp"] = datetime.utcnow().isoformat()
        events.append(event)
        self._save_file(self.timeline_file, events)

    def get_timeline(self):
        return self._load_file(self.timeline_file, [])

    def save_self_model(self, model_state):
        self._save_file(self.self_model_file, model_state)

    def load_self_model(self):
        return self._load_file(self.self_model_file, {})

    def _load_file(self, path, default):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return default

    def _save_file(self, path, data):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
