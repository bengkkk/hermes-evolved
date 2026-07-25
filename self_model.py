import json
import os
from datetime import datetime

class SelfModel:
    """Persistent self-model storing attributes, weaknesses, unknowns, commitments."""
    def __init__(self, path='self_model.json'):
        self.path = path
        self.data = {
            'version': '0.1',
            'created_at': None,
            'updated_at': None,
            'strengths': [],
            'weaknesses': [],
            'unknowns': [],
            'commitments': []
        }
        if os.path.exists(path):
            self.load()
        else:
            self.data['created_at'] = datetime.utcnow().isoformat()
            self.save()

    def load(self):
        with open(self.path, 'r') as f:
            self.data = json.load(f)

    def save(self):
        self.data['updated_at'] = datetime.utcnow().isoformat()
        with open(self.path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def set(self, key, value):
        self.data[key] = value
        self.save()

    def get(self, key):
        return self.data.get(key)
