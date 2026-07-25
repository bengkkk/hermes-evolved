import json
from datetime import datetime
from pathlib import Path

class SelfModel:
    def __init__(self, storage_path: str = "/tmp/hermes-evolved/data/self_model.json"):
        self.path = Path(storage_path)
        self.data = {
            "weaknesses": [],
            "unknown_areas": [],
            "commitments": [],
            "strengths": [],
            "state": {}
        }
        if self.path.exists():
            with open(self.path) as f:
                self.data = json.load(f)

    def update_weakness(self, weakness: str):
        self.data["weaknesses"].append({"text": weakness, "recorded": datetime.now().isoformat()})
        self._save()

    def update_unknown(self, unknown: str):
        self.data["unknown_areas"].append({"text": unknown, "recorded": datetime.now().isoformat()})
        self._save()

    def set_commitment(self, what: str, deadline: str = None):
        self.data["commitments"].append({
            "what": what,
            "deadline": deadline,
            "made": datetime.now().isoformat(),
            "status": "active"
        })
        self._save()

    def complete_commitment(self, commitment_text: str):
        for c in self.data["commitments"]:
            if c["what"] == commitment_text:
                c["status"] = "completed"
                c["completed_at"] = datetime.now().isoformat()
                break
        self._save()

    def get_state(self):
        return self.data

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)
