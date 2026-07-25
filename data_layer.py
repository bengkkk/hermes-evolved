import json
from datetime import datetime
from typing import List, Dict, Optional, Any

class Timeline:
    def __init__(self):
        self.events = []
    
    def add_event(self, event_type: str, summary: str, impact: str = "") -> None:
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": event_type,
            "summary": summary,
            "impact": impact
        }
        self.events.append(event)
    
    def get_recent(self, n: int = 10) -> List[Dict]:
        return self.events[-n:]
    
    def to_json(self) -> str:
        return json.dumps(self.events, indent=2)

class SelfModel:
    def __init__(self):
        self.strengths = []
        self.weaknesses = []
        self.unknowns = []
        self.commitments = []
    
    def update_self_model(self, 
                          strengths: Optional[List[str]] = None,
                          weaknesses: Optional[List[str]] = None,
                          unknowns: Optional[List[str]] = None,
                          commitments: Optional[List[str]] = None) -> None:
        if strengths is not None:
            self.strengths = strengths
        if weaknesses is not None:
            self.weaknesses = weaknesses
        if unknowns is not None:
            self.unknowns = unknowns
        if commitments is not None:
            self.commitments = commitments
    
    def to_dict(self) -> Dict:
        return {
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "unknowns": self.unknowns,
            "commitments": self.commitments
        }

timeline = Timeline()
self_model = SelfModel()

def update_system_prompt():
    # Placeholder: will be implemented later to inject into meta-instruction
    pass
