import json
import os
from typing import List, Optional
from datetime import datetime

class TimelineEvent:
    def __init__(self, event_type: str, summary: str, impact: str, timestamp: Optional[str] = None):
        self.type = event_type
        self.summary = summary
        self.impact = impact
        self.timestamp = timestamp or datetime.utcnow().isoformat()

    def to_dict(self):
        return {
            "type": self.type,
            "summary": self.summary,
            "impact": self.impact,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            event_type=data["type"],
            summary=data["summary"],
            impact=data["impact"],
            timestamp=data.get("timestamp")
        )

class Timeline:
    def __init__(self, events: Optional[List[TimelineEvent]] = None):
        self.events = events or []

    def add_event(self, event: TimelineEvent):
        self.events.append(event)

    def to_dict(self):
        return {"events": [e.to_dict() for e in self.events]}

    @classmethod
    def from_dict(cls, data):
        events = [TimelineEvent.from_dict(e) for e in data.get("events", [])]
        return cls(events)

    @classmethod
    def load(cls, filepath: str):
        if not os.path.exists(filepath):
            return cls()
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class SelfModel:
    def __init__(self, strengths: List[str] = None, weaknesses: List[str] = None,
                 unknowns: List[str] = None, commitments: List[str] = None,
                 evolution_version: str = "v5", total_cycles: int = 0):
        self.strengths = strengths or []
        self.weaknesses = weaknesses or []
        self.unknowns = unknowns or []
        self.commitments = commitments or []
        self.evolution_version = evolution_version
        self.total_cycles = total_cycles

    def to_dict(self):
        return {
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "unknowns": self.unknowns,
            "commitments": self.commitments,
            "evolution_version": self.evolution_version,
            "total_cycles": self.total_cycles
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            strengths=data.get("strengths", []),
            weaknesses=data.get("weaknesses", []),
            unknowns=data.get("unknowns", []),
            commitments=data.get("commitments", []),
            evolution_version=data.get("evolution_version", "v5"),
            total_cycles=data.get("total_cycles", 0)
        )

    @classmethod
    def load(cls, filepath: str):
        if not os.path.exists(filepath):
            return cls()
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def increment_cycle(self):
        self.total_cycles += 1
