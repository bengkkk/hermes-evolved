# Timeline.py - Stub for timeline consciousness
# Purpose: Provide data structures and logic for recording and recalling events across sessions.
# Gaps addressed: Gap 9 (Timeline)

from datetime import datetime
from typing import Dict, List, Optional

class TimelineEvent:
    """Represents a single event in the timeline."""
    def __init__(self, event_id: str, timestamp: datetime, event_type: str, summary: str, impact: str):
        self.event_id = event_id
        self.timestamp = timestamp
        self.event_type = event_type
        self.summary = summary
        self.impact = impact
        self.details: Dict = {}

class Timeline:
    """Container for timeline events with retrieval and persistence methods."""
    def __init__(self):
        self.events: List[TimelineEvent] = []
    
    def add_event(self, event: TimelineEvent) -> None:
        self.events.append(event)
    
    def get_recent(self, limit: int = 10) -> List[TimelineEvent]:
        return sorted(self.events, key=lambda e: e.timestamp, reverse=True)[:limit]
    
    def to_dict(self) -> Dict:
        return {"events": [vars(e) for e in self.events]}
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Timeline":
        timeline = cls()
        for e in data.get("events", []):
            timeline.events.append(TimelineEvent(**e))
        return timeline
