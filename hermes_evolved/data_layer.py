import json
import os
from datetime import datetime


class TimelineEvent:
    """Represents a single event in the timeline."""
    def __init__(self, event_type, summary, impact=None):
        self.id = None  # auto-assigned on append
        self.timestamp = datetime.utcnow().isoformat()
        self.event_type = event_type  # 'milestone', 'decision', 'reflection'
        self.summary = summary
        self.impact = impact

    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'event_type': self.event_type,
            'summary': self.summary,
            'impact': self.impact
        }


class Timeline:
    """Ordered timeline of events."""
    def __init__(self, max_events=100):
        self.events = []
        self.max_events = max_events

    def append(self, event):
        event.id = len(self.events) + 1
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

    def to_dict(self):
        return {
            'max_events': self.max_events,
            'events': [e.to_dict() for e in self.events]
        }

    @classmethod
    def from_dict(cls, data):
        tl = cls(max_events=data.get('max_events', 100))
        for ed in data.get('events', []):
            e = TimelineEvent(ed['event_type'], ed['summary'], ed.get('impact'))
            e.id = ed['id']
            e.timestamp = ed['timestamp']
            tl.events.append(e)
        return tl

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class SelfModel:
    """Represents the system's self-model (strengths, weaknesses, etc.)."""
    def __init__(self):
        self.strengths = []
        self.weaknesses = []
        self.unknowns = []
        self.commitments = []

    def add_strength(self, s):
        if s not in self.strengths:
            self.strengths.append(s)

    def add_weakness(self, w):
        if w not in self.weaknesses:
            self.weaknesses.append(w)

    def add_unknown(self, u):
        if u not in self.unknowns:
            self.unknowns.append(u)

    def add_commitment(self, c):
        self.commitments.append(c)

    def to_dict(self):
        return {
            'strengths': self.strengths,
            'weaknesses': self.weaknesses,
            'unknowns': self.unknowns,
            'commitments': self.commitments
        }

    @classmethod
    def from_dict(cls, data):
        sm = cls()
        sm.strengths = data.get('strengths', [])
        sm.weaknesses = data.get('weaknesses', [])
        sm.unknowns = data.get('unknowns', [])
        sm.commitments = data.get('commitments', [])
        return sm

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
