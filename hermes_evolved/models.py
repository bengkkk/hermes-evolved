#!/usr/bin/env python3
"""Timeline and SelfModel data layer for Hermes Evolved."""

import json
import os
from typing import Any, Dict, List, Optional


class Timeline:
    """Append-only event log for the system."""

    def __init__(self, path: str = "timeline.jsonl"):
        self.path = path
        self.events: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

    def append(self, event: dict):
        self.events.append(event)
        with open(self.path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def get_recent(self, n: int = 10) -> List[Dict]:
        return self.events[-n:]

    def get_all(self) -> List[Dict]:
        return self.events


class SelfModel:
    """Persistent self-model: stores state, strengths, weaknesses, goals, etc."""

    def __init__(self, path: str = "self_model.json"):
        self.path = path
        self.state: Dict[str, Any] = {}
        self.strengths: List[str] = []
        self.weaknesses: List[str] = []
        self.unknowns: List[str] = []
        self.goals: List[Dict] = []
        self.commitments: List[str] = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                try:
                    data = json.load(f)
                    self.state = data.get("state", {})
                    self.strengths = data.get("strengths", [])
                    self.weaknesses = data.get("weaknesses", [])
                    self.unknowns = data.get("unknowns", [])
                    self.goals = data.get("goals", [])
                    self.commitments = data.get("commitments", [])
                except json.JSONDecodeError:
                    pass

    def save(self):
        data = {
            "state": self.state,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "unknowns": self.unknowns,
            "goals": self.goals,
            "commitments": self.commitments,
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def update_state(self, key: str, value: Any):
        self.state[key] = value

    def add_goal(self, goal: dict):
        self.goals.append(goal)

    def recent_goals(self, n: int = 5) -> List[Dict]:
        return self.goals[-n:]
