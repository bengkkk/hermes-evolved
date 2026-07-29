"""
Data layer for Hermes Evolved — single authoritative source for Timeline,
SelfModel, and persistent state management.

Consolidates 10+ fragmented implementations into one canonical module with:
- File-backed persistence with atomic writes
- Modern datetime (timezone-aware, no deprecated utcnow())
- Robust JSON I/O with version validation and error handling
- Full type hints
- Both in-memory and persistent access patterns

Usage:
    from data_layer import Timeline, SelfModel, safe_read_json, safe_write_json

    # In-memory
    tl = Timeline()
    tl.add_event("milestone", "Did something cool")

    # File-backed
    sm = SelfModel.load()  # or SelfModel() to start fresh
    sm.add_strength("file-backed persistence")
    sm.save()

    # Low-level utilities (shared by daemon and self_evolve)
    data = safe_read_json(path, default={})
    safe_write_json(path, data)

Storage paths default to ~/.hermes/evolve/ (same as agent/self_evolve.py),
override via HERMES_EVOLVE_DIR env var.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ── Default storage paths ─────────────────────────────────────────
# Respects HERMES_HOME for profile awareness, with explicit override
_EVOLVE_DIR = Path(
    os.environ.get(
        "HERMES_EVOLVE_DIR",
        os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")),
    )
) / "evolve"


# ═══════════════════════════════════════════════════════════════════
#  Public helpers — reusable across daemon, self_evolve, and tests
# ═══════════════════════════════════════════════════════════════════

def get_evolve_dir() -> Path:
    """Return the evolve data directory (created on first access)."""
    _EVOLVE_DIR.mkdir(parents=True, exist_ok=True)
    return _EVOLVE_DIR


def safe_read_json(path: Path, default: Any = None) -> Any:
    """Read and parse a JSON file with version validation and resilience.

    Args:
        path: Path to the JSON file.
        default: Value returned on failure. If dict with a ``"version"`` key,
                 used for schema version validation (rejects versions more
                 than 1 ahead of the default).

    Returns:
        Parsed data on success, *default* on failure or version mismatch.
    """
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            # Schema version check — guard against downgrade corruption
            if isinstance(data, dict) and isinstance(default, dict):
                dv = data.get("version", 1)
                dd = default.get("version", 1)
                if dv > dd + 1:
                    logger.warning(
                        "%s has version %d, expected <= %d — resetting",
                        path.name, dv, dd,
                    )
                    return default
            return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not read %s: %s — returning default", path.name, e)
    return default


def safe_write_json(path: Path, data: Any) -> None:
    """Atomically write a JSON file with crash safety and error resilience.

    Writes to a ``.tmp`` staging file, then renames atomically.
    Logs a warning on failure instead of raising.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError as e:
        logger.warning("Could not write %s: %s", path.name, e)


# ═══════════════════════════════════════════════════════════════════
#  Internal helpers (used by classes below; prefer safe_read/write for external use)
# ═══════════════════════════════════════════════════════════════════

def now_iso() -> str:
    """Current UTC timestamp as ISO-8601 string (public)."""
    return datetime.now(timezone.utc).isoformat()


def now_compact() -> str:
    """Compact timestamp safe for filenames / IDs (public)."""
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _atomic_write(path: Path, data: Any) -> Path:
    """Write JSON atomically: write to .tmp, then replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def _read_json(path: Path, default: Any = None) -> Any:
    """Read and parse JSON, returning default on any failure."""
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read %s: %s", path, e)
    return default


# ═══════════════════════════════════════════════════════════════════
#  Timeline — append-only event log
# ═══════════════════════════════════════════════════════════════════

class Timeline:
    """Ordered event log tracking the system's evolution history.

    Supports in-memory and file-backed usage. Stored as a JSON list
    under ``~/.hermes/evolve/timeline.json`` by default.
    """

    def __init__(self, events: Optional[List[Dict[str, Any]]] = None):
        self.events: List[Dict[str, Any]] = events if events is not None else []

    # ── Mutators ──────────────────────────────────────────────────

    def add_event(
        self,
        event_type: str,
        summary: str,
        impact: str = "",
    ) -> Dict[str, Any]:
        """Append an event and return it.

        Args:
            event_type: ``"milestone"`` | ``"decision"`` | ``"reflection"`` | …
            summary: One-line description.
            impact: Why this matters / what it enables.
        """
        event: Dict[str, Any] = {
            "id": now_compact(),
            "timestamp": now_iso(),
            "type": event_type,
            "summary": summary,
            "impact": impact,
        }
        self.events.append(event)
        return event

    # ── Queries ───────────────────────────────────────────────────

    def get_recent(self, count: int = 10) -> List[Dict[str, Any]]:
        """Return the *count* most recent events."""
        return self.events[-count:]

    def get_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """Filter events by type string."""
        return [e for e in self.events if e.get("type") == event_type]

    def get_all(self) -> List[Dict[str, Any]]:
        """Return every recorded event (ordered oldest → newest)."""
        return list(self.events)

    def get_by_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Look up a single event by its ``id`` field."""
        for e in self.events:
            if e.get("id") == event_id:
                return e
        return None

    # ── Serialisation ─────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Full dict including version for schema tracking."""
        return {"version": 2, "events": self.events}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Timeline":
        """Build a Timeline from a dict (output of :meth:`to_dict`)."""
        raw = data.get("events", [])
        # v1 stores events directly in a list; v2 nests under "events"
        events = raw if isinstance(raw, list) else data.get("events", [])
        return cls(events=events)

    # ── File-backed persistence ───────────────────────────────────

    @staticmethod
    def storage_path() -> Path:
        return _EVOLVE_DIR / "timeline.json"

    def save(self, path: Optional[Path] = None) -> Path:
        """Persist all events as a versioned JSON dict.

        Uses atomic write (``.tmp`` → final) for crash safety.
        Writes via :meth:`to_dict` so the versioned format is consistent
        with :meth:`from_dict` and :meth:`load`.
        """
        target = path or self.storage_path()
        _atomic_write(target, self.to_dict())
        return target

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Timeline":
        """Load events from disk, returning an empty Timeline on failure."""
        target = path or cls.storage_path()
        data = _read_json(target)
        if data is None:
            return cls()
        # Support both list and nested {"events": [...]} formats
        if isinstance(data, list):
            return cls(events=data)
        if isinstance(data, dict):
            return cls.from_dict(data)
        return cls()

    # ── Convenience ───────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.events)

    def __bool__(self) -> bool:
        return bool(self.events)

    def __repr__(self) -> str:
        return f"<Timeline events={len(self.events)}>"


# ═══════════════════════════════════════════════════════════════════
#  SelfModel — identity, capabilities, and metacognitive state
# ═══════════════════════════════════════════════════════════════════

class SelfModel:
    """Persistent self-representation of the evolving system.

    Tracks what the system knows about itself: identity, strengths,
    weaknesses, unknown areas, commitments, and state flags such as
    the current *gap focus* and *evolution version*.

    Stored as a JSON dict under ``~/.hermes/evolve/self_model.json``.
    """

    _DEFAULT_DATA: Dict[str, Any] = {
        "identity": {
            "name": "Hermes (evolved)",
            "role": "Self-evolving AI system",
        },
        "state": {
            "evolution_version": 5,
            "current_gap_focus": "",
            "total_cycles": 0,
        },
        "capabilities": {
            "strengths": [],
            "weaknesses": [],
            "unknown_areas": [],
        },
        "commitments": {
            "promised_features": [],
            "active_obligations": [],
        },
    }

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self.data: Dict[str, Any] = (
            self._merge_defaults(data) if data else copy.deepcopy(SelfModel._DEFAULT_DATA)
        )

    # ── Internal ──────────────────────────────────────────────────

    @staticmethod
    def _merge_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
        """Deep-merge loaded data with defaults so new keys appear.

        Uses *deep* copy so class-level ``_DEFAULT_DATA`` is never mutated
        when loading saved state — without this, every ``SelfModel(data=…)``
        call would leak saved keys into the shared defaults dict.
        """
        merged = copy.deepcopy(SelfModel._DEFAULT_DATA)
        for section in ("identity", "state", "capabilities", "commitments"):
            if section in data:
                merged[section].update(data[section])
        # Absorb top-level keys that don't fit a section
        for k, v in data.items():
            if k not in merged:
                merged[k] = v
        return merged

    def _section(self, name: str) -> Dict[str, Any]:
        return self.data.setdefault(name, {})

    # ── Identity ──────────────────────────────────────────────────

    @property
    def identity(self) -> Dict[str, str]:
        return self._section("identity")

    @identity.setter
    def identity(self, value: Dict[str, str]) -> None:
        self.data["identity"] = value

    # ── State ─────────────────────────────────────────────────────

    @property
    def state(self) -> Dict[str, Any]:
        return self._section("state")

    def get_state_value(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def set_state_value(self, key: str, value: Any) -> None:
        self.state[key] = value

    # ── Capabilities ──────────────────────────────────────────────

    @property
    def capabilities(self) -> Dict[str, Any]:
        return self._section("capabilities")

    def _cap_list(self, name: str) -> List[str]:
        return self.capabilities.setdefault(name, [])

    def add_strength(self, strength: str) -> bool:
        """Register a new strength. Returns True if added (not duplicate)."""
        lst = self._cap_list("strengths")
        if strength not in lst:
            lst.append(strength)
            return True
        return False

    def add_weakness(self, weakness: str) -> bool:
        """Register a new weakness. Returns True if added."""
        lst = self._cap_list("weaknesses")
        if weakness not in lst:
            lst.append(weakness)
            return True
        return False

    def add_unknown(self, unknown: str) -> bool:
        """Register a new unknown area. Returns True if added."""
        lst = self._cap_list("unknown_areas")
        if unknown not in lst:
            lst.append(unknown)
            return True
        return False

    def remove_weakness(self, pattern: str) -> int:
        """Remove weaknesses matching *pattern* (substring match).

        Returns the number of items removed.
        """
        lst = self._cap_list("weaknesses")
        before = len(lst)
        self.capabilities["weaknesses"] = [
            w for w in lst if pattern.lower() not in w.lower()
        ]
        return before - len(self.capabilities["weaknesses"])

    # ── Commitments ───────────────────────────────────────────────

    def set_commitment(
        self,
        what: str,
        deadline: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record a new active commitment."""
        c: Dict[str, Any] = {
            "what": what,
            "deadline": deadline,
            "status": "active",
            "created_at": now_iso(),
        }
        self.data.setdefault("commitments", {})
        self.data["commitments"].setdefault("active_obligations", []).append(c)
        return c

    def complete_commitment(self, what: str) -> bool:
        """Mark a commitment as completed by its ``what`` text."""
        for c in self.data.get("commitments", {}).get("active_obligations", []):
            if c.get("what") == what:
                c["status"] = "completed"
                c["completed_at"] = now_iso()
                return True
        return False

    # ── Snapshot / summary ────────────────────────────────────────

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Compact dict suitable for injecting into a system prompt."""
        caps = self.capabilities
        return {
            "identity": self.identity.get("name", "?"),
            "evolution_version": self.state.get("evolution_version"),
            "gap_focus": self.state.get("current_gap_focus", ""),
            "strengths": caps.get("strengths", [])[:8],
            "weaknesses": caps.get("weaknesses", [])[:5],
            "unknown_areas": caps.get("unknown_areas", [])[:5],
        }

    def summary(self) -> str:
        """Human-readable one-liner."""
        id_ = self.identity.get("name", "?")
        ver = self.state.get("evolution_version", "?")
        cyc = self.state.get("total_cycles", 0)
        return f"{id_} — v{ver} — {cyc} cycles"

    # ── File-backed persistence ───────────────────────────────────

    @staticmethod
    def storage_path() -> Path:
        return _EVOLVE_DIR / "self_model.json"

    def save(self, path: Optional[Path] = None) -> Path:
        """Persist to disk as JSON (atomic write)."""
        target = path or self.storage_path()
        _atomic_write(target, self.data)
        return target

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "SelfModel":
        """Load from disk, returning a fresh SelfModel on failure."""
        target = path or cls.storage_path()
        data = _read_json(target)
        return cls(data=data) if isinstance(data, dict) else cls()

    # ── Convenience ───────────────────────────────────────────────

    def __repr__(self) -> str:
        n = self.identity.get("name", "?")
        g = self.state.get("current_gap_focus", "")
        return f"<SelfModel {n} gap={g!r}>"


# ═══════════════════════════════════════════════════════════════════
#  Orientation — lightweight focus tracking
# ═══════════════════════════════════════════════════════════════════

class Orientation:
    """Records the system's current focus, insight history, and next steps.

    A simpler, free-form companion to SelfModel that tracks the
    moment-to-moment thinking direction rather than permanent traits.
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self.data: Dict[str, Any] = data or {
            "focus": "",
            "insights": [],
            "next_steps": [],
        }

    @property
    def focus(self) -> str:
        return self.data.get("focus", "")

    @focus.setter
    def focus(self, value: str) -> None:
        self.data["focus"] = value

    def add_insight(self, text: str) -> None:
        self.data.setdefault("insights", [])
        if not self.data["insights"] or self.data["insights"][-1] != text:
            self.data["insights"].append(text)
            self.data["insights"] = self.data["insights"][-20:]  # cap

    def add_next_step(self, step: str) -> None:
        self.data.setdefault("next_steps", [])
        if not self.data["next_steps"] or self.data["next_steps"][-1] != step:
            self.data["next_steps"].append(step)
            self.data["next_steps"] = self.data["next_steps"][-10:]

    # Persistence
    @staticmethod
    def storage_path() -> Path:
        return _EVOLVE_DIR / "orientation.json"

    def save(self, path: Optional[Path] = None) -> Path:
        target = path or self.storage_path()
        _atomic_write(target, self.data)
        return target

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Orientation":
        target = path or cls.storage_path()
        data = _read_json(target)
        return cls(data=data) if isinstance(data, dict) else cls()

    def __repr__(self) -> str:
        return f"<Orientation focus={self.focus!r}>"


# ═══════════════════════════════════════════════════════════════════
#  Memory — multi-type store (episodic / semantic / procedural)
# ═══════════════════════════════════════════════════════════════════

_DEFAULT_MEMORY: Dict[str, Any] = {
    "version": 1,
    "episodic": [],
    "semantic": [],
    "procedural": [],
}


class Memory:
    """Multi-type memory store for the evolving system.

    Three memory types:
    - **Episodic**: experiences, observations, interactions (salience-tagged)
    - **Semantic**: facts, knowledge, concepts (confidence-weighted)
    - **Procedural**: how-to patterns, behaviors (success-tracked)

    Stored as a JSON dict under ``~/.hermes/evolve/memory.json``.
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self.data: Dict[str, Any] = data if data else copy.deepcopy(_DEFAULT_MEMORY)

    # ── Episodic ───────────────────────────────────────────────────

    def add_episodic(
        self,
        mtype: str = "observation",
        summary: str = "",
        details: str = "",
        tags: Optional[List[str]] = None,
        salience: float = 0.5,
    ) -> str:
        """Record an episodic memory (experience / observation / interaction).

        Returns the memory ID.
        """
        mem_id = f"ep_{now_compact()}"
        entry: Dict[str, Any] = {
            "id": mem_id,
            "timestamp": now_iso(),
            "type": mtype,
            "summary": summary,
            "details": details,
            "salience": salience,
            "tags": tags or [],
        }
        self.data.setdefault("episodic", []).append(entry)
        self.data["episodic"] = self.data["episodic"][-200:]
        return mem_id

    def search_episodic(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Simple substring search across episodic entries."""
        q = query.lower()
        return [
            e for e in self.data.get("episodic", [])
            if q in json.dumps(e).lower()
        ][-limit:]

    # ── Semantic ───────────────────────────────────────────────────

    def add_semantic(
        self,
        topic: str,
        fact: str,
        source: str = "experience",
        confidence: float = 0.7,
    ) -> str:
        """Record a semantic fact (knowledge / concept learned).

        Deduplicates on (topic + fact), updating confidence on repeat.
        Returns the memory ID.
        """
        mem_id = f"sem_{now_compact()}"
        # Deduplicate
        for existing in self.data.get("semantic", []):
            if existing.get("topic") == topic and existing.get("fact") == fact:
                existing["last_accessed"] = now_iso()
                existing["confidence"] = max(existing.get("confidence", 0), confidence)
                return existing["id"]
        entry: Dict[str, Any] = {
            "id": mem_id,
            "timestamp": now_iso(),
            "topic": topic,
            "fact": fact,
            "source": source,
            "confidence": confidence,
            "last_accessed": now_iso(),
        }
        self.data.setdefault("semantic", []).append(entry)
        self.data["semantic"] = self.data["semantic"][-500:]
        return mem_id

    def get_semantic_by_topic(self, topic: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get semantic facts matching a topic (case-insensitive)."""
        tl = topic.lower()
        return [
            e for e in self.data.get("semantic", [])
            if tl in e.get("topic", "").lower() or tl in e.get("fact", "").lower()
        ][-limit:]

    # ── Procedural ─────────────────────────────────────────────────

    def add_procedural(self, pattern: str, trigger: str, procedure: str) -> str:
        """Record a procedural memory (how-to pattern).

        Increments success_count if the pattern already exists.
        Returns the memory ID.
        """
        mem_id = f"pro_{now_compact()}"
        for existing in self.data.get("procedural", []):
            if existing.get("pattern") == pattern:
                existing["success_count"] = existing.get("success_count", 0) + 1
                return existing["id"]
        entry: Dict[str, Any] = {
            "id": mem_id,
            "timestamp": now_iso(),
            "pattern": pattern,
            "trigger": trigger,
            "procedure": procedure,
            "success_count": 1,
            "failure_count": 0,
        }
        self.data.setdefault("procedural", []).append(entry)
        self.data["procedural"] = self.data["procedural"][-100:]
        return mem_id

    # ── Cross-type search ──────────────────────────────────────────

    def search(
        self,
        query: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Simple text search across one or more memory types.

        Args:
            query: Search term (case-insensitive substring).
            memory_types: Which types to search (default: all three).
            limit: Max results per type.

        Returns:
            Dict mapping type name to list of matching entries.
        """
        types = memory_types or ["episodic", "semantic", "procedural"]
        q = query.lower()
        results: Dict[str, List[Dict[str, Any]]] = {}
        for mt in types:
            matches = [
                e for e in self.data.get(mt, [])
                if q in json.dumps(e).lower()
            ]
            results[mt] = matches[-limit:]
        return results

    # ── Format helpers ─────────────────────────────────────────────

    def format_context(self, max_episodic: int = 3, max_semantic: int = 3,
                       max_procedural: int = 2) -> str:
        """Format recent memories as a readable context block."""
        parts: List[str] = []
        episodic = self.data.get("episodic", [])
        recent_ep = episodic[-max_episodic:] if episodic else []
        if recent_ep:
            parts.append("Recent experiences:")
            for e in recent_ep:
                sal = "★" if e.get("salience", 0) >= 0.8 else "·"
                parts.append(f"  {sal} {e['summary'][:80]}")

        semantic = self.data.get("semantic", [])
        recent_sem = semantic[-max_semantic:] if semantic else []
        if recent_sem:
            parts.append("Knowledge gained:")
            for s in recent_sem:
                conf = f" ({s.get('confidence', 0):.0%})" if s.get("confidence") else ""
                parts.append(f"  · {s['topic']}: {s['fact'][:100]}{conf}")

        procedural = self.data.get("procedural", [])
        recent_pro = procedural[-max_procedural:] if procedural else []
        if recent_pro:
            parts.append("Learned patterns:")
            for p in recent_pro:
                sc = p.get("success_count", 0)
                fc = p.get("failure_count", 0)
                parts.append(f"  · {p['pattern']} (✓{sc} ✗{fc})")

        if not (recent_ep or recent_sem or recent_pro):
            parts.append("  (no recent memories yet)")

        return "\n".join(parts)

    # ── Persistence ───────────────────────────────────────────────

    @staticmethod
    def storage_path() -> Path:
        return _EVOLVE_DIR / "memory.json"

    def save(self, path: Optional[Path] = None) -> Path:
        target = path or self.storage_path()
        _atomic_write(target, self.data)
        return target

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Memory":
        target = path or cls.storage_path()
        data = _read_json(target)
        if isinstance(data, dict):
            merged = dict(_DEFAULT_MEMORY)
            merged.update(data)
            for section in ("episodic", "semantic", "procedural"):
                if section in data and isinstance(data[section], list):
                    merged[section] = data[section]
            return cls(data=merged)
        return cls()

    # ── Convenience ───────────────────────────────────────────────

    def __repr__(self) -> str:
        ep = len(self.data.get("episodic", []))
        sem = len(self.data.get("semantic", []))
        pro = len(self.data.get("procedural", []))
        return f"<Memory episodic={ep} semantic={sem} procedural={pro}>"


# ═══════════════════════════════════════════════════════════════════
#  Goals — self-generated goal lifecycle
# ═══════════════════════════════════════════════════════════════════

_DEFAULT_GOALS: Dict[str, Any] = {
    "version": 1,
    "goals": [],
}


class Goals:
    """Self-generated goal store with lifecycle tracking.

    Each goal progresses through: proposed → active → in_progress → completed/abandoned.
    Goals are stored as a JSON dict under ``~/.hermes/evolve/goals.json``.
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self.data: Dict[str, Any] = data if data else copy.deepcopy(_DEFAULT_GOALS)

    def propose(
        self,
        title: str,
        description: str,
        rationale: str = "",
        gap_reference: str = "",
        verification_criteria: str = "",
        priority: int = 3,
        dependencies: Optional[List[str]] = None,
    ) -> str:
        """Propose a new self-generated goal. Returns the goal ID."""
        goal_id = f"goal_{now_compact()}"
        goal: Dict[str, Any] = {
            "id": goal_id,
            "title": title,
            "description": description,
            "rationale": rationale,
            "priority": priority,
            "status": "proposed",
            "gap_reference": gap_reference,
            "dependencies": dependencies or [],
            "estimated_effort": "",
            "verification_criteria": verification_criteria,
            "created_at": now_iso(),
            "completed_at": None,
            "notes": "",
        }
        self.data.setdefault("goals", []).append(goal)
        self.data["goals"] = self.data["goals"][-100:]
        return goal_id

    def update_status(self, goal_id: str, new_status: str, note: str = "") -> bool:
        """Update a goal's lifecycle status.

        Valid statuses: ``active``, ``in_progress``, ``completed``, ``abandoned``.
        Returns True if the goal was found.
        """
        for g in self.data.get("goals", []):
            if g.get("id") == goal_id:
                g["status"] = new_status
                if new_status in ("completed", "abandoned"):
                    g["completed_at"] = now_iso()
                if note:
                    g["notes"] = note
                return True
        return False

    def get_active(
        self,
        status_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get goals filtered by status.

        Default: proposed + active + in_progress (i.e. not completed/abandoned).
        Results sorted by priority (lower = higher), then by creation time.
        """
        statuses = status_filter or ["proposed", "active", "in_progress"]
        filtered = [g for g in self.data.get("goals", []) if g.get("status") in statuses]
        filtered.sort(key=lambda g: (g.get("priority", 5), g.get("created_at", "")))
        return filtered

    def format_context(self) -> str:
        """Format pending goals for the system prompt."""
        pending = self.get_active()
        parts: List[str] = ["## Self-generated Goals"]
        if not pending:
            parts.append("  (no self-generated goals yet)")
            return "\n".join(parts)

        for g in pending:
            sym = {"proposed": "◇", "active": "○", "in_progress": "◎"}.get(
                g.get("status", ""), "·"
            )
            deps = (
                f" [depends: {', '.join(g['dependencies'][:3])}]"
                if g.get("dependencies")
                else ""
            )
            gap = f" [{g['gap_reference']}]" if g.get("gap_reference") else ""
            parts.append(f"  {sym} P{g.get('priority', 3)} — {g['title']}{deps}{gap}")
            parts.append(f"      {g.get('description', '')[:80]}")

        active_count = sum(
            1 for g in pending if g.get("status") in ("active", "in_progress")
        )
        proposed_count = sum(1 for g in pending if g.get("status") == "proposed")
        parts.append(f"  ({active_count} active, {proposed_count} proposed)")
        return "\n".join(parts)

    # ── Persistence ───────────────────────────────────────────────

    @staticmethod
    def storage_path() -> Path:
        return _EVOLVE_DIR / "goals.json"

    def save(self, path: Optional[Path] = None) -> Path:
        target = path or self.storage_path()
        _atomic_write(target, self.data)
        return target

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Goals":
        target = path or cls.storage_path()
        data = _read_json(target)
        if isinstance(data, dict):
            merged = dict(_DEFAULT_GOALS)
            merged.update(data)
            if "goals" in data and isinstance(data["goals"], list):
                merged["goals"] = data["goals"]
            return cls(data=merged)
        return cls()

    def __repr__(self) -> str:
        n = len(self.data.get("goals", []))
        return f"<Goals count={n}>"


# ═══════════════════════════════════════════════════════════════════
#  Plan management — stored inside timeline.json (future.plans[])
# ═══════════════════════════════════════════════════════════════════

_DEFAULT_TIMELINE_DICT: Dict[str, Any] = {
    "version": 1,
    "past": {
        "events": [],
        "completed_sessions": [],
        "outcomes": [],
    },
    "present": {
        "active_project": None,
        "active_tasks": [],
        "waiting_for": [],
        "last_session_focus": None,
        "commitments": [],
    },
    "future": {
        "goals": [],
        "scheduled_actions": [],
        "contingencies": [],
        "predictions": [],
        "plans": [],
    },
}


def load_timeline_dict() -> Dict[str, Any]:
    """Load the dict-format timeline (past/present/future).

    Public entry point to the structured timeline. Returns a merged
    dict with default values for any missing sections.
    """
    return _load_timeline_dict()


def save_timeline_dict(data: Dict[str, Any]) -> None:
    """Save the dict-format timeline (public wrapper)."""
    _save_timeline_dict(data)


def _load_timeline_dict() -> Dict[str, Any]:
    """Load the dict-format timeline (past/present/future)."""
    path = _EVOLVE_DIR / "timeline.json"
    data = _read_json(path)
    if not isinstance(data, dict):
        return dict(_DEFAULT_TIMELINE_DICT)
    merged = dict(_DEFAULT_TIMELINE_DICT)
    for section in ("past", "present", "future"):
        if section in data and isinstance(data[section], dict):
            merged[section].update(data[section])
    for k, v in data.items():
        if k not in ("past", "present", "future"):
            merged[k] = v
    return merged


def _save_timeline_dict(data: Dict[str, Any]) -> None:
    """Save the dict-format timeline."""
    safe_write_json(_EVOLVE_DIR / "timeline.json", data)


# ── Dict-format timeline event helpers ──────────────────────────────


def record_event(
    event_type: str, summary: str, impact: str = ""
) -> None:
    """Record an event in the timeline's past section."""
    timeline = _load_timeline_dict()
    event = {
        "id": now_compact(),
        "type": event_type,
        "timestamp": now_iso(),
        "summary": summary,
        "impact": impact,
    }
    timeline["past"]["events"].append(event)
    timeline["past"]["events"] = timeline["past"]["events"][-50:]
    _save_timeline_dict(timeline)


def record_session_completion(
    session_id: str, focus: str, outcomes: list
) -> None:
    """Record a completed session in the timeline."""
    timeline = _load_timeline_dict()
    entry = {
        "session_id": session_id,
        "timestamp": now_iso(),
        "focus": focus,
        "outcomes": outcomes,
    }
    timeline["past"]["completed_sessions"].append(entry)
    timeline["past"]["completed_sessions"] = \
        timeline["past"]["completed_sessions"][-20:]
    _save_timeline_dict(timeline)


def record_outcome(event_id: str, summary: str, impact: str = "") -> None:
    """Record an outcome linked to a past event."""
    timeline = _load_timeline_dict()
    outcome = {
        "id": now_compact(),
        "event_id": event_id,
        "timestamp": now_iso(),
        "summary": summary,
        "impact": impact,
    }
    timeline["past"]["outcomes"].append(outcome)
    timeline["past"]["outcomes"] = timeline["past"]["outcomes"][-50:]
    _save_timeline_dict(timeline)


def add_commitment(
    what: str, deadline: Optional[str] = None, status: str = "active"
) -> None:
    """Track a commitment with an optional deadline."""
    timeline = _load_timeline_dict()
    commit = {
        "id": now_compact(),
        "what": what,
        "deadline": deadline,
        "status": status,
        "created_at": now_iso(),
    }
    timeline["present"]["commitments"].append(commit)
    timeline["present"]["commitments"] = \
        timeline["present"]["commitments"][-30:]
    _save_timeline_dict(timeline)


def update_commitment(commit_id: str, status: str = "done") -> None:
    """Mark a commitment as done/expired."""
    timeline = _load_timeline_dict()
    for c in timeline["present"]["commitments"]:
        if c["id"] == commit_id:
            c["status"] = status
            break
    _save_timeline_dict(timeline)


def add_prediction(
    text: str,
    timeframe: Optional[str] = None,
    confidence: Optional[float] = None,
    basis: Optional[str] = None,
) -> None:
    """Record a prediction about future outcomes."""
    timeline = _load_timeline_dict()
    pred = {
        "id": now_compact(),
        "text": text,
        "timeframe": timeframe,
        "confidence": confidence,
        "basis": basis,
        "created_at": now_iso(),
    }
    timeline["future"]["predictions"].append(pred)
    timeline["future"]["predictions"] = \
        timeline["future"]["predictions"][-50:]
    _save_timeline_dict(timeline)


def update_present_state(
    active_project: Optional[str] = None,
    tasks: Optional[list] = None,
    waiting_for: Optional[list] = None,
    focus: Optional[str] = None,
) -> None:
    """Update the present state section of the timeline."""
    timeline = _load_timeline_dict()
    if active_project is not None:
        timeline["present"]["active_project"] = active_project
    if tasks is not None:
        timeline["present"]["active_tasks"] = tasks
    if waiting_for is not None:
        timeline["present"]["waiting_for"] = waiting_for
    if focus is not None:
        timeline["present"]["last_session_focus"] = focus
    _save_timeline_dict(timeline)


def update_goals(
    goals: Optional[list] = None,
    scheduled: Optional[list] = None,
    contingencies: Optional[list] = None,
) -> None:
    """Update the future goals/plans/scheduled actions in the timeline."""
    timeline = _load_timeline_dict()
    if goals is not None:
        timeline["future"]["goals"] = goals
    if scheduled is not None:
        timeline["future"]["scheduled_actions"] = scheduled
    if contingencies is not None:
        timeline["future"]["contingencies"] = contingencies
    _save_timeline_dict(timeline)


def create_plan(goal: str, steps: Optional[List[Dict[str, Any]]] = None) -> str:
    """Create a new plan in the timeline's future section.

    Each step can have: id, description, verification, status, blocked_by,
    assigned_to, completed_at, note.
    Returns the plan ID.
    """
    timeline = _load_timeline_dict()
    plan_id = f"plan_{now_compact()}"
    normalized_steps = []
    for i, s in enumerate(steps or []):
        normalized_steps.append({
            "id": s.get("id", f"step_{i + 1}"),
            "description": s.get("description", ""),
            "verification": s.get("verification", ""),
            "status": s.get("status", "pending"),
            "blocked_by": s.get("blocked_by"),
            "assigned_to": s.get("assigned_to", "user"),
            "completed_at": s.get("completed_at"),
            "note": s.get("note"),
        })
    plan: Dict[str, Any] = {
        "id": plan_id,
        "goal": goal,
        "steps": normalized_steps,
        "status": "active",
        "progress": f"0/{len(normalized_steps)} steps" if normalized_steps else "0 steps",
        "created_at": now_iso(),
        "completed_at": None,
    }
    timeline.setdefault("future", {}).setdefault("plans", []).append(plan)
    timeline["future"]["plans"] = timeline["future"]["plans"][-10:]
    _save_timeline_dict(timeline)
    return plan_id


def get_active_plan() -> Optional[Dict[str, Any]]:
    """Return the first active plan from the timeline, or None."""
    timeline = _load_timeline_dict()
    for p in timeline.get("future", {}).get("plans", []):
        if p.get("status") == "active":
            return p
    return None


def update_plan_step(
    plan_id: str, step_id: str, new_status: str, note: str = ""
) -> bool:
    """Update a single step's status in a plan. Returns True if found."""
    timeline = _load_timeline_dict()
    for p in timeline.get("future", {}).get("plans", []):
        if p.get("id") == plan_id:
            for s in p.get("steps", []):
                if s.get("id") == step_id:
                    s["status"] = new_status
                    if note:
                        s["note"] = note
                    if new_status in ("complete", "failed"):
                        s["completed_at"] = now_iso()
                    total = len(p["steps"])
                    done = sum(1 for st in p["steps"] if st.get("status") == "complete")
                    p["progress"] = f"{done}/{total} steps"
                    _save_timeline_dict(timeline)
                    return True
    return False


def complete_plan(plan_id: str, status: str = "complete") -> bool:
    """Mark a plan as complete or failed. Returns True if found."""
    timeline = _load_timeline_dict()
    for p in timeline.get("future", {}).get("plans", []):
        if p.get("id") == plan_id:
            p["status"] = status
            p["completed_at"] = now_iso()
            _save_timeline_dict(timeline)
            return True
    return False


# ── Self-model helpers (thin wrappers around SelfModel class) ──────


def load_self_model() -> Dict[str, Any]:
    """Load the self-model from disk."""
    sm = SelfModel.load()
    return sm.data


def save_self_model(model: Dict[str, Any]) -> None:
    """Persist the self-model to disk."""
    sm = SelfModel(data=model)
    sm.save()


def update_self_model(**updates) -> None:
    """Update specific fields of the self-model."""
    sm = SelfModel.load()
    for section, data in updates.items():
        if section in sm.data and isinstance(data, dict):
            sm.data[section].update(data)
    sm.save()


def format_self_model_context() -> str:
    """Format the self-model as a natural text block for the system prompt."""
    sm = SelfModel.load()
    parts: List[str] = ["## Self Model"]
    parts.append(
        f"Identity: {sm.identity.get('name', '—')} — {sm.identity.get('role', '—')}"
    )
    ver = sm.state.get("evolution_version", 0)
    gap = sm.state.get("current_gap_focus")
    gap_str = f", current gap focus: {gap}" if gap else ""
    parts.append(f"Evolution: v{ver}{gap_str}")
    weaknesses = sm.capabilities.get("weaknesses", [])
    if weaknesses:
        parts.append(f"Known weaknesses: {'; '.join(weaknesses[:3])}")
    unknowns = sm.capabilities.get("unknown_areas", [])
    if unknowns:
        parts.append(f"Areas to learn: {'; '.join(unknowns[:3])}")
    project = sm.data.get("commitments", {}).get("current_project")
    if project:
        parts.append(f"Committed to: {project}")
    return "\n".join(parts)


def format_plan_context() -> str:
    """Format the active plan as readable text for the system prompt."""
    active = get_active_plan()
    if not active:
        return ""
    goal = active.get("goal", "")
    progress = active.get("progress", "0/0 steps")
    steps = active.get("steps", [])
    lines = [f"Active plan: {goal} ({progress})"]
    for s in steps:
        icon = {
            "complete": "✓",
            "blocked": "⊘",
            "in_progress": "●",
            "pending": "→",
        }.get(s.get("status", "pending"), "·")
        lines.append(f"  {icon} {s['description']}")
        if s.get("note"):
            lines.append(f"     note: {s['note']}")
    return "\n".join(lines)


def format_timeline_context() -> str:
    """Format timeline as natural narrative for the system prompt.

    Reads the dict-format timeline (past/present/future) from disk and
    produces a concise briefing.
    """
    timeline = _load_timeline_dict()
    narrative: List[str] = ["## Timeline"]

    # Past: last session
    completed = timeline.get("past", {}).get("completed_sessions", [])
    recent_session = completed[-1] if completed else None
    if recent_session:
        focus = recent_session.get("focus", "—")
        n_outcomes = len(recent_session.get("outcomes", []))
        narrative.append(f"Last session focus: {focus}")
        if n_outcomes:
            narrative.append(f"Completed {n_outcomes} items.")

    # Past: recent events
    events = timeline.get("past", {}).get("events", [])
    recent = events[-3:] if events else []
    if recent:
        narrative.append("Recent events:")
        for e in recent:
            narrative.append(f"  · {e.get('summary', '—')}")

    # Outcomes
    outcomes = timeline.get("past", {}).get("outcomes", [])
    recent_outcomes = outcomes[-2:] if outcomes else []
    if recent_outcomes:
        narrative.append("Results:")
        for o in recent_outcomes:
            s = f"  · {o.get('summary', '—')}"
            if o.get("impact"):
                s += f" ({o['impact']})"
            narrative.append(s)

    # Present: active context
    present = timeline.get("present", {})
    project = present.get("active_project")
    tasks = present.get("active_tasks", [])
    waiting = present.get("waiting_for", [])
    bits = []
    if project:
        bits.append(f"project: {project}")
    if tasks:
        bits.append(f"tasks: {'; '.join(tasks[:3])}")
    if waiting:
        bits.append(f"waiting: {'; '.join(waiting[:2])}")
    if bits:
        narrative.append("Current state — " + "; ".join(bits))

    # Commitments
    commitments = present.get("commitments", [])
    active_commits = [c for c in commitments if c.get("status") == "active"]
    if active_commits:
        narrative.append("Active commitments:")
        for c in active_commits[:3]:
            dl = f" (by {c['deadline']})" if c.get("deadline") else ""
            narrative.append(f"  · {c['what']}{dl}")

    # Future: plan
    plan_text = format_plan_context()
    if plan_text:
        narrative.append(plan_text)

    # Future: goals + predictions
    future = timeline.get("future", {})
    goals = future.get("goals", [])
    if goals:
        narrative.append(f"Goals: {' → '.join(goals[:3])}")
    predictions = future.get("predictions", [])
    if predictions:
        last_pred = predictions[-1]
        tf = f" [{last_pred.get('timeframe', '')}]" if last_pred.get("timeframe") else ""
        conf = f" (confidence: {last_pred.get('confidence', '')})" if last_pred.get("confidence") else ""
        narrative.append(f"Prediction:{tf} {last_pred.get('text', '')}{conf}")

    return "\n".join(narrative)


# ═══════════════════════════════════════════════════════════════════
#  Unified context injection — used by system_prompt.py
# ═══════════════════════════════════════════════════════════════════

def format_orientation_context() -> str:
    """Format orientation + timeline + self-model + memory + goals for the
    system prompt.

    Returns a multi-section string that gives the agent cross-session
    continuity: previous focus, timeline awareness, self-model identity,
    multi-type memory, and goal status.
    """
    parts: List[str] = []

    # Section 1: Session orientation
    orient = Orientation.load()
    if orient.focus or orient.data.get("insights") or orient.data.get("next_steps"):
        parts.append("## Session Orientation")
        if orient.focus:
            parts.append(f"Previous focus: {orient.focus}")
        insights = orient.data.get("insights", [])
        if insights:
            parts.append("Recent insights:")
            for ins in insights[-3:]:
                parts.append(f"  - {ins}")
        next_steps = orient.data.get("next_steps", [])
        if next_steps:
            parts.append("Unfinished direction:")
            for step in next_steps[-3:]:
                parts.append(f"  - {step}")

    # Section 2: Timeline
    tl = format_timeline_context()
    if tl and tl != "## Timeline":
        parts.append(tl)

    # Section 3: Multi-type Memory
    mem = Memory.load()
    mem_str = mem.format_context()
    if mem_str:
        parts.append(f"## Recent Memories\n{mem_str}")

    # Section 4: Self-generated Goals
    goals = Goals.load()
    goals_str = goals.format_context()
    if goals_str:
        parts.append(goals_str)

    # Section 5: Self Model (reuses dedicated formatter)
    self_model_str = format_self_model_context()
    if self_model_str:
        parts.append(self_model_str)

    if not parts:
        return ""

    return "\n\n".join(parts)
