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

def _now_iso() -> str:
    """Current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _now_compact() -> str:
    """Compact timestamp safe for filenames / IDs."""
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _atomic_write(path: Path, data: Any) -> Path:
    """Write JSON atomically: write to .tmp, then replace."""
    _ensure_dir(path)
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
            "id": _now_compact(),
            "timestamp": _now_iso(),
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
        """Persist all events as a JSON list.

        Uses atomic write (``.tmp`` → final) for crash safety.
        """
        target = path or self.storage_path()
        _atomic_write(target, self.events)
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
            self._merge_defaults(data) if data else dict(SelfModel._DEFAULT_DATA)
        )

    # ── Internal ──────────────────────────────────────────────────

    @staticmethod
    def _merge_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
        """Deep-merge loaded data with defaults so new keys appear."""
        merged = dict(SelfModel._DEFAULT_DATA)
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
            "created_at": _now_iso(),
        }
        self.data.setdefault("commitments", {})
        self.data["commitments"].setdefault("active_obligations", []).append(c)
        return c

    def complete_commitment(self, what: str) -> bool:
        """Mark a commitment as completed by its ``what`` text."""
        for c in self.data.get("commitments", {}).get("active_obligations", []):
            if c.get("what") == what:
                c["status"] = "completed"
                c["completed_at"] = _now_iso()
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
