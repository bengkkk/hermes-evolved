"""Tests for the World Model → Self-Model bridge (think_daemon._bridge_world_model_to_self_model)."""

import sys
from pathlib import Path

# Ensure we can import from the repo root
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from world_model import WorldModel


def _import_bridge():
    """Import the bridge function from think_daemon."""
    from think_daemon import _bridge_world_model_to_self_model
    return _bridge_world_model_to_self_model


def _add_triple(wm: WorldModel, action_type: str, desc: str, error: float):
    """Add a completed action triple with a directly-set prediction error.

    Uses record_action + record_action_complete but then overrides the
    prediction_error field so tests are deterministic regardless of the
    text-comparison heuristic.  The bridge reads prediction_error directly
    from triples, so this is the correct level of control.
    """
    result = wm.record_action_complete(
        action_type, desc,
        action_output=f"output (error={error:.2f})",
        expected_outcome=f"expected {desc}",
    )
    # Override the computed error with the exact value we want for testing
    for t in wm.data.get("action_triples", []):
        if t.get("id") == result["id"]:
            t["prediction_error"] = error
            break
    wm._update_accuracy_stats()
    return result


class TestWmSelfBridge:
    """Tests for auto-syncing world model patterns into self-model weaknesses."""

    def test_no_patterns_no_weakness(self):
        """Empty world model should not add weaknesses."""
        bridge = _import_bridge()
        wm = WorldModel()
        sm = {"capabilities": {"weaknesses": []}}

        added = bridge(wm, sm)
        assert added == 0
        assert sm["capabilities"]["weaknesses"] == []

    def test_high_error_pattern_adds_weakness(self):
        """High-error shell pattern should produce a weakness."""
        bridge = _import_bridge()
        wm = WorldModel()
        for i in range(3):
            _add_triple(wm, "shell", f"shell action {i}", error=0.8)
        sm = {"capabilities": {"weaknesses": []}}

        added = bridge(wm, sm)
        assert added == 1, f"Expected 1 weakness added, got {added}"

        weaknesses = sm["capabilities"]["weaknesses"]
        assert len(weaknesses) == 1
        assert "Systematic prediction bias:" in weaknesses[0]
        assert "shell" in weaknesses[0]

    def test_low_error_pattern_no_weakness(self):
        """Low-error pattern should NOT produce a weakness (threshold check)."""
        bridge = _import_bridge()
        wm = WorldModel()
        for i in range(3):
            _add_triple(wm, "shell", f"shell action {i}", error=0.25)
        sm = {"capabilities": {"weaknesses": []}}

        added = bridge(wm, sm)
        assert added == 0, f"Expected 0 weaknesses for low error, got {added}"

    def test_insufficient_samples_no_weakness(self):
        """Pattern with <2 samples should not produce a weakness."""
        bridge = _import_bridge()
        wm = WorldModel()
        _add_triple(wm, "shell", "single action", error=0.9)
        sm = {"capabilities": {"weaknesses": []}}

        added = bridge(wm, sm)
        assert added == 0, f"Expected 0 weaknesses for single outlier, got {added}"

    def test_multiple_action_types_multiple_weaknesses(self):
        """Multiple high-error types should each get a weakness."""
        bridge = _import_bridge()
        wm = WorldModel()
        # Add high-error shell actions
        for i in range(3):
            _add_triple(wm, "shell", f"shell {i}", error=0.8)
        # Add high-error write_file actions
        for i in range(3):
            _add_triple(wm, "write_file", f"write {i}", error=0.8)
        sm = {"capabilities": {"weaknesses": []}}

        added = bridge(wm, sm)
        assert added >= 2, f"Expected at least 2 weaknesses, got {added}"
        weaknesses = sm["capabilities"]["weaknesses"]
        shell_w = [w for w in weaknesses if "shell" in w]
        write_w = [w for w in weaknesses if "write_file" in w]
        assert len(shell_w) >= 1, f"No shell weakness in {weaknesses}"
        assert len(write_w) >= 1, f"No write_file weakness in {weaknesses}"

    def test_preserves_manual_weaknesses(self):
        """LLM-generated weaknesses should not be removed."""
        bridge = _import_bridge()
        wm = WorldModel()
        for i in range(3):
            _add_triple(wm, "shell", f"shell {i}", error=0.8)
        sm = {
            "capabilities": {
                "weaknesses": [
                    "LLM API unreliable (timeout during thinking cycle)",
                    "Not enough training data for X",
                ]
            }
        }

        added = bridge(wm, sm)
        assert added == 1
        weaknesses = sm["capabilities"]["weaknesses"]
        assert "LLM API unreliable" in str(weaknesses)
        assert "Not enough training data" in str(weaknesses)

    def test_stale_auto_weakness_removed(self):
        """Auto-weakness should be removed when its pattern resolves.

        Uses the recency-based decay in _update_discrepancy_patterns:
        when the 5 most recent actions of a type all have error <= 0.3,
        the pattern is suppressed even if earlier actions were high-error.
        """
        bridge = _import_bridge()
        wm = WorldModel()
        # Start with 3 high-error shell actions
        for i in range(3):
            _add_triple(wm, "shell", f"old shell {i}", error=0.8)
        sm = {"capabilities": {"weaknesses": []}}

        added = bridge(wm, sm)
        assert added == 1

        # Now add 5 consecutive low-error shell actions (triggers recency decay)
        for i in range(5):
            _add_triple(wm, "shell", f"fixed shell {i}", error=0.15)

        # Bridge again — the auto weakness should be removed because recency
        # decay detects all 5 most recent have error <= 0.3.
        added2 = bridge(wm, sm)
        weaknesses = sm["capabilities"]["weaknesses"]
        auto_weaknesses = [w for w in weaknesses if w.startswith("Systematic prediction bias:")]
        assert len(auto_weaknesses) == 0, (
            f"Expected auto weakness removed by recency decay, "
            f"got {auto_weaknesses}"
        )

    def test_stale_auto_weakness_persists_with_recent_failure(self):
        """Auto-weakness should persist if a recent action still has high error."""
        bridge = _import_bridge()
        wm = WorldModel()
        for i in range(3):
            _add_triple(wm, "shell", f"old shell {i}", error=0.8)

        sm = {"capabilities": {"weaknesses": []}}
        added = bridge(wm, sm)
        assert added == 1

        # Add 4 good actions — not enough for recency decay (need 5)
        for i in range(4):
            _add_triple(wm, "shell", f"ok shell {i}", error=0.15)

        # And 1 recent failure
        _add_triple(wm, "shell", "recent fail", error=0.85)

        added2 = bridge(wm, sm)
        weaknesses = sm["capabilities"]["weaknesses"]
        auto_weaknesses = [w for w in weaknesses if w.startswith("Systematic prediction bias:")]
        # Pattern persists because the 5th most recent action is a failure
        assert len(auto_weaknesses) >= 1, (
            f"Expected auto weakness to persist with recent failure, "
            f"got {auto_weaknesses}"
        )

    def test_capped_at_ten(self):
        """Weakness list should be capped at 10 entries."""
        bridge = _import_bridge()
        wm = WorldModel()
        # Create high-error patterns for many action types
        for atype in [f"type_{i}" for i in range(8)]:
            for j in range(2):
                _add_triple(wm, atype, f"{atype} {j}", error=0.8)

        sm = {"capabilities": {"weaknesses": []}}
        added = bridge(wm, sm)
        assert added >= 1
        assert len(sm["capabilities"]["weaknesses"]) <= 10

    def test_import_from_think_daemon(self):
        """Verify the function is importable from think_daemon."""
        bridge = _import_bridge()
        assert callable(bridge)

    def test_removes_stale_weakness_and_adds_new(self):
        """Replacing one pattern with another removes the stale and adds the new."""
        bridge = _import_bridge()
        wm = WorldModel()
        # High-error shell
        for i in range(3):
            _add_triple(wm, "shell", f"shell {i}", error=0.8)
        sm = {"capabilities": {"weaknesses": []}}

        added = bridge(wm, sm)
        assert added == 1
        assert "shell" in str(sm["capabilities"]["weaknesses"])

        # Now resolve shell pattern and create a write_file pattern instead
        for i in range(5):
            _add_triple(wm, "shell", f"fixed shell {i}", error=0.15)
        for i in range(3):
            _add_triple(wm, "write_file", f"write {i}", error=0.85)

        added2 = bridge(wm, sm)
        weaknesses = sm["capabilities"]["weaknesses"]
        auto_weaknesses = [w for w in weaknesses if w.startswith("Systematic prediction bias:")]
        # Only write_file pattern should have an auto-weakness
        # Format: "Systematic prediction bias: <type> actions have ..."
        auto_types = [w.split(" ")[3] if len(w.split(" ")) > 3 else "" for w in auto_weaknesses]
        assert "shell" not in auto_types, (
            f"Stale shell auto-weakness not removed: {auto_weaknesses}"
        )
        assert "write_file" in auto_types, (
            f"Expected write_file auto-weakness, got {auto_types}"
        )
