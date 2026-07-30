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


class TestWmSelfBridge:
    """Tests for auto-syncing world model patterns into self-model weaknesses."""

    def _setup_wm_with_pattern(self, action_type: str = "shell",
                                count: int = 3, avg_error: float = 0.7):
        """Create a world model with controlled discrepancy patterns."""
        wm = WorldModel()
        for i in range(count):
            tid = wm.record_action(
                action_type,
                f"{action_type} action {i}",
                "expected success",
            )
            # All have the same error level
            if avg_error >= 0.8:
                wm.complete_action(tid, "exit=1: failure")
            elif avg_error >= 0.4:
                wm.complete_action(tid, "exit=0: some unexpected output")
            else:
                wm.complete_action(tid, "exit=0: exactly as expected")
        return wm

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
        wm = self._setup_wm_with_pattern("shell", count=3, avg_error=0.85)
        sm = {"capabilities": {"weaknesses": []}}

        added = bridge(wm, sm)
        assert added == 1, f"Expected 1 weakness added, got {added}"

        weaknesses = sm["capabilities"]["weaknesses"]
        assert len(weaknesses) == 1
        assert "Systematic prediction bias:" in weaknesses[0]
        assert "shell" in weaknesses[0]
        assert "0.85" in weaknesses[0] or "0.8" in weaknesses[0]

    def test_low_error_pattern_no_weakness(self):
        """Low-error pattern should NOT produce a weakness (threshold check)."""
        bridge = _import_bridge()
        wm = self._setup_wm_with_pattern("shell", count=3, avg_error=0.15)
        sm = {"capabilities": {"weaknesses": []}}

        added = bridge(wm, sm)
        assert added == 0, f"Expected 0 weaknesses for low error, got {added}"

    def test_insufficient_samples_no_weakness(self):
        """Pattern with <2 samples should not produce a weakness."""
        bridge = _import_bridge()
        wm = self._setup_wm_with_pattern("shell", count=1, avg_error=0.9)
        sm = {"capabilities": {"weaknesses": []}}

        added = bridge(wm, sm)
        assert added == 0, f"Expected 0 weaknesses for single outlier, got {added}"

    def test_multiple_action_types_multiple_weaknesses(self):
        """Multiple high-error types should each get a weakness."""
        bridge = _import_bridge()
        wm = WorldModel()
        # Add high-error shell actions
        for i in range(2):
            tid = wm.record_action("shell", f"shell {i}", "ok")
            wm.complete_action(tid, "exit=1: fail")
        # Add high-error write_file actions
        for i in range(2):
            tid = wm.record_action("write_file", f"write {i}", "ok")
            wm.complete_action(tid, "permission denied")
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
        wm = self._setup_wm_with_pattern("shell", count=3, avg_error=0.85)
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
        """Auto-weakness should be removed when its pattern resolves."""
        bridge = _import_bridge()
        # Start with shell having high error
        wm = self._setup_wm_with_pattern("shell", count=3, avg_error=0.85)
        sm = {"capabilities": {"weaknesses": []}}

        added = bridge(wm, sm)
        assert added == 1

        # Now simulate the pattern resolving: add more low-error shell actions
        # that make the recent 5 all good (triggering recency decay)
        for i in range(5):
            tid = wm.record_action("shell", f"fixed shell {i}", "ok")
            wm.complete_action(tid, "exit=0: perfect")

        # Bridge again — the auto weakness should be removed because the
        # recency-based decay in _update_discrepancy_patterns detects that
        # all 5 most recent shell actions have error <= 0.3.
        added2 = bridge(wm, sm)
        weaknesses = sm["capabilities"]["weaknesses"]
        auto_weaknesses = [w for w in weaknesses if w.startswith("Systematic prediction bias:")]
        # The pattern is suppressed by recency decay (5 consecutive good actions)
        # even if the overall type avg is still slightly above 0.4
        assert len(auto_weaknesses) == 0, (
            f"Expected auto weakness removed by recency decay, "
            f"got {auto_weaknesses}"
        )

    def test_stale_auto_weakness_persists_with_recent_failure(self):
        """Auto-weakness should persist if a recent action still has high error."""
        bridge = _import_bridge()
        wm = self._setup_wm_with_pattern("shell", count=3, avg_error=0.85)
        sm = {"capabilities": {"weaknesses": []}}

        added = bridge(wm, sm)
        assert added == 1

        # Add 4 good actions — not enough for recency decay (need 5)
        for i in range(4):
            tid = wm.record_action("shell", f"ok shell {i}", "ok")
            wm.complete_action(tid, "exit=0: perfect")

        # And 1 recent failure
        tid = wm.record_action("shell", "recent fail", "should work")
        wm.complete_action(tid, "exit=1: failed")

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
                tid = wm.record_action(atype, f"{atype} {j}", "ok")
                wm.complete_action(tid, "exit=1: error")

        sm = {"capabilities": {"weaknesses": []}}
        added = bridge(wm, sm)
        assert added >= 1
        assert len(sm["capabilities"]["weaknesses"]) <= 10

    def test_import_from_think_daemon(self):
        """Verify the function is importable from think_daemon."""
        bridge = _import_bridge()
        assert callable(bridge)
