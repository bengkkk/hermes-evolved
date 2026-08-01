"""Tests for _reconcile_goals_with_world — auto-completion of stale goals."""

import copy
import pytest
from world_model import WorldModel
from data_layer import Goals


@pytest.fixture
def wm_with_data() -> WorldModel:
    """World model with per-type accuracy that satisfies some sample goals."""
    wm = WorldModel()
    wm.data["per_type_accuracy"] = {
        "shell": {"count": 48, "avg_error": 0.23, "min_error": 0.15, "max_error": 0.85},
        "write_file": {"count": 6, "avg_error": 0.17, "min_error": 0.15, "max_error": 0.25},
        "git_commit": {"count": 4, "avg_error": 0.23, "min_error": 0.15, "max_error": 0.25},
        "install_package": {"count": 2, "avg_error": 0.15, "min_error": 0.15, "max_error": 0.15},
    }
    wm.data["prediction_accuracy"]["total_triples"] = 60
    wm.data["prediction_accuracy"]["avg_triple_error"] = 0.22
    return wm


@pytest.fixture
def goals_with_stale() -> dict:
    """Goals dict with a mix of stale and still-relevant goals."""
    return {
        "version": 1,
        "goals": [
            {
                "id": "g_stale_shell_samples",
                "title": "Gather more shell action samples for reliable calibration",
                "description": "Only 2 shell actions recorded.",
                "priority": 4, "status": "proposed",
                "gap_reference": "6",
                "created_at": "2026-07-29T22:32:58",
                "completed_at": None, "notes": "",
            },
            {
                "id": "g_valid_install_samples",
                "title": "Gather more install_package action samples for reliable calibration",
                "description": "Only 2 install_package actions recorded.",
                "priority": 4, "status": "proposed",
                "gap_reference": "6",
                "created_at": "2026-07-29T22:12:14",
                "completed_at": None, "notes": "",
            },
            {
                "id": "g_stale_investigate_shell",
                "title": "Investigate shell prediction failures (keywords: read, protected)",
                "description": "Shell actions have high prediction error.",
                "priority": 3, "status": "proposed",
                "gap_reference": "6",
                "created_at": "2026-07-30T04:01:01",
                "completed_at": None, "notes": "",
            },
            {
                "id": "g_stale_data_layer",
                "title": "Resolve data layer completeness",
                "description": "Ensure DataLayer and SelfModel classes work.",
                "priority": 1, "status": "proposed",
                "gap_reference": "8",
                "created_at": "2026-07-30T04:46:53",
                "completed_at": None, "notes": "",
            },
            {
                "id": "g_stale_overconfidence",
                "title": "Fix overconfidence at 60–80% confidence level",
                "description": "Predictions at 60-80% have avg error 0.38.",
                "priority": 3, "status": "proposed",
                "gap_reference": "6",
                "created_at": "2026-07-30T07:55:35",
                "completed_at": None, "notes": "",
            },
            {
                "id": "g_dup_older",
                "title": "Duplicate Goal",
                "description": "Older version",
                "priority": 3, "status": "proposed",
                "gap_reference": "6",
                "created_at": "2026-07-30T10:00:00",
                "completed_at": None, "notes": "",
            },
            {
                "id": "g_dup_newer",
                "title": "Duplicate Goal",
                "description": "Newer version",
                "priority": 1, "status": "proposed",
                "gap_reference": "8",
                "created_at": "2026-07-30T10:30:00",
                "completed_at": None, "notes": "",
            },
            {
                "id": "g_still_valid",
                "title": "Autonomous goal selection in think_daemon",
                "description": "Select goals autonomously.",
                "priority": 1, "status": "proposed",
                "gap_reference": "8",
                "created_at": "2026-07-30T07:29:21",
                "completed_at": None, "notes": "",
            },
        ],
    }


class TestReconcileGoalsWithWorld:
    """Tests for _reconcile_goals_with_world."""

    def test_imports(self):
        """Function is importable."""
        from think_daemon import _reconcile_goals_with_world
        assert callable(_reconcile_goals_with_world)

    def test_no_goals_returns_zero(self, wm_with_data):
        """No active goals → 0 completed."""
        from think_daemon import _reconcile_goals_with_world
        goals = {"version": 1, "goals": []}
        count = _reconcile_goals_with_world(wm_with_data, goals)
        assert count == 0

    def test_all_goals_still_valid(self, wm_with_data):
        """Goals that don't match any pattern stay unchanged."""
        from think_daemon import _reconcile_goals_with_world
        goals = {
            "version": 1,
            "goals": [
                {
                    "id": "g_active",
                    "title": "Build the ultimate AGI",
                    "description": "Real work.",
                    "priority": 1, "status": "active",
                    "created_at": "2026-07-30T00:00:00",
                    "completed_at": None, "notes": "",
                },
            ],
        }
        count = _reconcile_goals_with_world(wm_with_data, goals)
        assert count == 0
        g = Goals(data=goals)
        active = g.get_active()
        assert len(active) == 1
        assert active[0]["id"] == "g_active"

    def test_gather_more_samples_completes_when_enough_data(self, wm_with_data):
        """'Gather more shell action samples' completes when shell count >= 3."""
        from think_daemon import _reconcile_goals_with_world
        goals = {
            "version": 1,
            "goals": [
                {
                    "id": "g_shell",
                    "title": "Gather more shell action samples for reliable calibration",
                    "priority": 4, "status": "proposed",
                    "created_at": "2026-07-29T00:00:00",
                    "completed_at": None, "notes": "",
                },
            ],
        }
        count = _reconcile_goals_with_world(wm_with_data, goals)
        assert count == 1
        g = Goals(data=goals)
        assert g.get_active() == []  # no longer active

    def test_gather_more_samples_keeps_when_insufficient_data(self, wm_with_data):
        """'Gather more install_package samples' stays active (only 2 samples)."""
        from think_daemon import _reconcile_goals_with_world
        goals = {
            "version": 1,
            "goals": [
                {
                    "id": "g_install",
                    "title": "Gather more install_package action samples for reliable calibration",
                    "priority": 4, "status": "proposed",
                    "created_at": "2026-07-29T00:00:00",
                    "completed_at": None, "notes": "",
                },
            ],
        }
        count = _reconcile_goals_with_world(wm_with_data, goals)
        assert count == 0  # only 2 install_package samples < 3
        g = Goals(data=goals)
        assert len(g.get_active()) == 1

    def test_investigate_completes_when_error_dropped(self, wm_with_data):
        """'Investigate shell prediction failures' completes when error < 0.4."""
        from think_daemon import _reconcile_goals_with_world
        goals = {
            "version": 1,
            "goals": [
                {
                    "id": "g_investigate",
                    "title": "Investigate shell prediction failures (keywords: read, protected)",
                    "priority": 3, "status": "proposed",
                    "created_at": "2026-07-30T00:00:00",
                    "completed_at": None, "notes": "",
                },
            ],
        }
        count = _reconcile_goals_with_world(wm_with_data, goals)
        assert count == 1
        g = Goals(data=goals)
        assert g.get_active() == []

    def test_overconfidence_completes_when_low_error(self, wm_with_data):
        """'Fix overconfidence' completes when overall avg_error < 0.3."""
        from think_daemon import _reconcile_goals_with_world
        goals = {
            "version": 1,
            "goals": [
                {
                    "id": "g_overconf",
                    "title": "Fix overconfidence at 60–80% confidence level",
                    "priority": 3, "status": "proposed",
                    "created_at": "2026-07-30T00:00:00",
                    "completed_at": None, "notes": "",
                },
            ],
        }
        count = _reconcile_goals_with_world(wm_with_data, goals)
        assert count == 1
        g = Goals(data=goals)
        assert g.get_active() == []

    def test_overconfidence_keeps_when_error_high(self, wm_with_data):
        """'Fix overconfidence' stays when avg_error >= 0.3."""
        from think_daemon import _reconcile_goals_with_world
        wm_high = copy.deepcopy(wm_with_data)
        wm_high.data["prediction_accuracy"]["avg_triple_error"] = 0.45
        goals = {
            "version": 1,
            "goals": [
                {
                    "id": "g_overconf",
                    "title": "Fix overconfidence at 60–80% confidence level",
                    "priority": 3, "status": "proposed",
                    "created_at": "2026-07-30T00:00:00",
                    "completed_at": None, "notes": "",
                },
            ],
        }
        count = _reconcile_goals_with_world(wm_high, goals)
        assert count == 0
        g = Goals(data=goals)
        assert len(g.get_active()) == 1

    def test_data_layer_goal_completes_when_imports_work(self, wm_with_data):
        """'Resolve data layer completeness' completes when SelfModel imports."""
        from think_daemon import _reconcile_goals_with_world
        goals = {
            "version": 1,
            "goals": [
                {
                    "id": "g_dl",
                    "title": "Resolve data layer completeness",
                    "priority": 1, "status": "proposed",
                    "created_at": "2026-07-30T00:00:00",
                    "completed_at": None, "notes": "",
                },
            ],
        }
        count = _reconcile_goals_with_world(wm_with_data, goals)
        assert count == 1
        g = Goals(data=goals)
        assert g.get_active() == []

    def test_duplicate_deduplication(self, wm_with_data):
        """Duplicate titles: only the newest stays active."""
        from think_daemon import _reconcile_goals_with_world
        goals = {
            "version": 1,
            "goals": [
                {
                    "id": "g_old",
                    "title": "Some Goal",
                    "priority": 3, "status": "proposed",
                    "created_at": "2026-07-30T10:00:00",
                    "completed_at": None, "notes": "",
                },
                {
                    "id": "g_new",
                    "title": "Some Goal",
                    "priority": 1, "status": "proposed",
                    "created_at": "2026-07-30T10:30:00",
                    "completed_at": None, "notes": "",
                },
            ],
        }
        count = _reconcile_goals_with_world(wm_with_data, goals)
        assert count == 1  # only the older duplicate completed
        g = Goals(data=goals)
        active = g.get_active()
        assert len(active) == 1
        assert active[0]["id"] == "g_new"  # newest survived

    def test_full_reconciliation(self, wm_with_data, goals_with_stale):
        """Integration test: mixed stale/valid goals produce correct results."""
        from think_daemon import _reconcile_goals_with_world
        goals = copy.deepcopy(goals_with_stale)
        count = _reconcile_goals_with_world(wm_with_data, goals)
        # Expected completions:
        # - g_stale_shell_samples: shell has 48 samples >= 3
        # - g_stale_investigate_shell: shell avg 0.23 < 0.4
        # - g_stale_data_layer: SelfModel imports OK
        # - g_stale_overconfidence: avg_error 0.22 < 0.3
        # - g_dup_older: duplicate of g_dup_newer (older)
        # = 5 completions
        assert count == 5, f"Expected 5 completions, got {count}"

        g = Goals(data=goals)
        active = g.get_active()
        active_ids = {a["id"] for a in active}
        # These should remain active:
        assert "g_valid_install_samples" in active_ids  # install has 2 < 3
        assert "g_dup_newer" in active_ids  # newer duplicate kept
        assert "g_still_valid" in active_ids  # no pattern match
        # These should be completed:
        completed = [go for go in g.data["goals"] if go.get("status") == "completed"]
        completed_ids = {c["id"] for c in completed}
        assert "g_stale_shell_samples" in completed_ids
        assert "g_stale_investigate_shell" in completed_ids
        assert "g_stale_data_layer" in completed_ids
        assert "g_stale_overconfidence" in completed_ids
        assert "g_dup_older" in completed_ids

    def test_completed_goal_has_note(self, wm_with_data):
        """Completed goals get an informative note."""
        from think_daemon import _reconcile_goals_with_world
        goals = {
            "version": 1,
            "goals": [
                {
                    "id": "g_shell",
                    "title": "Gather more shell action samples for reliable calibration",
                    "priority": 4, "status": "proposed",
                    "created_at": "2026-07-29T00:00:00",
                    "completed_at": None, "notes": "",
                },
            ],
        }
        _reconcile_goals_with_world(wm_with_data, goals)
        g = Goals(data=goals)
        for goal in g.data["goals"]:
            if goal["id"] == "g_shell":
                assert goal["status"] == "completed"
                assert "Auto-completed:" in goal.get("notes", "")
                assert "48" in goal.get("notes", "")  # mentions sample count


class TestBridgeProofGoalReconciliation:
    """Auto-completion of the Gap 10 bridge-proof goal.

    The world model records api_call triples whose actual_outcome embeds the
    host-bridge result (``exit=0`` plus an HTTP status).  A goal titled
    "Prove the external action bridge with a GitHub read" must flip to
    completed exactly when such a successful triple exists.
    """

    def _bridge_wm(self, outcome: str) -> WorldModel:
        wm = WorldModel()
        wm.data["action_triples"] = [
            {
                "id": "act_test_1",
                "action_type": "api_call",
                "action_description": "GitHub read through the open gate",
                "action_parameters": {"endpoint": "https://api.github.com/", "method": "GET"},
                "expected_outcome": "HTTP 200 JSON body",
                "prediction_confidence": 0.65,
                "actual_outcome": outcome,
            }
        ]
        return wm

    def _bridge_goal(self) -> dict:
        return {
            "version": 1,
            "goals": [
                {
                    "id": "g_prove_bridge",
                    "title": "Prove the external action bridge with a GitHub read",
                    "description": "Turn Gap 10 into a working proof.",
                    "priority": 4, "status": "proposed",
                    "gap_reference": "10",
                    "created_at": "2026-08-01T18:51:00",
                    "completed_at": None, "notes": "",
                },
            ],
        }

    def test_completed_when_api_call_triple_is_http_success(self):
        from think_daemon import _reconcile_goals_with_world
        wm = self._bridge_wm('exit=0: {"status": 200, "bytes": 2262}')
        goals = self._bridge_goal()
        count = _reconcile_goals_with_world(wm, goals)
        assert count == 1
        g = Goals(data=goals)
        assert g.get_active() == []
        completed = g.data["goals"][0]
        assert completed["status"] == "completed"
        assert "Auto-completed:" in completed.get("notes", "")

    def test_stays_open_when_api_call_failed(self):
        from think_daemon import _reconcile_goals_with_world
        wm = self._bridge_wm('exit=1: {"status": 500, "bytes": 12}')
        goals = self._bridge_goal()
        count = _reconcile_goals_with_world(wm, goals)
        assert count == 0
        assert Goals(data=goals).get_active()[0]["id"] == "g_prove_bridge"

    def test_stays_open_without_api_call_triples(self):
        from think_daemon import _reconcile_goals_with_world
        wm = WorldModel()
        wm.data["action_triples"] = [
            {
                "id": "act_shell_1", "action_type": "shell",
                "action_description": "list files", "action_parameters": {},
                "expected_outcome": "listing", "actual_outcome": "file names",
            },
        ]
        goals = self._bridge_goal()
        count = _reconcile_goals_with_world(wm, goals)
        assert count == 0
        assert Goals(data=goals).get_active()[0]["id"] == "g_prove_bridge"
