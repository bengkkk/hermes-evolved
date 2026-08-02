"""Tests for scripts/fire_gap10_write.py — target-name resolution.

The daemon's LLM-generated shell commands historically passed the repo-relative
file path (``fire_gap10_write.py evidence/gap10-level1.md``) instead of the
logical target name (``evidence1``), producing a usage error (exit=2) and a
world-model triple with prediction_error 0.6. This pins the path-alias fix so
that failure mode cannot regress. Pure logic — no network, no bridge.

See world-model triple act_20260802225501_4 for the recorded discrepancy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fire_gap10_write as f  # noqa: E402


def test_logical_names_resolve_to_themselves():
    assert f._resolve_target("plan") == "plan"
    assert f._resolve_target("evidence1") == "evidence1"


def test_repo_relative_path_aliases_resolve():
    assert f._resolve_target("evidence/gap10-level1.md") == "evidence1"
    assert f._resolve_target("docs/gap10-level2-plan.md") == "plan"


def test_leading_dot_slash_path_alias_resolves():
    assert f._resolve_target("./evidence/gap10-level1.md") == "evidence1"


def test_unknown_target_resolves_to_none():
    assert f._resolve_target("evidence/gap10-level2.md") is None
    assert f._resolve_target("other.md") is None
    assert f._resolve_target("") is None


def test_daemon_failure_command_now_resolves():
    """The exact command the daemon generated in cycle 477 must now resolve
    to the evidence1 target instead of producing a usage error."""
    assert f._resolve_target("evidence/gap10-level1.md") == "evidence1"
