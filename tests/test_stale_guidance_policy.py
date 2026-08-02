"""Tests for stale_guidance_policy.py (Gap 8 slice 1 — stale-guidance detection).

Pins the policy contract's invariants:

1. **Conservative skip** — SKIP_STALE requires positive evidence; missing
   facts (empty RepoFacts) always yields EXECUTE.
2. **Three reference types** — commit SHAs, verification markers, and
   clean-file commit directives each skip only on their own positive
   evidence.
3. **No false skips** — unknown SHAs, unverified artifacts, and dirty files
   all execute.
"""

import pytest

import stale_guidance_policy as sgp
from stale_guidance_policy import (
    EXECUTE,
    SKIP_STALE,
    RepoFacts,
    extract_commit_shas,
    extract_file_paths,
    filter_stale,
    is_commit_directive,
    is_verify_directive,
    verdict_for,
)

FACTS = RepoFacts(
    commits=frozenset({"d5ae20b8a", "3b060ff4c"}),
    verified_markers=frozenset({"docs/gap10-level2-plan.md", "evidence/gap10-level1.md"}),
    clean_files=frozenset({"docs/gap10-level2-plan.md", "evidence/gap10-level1.md"}),
)
EMPTY = RepoFacts()


# ── Conservative default: no evidence ⇒ EXECUTE ──────────────────────────────

def test_empty_facts_never_skips():
    assert verdict_for("verify the bridge is up", EMPTY)[0] == EXECUTE
    assert verdict_for("re-check gap10 wiring", EMPTY)[0] == EXECUTE
    assert verdict_for("commit the plan doc", EMPTY)[0] == EXECUTE
    assert verdict_for("push the changes", EMPTY)[0] == EXECUTE
    assert verdict_for("confirm d5ae20b8a landed", EMPTY)[0] == EXECUTE


def test_blank_directives_execute():
    assert verdict_for("", EMPTY)[0] == EXECUTE
    assert verdict_for("   ", EMPTY)[0] == EXECUTE
    assert verdict_for(None, EMPTY)[0] == EXECUTE


def test_empty_facts_never_skips_even_with_real_paths():
    # A commit directive naming a real file must still EXECUTE with no
    # evidence: "clean" is a positive fact, not the absence of dirty state.
    assert verdict_for("commit docs/gap10-level2-plan.md", EMPTY)[0] == EXECUTE
    assert verdict_for("commit stale_guidance_policy.py", EMPTY)[0] == EXECUTE


# ── Commit-SHA references ────────────────────────────────────────────────────

def test_sha_in_history_skips():
    assert verdict_for("verify that d5ae20b8a landed", FACTS) == (
        SKIP_STALE,
        "commit d5ae20b8a already in history",
    )


def test_unknown_sha_executes():
    assert verdict_for("verify that 9999999 landed", FACTS)[0] == EXECUTE


def test_sha_extraction():
    assert extract_commit_shas("check 3b060ff4c and d5ae20b8a") == ["3b060ff4c", "d5ae20b8a"]
    # Short hex words (4 chars) are not SHAs.
    assert extract_commit_shas("face and beef are not shas") == []


# ── Verification markers ─────────────────────────────────────────────────────

def test_verified_artifact_skips():
    assert verdict_for("re-check docs/gap10-level2-plan.md", FACTS)[0] == SKIP_STALE
    assert verdict_for("verify evidence/gap10-level1.md again", FACTS)[0] == SKIP_STALE


def test_unverified_artifact_executes():
    assert verdict_for("re-check docs/other.md", FACTS)[0] == EXECUTE


def test_verify_intent_detection():
    assert is_verify_directive("re-verify the write path")
    assert is_verify_directive("check that the grant landed")
    assert is_verify_directive("confirm the bridge is up")
    assert not is_verify_directive("commit the plan doc")


# ── Commit directives on files ───────────────────────────────────────────────

def test_commit_checked_clean_file_skips():
    # Explicitly checked and clean ⇒ its state already changed.
    assert verdict_for("commit docs/gap10-level2-plan.md", FACTS)[0] == SKIP_STALE


def test_commit_unrecorded_file_executes():
    # Not in clean_files ⇒ unknown ⇒ execute (conservative).
    assert verdict_for("commit stale_guidance_policy.py", FACTS)[0] == EXECUTE


def test_commit_intent_detection():
    assert is_commit_directive("commit the changes")
    assert is_commit_directive("push the write triple")
    assert not is_commit_directive("verify the bridge")


# ── Path extraction ──────────────────────────────────────────────────────────

def test_path_extraction():
    assert "docs/gap10-level2-plan.md" in extract_file_paths(
        "check docs/gap10-level2-plan.md is correct"
    )
    assert extract_file_paths("no file refs here") == []


# ── filter_stale ─────────────────────────────────────────────────────────────

def test_filter_stale_keeps_only_executable():
    directives = [
        "verify that d5ae20b8a landed",  # SHA in history ⇒ stale
        "commit stale_guidance_policy.py",  # unrecorded ⇒ execute
        "re-check docs/gap10-level2-plan.md",  # verified ⇒ stale
        "push the write triple",  # no evidence ⇒ execute
    ]
    assert filter_stale(directives, FACTS) == [
        "commit stale_guidance_policy.py",
        "push the write triple",
    ]


def test_filter_stale_empty_facts_keeps_everything():
    directives = ["verify d5ae20b8a", "commit docs/gap10-level2-plan.md"]
    assert filter_stale(directives, EMPTY) == directives
