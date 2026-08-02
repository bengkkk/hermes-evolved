"""stale_guidance_policy.py — stale-guidance detection policy (Gap 8, slice 1).

Goal: "Enforce stale-guidance detection before directive-driven actions —
before executing a directive that references a file, commit, or verification
(e.g. 'verify X', 're-check Y', 'commit Z'), check whether the referenced
state has already changed (committed/verified) and skip the action if stale."

This module is the standalone, importable policy contract (same shape as
gap10_level2_policy.py). It is NOT yet wired into think_daemon.py's loop —
that wiring is a later bounded slice. The interface is designed so the
daemon can call ``filter_stale(directives, facts)`` once per cycle with the
directives it is considering, and only execute the ones that come back.

Design invariants (must never drift from the enforcing code, and are pinned
by tests/test_stale_guidance_policy.py):

1. **Conservative skip.** ``SKIP_STALE`` is returned ONLY on positive
   evidence that the referenced state already changed (a commit SHA that
   that exists in history, an artifact that carries a verification marker, or a
   file explicitly checked and found clean). Missing/unknown evidence ⇒
   ``EXECUTE``.
   The cost of a wrong skip (dropping a needed action) is worse than the
   cost of a redundant execute, so the default is always execute.
2. **Pure.** No I/O, no git calls, no network. Callers inject repo facts via
   ``RepoFacts`` (the daemon will populate it from ``git log`` / file
   state / marker files). This keeps the policy unit-testable and lets the
   daemon decide *when* to gather evidence.
3. **Three reference types.** Commit SHAs (7-40 hex), file paths, and
   verification verbs. A directive only skips when its own referenced state
   is provably already present.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

# ── Reference extraction ──────────────────────────────────────────────────────

# 7-40 lowercase hex chars bounded by word boundaries — a git SHA short form
# (7 chars) up to the full 40-char hash.
_SHA_RE = re.compile(r"\b[0-9a-f]{7,40}\b")

# Path-like tokens ending in a common source/doc extension. Bounded on
# purpose: we do not claim to parse free text, only to spot file references.
_PATH_RE = re.compile(r"(?<![\w])(?:[\w./-]+\.(?:md|py|json|txt|yaml|yml|toml|sh))(?![\w])")

# Verification intent verbs — the "re-check / verify / confirm" family that
# signals the directive is about confirming already-done state.
_VERIFY_VERB_RE = re.compile(
    r"\b(?:re-?verify|re-?check|verify|confirm|revalidate|check)\b", re.IGNORECASE
)

# Commit/push intent verbs — the "commit Z / push Z" family.
_COMMIT_VERB_RE = re.compile(r"\b(?:commit|push|land)\b", re.IGNORECASE)

# Verdict constants (strings so callers can log them directly).
EXECUTE = "EXECUTE"
SKIP_STALE = "SKIP_STALE"


@dataclass(frozen=True)
class RepoFacts:
    """Evidence about the repo's current state, gathered by the caller.

    All fields are *positive evidence*: anything absent counts as unknown,
    and unknown ⇒ EXECUTE (conservative). In particular, "clean" is a
    positive fact — a file the caller actually checked and found free of
    uncommitted changes — NOT the absence of a dirty-file record. Absence of
    evidence must never produce a skip.
    """

    commits: frozenset = frozenset()  # commit SHAs already in git history
    verified_markers: frozenset = frozenset()  # artifact keys already verified
    clean_files: frozenset = frozenset()  # files checked and free of pending changes


def extract_commit_shas(text: str) -> List[str]:
    """Return all 7-40 hex tokens in ``text`` (candidate commit SHAs)."""
    if not text:
        return []
    return _SHA_RE.findall(text)


def extract_file_paths(text: str) -> List[str]:
    """Return path-like tokens in ``text`` (candidate file references)."""
    if not text:
        return []
    return _PATH_RE.findall(text)


def is_verify_directive(text: str) -> bool:
    """True when the directive uses a verification/re-check verb."""
    return bool(text and _VERIFY_VERB_RE.search(text))


def is_commit_directive(text: str) -> bool:
    """True when the directive uses a commit/push/land verb."""
    return bool(text and _COMMIT_VERB_RE.search(text))


def verdict_for(directive: str, facts: Optional[RepoFacts] = None) -> Tuple[str, str]:
    """Decide whether ``directive`` is stale. Returns ``(verdict, reason)``.

    Verdict is one of ``EXECUTE`` / ``SKIP_STALE``. Skip requires positive
    evidence; every uncertain case executes. ``None``/empty directives
    execute (there is nothing provably stale about them).
    """
    if not directive or not directive.strip():
        return EXECUTE, "empty directive; nothing provably stale"

    facts = facts or RepoFacts()
    text = directive.strip()
    shas = extract_commit_shas(text)
    paths = extract_file_paths(text)
    verify_intent = is_verify_directive(text)
    commit_intent = is_commit_directive(text)

    # 1. Commit-SHA references: the referenced state is already in history.
    for sha in shas:
        if sha in facts.commits:
            return SKIP_STALE, "commit {} already in history".format(sha)

    # 2. Verification directives: skip only when the named artifact carries
    #    a verification marker (positive proof it was already verified).
    if verify_intent:
        for path in paths:
            if path in facts.verified_markers:
                return SKIP_STALE, "artifact {} already verified".format(path)

    # 3. Commit directives naming a file: a file explicitly checked and
    #    found clean is a no-op to commit — its state already changed (i.e.
    #    is already committed/up to date). Only POSITIVE clean evidence
    #    skips; an unrecorded file stays unknown ⇒ EXECUTE.
    if commit_intent:
        for path in paths:
            if path in facts.clean_files:
                return SKIP_STALE, "no pending changes for {}".format(path)

    return EXECUTE, "no positive stale evidence"


def filter_stale(
    directives: Iterable[str], facts: Optional[RepoFacts] = None
) -> List[str]:
    """Return only the directives that should execute (non-stale)."""
    return [d for d in directives if verdict_for(d, facts)[0] == EXECUTE]


if __name__ == "__main__":
    # Bounded self-verification (also exercised via the test suite).
    facts = RepoFacts(
        commits=frozenset({"d5ae20b8a"}),
        verified_markers=frozenset({"docs/gap10-level2-plan.md"}),
        clean_files=frozenset({"docs/gap10-level2-plan.md"}),
    )
    empty = RepoFacts()

    # Conservative default: no evidence ⇒ execute.
    assert verdict_for("re-verify the bridge", empty)[0] == EXECUTE
    assert verdict_for("commit the plan doc", empty)[0] == EXECUTE
    assert verdict_for("", empty)[0] == EXECUTE
    assert verdict_for(None, empty)[0] == EXECUTE

    # Positive commit-SHA evidence ⇒ skip.
    assert verdict_for("verify that d5ae20b8a landed", facts)[0] == SKIP_STALE
    # SHA not in history ⇒ execute.
    assert verdict_for("verify that 9999999 landed", facts)[0] == EXECUTE

    # Positive verification marker ⇒ skip; unverified artifact ⇒ execute.
    assert verdict_for("re-check docs/gap10-level2-plan.md", facts)[0] == SKIP_STALE
    assert verdict_for("re-check docs/other.md", facts)[0] == EXECUTE

    # Commit directive on a checked-clean file ⇒ skip; on an unrecorded
    # (possibly dirty) file ⇒ execute.
    assert verdict_for("commit docs/gap10-level2-plan.md", facts)[0] == SKIP_STALE
    assert verdict_for("commit stale_guidance_policy.py", facts)[0] == EXECUTE

    # filter_stale keeps only executable directives.
    kept = filter_stale(
        [
            "verify that d5ae20b8a landed",
            "commit stale_guidance_policy.py",
            "re-check docs/gap10-level2-plan.md",
        ],
        facts,
    )
    assert kept == ["commit stale_guidance_policy.py"], kept

    print("stale_guidance_policy selftest OK")
