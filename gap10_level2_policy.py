"""gap10_level2_policy.py — reusable deny-by-default pre-flight policy (Gap 10 Level 2).

Standalone reference implementation of the method-aware pre-flight gate that
evolve_bridge.py and think_daemon.py enforce at runtime. This module is NOT
wired into execution — the daemon and bridge are the authoritative enforcers;
this file is the readable, importable policy contract.

Security invariants (must never drift from the enforcing code):

1. **Exact-path write allowlist, never regex/prefix.** A write entry names
   ONE file (method + host + full path). Matching by pattern or prefix would
   let a crafted path (``plan.md-evil``, ``other.md``) widen to siblings.
   Earlier draft of this module used ``contents/.+$`` and would have allowed
   PUTs the bridge correctly blocks — the equivalence test
   (tests/test_evolve_bridge.py::test_policy_matches_enforced_allowlists)
   pins the two representations together so that cannot come back.
2. **Deny-by-default.** Unknown method, unknown host, no allowlist entry, or
   a missing/false permission flag for the method's action ⇒ deny. The only
   way to pass is an explicit allowlist entry PLUS an explicit grant.
3. **Method → action mapping.** GET/HEAD ⇒ ``read``; PUT/POST/PATCH/DELETE
   ⇒ ``write``. Any method absent from METHOD_ACTION can never pass.

``permissions`` uses the self_model.json shape (resource → flag dict), the
same shape ``SelfModel.check_permission`` consumes at runtime.

The test suite asserts this module's allowlists equal the daemon's lists and
that its decisions match the bridge gate on the exact Level 2 endpoints, so a
change to this file (or to the enforcers) that widens execution fails CI.
"""

from __future__ import annotations

import urllib.parse
from typing import Any, Dict, Optional, Tuple

# ── Allowlists (must stay identical to think_daemon._API_CALL_ALLOWLIST /
#    _API_WRITE_ALLOWLIST and evolve_bridge.BRIDGE_ALLOWLIST /
#    BRIDGE_WRITE_ALLOWLIST — the drift-guard test asserts equality) ──────────

READ_ALLOWLIST: list[Dict[str, Any]] = [
    {
        "method": "GET",
        "host": "api.github.com",
        "path_prefix": "/",
        "resource": "github",
    },
]

WRITE_ALLOWLIST: list[Dict[str, Any]] = [
    {
        "method": "PUT",
        "host": "api.github.com",
        "path": "/repos/bengkkk/hermes-evolved/contents/docs/gap10-level2-plan.md",
        "resource": "github",
    },
    {
        "method": "PUT",
        "host": "api.github.com",
        "path": "/repos/bengkkk/hermes-evolved/contents/evidence/gap10-level1.md",
        "resource": "github",
    },
]

# HTTP method → permission action (single source of truth for the reference;
# mirrors evolve_bridge._METHOD_ACTION / think_daemon._API_METHOD_ACTION).
METHOD_ACTION: Dict[str, str] = {
    "GET": "read",
    "HEAD": "read",
    "PUT": "write",
    "POST": "write",
    "PATCH": "write",
    "DELETE": "write",
}


def _read_allowlist_entry(method: str, endpoint: str) -> Optional[Dict[str, Any]]:
    """Prefix match on path (read semantics) — mirrors the bridge/daemon."""
    try:
        p = urllib.parse.urlparse(endpoint)
    except Exception:
        return None
    host = (p.hostname or "").lower()
    path = p.path or "/"
    method = (method or "GET").upper()
    for entry in READ_ALLOWLIST:
        if method != entry.get("method"):
            continue
        if host != entry.get("host"):
            continue
        if not path.startswith(entry.get("path_prefix", "/")):
            continue
        return entry
    return None


def _write_allowlist_entry(method: str, endpoint: str) -> Optional[Dict[str, Any]]:
    """Exact match on method + host + FULL path (query excluded).

    This is the load-bearing invariant: a write entry names one file and must
    never widen to sibling paths.
    """
    try:
        p = urllib.parse.urlparse(endpoint)
    except Exception:
        return None
    host = (p.hostname or "").lower()
    path = p.path or "/"
    method = (method or "GET").upper()
    for entry in WRITE_ALLOWLIST:
        if method != entry.get("method"):
            continue
        if host != entry.get("host"):
            continue
        if path != entry.get("path"):
            continue
        return entry
    return None


def method_aware_preflight(
    endpoint: str, method: str, permissions: Dict[str, Dict[str, Any]]
) -> Tuple[bool, str]:
    """Deny-by-default pre-flight gate. Returns ``(ok, reason)``.

    ``permissions`` maps resource → flag dict (self_model.json shape), e.g.
    ``{"github": {"read": True, "write": False}}``. No entry for a resource,
    a missing flag, or a false flag ⇒ deny. Mirrors the bridge's layers 2–3
    (allowlist → method-action → permission flag); body validation is a
    separate layer owned by the bridge.
    """
    method = (method or "GET").upper()

    entry = _read_allowlist_entry(method, endpoint)
    write_entry = _write_allowlist_entry(method, endpoint)
    if entry is None and write_entry is None:
        return False, "endpoint not in allowlist: {} {}".format(method, endpoint[:120])
    resource = (entry or write_entry).get("resource", "")

    action = METHOD_ACTION.get(method)
    if action is None:
        return False, "method not allowed: {}".format(method)

    flags = (permissions or {}).get(resource) or {}
    if not flags.get(action):
        return False, "no {} grant for resource {!r}".format(action, resource)

    return True, "ok"


if __name__ == "__main__":
    # Bounded self-verification (also exercised via the test suite).
    read_only = {"github": {"read": True, "write": False}}
    write = {"github": {"read": True, "write": True}}
    empty: Dict[str, Dict[str, Any]] = {}

    plan_path = (
        "https://api.github.com/repos/bengkkk/hermes-evolved/"
        "contents/docs/gap10-level2-plan.md"
    )

    # Read semantics: any api.github.com GET with a read grant passes.
    assert method_aware_preflight("https://api.github.com/repos/x", "GET", read_only) == (True, "ok")
    assert method_aware_preflight("https://api.github.com/repos/x", "GET", empty)[0] is False
    assert method_aware_preflight("https://evil.example.com/", "GET", read_only)[0] is False

    # Write semantics: exact-path allowlist + explicit write grant.
    assert method_aware_preflight(plan_path, "PUT", write) == (True, "ok")
    assert method_aware_preflight(plan_path, "PUT", read_only)[0] is False
    assert method_aware_preflight(plan_path, "PUT", empty)[0] is False

    # The security-critical cases: a write grant never widens the allowlist.
    assert method_aware_preflight(
        "https://api.github.com/repos/bengkkk/hermes-evolved/contents/other.md",
        "PUT", write,
    )[0] is False
    assert method_aware_preflight(plan_path + "-evil", "PUT", write)[0] is False
    assert method_aware_preflight(
        "https://api.github.com/repos/other/repo/contents/docs/gap10-level2-plan.md",
        "PUT", write,
    )[0] is False
    assert method_aware_preflight(plan_path, "DELETE", write)[0] is False
    assert method_aware_preflight("https://api.github.com/repos/x", "POST", write)[0] is False

    print("gap10_level2_policy selftest OK")
