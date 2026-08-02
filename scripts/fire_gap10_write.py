#!/usr/bin/env python3
"""fire_gap10_write.py — fire one allowlisted Gap 10 Level 2 write triple.

Usage:
  python3 scripts/fire_gap10_write.py plan        # PUT docs/gap10-level2-plan.md
  python3 scripts/fire_gap10_write.py evidence1   # PUT evidence/gap10-level1.md
  python3 scripts/fire_gap10_write.py --list      # show allowlisted write targets

Flow (the "write triple", mirrors think_daemon._execute_api_call's path):
  1. Pre-flight — gap10_level2_policy.method_aware_preflight against the LIVE
     permission registry (deny-by-default: unknown endpoint, no grant, or a
     drifted policy ⇒ abort before any outbound call).
  2. Read probe — GET the contents endpoint through the bridge. 404 ⇒ create
     (no sha); 200 ⇒ pass the returned blob sha for an update. The bridge
     surfaces sha/size in its compact output (evolve_bridge._compact_get_summary),
     so this works even though the content field is truncated.
  3. PUT — base64 content + commit message through the bridge (the bridge is
     the ONLY component that touches the GitHub credential; this script never
     reads ~/.git-credentials or any GitHub token).
  4. Verify — the PUT response carries the new sha and size; compare size
     against the local file and print the outcome.

Exit codes: 0 = write executed (or verified already-current), 1 = blocked or
failed, 2 = usage error.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import request as _request
from urllib.error import URLError as _URLError

# Import from the repo root (data_layer + gap10_level2_policy are top-level).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from data_layer import SelfModel, get_evolve_dir  # noqa: E402
from gap10_level2_policy import WRITE_ALLOWLIST, method_aware_preflight  # noqa: E402

BRIDGE_URL = os.environ.get("HERMES_EVOLVED_BRIDGE_URL", "http://127.0.0.1:8791")
BRIDGE_TIMEOUT = 30.0

# Logical target name → (local file relative to repo root, commit message).
# The endpoints themselves come from WRITE_ALLOWLIST (single source of truth),
# keyed by matching the allowlisted path suffix.
_TARGETS: Dict[str, Dict[str, str]] = {
    "plan": {
        "local": "docs/gap10-level2-plan.md",
        "message": "docs(gap10): publish Level 2 plan — first allowlisted write under the github.write grant",
    },
    "evidence1": {
        "local": "evidence/gap10-level1.md",
        "message": "docs(gap10): publish Level 1 evidence — allowlisted write under the github.write grant",
    },
}


def _bridge_token() -> str:
    tok = os.environ.get("HERMES_EVOLVED_BRIDGE_TOKEN", "") or ""
    if tok:
        return tok
    try:
        p = get_evolve_dir() / "bridge_token"
        return p.read_text(encoding="utf-8").strip() if p.exists() else ""
    except Exception:
        return ""


def _bridge_call(endpoint: str, method: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """POST one exec request to the bridge (same wire format as the daemon)."""
    payload = json.dumps(
        {"endpoint": endpoint, "method": method.upper(), "body": body or {}}
    ).encode("utf-8")
    req = _request.Request(
        BRIDGE_URL.rstrip("/") + "/bridge/v1/exec",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    token = _bridge_token()
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with _request.urlopen(req, timeout=BRIDGE_TIMEOUT) as resp:
            raw = resp.read(16384).decode("utf-8", "replace")
    except _URLError as e:
        return {"exit": 1, "output": f"bridge unavailable ({getattr(e, 'reason', e)})"}
    except Exception as e:  # noqa: BLE001 — outcome string, never raise
        return {"exit": 1, "output": f"bridge call failed: {e}"}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "exit" in parsed:
            return parsed
    except Exception:  # noqa: BLE001
        pass
    return {"exit": 1, "output": raw[:400]}


def _parse_output(output: str) -> Dict[str, Any]:
    """Bridge output is a JSON string; parse it defensively."""
    try:
        parsed = json.loads(output)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _endpoint_for(path: str) -> str:
    return "https://api.github.com" + path


def _git_blob_sha(data: bytes) -> str:
    """Compute the git blob sha1 of *data* (the Contents API 'sha' field).

    GitHub's contents-API sha is the git blob hash: sha1('blob <len>\\0' +
    content). Used to detect "already published" without a write.
    """
    import hashlib

    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # noqa: S324


def _target_endpoint(name: str) -> Optional[str]:
    """Resolve the allowlisted endpoint for a target name (exact-path match)."""
    for entry in WRITE_ALLOWLIST:
        path = entry.get("path", "")
        if path.endswith(_TARGETS[name]["local"]):
            return _endpoint_for(path)
    return None


def fire(name: str) -> int:
    if name not in _TARGETS:
        print(f"ERROR: unknown target {name!r} (use --list)", file=sys.stderr)
        return 2
    local_path = _REPO_ROOT / _TARGETS[name]["local"]
    if not local_path.exists():
        print(f"ERROR: local file missing: {local_path}", file=sys.stderr)
        return 2

    endpoint = _target_endpoint(name)
    if endpoint is None:
        print(f"ERROR: no allowlisted endpoint for target {name!r}", file=sys.stderr)
        return 2

    # Layer 1 — pre-flight against the LIVE permission registry.
    try:
        sm = SelfModel.load()
        permissions = sm.permissions
    except Exception as exc:  # noqa: BLE001 — a broken registry must DENY
        print(f"BLOCKED: cannot load permission registry ({exc})", file=sys.stderr)
        return 1
    ok, reason = method_aware_preflight(endpoint, "PUT", permissions or {})
    if not ok:
        print(f"BLOCKED: {reason}", file=sys.stderr)
        return 1
    print(f"preflight OK: {endpoint}")

    # Layer 2 — read probe: does the file already exist remotely?
    probe = _bridge_call(endpoint, "GET")
    probe_summary = _parse_output(probe.get("output", ""))
    status = probe_summary.get("status")
    sha = probe_summary.get("sha") if isinstance(probe_summary.get("sha"), str) else None
    if status == 404:
        print(f"probe: 404 — file does not exist remotely; will CREATE")
    elif status is not None and status < 400:
        print(f"probe: {status} — file exists (sha={sha}); will UPDATE")
        # Idempotency: if the remote blob sha equals the LOCAL file's git blob
        # sha, the content is already published — nothing to write.
        local_blob_sha = _git_blob_sha(local_path.read_bytes())
        if sha and sha == local_blob_sha:
            print(f"ALREADY CURRENT: remote sha matches local blob sha ({sha}) — no PUT needed")
            return 0
        print(f"  (local blob sha {local_blob_sha} differs — content changed, updating)")
    else:
        print(f"probe: unexpected ({probe.get('exit')}: {probe.get('output', '')[:160]})")
        return 1

    # Layer 3 — PUT the file through the bridge.
    content = base64.b64encode(local_path.read_bytes()).decode("ascii")
    body: Dict[str, Any] = {
        "message": _TARGETS[name]["message"],
        "content": content,
    }
    if sha:
        body["sha"] = sha
    result = _bridge_call(endpoint, "PUT", body)
    summary = _parse_output(result.get("output", ""))
    print(f"PUT exit={result.get('exit')} status={summary.get('status')} "
          f"sha={summary.get('sha')} size={summary.get('size')}")

    if result.get("exit") != 0 or not summary.get("sha"):
        print(f"WRITE FAILED: {result.get('output', '')[:300]}", file=sys.stderr)
        return 1

    # Layer 4 — verify: size must match the local file; sha must be present.
    local_size = local_path.stat().st_size
    if summary.get("size") == local_size:
        print(f"VERIFIED: {name} -> {summary.get('sha')} ({local_size} bytes, "
              f"status={summary.get('status')})")
        return 0
    print(f"WARNING: size mismatch remote={summary.get('size')} local={local_size} "
          f"(sha={summary.get('sha')})", file=sys.stderr)
    return 1


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 2
    if argv[0] == "--list":
        for name, meta in _TARGETS.items():
            ep = _target_endpoint(name)
            print(f"{name:10s} {ep}  <-  {meta['local']}")
        return 0
    if argv[0] in ("plan", "evidence1"):
        return fire(argv[0])
    print(f"usage: {Path(sys.argv[0]).name} {{plan|evidence1|--list}}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
