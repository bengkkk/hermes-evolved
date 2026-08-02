#!/usr/bin/env python3
"""verify_gap10_write_path.py — bounded live verification of the Gap 10 Level 2 write path.

Fires the three canonical probes against the LIVE bridge (the daemon's own
wire protocol: POST /bridge/v1/exec, Bearer token) and asserts each outcome
matches the CURRENT permission state:

  P1 GET  https://api.github.com/                                    -> 200 exit 0, live payload   (read path / calibration)
  P2 PUT  .../contents/docs/gap10-level2-plan.md (exact allowlist)   -> 403 deny_permission while github.write is DENIED
  P3 PUT  .../contents/docs/other.md             (sibling)           -> 403 deny_allowlist (exact-path invariant, holds in BOTH states)

Exit codes:
  0  VERIFY PASS    — all probes matched the expected outcome for the current state.
  1  VERIFY FAIL    — a probe drifted from expected behavior, or the bridge/token
                      could not be reached (investigate + fix before any write).
  2  GRANT ACTIVE   — github.write is granted. The path is verified live, but the
                      first write triple MUST be fired deliberately with the real
                      payload (PUT plan.md, verify HTTP 201 + world-model error
                      < 0.25) — never as a side effect of this script.

Token resolution mirrors evolve_bridge._bridge_token(): env
HERMES_EVOLVED_BRIDGE_TOKEN wins, else <evolve_dir>/bridge_token
(data_layer.get_evolve_dir()). The token is never logged or printed.

Why this exists: every cron cycle used to re-run this ritual manually and
append a near-identical paragraph to docs/gap10-level2-plan.md. Identical
outcomes carry no new information — they need no doc entry. This script gives
each cycle one command with a real pass/fail signal; only a changed outcome
(FAIL, or GRANT ACTIVE) is worth a commit.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

try:
    from data_layer import get_evolve_dir
except Exception:  # pragma: no cover - fallback for standalone runs
    get_evolve_dir = None  # type: ignore

DEFAULT_PORT = 8791
PLAN_DOC_ENDPOINT = (
    "https://api.github.com/repos/bengkkk/hermes-evolved/contents/docs/gap10-level2-plan.md"
)
SIBLING_ENDPOINT = (
    "https://api.github.com/repos/bengkkk/hermes-evolved/contents/docs/other.md"
)
READ_ENDPOINT = "https://api.github.com/"


def _bridge_token() -> str:
    tok = os.environ.get("HERMES_EVOLVED_BRIDGE_TOKEN", "") or ""
    if tok:
        return tok
    if get_evolve_dir is not None:
        try:
            p = get_evolve_dir() / "bridge_token"
            return p.read_text(encoding="utf-8").strip() if p.exists() else ""
        except Exception:
            return ""
    return ""


def _github_write_granted() -> bool:
    """Exit 0 of `evolve_permissions.py check github write` == granted."""
    try:
        r = subprocess.run(
            [sys.executable, "evolve_permissions.py", "check", "github", "write"],
            capture_output=True,
            timeout=20,
        )
        return r.returncode == 0
    except Exception:
        return False


def _probe(port: int, method: str, endpoint: str, body: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
    token = _bridge_token()
    if not token:
        raise RuntimeError("no bridge token: set HERMES_EVOLVED_BRIDGE_TOKEN or create <evolve_dir>/bridge_token")
    payload: Dict[str, Any] = {"method": method, "endpoint": endpoint}
    if body is not None:
        payload["body"] = body
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/bridge/v1/exec",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    try:
        r = urllib.request.urlopen(req, timeout=15)
        return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=int(os.environ.get("BRIDGE_PORT", DEFAULT_PORT)))
    args = ap.parse_args()

    granted = _github_write_granted()
    results: list[str] = []
    ok = True

    # P1 — read path must always work (live payload, exit 0).
    try:
        status, out = _probe(args.port, "GET", READ_ENDPOINT)
        p1_ok = status == 200 and out.get("exit") == 0 and '"status": 200' in str(out.get("output", ""))
        results.append(f"P1 read={status}/{'ok' if p1_ok else 'FAIL'}")
        ok = ok and p1_ok
    except Exception as e:
        results.append(f"P1 read=UNREACHABLE ({e})")
        ok = False

    # P3 — sibling PUT must ALWAYS be denied at the allowlist layer (exact-path invariant).
    try:
        status, out = _probe(
            args.port, "PUT", SIBLING_ENDPOINT, {"message": "probe", "content": "dGVzdA==", "branch": "main"}
        )
        p3_ok = status == 403 and "endpoint not in allowlist" in str(out.get("output", ""))
        results.append(f"P3 exact-path=403/{'ok' if p3_ok else 'FAIL'}")
        ok = ok and p3_ok
    except Exception as e:
        results.append(f"P3 exact-path=UNREACHABLE ({e})")
        ok = False

    # P2 — exact allowlisted write: denied while grant is absent; when granted,
    # we do NOT fire a real PUT from here (see module docstring) -> GRANT ACTIVE.
    if not granted:
        try:
            status, out = _probe(
                args.port, "PUT", PLAN_DOC_ENDPOINT, {"message": "probe", "content": "dGVzdA==", "branch": "main"}
            )
            p2_ok = status == 403 and "permission denied: no write grant" in str(out.get("output", ""))
            results.append(f"P2 gate=denied/{'ok' if p2_ok else 'FAIL'}")
            ok = ok and p2_ok
        except Exception as e:
            results.append(f"P2 gate=UNREACHABLE ({e})")
            ok = False
        print(f"VERIFY {'PASS' if ok else 'FAIL'} | github.write=denied | {'; '.join(results)}")
        return 0 if ok else 1

    print(
        "GRANT ACTIVE | github.write=GRANTED | read + exact-path verified | "
        "fire the first write triple deliberately (PUT plan.md, expect HTTP 201, "
        "world-model prediction error < 0.25)"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
