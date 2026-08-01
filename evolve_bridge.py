#!/usr/bin/env python3
"""evolve_bridge.py — host bridge for the daemon's ``api_call`` actions.

Gap 10 step 3 (see docs/gap10-bridge-design.md). The daemon
(think_daemon.py) executes external actions by POSTing a structured
request to this bridge; the bridge is the ONLY component that ever
touches credentials (``~/.git-credentials``) and makes outbound HTTP
calls. The daemon never sees secrets.

Security model — deny by default, three independent layers, every one
must pass before any outbound call:

  1. **Bearer token auth** — ``HERMES_EVOLVED_BRIDGE_TOKEN`` (or the
     token file ``<evolve_dir>/bridge_token``, chmod 600, shared with
     the daemon by evolve_daemon.sh). Constant-time comparison.
  2. **Endpoint allowlist** — method + host + path prefix, mirrored
     from the daemon's ``_API_CALL_ALLOWLIST`` (a test asserts the two
     lists stay identical so drift cannot widen execution). Currently
     only ``GET https://api.github.com/*``.
  3. **Permission registry** — the request's resource (derived from the
     allowlist entry) must carry a truthy ``read`` flag in
     ``self_model.json`` permissions (``SelfModel.check_permission``
     semantics, defaults merged — deny-by-default ``github`` entry).

Every accepted execution AND every blocked attempt is appended to
``<evolve_dir>/bridge_audit.log`` (design verification criterion:
"timeline shows an audit entry for every external call").

Run::

    HERMES_EVOLVED_BRIDGE_TOKEN=... .venv/bin/python3 evolve_bridge.py --port 8791

Transport: localhost TCP only (binds 127.0.0.1). Stdlib + fastapi +
uvicorn + httpx — all already present in the project venv.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from data_layer import SelfModel, get_evolve_dir

logger = logging.getLogger("evolve_bridge")

# ── Endpoint allowlist (must stay identical to think_daemon._API_CALL_ALLOWLIST) ──
# Defense-in-depth: the bridge owns its own copy so a change to the daemon
# list alone can never widen what actually executes. tests/test_evolve_bridge.py
# asserts the two lists are equal.
BRIDGE_ALLOWLIST: list[Dict[str, Any]] = [
    {
        "method": "GET",
        "host": "api.github.com",
        "path_prefix": "/",
        "resource": "github",
    },
]

DEFAULT_PORT = 8791
OUTBOUND_TIMEOUT = 30.0
MAX_OUTPUT_BYTES = 4000  # response body captured into the audit / world model

# GitHub REST API host we forward to (the only host currently allowlisted).
_GITHUB_API_BASE = "https://api.github.com"


class ExecRequest(BaseModel):
    """Wire format sent by the daemon's ``_execute_api_call``."""

    endpoint: str = ""
    method: str = "GET"
    body: Dict[str, Any] = {}
    expected_outcome: str = ""


# ── Token handling ────────────────────────────────────────────────────────────

def _token_from_env() -> str:
    return os.environ.get("HERMES_EVOLVED_BRIDGE_TOKEN", "") or ""


def _token_from_file() -> str:
    """Read the shared token file (written by evolve_daemon.sh, chmod 600)."""
    try:
        p = get_evolve_dir() / "bridge_token"
        return p.read_text(encoding="utf-8").strip() if p.exists() else ""
    except Exception:
        return ""


def _bridge_token() -> str:
    """Resolve the expected token: env wins, token file is the fallback."""
    tok = _token_from_env()
    if tok:
        return tok
    return _token_from_file()


def _token_ok(provided: str) -> bool:
    expected = _bridge_token()
    if not expected:
        return False
    return hashlib.sha256(provided.encode("utf-8")).digest() == hashlib.sha256(
        expected.encode("utf-8")
    ).digest()


# ── Allowlist + permission checks (mirror the daemon's semantics) ─────────────

def _allowlist_entry(method: str, endpoint: str) -> Optional[Dict[str, Any]]:
    """Return the matching allowlist entry or None (deny)."""
    try:
        p = urllib.parse.urlparse(endpoint)
    except Exception:
        return None
    host = (p.hostname or "").lower()
    path = p.path or "/"
    method = (method or "GET").upper()
    for entry in BRIDGE_ALLOWLIST:
        if method != entry.get("method"):
            continue
        if host != entry.get("host"):
            continue
        if not path.startswith(entry.get("path_prefix", "/")):
            continue
        return entry
    return None


def _permission_granted(resource: str, action: str = "read") -> bool:
    """Deny-by-default permission check against the live self-model.

    Uses ``SelfModel.load()`` so the bridge sees exactly what the daemon
    sees (defaults merged, deny-by-default ``github`` entry present even
    when the file lacks a permissions section).
    """
    try:
        return SelfModel.load().check_permission(resource, action)
    except Exception as exc:  # a broken self-model must DENY, never allow
        logger.warning("permission check failed (%s) — denying", exc)
        return False


# ── Credential access (the only component that ever reads secrets) ───────────

def _read_git_credentials_token(host: str, path: Optional[Path] = None) -> str:
    """Extract the token for *host* from a ``~/.git-credentials`` file.

    Format per line: ``https://user:token@host/`` (token may be
    percent-encoded). Returns "" when absent. Never logs the token.
    """
    cred_file = path or Path.home() / ".git-credentials"
    try:
        lines = cred_file.read_text(encoding="utf-8").splitlines()
    except Exception:
        return ""
    wanted = host.lower()
    for line in lines:
        try:
            u = urllib.parse.urlparse(line.strip())
            if (u.hostname or "").lower() != wanted:
                continue
            if u.username is None or u.password is None:
                continue
            return urllib.parse.unquote(u.password)
        except Exception:
            continue
    return ""


# ── Execution ────────────────────────────────────────────────────────────────

def _github_get(endpoint: str) -> Dict[str, Any]:
    """Perform the allowlisted GitHub GET. Returns ``{exit, output}``.

    ``output`` is a compact JSON summary (status + truncated body) —
    never raw credentials, never unbounded payloads.
    """
    p = urllib.parse.urlparse(endpoint)
    path = p.path or "/"
    if p.query:
        path += "?" + p.query
    token = _read_git_credentials_token("github.com")
    if not token:
        return {"exit": 1, "output": "no github.com credential in ~/.git-credentials"}
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "hermes-evolved-bridge/1.0",
    }
    try:
        with httpx.Client(timeout=OUTBOUND_TIMEOUT) as client:
            resp = client.get(_GITHUB_API_BASE + path, headers=headers)
        body = resp.text[:MAX_OUTPUT_BYTES]
        return {
            "exit": 0 if resp.status_code < 400 else 1,
            "output": json.dumps(
                {"status": resp.status_code, "bytes": len(resp.text), "body": body},
                ensure_ascii=False,
            )[: MAX_OUTPUT_BYTES + 500],
        }
    except httpx.HTTPError as exc:
        return {"exit": 1, "output": f"github GET failed: {exc.__class__.__name__}: {exc}"[:400]}


def _audit(entry: Dict[str, Any]) -> None:
    """Append one JSON line per decision (executed AND blocked)."""
    try:
        audit_file = get_evolve_dir() / "bridge_audit.log"
        entry["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with open(audit_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:  # auditing must never take the bridge down
        logger.warning("audit write failed: %s", exc)


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="hermes-evolved-bridge", version="1.0")


@app.get("/bridge/v1/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "version": 1, "pid": os.getpid()}


@app.post("/bridge/v1/exec")
async def exec_action(req: ExecRequest, request: Request) -> JSONResponse:
    """Validate (auth → allowlist → permission) then execute. Never raises."""
    method = (req.method or "GET").upper()
    endpoint = req.endpoint or ""

    # Layer 1 — auth
    auth = request.headers.get("Authorization", "")
    provided = auth[7:] if auth.lower().startswith("bearer ") else ""
    if not _token_ok(provided):
        _audit({"event": "deny_auth", "method": method, "endpoint": endpoint[:120]})
        return JSONResponse(
            status_code=401,
            content={"exit": 1, "output": "unauthorized: bad or missing bearer token"},
        )

    # Layer 2 — endpoint allowlist
    entry = _allowlist_entry(method, endpoint)
    if entry is None:
        _audit({"event": "deny_allowlist", "method": method, "endpoint": endpoint[:120]})
        return JSONResponse(
            status_code=403,
            content={
                "exit": 1,
                "output": f"BLOCKED: endpoint not in allowlist: {method} {endpoint[:120]}",
            },
        )
    resource = entry.get("resource", "")

    # Layer 3 — permission registry
    if not _permission_granted(resource, "read"):
        _audit(
            {
                "event": "deny_permission",
                "method": method,
                "endpoint": endpoint[:120],
                "resource": resource,
            }
        )
        return JSONResponse(
            status_code=403,
            content={
                "exit": 1,
                "output": f"BLOCKED: permission denied: no read grant for resource {resource!r}",
            },
        )

    # Execute
    if method == "GET":
        result = _github_get(endpoint)
    else:
        result = {"exit": 1, "output": f"unsupported method: {method}"}

    _audit(
        {
            "event": "exec",
            "method": method,
            "endpoint": endpoint[:120],
            "resource": resource,
            "exit": result.get("exit"),
        }
    )
    return JSONResponse(status_code=200, content=result)


# ── CLI entry ────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    port = DEFAULT_PORT
    if "--port" in argv:
        try:
            port = int(argv[argv.index("--port") + 1])
        except (ValueError, IndexError):
            print("usage: evolve_bridge.py [--port PORT]", file=sys.stderr)
            return 2
    if not _bridge_token():
        print(
            "evolve_bridge.py: refusing to start without a token — set "
            "HERMES_EVOLVED_BRIDGE_TOKEN or create <evolve_dir>/bridge_token",
            file=sys.stderr,
        )
        return 2
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger.info(
        "bridge listening on 127.0.0.1:%d (allowlist: %d entries, audit: %s)",
        port,
        len(BRIDGE_ALLOWLIST),
        get_evolve_dir() / "bridge_audit.log",
    )
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
