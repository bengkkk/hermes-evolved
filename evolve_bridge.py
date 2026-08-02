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

import base64
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

# ── Write allowlist (Gap 10 Level 2 — low-risk writes, deny-until-granted) ──
# Mirrors think_daemon._API_WRITE_ALLOWLIST (tests assert equality so drift
# cannot widen execution). Entries match the FULL path exactly (NOT prefix):
# a write entry names one file the daemon may PUT; prefix matching here
# would let a crafted path escape to sibling files. Execution additionally
# requires a truthy ``write`` flag on the resource in the permission
# registry — these entries exist NOW so the write path is armed and ready,
# but nothing can execute until the user issues the explicit Level 2 grant
# (evolve_permissions.py grant github write, see docs/gap10-level2-plan.md).
BRIDGE_WRITE_ALLOWLIST: list[Dict[str, Any]] = [
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

# HTTP method → permission action mapping (single source of truth for the
# method-aware permission check; any method absent here can never pass).
_METHOD_ACTION: Dict[str, str] = {
    "GET": "read",
    "HEAD": "read",
    "PUT": "write",
    "POST": "write",
    "PATCH": "write",
    "DELETE": "write",
}

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


def _method_action(method: str) -> Optional[str]:
    """Map an HTTP method to the permission action it requires.

    ``None`` for unknown methods — such a method can never pass pre-flight.
    """
    return _METHOD_ACTION.get((method or "GET").upper())


def _write_allowlist_entry(method: str, endpoint: str) -> Optional[Dict[str, Any]]:
    """Return the matching write-allowlist entry or None (deny).

    Exact match on method + host + full path (query excluded). Unlike the
    read allowlist's prefix match, write entries must be exact so an entry
    for one file can never widen to sibling paths (e.g. a PUT to
    ``.../plan.md-evil`` or ``.../other.md`` stays denied).
    """
    try:
        p = urllib.parse.urlparse(endpoint)
    except Exception:
        return None
    host = (p.hostname or "").lower()
    path = p.path or "/"
    method = (method or "GET").upper()
    for entry in BRIDGE_WRITE_ALLOWLIST:
        if method != entry.get("method"):
            continue
        if host != entry.get("host"):
            continue
        if path != entry.get("path"):
            continue
        return entry
    return None


def _validate_contents_body(body: Any) -> tuple:
    """Validate a GitHub Contents API PUT body. Returns ``(ok, error)``.

    Wire contract (mirrors think_daemon._validate_contents_body — the
    bridge is authoritative, so a mismatch there can only tighten, never
    widen): ``message`` (non-empty str) and ``content`` (non-empty base64
    str) are required; ``sha`` is optional and, when present, must be a
    str. Rejects non-dict bodies, missing/empty fields, and non-base64
    content so a malformed write never reaches the wire.
    """
    if not isinstance(body, dict):
        return False, f"body must be a JSON object, got {type(body).__name__}"
    msg = body.get("message")
    content = body.get("content")
    if not isinstance(msg, str) or not msg.strip():
        return False, "body.message must be a non-empty string"
    if not isinstance(content, str) or not content.strip():
        return False, "body.content must be a non-empty base64 string"
    try:
        base64.b64decode(content, validate=True)
    except Exception:
        return False, "body.content is not valid base64"
    if "sha" in body and body["sha"] is not None and not isinstance(body["sha"], str):
        return False, "body.sha must be a string when present"
    return True, ""


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

def _compact_get_summary(status: int, text: str) -> Dict[str, Any]:
    """Compact summary of a GitHub API response for the bridge's output slot.

    Keeps the truncated body (never unbounded payloads) and additionally
    surfaces the fields a read-then-write flow needs: for a Contents API
    object the current blob ``sha`` + ``size`` (so an update PUT can pass
    the sha back without re-fetching unbounded content), for an error
    object the ``message``, and for a list a ``count``. Non-JSON bodies
    degrade to the status/bytes/body shape — never raises.
    """
    summary: Dict[str, Any] = {
        "status": status,
        "bytes": len(text),
        "body": text[:MAX_OUTPUT_BYTES],
    }
    try:
        obj = json.loads(text)
    except Exception:
        return summary
    if isinstance(obj, dict):
        # Contents API PUT responses wrap the file object under ``content``
        # (create AND update); a GET returns it at the top level. Unwrap so
        # sha/size are surfaced for both shapes.
        inner = obj.get("content") if isinstance(obj.get("content"), dict) else None
        src = inner if inner is not None else obj
        if isinstance(src.get("sha"), str):
            summary["sha"] = src["sha"]
        if isinstance(src.get("size"), int):
            summary["size"] = src["size"]
        if isinstance(obj.get("message"), str) and "sha" not in summary:
            summary["error"] = obj["message"][:200]
    elif isinstance(obj, list):
        summary["count"] = len(obj)
    return summary


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
        return {
            "exit": 0 if resp.status_code < 400 else 1,
            "output": json.dumps(
                _compact_get_summary(resp.status_code, resp.text),
                ensure_ascii=False,
            )[: MAX_OUTPUT_BYTES + 500],
        }
    except httpx.HTTPError as exc:
        return {"exit": 1, "output": f"github GET failed: {exc.__class__.__name__}: {exc}"[:400]}


def _github_put(endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Perform an allowlisted GitHub Contents API PUT. Returns ``{exit, output}``.

    Sends ONLY the validated subset of *body* (message/content/sha) so an
    over-permissive daemon payload can never smuggle extra fields onto the
    wire. Same credential + truncation rules as ``_github_get``; the token
    is never logged or echoed back.
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
    payload: Dict[str, Any] = {
        "message": body.get("message", ""),
        "content": body.get("content", ""),
    }
    if body.get("sha") is not None:
        payload["sha"] = body["sha"]
    try:
        with httpx.Client(timeout=OUTBOUND_TIMEOUT) as client:
            resp = client.put(_GITHUB_API_BASE + path, headers=headers, json=payload)
        return {
            "exit": 0 if resp.status_code < 400 else 1,
            "output": json.dumps(
                _compact_get_summary(resp.status_code, resp.text),
                ensure_ascii=False,
            )[: MAX_OUTPUT_BYTES + 500],
        }
    except httpx.HTTPError as exc:
        return {"exit": 1, "output": f"github PUT failed: {exc.__class__.__name__}: {exc}"[:400]}


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

    # Layer 2 — endpoint allowlist. Read entries match method + host + path
    # PREFIX; write entries match method + host + FULL path exactly (a write
    # entry names one file and must never widen to siblings). A request is
    # only allowlisted when it matches one of the two lists.
    entry = _allowlist_entry(method, endpoint)
    write_entry = _write_allowlist_entry(method, endpoint)
    if entry is None and write_entry is None:
        _audit({"event": "deny_allowlist", "method": method, "endpoint": endpoint[:120]})
        return JSONResponse(
            status_code=403,
            content={
                "exit": 1,
                "output": f"BLOCKED: endpoint not in allowlist: {method} {endpoint[:120]}",
            },
        )
    resource = (entry or write_entry).get("resource", "")

    # Layer 3 — permission registry, method-aware (GET→read, PUT/POST/etc →
    # write). The Level 2 gate lives HERE: write allowlist entries exist, but
    # without a truthy ``write`` grant the request is blocked deny-by-default.
    action = _method_action(method)
    if action is None or not _permission_granted(resource, action):
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
                "output": (
                    f"BLOCKED: permission denied: no {action or 'unknown'} "
                    f"grant for resource {resource!r}"
                ),
            },
        )

    # Layer 4 — write-body validation (write methods only). A malformed
    # body must be rejected before anything reaches the wire.
    if action == "write":
        _ok, _err = _validate_contents_body(req.body)
        if not _ok:
            _audit(
                {
                    "event": "deny_body",
                    "method": method,
                    "endpoint": endpoint[:120],
                    "resource": resource,
                }
            )
            return JSONResponse(
                status_code=400,
                content={"exit": 1, "output": f"BLOCKED: invalid body: {_err}"},
            )

    # Execute
    if method == "GET":
        result = _github_get(endpoint)
    elif method == "PUT":
        result = _github_put(endpoint, req.body)
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
