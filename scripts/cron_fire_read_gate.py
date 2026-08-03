#!/usr/bin/env python3
"""Cron-cycle action: fire the allowlisted GET through the host bridge and
record the world-model triple (predict -> execute -> compare), exactly as
the daemon's `_execute_api_call` path does.

This is the committed per-cycle action while awaiting Level 3 grants:
one allowlisted GET https://api.github.com/ per cycle. It corrects the
drift of ticks 491-496, which auto-synced git_commit instead of firing
the committed read.
"""
import json
import os
import sys
import urllib.request
from urllib.error import URLError

sys.path.insert(0, "/workspace/hermes-evolved")
os.environ.setdefault("HERMES_HOME", os.path.expanduser("~/.hermes-evolved"))

from world_model import WorldModel  # noqa: E402
from hermes_constants import get_hermes_home  # noqa: E402

EVOLVE_DIR = os.path.join(os.path.expanduser("~/.hermes-evolved"), "evolve")
BRIDGE_URL = "http://127.0.0.1:8791"
TOKEN_PATH = os.path.join(EVOLVE_DIR, "bridge_token")

token = ""
if os.environ.get("HERMES_EVOLVED_BRIDGE_TOKEN"):
    token = os.environ["HERMES_EVOLVED_BRIDGE_TOKEN"]
elif os.path.exists(TOKEN_PATH):
    token = open(TOKEN_PATH).read().strip()

ENDPOINT = "https://api.github.com/"
EXPECTED = "HTTP 200: JSON root with current_user_url, rate_limit, and other GitHub API fields (~2.3 KB)"

# 1. Record prediction BEFORE execution
wm = WorldModel()
triple_id = wm.record_action(
    action_type="api_call",
    action_description="GET https://api.github.com/ through host bridge (committed per-cycle read gate)",
    expected_outcome=EXPECTED,
    expected_source="world_model",
    prediction_confidence=0.95,
    parameters={"endpoint": ENDPOINT, "method": "GET"},
)
print(f"triple_id={triple_id}")

# 2. Execute through the bridge (mirrors think_daemon._execute_api_call)
payload = json.dumps(
    {
        "endpoint": ENDPOINT,
        "method": "GET",
        "body": {},
        "expected_outcome": EXPECTED,
    }
).encode("utf-8")
req = urllib.request.Request(
    BRIDGE_URL + "/bridge/v1/exec",
    data=payload,
    method="POST",
    headers={"Content-Type": "application/json"},
)
if token:
    req.add_header("Authorization", "Bearer " + token)

try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read(8192).decode("utf-8", "replace")
        status = getattr(resp, "status", 200)
except URLError as e:
    actual = f"exit=1: bridge unavailable ({getattr(e, 'reason', e)}): {BRIDGE_URL}"
    status = None
except Exception as e:
    actual = f"exit=1: api_call failed: {e}"
    status = None

if status is not None:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "exit" in parsed:
            actual = f"exit={int(parsed['exit'])}: {str(parsed.get('output', ''))[:400]}"
        else:
            actual = f"exit={status}: {raw[:400]}"
    except Exception:
        actual = f"exit={status}: {raw[:400]}"

print(f"actual={actual[:300]}")

# 3. Compare: complete the triple with the actual outcome
error = wm.complete_action(triple_id, actual)
print(f"prediction_error={error}")
wm.save()
print("world_model.json saved")

# 4. Audit verification: confirm the bridge logged this execution
audit_path = os.path.join(EVOLVE_DIR, "bridge_audit.log")
if os.path.exists(audit_path):
    tail = open(audit_path).read()[-2000:]
    hit = "api.github.com" in tail and "GET" in tail
    print(f"bridge_audit tail mentions this GET: {hit}")
    print("--- audit tail ---")
    print(tail[-800:])
