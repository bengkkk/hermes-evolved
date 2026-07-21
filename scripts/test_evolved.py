"""Test that SOUL.md passes threat scan and system prompt has evolved identity."""
import os
import sys

sys.path.insert(0, "/opt/hermes")

# Test 1: SOUL.md must pass threat scan
os.environ["HERMES_HOME"] = "/root/.hermes"
from tools.threat_patterns import scan_for_threats

content = open("/root/.hermes/SOUL.md").read()
findings = scan_for_threats(content, scope="context")
assert not findings, f"SOUL.md blocked: {findings}"
print("1. SOUL.md: passes threat scan")

# Test 2: System prompt must have evolved identity
from agent.system_prompt import build_system_prompt
from types import SimpleNamespace

ma = SimpleNamespace()
ma._memory_store = None
ma._memory_enabled = False
ma._user_profile_enabled = False
ma._memory_manager = None
ma.load_soul_identity = True
ma.skip_context_files = True
ma.valid_tool_names = []
ma._task_completion_guidance = False
ma._parallel_tool_call_guidance = False
ma._thinking_protocol = True
ma._self_evolve = False
ma._tool_use_enforcement = False
ma.provider = "test"
ma.model = "test"
ma.platform = "cli"
ma.pass_session_id = False
ma.session_id = None

p = build_system_prompt(ma)
assert "thinking entity" in p, "Missing evolved identity"
assert "BLOCKED" not in p, "System prompt has blocked content"
assert "How You Think" in p or "three questions" in p, "Missing thinking protocol"
print("2. System prompt: evolved identity + thinking protocol confirmed")
print("3. No blocked content in system prompt")
print("All tests passed!")
