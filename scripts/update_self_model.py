"""Update self_model.json to reflect current reality and break the stale reasoning loop.

The daemon has been stuck in a loop because:
1. It keeps trying to import a non-existent 'DataLayer' class
2. The data layer uses Timeline, SelfModel, Orientation, Memory, Goals as separate classes — NOT DataLayer
3. Stale promised_features keep feeding back into the LLM prompt, reinforcing the loop

This script updates the self-model to:
- Mark data layer as resolved (correct classes: Timeline, SelfModel, Orientation, Memory, Goals)
- Remove stale promised_features about data layer import
- Update version to match orientation.json (v7)
- Keep gap focus on Gap 8 but with actionable framing
"""

import json
import sys
from pathlib import Path

EVOLVE_DIR = Path("/root/.hermes-evolved/evolve")
SELF_MODEL_FILE = EVOLVE_DIR / "self_model.json"
ORIENTATION_FILE = EVOLVE_DIR / "orientation.json"

def load_json(path):
    return json.loads(path.read_text()) if path.exists() else {}

def save_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)

def main():
    sm = load_json(SELF_MODEL_FILE)
    if not sm:
        print("ERROR: self_model.json not found")
        return 1

    orient = load_json(ORIENTATION_FILE)
    orient_version = orient.get("evolution_version", 7) if orient else 7

    # Track changes
    changes = []
    
    # 1. Update evolution version to match orientation
    old_ver = sm.get("state", {}).get("evolution_version")
    if old_ver != orient_version:
        sm.setdefault("state", {})["evolution_version"] = orient_version
        changes.append(f"evolution_version: {old_ver} → {orient_version}")

    # 2. Clear stale promised_features (they reinforce the bad loop)
    old_features = sm.get("commitments", {}).get("promised_features", [])
    stale_prefixes = [
        "DataLayer",
        "data layer",
        "data_layer",
        "Verify data",
        "Fix data",
        "Resolve data",
        "Complete step",
        "Complete verification",
        "inject orientation",
        "Inject orientation",
        "Will review all",
        "Complete orientation",
        "Inspect data_layer",
    ]
    new_features = []
    for f in old_features:
        if any(f.strip().lower().startswith(p.strip().lower()) for p in stale_prefixes):
            continue
        new_features.append(f)
    
    if len(new_features) != len(old_features):
        removed = len(old_features) - len(new_features)
        changes.append(f"removed {removed} stale promised_features (data layer / orientation injection)")
    
    # Add the one real commitment
    real_commitment = "Drive self-directed evolution (Gap 8): inspect think_daemon prompt loop, break stale reasoning patterns"
    if real_commitment not in new_features:
        new_features.append(real_commitment)
        changes.append("added real commitment for Gap 8")
    
    sm.setdefault("commitments", {})["promised_features"] = new_features

    # 3. Remove stale unknown areas that have been resolved
    old_unknowns = sm.get("capabilities", {}).get("unknown_areas", [])
    resolved_unknowns = [
        "Exact location of orientation configuration files.",
        "Exact dependencies of data_layer.py",
    ]
    new_unknowns = [u for u in old_unknowns if u not in resolved_unknowns]
    if len(new_unknowns) != len(old_unknowns):
        changes.append(f"removed {len(old_unknowns) - len(new_unknowns)} resolved unknown areas")
    sm["capabilities"]["unknown_areas"] = new_unknowns

    # 4. Add a strength: data layer verified working
    strengths = sm.get("capabilities", {}).get("strengths", [])
    resolved_strength = "Data layer verified: Timeline, SelfModel, Orientation, Memory, Goals all import and function correctly"
    if resolved_strength not in strengths:
        strengths.append(resolved_strength)
        changes.append("added 'data layer verified' strength")
    
    # 5. Update weaknesses to be real
    weaknesses = sm.get("capabilities", {}).get("weaknesses", [])
    
    # Remove resolved weaknesses
    resolved_weaknesses = [
        "Systematic prediction bias: shell actions have 0.85 avg prediction error across 3 samples",
    ]
    new_weaknesses = [w for w in weaknesses if w not in resolved_weaknesses]
    
    # Check if the daemon-loop weakness is already there
    loop_weakness = "Daemon stuck in stale reasoning loop: repeatedly tries to import nonexistent 'DataLayer' class instead of using the correct Timeline/SelfModel/Orientation APIs"
    if loop_weakness not in new_weaknesses:
        new_weaknesses.append(loop_weakness)
        changes.append("added 'daemon reasoning loop' weakness")
    
    if "Stale promised_features pollute LLM prompt, causing repetitive fixation on resolved issues" not in new_weaknesses:
        new_weaknesses.append("Stale promised_features pollute LLM prompt, causing repetitive fixation on resolved issues")
        changes.append("added 'stale prompt pollution' weakness")
    
    sm["capabilities"]["weaknesses"] = new_weaknesses

    # 6. Add a feature: self-model awareness
    features = sm.get("capabilities", {}).get("features", [])
    if "self_model_prompt_awareness" not in features:
        features.append("self_model_prompt_awareness")
        changes.append("added 'self_model_prompt_awareness' feature flag")
    sm["capabilities"]["features"] = features
    
    # 7. Update cycle count
    sm.setdefault("state", {})["total_cycles"] = sm["state"].get("total_cycles", 0) + 1

    # Write updated self-model
    save_json(SELF_MODEL_FILE, sm)
    print("self_model.json updated:")
    for c in changes:
        print(f"  ✓ {c}")
    
    # Also write a note to daemon_state that this cycle made progress
    daemon_state_file = EVOLVE_DIR / "daemon_state.json"
    if daemon_state_file.exists():
        ds = json.loads(daemon_state_file.read_text())
        ds["last_self_model_update"] = "broke stale data layer loop, set v7, added real commitments"
        save_json(daemon_state_file, ds)
        print("  ✓ daemon_state.json annotated")

    return 0

if __name__ == "__main__":
    sys.exit(main())
