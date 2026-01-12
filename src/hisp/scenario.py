"""
Scenario and Pulse are defined in PFC-Tritium-Transport to keep CSV-driven
scenario definitions in the PFC repository. This module acts as a shim and
re-exports `Scenario` and `Pulse` from PFC-Tritium-Transport/scenario.py.

The import strategy mirrors `hisp.bin`'s approach: try an env var `PFC_TT_PATH`,
then try common sibling locations and insert the repo root into `sys.path` so
we can `from scenario import Scenario, Pulse`.
"""
import os
import sys
from pathlib import Path

# Candidate paths where PFC-Tritium-Transport might live
candidate_paths = []

# 1) Environment variable override
env_path = os.environ.get("PFC_TT_PATH") or os.environ.get("HISP_PFC_TT_PATH")
if env_path:
    candidate_paths.append(Path(env_path))

# 2) Common relative locations (when hisp and PFC-TT are sibling folders)
here = Path(__file__).resolve()
parents = here.parents
for idx in (3, 4, 5):
    if len(parents) > idx:
        candidate_paths.append(parents[idx] / "PFC-Tritium-Transport")

# De-duplicate while preserving order
seen = set()
unique_candidates = []
for p in candidate_paths:
    try:
        sp = str(p)
    except Exception:
        continue
    if sp not in seen:
        seen.add(sp)
        unique_candidates.append(p)

resolved_pfc_path = None
for p in unique_candidates:
    if (p / "scenario.py").exists():
        resolved_pfc_path = p
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
        break

try:
    from scenario import Scenario, Pulse
except ImportError as e:
    tried = ", ".join(str(p) for p in unique_candidates)
    hint = "Set env var PFC_TT_PATH to your PFC-Tritium-Transport folder."
    raise ImportError(
        "Could not import Scenario/Pulse from PFC-Tritium-Transport. "
        f"Tried: {tried}. {hint} Error: {e}"
    )

__all__ = ["Scenario", "Pulse"]
