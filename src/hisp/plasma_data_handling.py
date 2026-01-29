"""
Plasma Data Handling - imported from PFC-Tritium-Transport

This module provides a bridge to import PlasmaDataHandling from PFC-Tritium-Transport.
All data management logic should be implemented in PFC-Tritium-Transport.
"""
import os
import sys
from pathlib import Path

# Add PFC-Tritium-Transport to path
# Priority: env var -> sibling of repo roots -> fallback
candidate_paths = []

# 1) Environment variable override
env_path = os.environ.get("PFC_TT_PATH") or os.environ.get("HISP_PFC_TT_PATH")
if env_path:
    candidate_paths.append(Path(env_path))

# 2) Common relative locations (when hisp and PFC-TT are sibling folders)
here = Path(__file__).resolve()
parents = here.parents
# Try siblings at different levels depending on layout
for idx in (3, 4, 5):
    if len(parents) > idx:
        candidate_paths.append(parents[idx] / "PFC-Tritium-Transport")

# 3) De-duplicate, keep order
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
    # Check if plasma_data_handling module exists
    if (p / "plasma_data_handling" / "__init__.py").exists():
        resolved_pfc_path = p
        if sp := str(p):
            if sp not in sys.path:
                sys.path.insert(0, sp)
        break

# Import PlasmaDataHandling from PFC-Tritium-Transport
try:
    from plasma_data_handling import PlasmaDataHandling
except ImportError as e:
    tried = ", ".join(str(p) for p in unique_candidates)
    hint = "Set env var PFC_TT_PATH to your PFC-Tritium-Transport folder."
    raise ImportError(
        "Could not import PlasmaDataHandling from PFC-Tritium-Transport. "
        f"Tried: {tried}. {hint} Error: {e}"
    )

# Re-export for backwards compatibility
__all__ = ['PlasmaDataHandling']
