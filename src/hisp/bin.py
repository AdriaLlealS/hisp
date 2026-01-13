"""
CSV-driven bin classes - imported from PFC-Tritium-Transport
This module intentionally avoids defining any legacy HISP bin classes.
"""
# =============================================================================
# CSV-driven bin classes - imported from PFC-Tritium-Transport
# =============================================================================
import os
import sys
from pathlib import Path

# Add PFC-Tritium-Transport to path to import CSV bin classes
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
    if (p / "csv_bin.py").exists():
        resolved_pfc_path = p
        if sp := str(p):
            if sp not in sys.path:
                sys.path.insert(0, sp)
        break

# Import CSV bin classes from PFC-Tritium-Transport
# This avoids duplication — all changes should be made in PFC-Tritium-Transport/csv_bin.py
try:
    from bins_from_csv.csv_bin import BinConfiguration, Bin, BinCollection, Reactor
except ImportError as e:
    tried = ", ".join(str(p) for p in unique_candidates)
    hint = "Set env var PFC_TT_PATH to your PFC-Tritium-Transport folder."
    raise ImportError(
        "Could not import CSV bin classes from PFC-Tritium-Transport. "
        f"Tried: {tried}. {hint} Error: {e}"
    )

# Also import the Material class from the PFC-Tritium-Transport package so
# HISP code can reference materials via `hisp.bin.Material` in the same way
# it references the CSV-driven Bin classes above.
try:
    from materials.materials import Material
except ImportError as e:
    tried = ", ".join(str(p) for p in unique_candidates)
    hint = "Set env var PFC_TT_PATH to your PFC-Tritium-Transport folder."
    raise ImportError(
        "Could not import Material from PFC-Tritium-Transport. "
        f"Tried: {tried}. {hint} Error: {e}"
    )

# For backwards compatibility, re-export the imported classes
# Re-export the new names
__all__ = ['BinConfiguration', 'Bin', 'BinCollection', 'Reactor', 'Material']

# These classes are imported from PFC-Tritium-Transport/csv_bin.py
# =============================================================================

# NOTE: Monkeypatch removed - new code expects bin.material to be a Material object
# Legacy code that expects bin.material to be a string should use bin.material_name instead
