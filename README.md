# HISP

Hydrogen Inventory Simulations for PFCs (HISP) is a series of code that uses FESTIM to simulate deuterium and tritium inventories in a fusion tokamak first wall and divertor PFCs.

## This Version / Origin

This repository is a modified version of the original `hisp` project initially developed by Kaelyn Dunnell at MIT. This particular fork was developed by Adrià Lleal during an internship at the ITER Organization and has been adapted to work with a more open and general `PFC-Tritium-Transport` workflow. It is tailored for estimations of tritium/hydrogen retention on fusion reactor plasma-facing components.

## What HISP Does

HISP receives bin definitions, material properties, time-dependent particle fluxes and heat loads, and a scenario specification (pulses, durations, repetition) — typically provided by `PFC-Tritium-Transport` via a CSV input table. For each bin it constructs a FESTIM simulation: it translates the bin geometry (start/end coordinates, thickness, optional Cu layer and surface area) into the model domain, assigns material parameters from a CSV materials input table, builds time-dependent boundary conditions and source/flux expressions, and selects appropriate boundary-condition types (Robin/Neumann) before assembling and solving the transport equations with FESTIM. The per-bin outputs (surface concentrations, retained inventory, implanted fractions, and time traces) are exported so they can be analysed individually or aggregated across bins for inventory estimates.

## Dependencies

HISP depends on:
- [FESTIM](https://github.com/festim-dev/festim) — finite element solver for hydrogen transport, installed automatically via the PFC-TT conda environment
- [PFC-Tritium-Transport](https://github.com/iterorganization/PFC-Tritium-Transport) — provides bin definitions, material classes, scenario handling, and plasma data. It cannot be run without a local clone of that repository.

All dependencies are installed as part of the PFC-Tritium-Transport setup instructions below.

## How to Install

Follow the full installation instructions in the [PFC-Tritium-Transport README](https://github.com/iterorganization/PFC-Tritium-Transport). In summary:

### 1. Clone PFC-Tritium-Transport
```bash
git clone --branch main https://github.com/iterorganization/PFC-Tritium-Transport.git
```

### 2. Create the conda environment
This step installs all core simulation dependencies including **FESTIM** (required by both PFC-Tritium-Transport and HISP), FEniCS-DOLFINx, PETSc, and all other required packages:
```bash
conda config --set channel_priority flexible
conda env create -f PFC-TT.yml
conda activate PFC-TT
```

### 3. Register the PFC-TT path
HISP imports bin definitions, material classes, and scenario handling directly from this repository at runtime. For this to work, HISP needs to know where PFC-Tritium-Transport is located on your system. Register the path once in your conda environment (replace with your actual clone location):
```bash
conda env config vars set PFC_TT_PATH="/path/to/your/PFC-Tritium-Transport"
conda deactivate && conda activate PFC-TT
```

You can verify it was set correctly with:
```bash
conda env config vars list
```

### 4. Install HISP
HISP is installed without dependencies since FESTIM and all other requirements are already provided by the PFC-TT conda environment:
```bash
pip install --no-deps git+https://github.com/AdriaLlealS/hisp.git@main
pip install h_transport_materials
```

## Running Tests

With the conda environment active and `PFC_TT_PATH` set:
```bash
cd /path/to/hisp
python -m pytest tests/ -v
```