# HISP

Hydrogen Inventory Simulations for PFCs (HISP) is a series of code that uses FESTIM to simulate deuterium and tritium inventories in a fusion tokamak first wall and divertor PFCs. 

## This Version / Origin

This repository is a modified version of the original `hisp` project initially developed by Kaelyn Dunnell at MIT. This particular fork was developed by Adrià Lleal during an internship at the ITER Organization and has been adapted to work with a more open and general `PFC-Tritium-Transport` workflow. It is tailored for estimations of tritium/hydrogen retention on fusion reactor plasma-facing components.

Summary of what HISP does

HISP receives bin definitions, material properties, time-dependent particle fluxes and heat loads, and a scenario specification (pulses, durations, repetition) — typically provided by the `PFC-Tritium-Transport` via a CSV input table. For each bin it constructs a FESTIM simulation: it translates the bin geometry (start/end coordinates, thickness, optional Cu layer and surface area) into the model domain, assigns material parameters from a CSV materials input table, builds time-dependent boundary conditions and source/flux expressions, and selects appropriate boundary-condition types (Robin/Neumann) before assembling and solving the transport equations with FESTIM. The per-bin outputs (surface concentrations, retained inventory, implanted fractions, and time traces) are exported so they can be analysed individually or aggregated across bins for inventory estimates.

