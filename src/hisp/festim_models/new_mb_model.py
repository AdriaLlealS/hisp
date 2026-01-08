"""
Dynamic FESTIM model builder based on Bin configuration.

This module creates FESTIM models dynamically using the material properties,
trap parameters, and simulation settings stored in a Bin object.
"""

from typing import Callable, Tuple, Dict, Union, List, Optional
from numpy.typing import NDArray
import numpy as np
import festim as F

from hisp.scenario import Scenario
from hisp.plasma_data_handling import PlasmaDataHandling
from hisp.h_transport_class import CustomProblem
from hisp.settings import CustomSettings
from hisp.helpers import Stepsize
from hisp.helpers import gaussian_implantation_ufl

# Constants
kB_J = 1.380649e-23      # J/K
eV_to_J = 1.602176634e-19  # J/eV
implantation_range = 3e-9  # m (TODO: make this depend on incident energy)
width = 1e-9  # m (implantation distribution sigma)


def build_vertices_adaptive(L: float) -> np.ndarray:
    """
    Return an np.ndarray of vertices on [0, L] with adaptive meshing:
    - If L >= 5e-5: graded (h0=1e-10, r=1.1) until dL=1e-5, then constant dL=1e-5
    - If L < 5e-5: graded (h0=1e-10, r=1.015) until reaching L
    
    Args:
        L: Domain length (m)
        
    Returns:
        np.ndarray of vertex positions
    """
    if L <= 0.0:
        raise ValueError("L must be positive.")

    xs = [0.0]
    eps = 1e-18  # tolerance to avoid duplicates

    if L >= 5e-5:
        # --- GRADED REGION ---
        h = 1e-10
        r = 1.1
        dL_const = 1e-5

        while True:
            next_x = xs[-1] + h
            if next_x >= L - eps:
                if L - xs[-1] > eps:
                    xs.append(L)
                break
            if h < dL_const - eps:
                xs.append(next_x)
                h *= r
            else:
                break

        # --- CONSTANT REGION ---
        if xs[-1] < L - eps:
            start = xs[-1] + dL_const
            uniform = np.arange(start, L - eps, dL_const)
            xs.extend(uniform.tolist())
            if xs[-1] < L - eps:
                xs.append(L)
    else:
        # --- PURE GRADED REGION ---
        h = 1e-10
        r = 1.015
        while True:
            next_x = xs[-1] + h
            if next_x < L - eps:
                xs.append(next_x)
                h *= r
            else:
                if L - xs[-1] > eps:
                    xs.append(L)
                break

    # --- ENSURE NO DUPLICATES ---
    out = [xs[0]]
    for v in xs[1:]:
        if abs(v - out[-1]) > eps:
            out.append(v)

    return np.array(out)


def make_surface_concentration_time_function(
    T_fun: Callable,
    flux_fun: Callable,
    D0: float,
    E_eV: float,
    R_p: float,
    surface_x: float = 0.0
) -> Callable[[float], float]:
    """
    Create a surface concentration function for Dirichlet BC.
    
    Args:
        T_fun: Temperature function T(x, t) returning temperature in K
        flux_fun: Flux function returning particle flux in part/m^2/s
        D0: Diffusivity pre-exponential (m^2/s)
        E_eV: Diffusion activation energy (eV)
        R_p: Implantation range (m)
        surface_x: Surface position (m)
        
    Returns:
        Callable that returns surface concentration (part/m^3) at time t
    """
    x_surf = np.array([[float(surface_x)]])
    E_J = float(E_eV) * eV_to_J

    def c_S(t):
        t = float(t)
        T_surf = float(T_fun(x_surf, t)[0])
        phi = float(flux_fun(t))
        D_T = D0 * np.exp(-E_J / (kB_J * T_surf))
        val = (phi * float(R_p)) / D_T
        return float(val)
    
    return c_S


def create_species_and_traps(
    material,
    volume_subdomain: F.VolumeSubdomain1D
) -> Tuple[List[F.Species], List[F.Reaction]]:
    """
    Create FESTIM species, traps, and reactions based on material properties.
    
    Args:
        material: Material object with trap information
        volume_subdomain: FESTIM volume subdomain for reactions
        
    Returns:
        Tuple of (species_list, implicit_species_list, reactions_list)
    """
    # Mobile species
    mobile_D = F.Species("D")
    mobile_T = F.Species("T")
    species_list = [mobile_D, mobile_T]
    
    # Create trap species (one for each trap, each isotope)
    trap_list = []
    
    n_traps = material.N_traps
    mat_density = material.Mat_density  # atoms/m³
    print(f"\n=== DEBUG: Creating traps for {material.name} with N_traps={n_traps}, Mat_density={mat_density} ===")
    for i in range(1, n_traps + 1):
        # Get trap parameters
        trap_params = material.traps[i - 1]
        # Convert atomic fraction to absolute density (atoms/m³)
        trap_density = trap_params.Trap_density * mat_density
        
        # Debug output
        print(f"Trap {i}: Trap_density={trap_density} (from {trap_params.Trap_density} at.fr.), k_0={trap_params.k_0}, E_k={trap_params.E_k}, p_0={trap_params.p_0}, E_p={trap_params.E_p}")
        
        # Create trapped species for D and T in this trap
        trap_D = F.Species(f"trap{i}_D", mobile=False)
        trap_T = F.Species(f"trap{i}_T", mobile=False)
        species_list.extend([trap_D, trap_T])
        
        # Create implicit species (empty trap) - shared by both D and T
        empty_trap = F.ImplicitSpecies(
            n=trap_density,
            others=[trap_T, trap_D],
            name=f"empty_trap{i}",
        )
        
        trap_list.append({
            'index': i,
            'trap_D': trap_D,
            'trap_T': trap_T,
            'empty_trap': empty_trap,
            'params': trap_params,
        })
    print("=== DEBUG: Trap creation complete ===\n")
    
    # Create reactions
    reactions_list = []
    
    print(f"\n=== DEBUG: Creating reactions ===")
    for trap_info in trap_list:
        trap_params = trap_info['params']
        trap_idx = trap_info['index']
        
        # Use trap-specific parameters from CSV (all required)
        k_0 = trap_params.k_0
        E_k = trap_params.E_k
        p_0 = trap_params.p_0
        E_p = trap_params.E_p
        
        print(f"Trap {trap_idx} reactions: k_0={k_0}, E_k={E_k}, p_0={p_0}, E_p={E_p}")
        
        # Reaction for D in this trap
        reactions_list.append(
            F.Reaction(
                k_0=k_0,
                E_k=E_k,
                p_0=p_0,
                E_p=E_p,
                volume=volume_subdomain,
                reactant=[mobile_D, trap_info['empty_trap']],
                product=trap_info['trap_D'],
            )
        )
        
        # Reaction for T in this trap
        reactions_list.append(
            F.Reaction(
                k_0=k_0,
                E_k=E_k,
                p_0=p_0,
                E_p=E_p,
                volume=volume_subdomain,
                reactant=[mobile_T, trap_info['empty_trap']],
                product=trap_info['trap_T'],
            )
        )
    print("=== DEBUG: Reaction creation complete ===\n")
    
    return species_list, reactions_list


def make_dynamic_mb_model(
    bin,  # Bin object with material and configuration
    temperature: Callable,
    deuterium_ion_flux: Callable,
    tritium_ion_flux: Callable,
    deuterium_atom_flux: Callable,
    tritium_atom_flux: Callable,
    final_time: float,
    folder: str,
    exports: bool = False,
) -> Tuple[F.HydrogenTransportProblem, Dict[str, F.TotalVolume]]:
    """
    Create a FESTIM model dynamically based on bin properties.
    
    Args:
        bin: Bin object containing material, thickness, and configuration
        temperature: Temperature function T(x, t) in K
        deuterium_ion_flux: Deuterium ion flux function (part/m^2/s)
        tritium_ion_flux: Tritium ion flux function (part/m^2/s)
        deuterium_atom_flux: Deuterium atom flux function (part/m^2/s)
        tritium_atom_flux: Tritium atom flux function (part/m^2/s)
        final_time: Final simulation time (s)
        folder: Output folder for results
        exports: Whether to export detailed outputs
        
    Returns:
        Tuple of (festim_model, quantities_dict)
    """
    my_model = CustomProblem()
    
    # --- GEOMETRY AND MESH ---
    L = bin.thickness  # Domain length from bin
    vertices = build_vertices_adaptive(L)
    my_model.mesh = F.Mesh1D(vertices)
    
    # --- MATERIAL ---
    material = bin.material
    festim_material = F.Material(
        D_0=material.D0,
        E_D=material.E_D,
        name=material.name,
    )
    
    # --- SUBDOMAINS ---
    volume_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, L], material=festim_material)
    inlet = F.SurfaceSubdomain1D(id=1, x=0)
    outlet = F.SurfaceSubdomain1D(id=2, x=L)
    my_model.subdomains = [volume_subdomain, inlet, outlet]
    
    # --- SPECIES, TRAPS, AND REACTIONS ---
    species_list, reactions_list = create_species_and_traps(
        material, volume_subdomain
    )
    my_model.species = species_list
    my_model.reactions = reactions_list
    
    # --- TEMPERATURE ---
    my_model.temperature = temperature
    
    # --- BOUNDARY CONDITIONS ---
    # Total flux functions
    def Gamma_D_total(t):
        return deuterium_ion_flux(t) + deuterium_atom_flux(t)

    def Gamma_T_total(t):
        return tritium_ion_flux(t) + tritium_atom_flux(t)

    # Get BC type from bin configuration
    bc_plasma_facing = bin.bin_configuration.bc_plasma_facing_surface
    bc_rear = bin.bin_configuration.bc_rear_surface

    # Get species objects once for use in all BC branches
    mobile_D = next(s for s in my_model.species if s.name == "D")
    mobile_T = next(s for s in my_model.species if s.name == "T")

    boundary_conditions = []

    # --- Plasma-facing surface (inlet) BC choices ---
    # Options supported:
    #  - "Robin - Surf. Rec. + Implantation"
    #  - "Dirichlet - 0 concentration + Implantation"
    #  - "Dirichlet - Analyttical implantation approximation"
    if bc_plasma_facing == "Robin - Surf. Rec. + Implantation":
        # Use volumetric implantation sources (gaussian) + Dirichlet 0 at surface
        distribution = gaussian_implantation_ufl(implantation_range, width, thickness=L)

        my_model.sources = [
            F.ParticleSource(value=lambda x, t: deuterium_ion_flux(t) * distribution(x), volume=volume_subdomain, species=mobile_D),
            F.ParticleSource(value=lambda x, t: deuterium_atom_flux(t) * distribution(x), volume=volume_subdomain, species=mobile_D),
            F.ParticleSource(value=lambda x, t: tritium_ion_flux(t) * distribution(x), volume=volume_subdomain, species=mobile_T),
            F.ParticleSource(value=lambda x, t: tritium_atom_flux(t) * distribution(x), volume=volume_subdomain, species=mobile_T),
        ]

        # --- Surface recombination (Robin-like) ---
        # Read recombination parameters from material if available, otherwise use defaults
        k_r0 = getattr(material, "K_R", 7.94e-17)
        E_kr = getattr(material, "E_R", -2.0)
        k_d0 = getattr(material, "k_d0", 0.0)
        E_kd = getattr(material, "E_kd", 0.0)

        surface_reaction_dd = F.SurfaceReactionBC(
            reactant=[mobile_D, mobile_D],
            gas_pressure=0,
            k_r0=k_r0,
            E_kr=E_kr,
            k_d0=k_d0,
            E_kd=E_kd,
            subdomain=inlet,
        )

        surface_reaction_tt = F.SurfaceReactionBC(
            reactant=[mobile_T, mobile_T],
            gas_pressure=0,
            k_r0=k_r0,
            E_kr=E_kr,
            k_d0=k_d0,
            E_kd=E_kd,
            subdomain=inlet,
        )

        surface_reaction_dt = F.SurfaceReactionBC(
            reactant=[mobile_D, mobile_T],
            gas_pressure=0,
            k_r0=k_r0,
            E_kr=E_kr,
            k_d0=k_d0,
            E_kd=E_kd,
            subdomain=inlet,
        )

        # Add surface reactions to BCs (keep fixed concentration too to mirror legacy)
        boundary_conditions.extend([
            surface_reaction_dd, 
            surface_reaction_dt, 
            surface_reaction_tt
        ])

    elif bc_plasma_facing == "Dirichlet - 0 concentration + Implantation":
        # Volumetric implantation + zero Dirichlet at surface
        distribution = gaussian_implantation_ufl(implantation_range, width, thickness=L)
        my_model.sources = [
            F.ParticleSource(value=lambda x, t: deuterium_ion_flux(t) * distribution(x), volume=volume_subdomain, species=mobile_D),
            F.ParticleSource(value=lambda x, t: deuterium_atom_flux(t) * distribution(x), volume=volume_subdomain, species=mobile_D),
            F.ParticleSource(value=lambda x, t: tritium_ion_flux(t) * distribution(x), volume=volume_subdomain, species=mobile_T),
            F.ParticleSource(value=lambda x, t: tritium_atom_flux(t) * distribution(x), volume=volume_subdomain, species=mobile_T),
        ]
        boundary_conditions.extend([
            F.FixedConcentrationBC(subdomain=inlet, value=0.0, species="D"),
            F.FixedConcentrationBC(subdomain=inlet, value=0.0, species="T"),
        ])

    elif bc_plasma_facing == "Dirichlet - Analyttical implantation approximation":
        # Use analytical surface concentration approximation (Dirichlet)
        c_sD = make_surface_concentration_time_function(
            temperature, Gamma_D_total, material.D0, material.E_D, implantation_range, surface_x=0.0
        )
        c_sT = make_surface_concentration_time_function(
            temperature, Gamma_T_total, material.D0, material.E_D, implantation_range, surface_x=0.0
        )
        boundary_conditions.extend([
            F.FixedConcentrationBC(subdomain=inlet, value=c_sD, species="D"),
            F.FixedConcentrationBC(subdomain=inlet, value=c_sT, species="T"),
        ])

    else:
        raise ValueError(f"Unsupported plasma-facing BC: {bc_plasma_facing!r}")

    # --- Rear surface (outlet) BC choices ---
    if bc_rear == "Dirichlet - 0 concentration":
        boundary_conditions.extend([
            F.FixedConcentrationBC(subdomain=outlet, value=0.0, species="D"),
            F.FixedConcentrationBC(subdomain=outlet, value=0.0, species="T"),
        ])
    elif bc_rear == "Neumann - no flux":
        # Explicit Neumann / no-flux at outlet
        boundary_conditions.extend([
            F.ParticleFluxBC(subdomain=outlet, value=0.0, species="D"),
            F.ParticleFluxBC(subdomain=outlet, value=0.0, species="T"),
        ])
    else:
        raise ValueError(f"Unsupported rear BC: {bc_rear!r}")

    my_model.boundary_conditions = boundary_conditions

    # --- DEBUG: print a concise summary of the boundary conditions and sources ---
    def _summarize_bc(bc):
        try:
            if isinstance(bc, F.SurfaceReactionBC):
                reactant = getattr(bc, "reactant", None)
                names = [r.name if hasattr(r, "name") else str(r) for r in reactant] if reactant else []
                return f"SurfaceReactionBC reactants={names} k_r0={getattr(bc, 'k_r0', None)}"
            if isinstance(bc, F.FixedConcentrationBC):
                return f"FixedConcentrationBC species={getattr(bc, 'species', None)} value={getattr(bc, 'value', None)}"
            if isinstance(bc, F.ParticleFluxBC):
                return f"ParticleFluxBC species={getattr(bc, 'species', None)} value={getattr(bc, 'value', None)}"
        except Exception:
            pass
        # Fallback representation
        return repr(bc)

    try:
        print(f"=== DEBUG: Selected BCs -> plasma_facing={bc_plasma_facing!r}, rear={bc_rear!r} ===")
        for i, bc in enumerate(boundary_conditions):
            try:
                summary = _summarize_bc(bc)
            except Exception as e:
                summary = f"<error summarizing: {e}>"
            print(f"BC[{i}]: {summary}")

        # Print sources if present
        sources = getattr(my_model, "sources", None)
        if sources:
            print(f"=== DEBUG: Found {len(sources)} volumetric source(s) ===")
            for j, src in enumerate(sources):
                try:
                    sps = [s.name for s in getattr(src, 'species', [])]
                except Exception:
                    sps = getattr(src, 'species', None)
                print(f"Source[{j}]: species={sps} volume={getattr(src,'volume',None)}")
        else:
            print("=== DEBUG: No volumetric sources defined ===")
    except Exception as e:
        print(f"=== DEBUG: Failed to print BC summary: {e} ===")
    
    # --- EXPORTS ---
    if exports:
        my_model.exports = [
            F.XDMFExport(
                field="solute",
                folder=folder,
                checkpoint=False,
            ),
        ]
    else:
        my_model.exports = []
    
    # --- QUANTITIES TO TRACK ---
    quantities = {}
    
    # Add total volume for each species
    for species in my_model.species:
        quantity = F.TotalVolume(field=species, volume=volume_subdomain)
        my_model.exports.append(quantity)
        quantities[species.name] = quantity
        
        # Add surface flux for mobile species at inlet and outlet
        if species.mobile:
            inlet_flux = F.SurfaceFlux(field=species, surface=inlet)
            my_model.exports.append(inlet_flux)
            quantities[f"{species.name}_inlet_flux"] = inlet_flux
            
            outlet_flux = F.SurfaceFlux(field=species, surface=outlet)
            my_model.exports.append(outlet_flux)
            quantities[f"{species.name}_outlet_flux"] = outlet_flux
    
    # --- SETTINGS ---
    bin_config = bin.bin_configuration
    my_model.settings = CustomSettings(
        atol=bin_config.atol,
        rtol=bin_config.rtol,
        max_iterations=100,
        final_time=final_time,
    )
    
    my_model.settings.stepsize = Stepsize(initial_value=1e-3)
    
    return my_model, quantities


def make_model_with_scenario(
    bin,
    scenario: Scenario,
    plasma_data_handling: PlasmaDataHandling,
    coolant_temp: float,
    exports: bool = False,
) -> Tuple[F.HydrogenTransportProblem, Dict[str, F.TotalVolume]]:
    """
    Create a FESTIM model using scenario-based flux and temperature functions.
    
    Args:
        bin: Bin object
        scenario: Scenario with pulse sequence
        plasma_data_handling: PlasmaDataHandling for flux/heat data
        coolant_temp: Coolant temperature (K)
        exports: Whether to export detailed outputs
        
    Returns:
        Tuple of (festim_model, quantities_dict)
    """
    from hisp.festim_models.mb_model import (
        make_temperature_function,
        make_particle_flux_function,
        compute_flux_values,
        build_ufl_flux_expression,
    )
    
    # Create temperature function from scenario
    temperature_function = make_temperature_function(
        scenario=scenario,
        plasma_data_handling=plasma_data_handling,
        bin=bin,
        coolant_temp=coolant_temp,
    )
    
    # Check BC type to decide which flux function type to use
    bc_plasma_facing = bin.bin_configuration.bc_plasma_facing_surface
    
    # For implantation BCs, use UFL flux expressions (required for ParticleSource)
    if bc_plasma_facing in ("Robin - Surf. Rec. + Implantation", "Dirichlet - 0 concentration + Implantation"):
        # Use UFL flux expressions for ParticleSource compatibility
        occurrences = compute_flux_values(scenario, plasma_data_handling, bin)
        deuterium_ion_flux, deuterium_atom_flux, tritium_ion_flux, tritium_atom_flux = build_ufl_flux_expression(occurrences)
    else:
        # For analytical Dirichlet BC (no volumetric sources), plain callables are fine
        deuterium_ion_flux = make_particle_flux_function(
            scenario=scenario,
            plasma_data_handling=plasma_data_handling,
            bin=bin,
            ion=True,
            tritium=False,
        )
        
        tritium_ion_flux = make_particle_flux_function(
            scenario=scenario,
            plasma_data_handling=plasma_data_handling,
            bin=bin,
            ion=True,
            tritium=True,
        )
        
        deuterium_atom_flux = make_particle_flux_function(
            scenario=scenario,
            plasma_data_handling=plasma_data_handling,
            bin=bin,
            ion=False,
            tritium=False,
        )
        
        tritium_atom_flux = make_particle_flux_function(
            scenario=scenario,
            plasma_data_handling=plasma_data_handling,
            bin=bin,
            ion=False,
            tritium=True,
        )
    
    # Create model
    return make_dynamic_mb_model(
        bin=bin,
        temperature=temperature_function,
        deuterium_ion_flux=deuterium_ion_flux,
        tritium_ion_flux=tritium_ion_flux,
        deuterium_atom_flux=deuterium_atom_flux,
        tritium_atom_flux=tritium_atom_flux,
        final_time=scenario.get_maximum_time(),
        folder=f"results_bin_{bin.bin_number}",
        exports=exports,
    )
