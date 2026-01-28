"""
Dynamic FESTIM model builder based on Bin configuration.

This module creates FESTIM models dynamically using the material properties,
trap parameters, and simulation settings stored in a Bin object.
"""

from typing import Callable, Tuple, Dict, Union, List, Optional
from numpy.typing import NDArray
import numpy as np
import festim as F
import scipy.stats
from hisp.scenario import Scenario
from hisp.plasma_data_handling import PlasmaDataHandling
from hisp.h_transport_class import CustomProblem
from hisp.settings import CustomSettings
from builtins import ValueError, bool, callable, float, int, isinstance, str, type
from hisp.h_transport_class import CustomProblem
from hisp.helpers import (
    PulsedSource,
    gaussian_distribution,
    Stepsize,
    periodic_pulse_function,
    gaussian_implantation_ufl,
    XDMFExportEveryDt
)
import hisp.bin
from ufl import conditional, lt, ge, And
import h_transport_materials as htm
from scipy.optimize import bisect
import math

# Constants
kB_J = 1.380649e-23      # J/K
eV_to_J = 1.602176634e-19  # J/eV
implantation_range = 3e-9  # m (TODO: make this depend on incident energy)
width = 1e-9  # m (implantation distribution sigma)


def graded_vertices(L, h0, r):
        xs = [0.0]; h = h0
        while xs[-1] + h < L:
            xs.append(xs[-1] + h); h *= r
        if xs[-1] < L: xs.append(L)
        return np.array(xs)


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
    occurrences: list = None,  # Optional: Pre-computed flux occurrences with steady-state values
    exports: bool = False,
) -> Tuple[F.HydrogenTransportProblem, Dict[str, F.TotalVolume]]:
    """
    Create a FESTIM model dynamically based on bin properties.
    
    Args:
        bin: Bin object containing material, thickness, configuration, and implantation_params
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
    
    # --- GET IMPLANTATION PARAMETERS FROM BIN ---
    # bin.implantation_params should have structure:
    # {'ion': {'implantation_range': ..., 'width': ..., 'reflection_coefficient': ...},
    #  'atom': {...}}
    implantation_params = getattr(bin, 'implantation_params', {
        'ion': {'implantation_range': implantation_range, 'width': width, 'reflection_coefficient': 0.0},
        'atom': {'implantation_range': implantation_range, 'width': width, 'reflection_coefficient': 0.0}
    })
    
    # Get parameters (use defaults if not available)
    ion_range = implantation_params.get('ion', {}).get('implantation_range', implantation_range)
    ion_width = implantation_params.get('ion', {}).get('width', width)
    atom_range = implantation_params.get('atom', {}).get('implantation_range', implantation_range)
    atom_width = implantation_params.get('atom', {}).get('width', width)
    
    # Reflection coefficients (will be applied to flux functions)
    ion_reflection = implantation_params.get('ion', {}).get('reflection_coefficient', 0.0)
    atom_reflection = implantation_params.get('atom', {}).get('reflection_coefficient', 0.0)
    
    # --- APPLY REFLECTION COEFFICIENTS TO FLUX FUNCTIONS ---
    # Wrap flux functions to reduce them by reflection coefficient
    def apply_reflection(flux_func, reflection_coeff):
        """Wrapper to apply reflection coefficient to a flux function"""
        def reflected_flux(t):
            return flux_func(t) * (1.0 - reflection_coeff)
        return reflected_flux
    
    # Apply reflection to ion and atom fluxes
    deuterium_ion_flux_reflected = apply_reflection(deuterium_ion_flux, ion_reflection)
    tritium_ion_flux_reflected = apply_reflection(tritium_ion_flux, ion_reflection)
    deuterium_atom_flux_reflected = apply_reflection(deuterium_atom_flux, atom_reflection)
    tritium_atom_flux_reflected = apply_reflection(tritium_atom_flux, atom_reflection)
    
    # --- GEOMETRY AND MESH ---
    L = bin.thickness  # Domain length from bin
    vertices = graded_vertices(L=L, h0=1e-10, r=1.1)
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
    
    # --- CALCULATE WEIGHTED AVERAGE IMPLANTATION RANGE (time-dependent) ---
    # For analytical approximation BC, use weighted average of ion and atom ranges
    # Weight by their respective steady-state flux values
    # This varies with time if different pulses have different flux values
    
    def get_weighted_implantation_ranges(t):
        """Calculate weighted implantation ranges for D and T at time t
        
        Returns:
            Tuple of (weighted_range_d, weighted_range_t)
        """
        weighted_range_d = ion_range  # default fallback
        weighted_range_t = ion_range  # default fallback
        
        if occurrences and len(occurrences) > 0:
            # Find the occurrence that contains time t
            for occurrence in occurrences:
                if occurrence['start'] <= t < occurrence['end']:
                    # Extract steady-state flux values for this pulse
                    d_ion_flux_ss = occurrence['D_ion']
                    d_atom_flux_ss = occurrence['D_atom']
                    t_ion_flux_ss = occurrence['T_ion']
                    t_atom_flux_ss = occurrence['T_atom']
                    
                    # Calculate weighted average range for deuterium
                    d_total_flux = d_ion_flux_ss + d_atom_flux_ss
                    if d_total_flux > 0:
                        weighted_range_d = (d_atom_flux_ss * atom_range + d_ion_flux_ss * ion_range) / d_total_flux
                    else:
                        weighted_range_d = ion_range
                    
                    # Calculate weighted average range for tritium
                    t_total_flux = t_ion_flux_ss + t_atom_flux_ss
                    if t_total_flux > 0:
                        weighted_range_t = (t_atom_flux_ss * atom_range + t_ion_flux_ss * ion_range) / t_total_flux
                    else:
                        weighted_range_t = ion_range
                    
                    break
        
        return weighted_range_d, weighted_range_t
    
    # --- BOUNDARY CONDITIONS ---
    # Total flux functions (using reflected fluxes)
    def Gamma_D_total(t):
        return deuterium_ion_flux_reflected(t) + deuterium_atom_flux_reflected(t)

    def Gamma_T_total(t):
        return tritium_ion_flux_reflected(t) + tritium_atom_flux_reflected(t)

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
        # Use ion parameters for ions, atom parameters for atoms
        distribution_ion = gaussian_implantation_ufl(ion_range, ion_width, thickness=L)
        distribution_atom = gaussian_implantation_ufl(atom_range, atom_width, thickness=L)

        my_model.sources = [
            F.ParticleSource(value=lambda x, t: deuterium_ion_flux_reflected(t) * distribution_ion(x), volume=volume_subdomain, species=mobile_D),
            F.ParticleSource(value=lambda x, t: deuterium_atom_flux_reflected(t) * distribution_atom(x), volume=volume_subdomain, species=mobile_D),
            F.ParticleSource(value=lambda x, t: tritium_ion_flux_reflected(t) * distribution_ion(x), volume=volume_subdomain, species=mobile_T),
            F.ParticleSource(value=lambda x, t: tritium_atom_flux_reflected(t) * distribution_atom(x), volume=volume_subdomain, species=mobile_T),
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
        distribution_ion = gaussian_implantation_ufl(ion_range, ion_width, thickness=L)
        distribution_atom = gaussian_implantation_ufl(atom_range, atom_width, thickness=L)
        
        my_model.sources = [
            F.ParticleSource(value=lambda x, t: deuterium_ion_flux_reflected(t) * distribution_ion(x), volume=volume_subdomain, species=mobile_D),
            F.ParticleSource(value=lambda x, t: deuterium_atom_flux_reflected(t) * distribution_atom(x), volume=volume_subdomain, species=mobile_D),
            F.ParticleSource(value=lambda x, t: tritium_ion_flux_reflected(t) * distribution_ion(x), volume=volume_subdomain, species=mobile_T),
            F.ParticleSource(value=lambda x, t: tritium_atom_flux_reflected(t) * distribution_atom(x), volume=volume_subdomain, species=mobile_T),
        ]
        boundary_conditions.extend([
            F.FixedConcentrationBC(subdomain=inlet, value=0.0, species="D"),
            F.FixedConcentrationBC(subdomain=inlet, value=0.0, species="T"),
        ])

    elif bc_plasma_facing == "Dirichlet - Analyttical implantation approximation":
        # Use analytical surface concentration approximation (Dirichlet)
        # Use separate weighted ranges for D and T based on their respective flux ratios
        def c_sD_time_dependent(t):
            weighted_range_d, _ = get_weighted_implantation_ranges(t)
            return make_surface_concentration_time_function(
                temperature, Gamma_D_total, material.D0, material.E_D, weighted_range_d, surface_x=0.0
            )(t)
        
        def c_sT_time_dependent(t):
            _, weighted_range_t = get_weighted_implantation_ranges(t)
            return make_surface_concentration_time_function(
                temperature, Gamma_T_total, material.D0, material.E_D, weighted_range_t, surface_x=0.0
            )(t)
        boundary_conditions.extend([
            F.FixedConcentrationBC(subdomain=inlet, value=c_sD_time_dependent, species="D"),
            F.FixedConcentrationBC(subdomain=inlet, value=c_sT_time_dependent, species="T"),
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
    
    # Create temperature function from scenario
    temperature_function = make_temperature_function(
        scenario=scenario,
        plasma_data_handling=plasma_data_handling,
        bin=bin,
        coolant_temp=coolant_temp,
    )
    
    # Check BC type to decide which flux function type to use
    bc_plasma_facing = bin.bin_configuration.bc_plasma_facing_surface
    
    # Always compute occurrences for steady-state flux values (used for weighted implantation range)
    occurrences = compute_flux_values(scenario, plasma_data_handling, bin)
    
    # For implantation BCs, use UFL flux expressions (required for ParticleSource)
    if bc_plasma_facing in ("Robin - Surf. Rec. + Implantation", "Dirichlet - 0 concentration + Implantation"):
        # Use UFL flux expressions for ParticleSource compatibility
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
        occurrences=occurrences,
        exports=exports,
    )

#Helper functions block to create temperature profiles and flux functions 

# calculate how the rear temperature of the W layer evolves with the surface temperature
# data from E.A. Hodille et al 2021 Nucl. Fusion 61 126003 10.1088/1741-4326/ac2abc (Table I)
heat_fluxes_hodille = [10e6, 5e6, 1e6]  # W/m2
T_rears_hodille = [552, 436, 347]  # K

slope_T_rear, intercept, r_value, p_value, std_err = scipy.stats.linregress(
    heat_fluxes_hodille, T_rears_hodille
)


def tungsten_slab_temperature(q_front, D_W, D_Cu, T_cool):
    """
    Calculate the temperature of the front and back surfaces of a tungsten slab
    with heat flux applied to the front surface and cooling via a copper slab.
    From T. Wauters

    Parameters:
    q_front (float): Heat flux at the front surface of the tungsten slab (W/m^2).
    D_W (float): Thickness of the tungsten slab (m).
    D_Cu (float): Thickness of the copper slab (m).
    T_cool (float): Cooling water temperature (K).

    Returns:
    tuple: (T_w_surf, T_w_interface) where:
        - T_w_surf is the front surface temperature of tungsten (K).
        - T_w_interface is the tungsten-copper interface temperature (K).
    """
    # Thermal conductivities (W/m·K) #TODO: add citations
    k_W = 170  # Tungsten thermal conductivity
    k_Cu = 400  # Copper thermal conductivity
    # Heat transfer coefficient from copper to water (W/m^2·K)
    h_Cu_water = 10_000  # Typical value for water cooling

    w_diffusivity = (
        htm.diffusivities.filter(material="tungsten")
        .filter(isotope="h")
        .filter(author="holzner")
    )

    # Temperature drop across tungsten slab
    delta_T_W = (q_front * D_W) / k_W
    # Temperature drop across copper slab
    delta_T_Cu = (q_front * D_Cu) / k_Cu
    # Temperature drop at the copper-water interface
    delta_T_interface = q_front / h_Cu_water

    # Compute temperatures
    T_w_interface = T_cool + delta_T_interface + delta_T_Cu
    T_w_surf = T_w_interface + delta_T_W

    return T_w_surf, T_w_interface


def calculate_temperature_W(
    x: float | NDArray,
    heat_flux: float,
    coolant_temp: float,
    thickness: float,
    copper_thickness: float | None,
) -> float | NDArray:
    """Calculates the temperature in the W layer based on coolant temperature and heat flux

    Reference:
    - Delaporte-Mathurin et al. Sci Rep 10, 17798 (2020) 10.1038/s41598-020-74844-w
    - E.A. Hodille et al 2021 Nucl. Fusion 61 126003 10.1088/1741-4326/ac2abc

    Args:
        x: position in m
        heat_flux: heat_flux in W/m2
        coolant_temp: coolant temperature in K
        thickness: thickness of the W layer in m

    Returns:
        temperature in K
    """

    # T_surface and T_rear calculations taken from tungsten/copper calculations
    # provided by T. Wauters
    if copper_thickness is not None:
        T_surface, T_rear = tungsten_slab_temperature(
            q_front=heat_flux, D_W=thickness, D_Cu=copper_thickness, T_cool=coolant_temp
        )
    else:
        # the evolution of T surface is taken from Delaporte-Mathurin et al. Sci Rep 10, 17798 (2020).
        # https://doi.org/10.1038/s41598-020-74844-w
        T_surface = 1.1e-4 * heat_flux + coolant_temp
        T_rear = slope_T_rear * heat_flux + coolant_temp

    a = (T_rear - T_surface) / thickness
    b = T_surface
    return a * x + b


def calculate_temperature_B(heat_flux: float, coolant_temp: float) -> float:
    """
    Calculates the temperature in the boron layer based on coolant temperature and heat flux.
    The temperature is assumed to be homogeneous in the B layer and is calculated based on the
    surface temperature of the W layer.

    T_B = R_c * q + T_surface_W

    where
    - R_c is the thermal contact resistance of the layer in m2 K/W
    - q is the heat flux in W/m2
    - T_surface_W is the surface temperature of the W layer in K

    References:
    - Delaporte-Mathurin et al. Sci Rep 10, 17798 (2020) 10.1038/s41598-020-74844-w
    - Jae-Sun Park et al 2023 Nucl. Fusion 63 076027 10.1088/1741-4326/acd9d9

    Args:
        heat_flux: heat flux in W/m2
        coolant_temp: coolant temperature in K

    Returns:
        temperature in K
    """
    # the evolution of T surface is taken from Delaporte-Mathurin et al. Sci Rep 10, 17798 (2020).
    # https://doi.org/10.1038/s41598-020-74844-w
    T_surf_tungsten = 1.1e-4 * heat_flux + coolant_temp
    R_c_jet = 5e-4  # m2 K/W  calculated from JET-ILW (JPN#98297)
    return R_c_jet * heat_flux + T_surf_tungsten


def make_temperature_function(
    scenario: Scenario,
    plasma_data_handling: PlasmaDataHandling,
    bin,  # Accept any bin type (SubBin, DivBin, or CSVBin)
    coolant_temp: float,
) -> Callable[[NDArray, float], NDArray]:
    """Returns a function that calculates the temperature of the bin based on time and position.

    Args:
        scenario: the Scenario object containing the pulses
        plasma_data_handling: the object containing the plasma data
        bin: the bin/subbin to get the temperature function for
        coolant_temp: the coolant temperature in K

    Returns:
        a callable of x, t returning the temperature in K
    """

    def T_function(x: NDArray, t: float) -> NDArray:
        # Handle FESTIM 2.0 passing dolfinx.Constant instead of float
        if hasattr(t, 'value'):
            t = float(t.value)
        elif not isinstance(t, (float, int)):
            raise TypeError(f"t should be a float or have a .value attribute, got {type(t)}")

        # get the pulse and time relative to the start of the pulse
        pulse = scenario.get_pulse(t)
        t_rel = t - scenario.get_time_start_current_pulse(t)
        relative_time_within_pulse = t_rel % pulse.total_duration

        if pulse.pulse_type == "BAKE":
            T_value = periodic_pulse_function(
                relative_time_within_pulse,
                pulse=pulse,
                value=483.15,  # K
                value_off=343.0,  # K
            )
            value = np.full_like(x[0], T_value)

        else:
            heat_flux = plasma_data_handling.get_heat(
                pulse, bin, relative_time_within_pulse
            )
            # Handle both string materials and Material objects
            material_name = bin.material.name if hasattr(bin.material, 'name') else bin.material
            if (
                material_name == "W" or material_name == "SS"
            ):  # FIXME: update ss temp when gven data:
                value = calculate_temperature_W(
                    x[0], heat_flux, coolant_temp, bin.thickness, bin.copper_thickness
                )
            elif material_name == "B":
                T_value = calculate_temperature_B(heat_flux, coolant_temp)
                value = np.full_like(x[0], T_value)
            else:
                raise ValueError(f"Unsupported material: {bin.material}")

        return value

    return T_function


def make_particle_flux_function(
    scenario: Scenario,
    plasma_data_handling: PlasmaDataHandling,
    bin,  # Accept any bin type (SubBin, DivBin, or CSVBin)
    ion: bool,
    tritium: bool,
) -> Callable[[float], float]:
    """Returns a function that calculates the particle flux based on time.

    Args:
        scenario: the Scenario object containing the pulses
        plasma_data_handling: the object containing the plasma data
        bin: the bin/subbin to get the temperature function for
        ion: whether to get the ion flux
        tritium: whether to get the tritium flux

    Returns:
        a callable of t returning the **incident** particle flux in m^-2 s^-1
    """

    def particle_flux_function(t: float) -> float:
        # Handle FESTIM 2.0 passing dolfinx.Constant instead of float
        if hasattr(t, 'value'):
            t = float(t.value)
        elif not isinstance(t, (float, int)):
            raise TypeError(f"t should be a float or have a .value attribute, got {type(t)}")

        # get the pulse and time relative to the start of the pulse
        pulse = scenario.get_pulse(t)
        relative_time = t - scenario.get_time_start_current_pulse(t)
        relative_time_within_pulse = relative_time % pulse.total_duration

        # get the incident particle flux
        incident_hydrogen_particle_flux = plasma_data_handling.get_particle_flux(
            pulse=pulse,
            bin=bin,
            t_rel=relative_time_within_pulse,
            ion=ion,
        )

        # if tritium is requested, multiply by tritium fraction
        if tritium:
            value = incident_hydrogen_particle_flux * pulse.tritium_fraction
        else:
            value = incident_hydrogen_particle_flux * (1 - pulse.tritium_fraction)

        return value

    return particle_flux_function


def compute_flux_values(scenario, plasma_data_handling, bin_):
    """
    Compute steady-state flux values for each pulse occurrence using get_particle_flux
    at the midpoint of the steady-state region.
    Returns a list of dicts with D_ion, D_atom, T_ion, T_atom.
    """
    occurrences = []
    current_time = 0.0
    for pulse in scenario.pulses:
        for _ in range(pulse.nb_pulses):
            # Pick a time inside steady state
            if pulse.steady_state > 0:
                t_rel = pulse.ramp_up + pulse.steady_state / 2
            else:
                t_rel = pulse.total_duration / 2  # fallback if no steady state

            # Compute hydrogen flux for ion and atom
            flux_ion = plasma_data_handling.get_particle_flux(pulse, bin_, t_rel, ion=True)
            flux_atom = plasma_data_handling.get_particle_flux(pulse, bin_, t_rel, ion=False)

            # Apply tritium fraction
            T_ion = flux_ion * pulse.tritium_fraction
            D_ion = flux_ion * (1 - pulse.tritium_fraction)
            T_atom = flux_atom * pulse.tritium_fraction
            D_atom = flux_atom * (1 - pulse.tritium_fraction)

            occurrences.append({
                'start': current_time,
                'end': current_time + pulse.total_duration,
                'pulse': pulse,
                'D_ion': D_ion,
                'D_atom': D_atom,
                'T_ion': T_ion,
                'T_atom': T_atom
            })
            current_time += pulse.total_duration
    return occurrences


def build_ufl_flux_expression(occurrences, value_off=0.0):
    """
    Returns four functions:
    (D_ion_fn, D_atom_fn, T_ion_fn, T_atom_fn)
    Each function accepts a UFL time variable `t` and returns the corresponding UFL expression.
    """

    def make_flux_fn(flux_key):
        def flux_builder(t):
            expr = value_off
            for occ in occurrences:
                p = occ['pulse']
                start, end = occ['start'], occ['end']

                in_window = And(ge(t, start), lt(t, end))
                t_rel = t - start

                ramp_up_cond = lt(t_rel, p.ramp_up)
                steady_cond = And(ge(t_rel, p.ramp_up), lt(t_rel, p.ramp_up + p.steady_state))

                # Ramp-up and ramp-down expressions
                ramp_up_expr = (occ[flux_key] - value_off) / p.ramp_up * t_rel + value_off if p.ramp_up > 0 else occ[flux_key]
                ramp_down_raw = occ[flux_key] - (occ[flux_key] - value_off) / p.ramp_down * (t_rel - (p.ramp_up + p.steady_state)) if p.ramp_down > 0 else occ[flux_key]
                ramp_down_expr = conditional(ge(ramp_down_raw, value_off), ramp_down_raw, value_off)

                pulse_flux = conditional(ramp_up_cond, ramp_up_expr,
                                         conditional(steady_cond, occ[flux_key], ramp_down_expr))

                expr += conditional(in_window, pulse_flux, 0.0)
            return expr
        return flux_builder

    # Return four callable builders
    return (
        make_flux_fn('D_ion'),
        make_flux_fn('D_atom'),
        make_flux_fn('T_ion'),
        make_flux_fn('T_atom'),
    )
