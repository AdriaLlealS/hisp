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

# Constants
kB_J = 1.380649e-23      # J/K
eV_to_J = 1.602176634e-19  # J/eV
implantation_range = 3e-9  # m (TODO: make this depend on incident energy)


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
    for i in range(1, n_traps + 1):
        # Get trap parameters
        trap_params = material.traps[i - 1]
        trap_density = trap_params.Trap_density
        
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
    
    # Create reactions
    reactions_list = []
    
    # Material diffusion parameters
    D_0 = material.D0
    E_D = material.E_D
    Mat_density = material.Mat_density
    
    # Default recombination coefficient if not provided
    # Use a simple estimate: k_0 = D_0 / (lattice_constant^2 * sites_per_atom * density)
    # For simplicity, assume interstitial_distance and sites_per_atom
    interstitial_distance = 1.117e-10  # m (typical for W, adjust if needed)
    interstitial_sites_per_atom = 6
    
    default_k_0 = D_0 / (interstitial_distance**2 * interstitial_sites_per_atom * Mat_density)
    
    for trap_info in trap_list:
        trap_params = trap_info['params']
        
        # Use trap-specific parameters if provided, otherwise use defaults
        k_0 = trap_params.k_0 if trap_params.k_0 is not None else default_k_0
        E_k = trap_params.E_k if trap_params.E_k is not None else E_D
        p_0 = trap_params.p_0 if trap_params.p_0 is not None else 1e13
        E_p = trap_params.E_p if trap_params.E_p is not None else 1.0  # Default detrapping energy
        
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
    my_model = F.HydrogenTransportProblem()
    
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
    
    boundary_conditions = []
    
    # Plasma-facing surface (inlet) boundary conditions
    if "Dirichlet" in bc_plasma_facing or "Robin" in bc_plasma_facing:
        # Use Dirichlet BC (surface concentration)
        c_sD = make_surface_concentration_time_function(
            temperature, Gamma_D_total, material.D0, material.E_D, implantation_range, surface_x=0.0
        )
        c_sT = make_surface_concentration_time_function(
            temperature, Gamma_T_total, material.D0, material.E_D, implantation_range, surface_x=0.0
        )
        
        bc_D = F.FixedConcentrationBC(subdomain=inlet, value=c_sD, species="D")
        bc_T = F.FixedConcentrationBC(subdomain=inlet, value=c_sT, species="T")
        boundary_conditions.extend([bc_D, bc_T])
    else:
        # Default: no flux or other BC
        boundary_conditions.extend([
            F.ParticleFluxBC(subdomain=inlet, value=0.0, species="D"),
            F.ParticleFluxBC(subdomain=inlet, value=0.0, species="T"),
        ])
    
    # Rear surface (outlet) boundary conditions
    if "Neumann" in bc_rear or "no flux" in bc_rear.lower():
        # No flux BC (Neumann)
        boundary_conditions.extend([
            F.ParticleFluxBC(subdomain=outlet, value=0.0, species="D"),
            F.ParticleFluxBC(subdomain=outlet, value=0.0, species="T"),
        ])
    else:
        # Default: no flux
        boundary_conditions.extend([
            F.ParticleFluxBC(subdomain=outlet, value=0.0, species="D"),
            F.ParticleFluxBC(subdomain=outlet, value=0.0, species="T"),
        ])
    
    my_model.boundary_conditions = boundary_conditions
    
    # --- EXPORTS ---
    if exports:
        my_model.exports = [
            F.XDMFExport(
                field="solute",
                folder=folder,
                checkpoint=False,
            ),
            F.XDMFExport(
                field="retention",
                folder=folder,
                checkpoint=False,
            ),
        ]
    
    # --- QUANTITIES TO TRACK ---
    quantities = {}
    for species in my_model.species:
        quantities[f"{species.name}_total_volume"] = F.TotalVolume(
            field=species.name, volume=volume_subdomain
        )
    
    # Total retention
    quantities["total_retention"] = F.TotalVolume(
        field="retention", volume=volume_subdomain
    )
    
    # --- SETTINGS ---
    bin_config = bin.bin_configuration
    my_model.settings = F.Settings(
        atol=bin_config.atol,
        rtol=bin_config.rtol,
        max_iterations=100,
        final_time=final_time,
    )
    
    my_model.settings.stepsize = F.Stepsize(initial_value=1e-3)
    
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
    )
    
    # Create temperature and flux functions from scenario
    temperature_function = make_temperature_function(
        scenario=scenario,
        plasma_data_handling=plasma_data_handling,
        bin=bin,
        coolant_temp=coolant_temp,
    )
    
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
