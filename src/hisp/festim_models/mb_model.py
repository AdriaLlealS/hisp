from builtins import ValueError, bool, callable, float, int, isinstance, str, type
from hisp.h_transport_class import CustomProblem
from hisp.helpers import (
    PulsedSource,
    gaussian_distribution,
    Stepsize,
    periodic_pulse_function,
    gaussian_implantation_ufl,
)
from hisp.scenario import Scenario
from hisp.plamsa_data_handling import PlasmaDataHandling
from hisp.settings import CustomSettings
import hisp.bin
from ufl import conditional, lt, ge, And
import numpy as np
import festim as F
import h_transport_materials as htm

from typing import Callable, Tuple, Dict, Union, List
from numpy.typing import NDArray

import math

# TODO this is hard coded and should depend on incident energy?
implantation_range = 3e-9  # m
width = 1e-9  # m

def graded_vertices(L, h0, r):
        xs = [0.0]; h = h0
        while xs[-1] + h < L:
            xs.append(xs[-1] + h); h *= r
        if xs[-1] < L: xs.append(L)
        return np.array(xs)


def make_W_mb_model(
    temperature: Callable | float | int,
    deuterium_ion_flux: Callable,
    tritium_ion_flux: Callable,
    deuterium_atom_flux: Callable,
    tritium_atom_flux: Callable,
    final_time: float,
    folder: str,
    L: float,
    custom_rtol: Union[
        float, Callable
    ] = 1e-8,  # default rtol unless otherwise specified, used for everything but BAKE
    exports=False,
) -> Tuple[CustomProblem, Dict[str, F.TotalVolume]]:
    """Create a FESTIM model for the W MB scenario.

    Args:
        temperature: the temperature in K.
        deuterium_ion_flux: the deuterium ion flux in m^-2 s^-1.
        tritium_ion_flux: the tritium ion flux in m^-2 s^-1.
        deuterium_atom_flux: the deuterium atom flux in m^-2 s^-1.
        tritium_atom_flux: the tritium atom flux in m^-2 s^-1.
        final_time: the final time in s.
        folder: the folder to save the results.
        L: the length of the domain in m.

    Returns:
        the FESTIM model, the quantities to export.
    """
    my_model = CustomProblem()

    ############# Material Parameters #############
    
    vertices_graded = graded_vertices(L=L, h0=L/12e9, r=1.01)

    my_model.mesh = F.Mesh1D(vertices_graded)

    # W material parameters
    w_density = 6.3382e28  # atoms/m3
    w_diffusivity = (
        htm.diffusivities.filter(material="tungsten")
        .filter(isotope="h")
        .filter(author="holzner")
    )
    w_diffusivity = w_diffusivity[0]
    D_0 = w_diffusivity.pre_exp.magnitude
    E_D = w_diffusivity.act_energy.magnitude
    tungsten = F.Material(
        D_0=D_0,
        E_D=E_D,
        name="tungsten",
    )

    # mb subdomains
    w_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, L], material=tungsten)
    inlet = F.SurfaceSubdomain1D(id=1, x=0)
    outlet = F.SurfaceSubdomain1D(id=2, x=L)

    my_model.subdomains = [w_subdomain, inlet, outlet]

    # hydrogen species
    mobile_D = F.Species("D")
    mobile_T = F.Species("T")

    trap1_D = F.Species("trap1_D", mobile=False)
    trap1_T = F.Species("trap1_T", mobile=False)
    trap2_D = F.Species("trap2_D", mobile=False)
    trap2_T = F.Species("trap2_T", mobile=False)
    # trap3_D = F.Species("trap3_D", mobile=False)
    # trap3_T = F.Species("trap3_T", mobile=False)

    # traps
    empty_trap1 = F.ImplicitSpecies(  # implicit trap 1
        n=6.338e24,  # 1e-4 at.fr.
        others=[trap1_T, trap1_D],
        name="empty_trap1",
    )

    empty_trap2 = F.ImplicitSpecies(  # implicit trap 2
        n=6.338e24,
        others=[trap2_T, trap2_D],
        name="empty_trap2",
    )

    # TODO: make trap space dependent (existing in only first 10nm). commenting out until implemented
    # density_func = lambda x: ufl.conditional(ufl.gt(x[0],10), 6.338e27, 0.0) #  small damanged zone in first 10nm, 1e-1 at.fr.
    # empty_trap3 = F.ImplicitSpecies(
    #     n=6.338e27,
    #     others=[trap3_T, trap3_D],
    #     name="empty_trap3",
    # )

    my_model.species = [
        mobile_D,
        mobile_T,
        trap1_D,
        trap1_T,
        trap2_D,
        trap2_T,
        # trap3_D,
        # trap3_T,
    ]

    interstitial_distance = 1.117e-10  # m
    interstitial_sites_per_atom = 6

    # hydrogen reactions - 1 per trap per species
    my_model.reactions = [
        F.Reaction(
            k_0=D_0
            / (interstitial_distance**2 * interstitial_sites_per_atom * w_density),
            E_k=E_D,
            p_0=1e13,
            E_p=0.85,
            volume=w_subdomain,
            reactant=[mobile_D, empty_trap1],
            product=trap1_D,
        ),
        F.Reaction(
            k_0=D_0
            / (interstitial_distance**2 * interstitial_sites_per_atom * w_density),
            E_k=E_D,
            p_0=1e13,
            E_p=0.85,
            volume=w_subdomain,
            reactant=[mobile_T, empty_trap1],
            product=trap1_T,
        ),
        F.Reaction(
            k_0=D_0
            / (interstitial_distance**2 * interstitial_sites_per_atom * w_density),
            E_k=E_D,
            p_0=1e13,
            E_p=1,
            volume=w_subdomain,
            reactant=[mobile_D, empty_trap2],
            product=trap2_D,
        ),
        F.Reaction(
            k_0=D_0
            / (interstitial_distance**2 * interstitial_sites_per_atom * w_density),
            E_k=E_D,
            p_0=1e13,
            E_p=1,
            volume=w_subdomain,
            reactant=[mobile_T, empty_trap2],
            product=trap2_T,
        ),
        # F.Reaction(
        #     k_0=D_0 / (interstitial_distance**2 * interstitial_sites_per_atom * w_density),
        #     E_k=E_D,
        #     p_0=1e13,
        #     E_p=1.5,
        #     volume=w_subdomain,
        #     reactant=[mobile_D, empty_trap3],
        #     product=trap3_D,
        # ),
        # F.Reaction(
        #     k_0=D_0 / (interstitial_distance**2 * interstitial_sites_per_atom * w_density),
        #     E_k=E_D,
        #     p_0=1e13,
        #     E_p=1.5,
        #     volume=w_subdomain,
        #     reactant=[mobile_T, empty_trap3],
        #     product=trap3_T,
        # ),
    ]

    #############   Parameters (K) #############

    my_model.temperature = temperature

    ############# Flux Parameters #############

    def Gamma_D_total(t): 
        return float(deuterium_ion_flux(t)+deuterium_atom_flux(t))

    def Gamma_T_total(t): 
        return float(tritium_atom_flux(t)+tritium_ion_flux(t))

    # Build the two BC callables
    c_sD = make_surface_concentration_time_function_J(temperature, Gamma_D_total, D_0, E_D, implantation_range, surface_x=0.0)
    c_sT = make_surface_concentration_time_function_J(temperature, Gamma_T_total, D_0, E_D, implantation_range, surface_x=0.0)

    # Register as Dirichlet BCs at the inlet (replace existing BCs if desired)
    bc_D = F.FixedConcentrationBC(subdomain=inlet, value=c_sD, species="D")
    bc_T = F.FixedConcentrationBC(subdomain=inlet, value=c_sT, species="T")



    my_model.boundary_conditions = [
        bc_D,
        bc_T
    ]

    ############# Exports #############
    if exports:
        my_model.exports = [
            F.VTXSpeciesExport(f"{folder}/mobile_concentration_t.bp", field=mobile_T),
            F.VTXSpeciesExport(f"{folder}/mobile_concentration_d.bp", field=mobile_D),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_d1.bp", field=trap1_D),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_t1.bp", field=trap1_T),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_d2.bp", field=trap2_D),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_t2.bp", field=trap2_T),
            # F.VTXSpeciesExport(f"{folder}/trapped_concentration_d3.bp", field=trap3_D),
            # F.VTXSpeciesExport(f"{folder}/trapped_concentration_t3.bp", field=trap3_T),
        ]

    quantities = {}
    for species in my_model.species:
        quantity = F.TotalVolume(field=species, volume=w_subdomain)
        my_model.exports.append(quantity)
        quantities[species.name] = quantity
        if species.mobile:
            flux = F.SurfaceFlux(field=species, surface=inlet)
            my_model.exports.append(flux)
            quantities[species.name + "_surface_flux"] = flux

    ############# Settings #############
    my_model.settings = CustomSettings(
        atol=1e10,
        rtol=custom_rtol,
        max_iterations=100,  # the first timestep needs about 66 iterations....
        final_time=final_time,
    )

    my_model.settings.stepsize = Stepsize(initial_value=1e-3)
    my_model.settings.linear_solver   = "preonly"  # one direct solve per Newton iteration
    my_model.settings.preconditioner  = "lu"       # LU factorization
    my_model._element_for_traps = "CG"

    return my_model, quantities


def make_B_mb_model(
    temperature: Callable | float | int,
    deuterium_ion_flux: Callable,
    tritium_ion_flux: Callable,
    deuterium_atom_flux: Callable,
    tritium_atom_flux: Callable,
    final_time: float,
    folder: str,
    L: float,
    custom_atol: Union[
        float, Callable
    ] = 1e10,  # default atol unless otherwise specified, used for FP, ICWC, RISP in hisp-for-iter
    custom_rtol: Union[
        float, Callable
    ] = 1e-10,  # default rtol unless otherwise specified, used for FP, ICWC, RISP in hisp-for-iter
    exports=False,
) -> Tuple[CustomProblem, Dict[str, F.TotalVolume]]:
    """Create a FESTIM model for the B MB scenario.

    Args:
        temperature: the temperature in K.
        deuterium_ion_flux: the deuterium ion flux in m^-2 s^-1.
        tritium_ion_flux: the tritium ion flux in m^-2 s^-1.
        deuterium_atom_flux: the deuterium atom flux in m^-2 s^-1.
        tritium_atom_flux: the tritium atom flux in m^-2 s^-1.
        final_time: the final time in s.
        folder: the folder to save the results.
        L: the length of the domain in m.

    Returns:
        the FESTIM model, the quantities to export.
    """
    my_model = CustomProblem()

    ############# Material Parameters #############

    vertices_graded = graded_vertices(L=L, h0=L/12e9, r=1.008)

    my_model.mesh = F.Mesh1D(vertices_graded)

    # B material parameters from Etienne Hodilles's unpublished TDS study for boron
    b_density = 1.34e29  # atoms/m3
    D_0 = 1.07e-6  # m^2/s
    E_D = 0.3  # eV
    boron = F.Material(
        D_0=D_0,
        E_D=E_D,
        name="boron",
    )

    # mb subdomains
    b_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, L], material=boron)
    inlet = F.SurfaceSubdomain1D(id=1, x=0)
    outlet = F.SurfaceSubdomain1D(id=2, x=L)

    my_model.subdomains = [b_subdomain, inlet, outlet]

    # hydrogen species
    mobile_D = F.Species("D")
    mobile_T = F.Species("T")

    trap1_D = F.Species("trap1_D", mobile=False)
    trap1_T = F.Species("trap1_T", mobile=False)
    trap2_D = F.Species("trap2_D", mobile=False)
    trap2_T = F.Species("trap2_T", mobile=False)
    trap3_D = F.Species("trap3_D", mobile=False)
    trap3_T = F.Species("trap3_T", mobile=False)
    trap4_D = F.Species("trap4_D", mobile=False)
    trap4_T = F.Species("trap4_T", mobile=False)
    # trap5_D = F.Species("trap5_D", mobile=False)
    # trap5_T = F.Species("trap5_T", mobile=False)

    # traps
    empty_trap1 = F.ImplicitSpecies(  # implicit trap 1
        n=6.867e-1
        * b_density,  # from Etienne Hodilles's unpublished TDS study for boron
        others=[trap1_T, trap1_D],
        name="empty_trap1",
    )

    empty_trap2 = F.ImplicitSpecies(  # implicit trap 2
        n=5.214e-1
        * b_density,  # from Etienne Hodilles's unpublished TDS study for boron
        others=[trap2_T, trap2_D],
        name="empty_trap2",
    )

    empty_trap3 = F.ImplicitSpecies(
        n=2.466e-1
        * b_density,  # from Etienne Hodilles's unpublished TDS study for boron
        others=[trap3_T, trap3_D],
        name="empty_trap3",
    )

    empty_trap4 = F.ImplicitSpecies(
        n=1.280e-1
        * b_density,  # from Etienne Hodilles's unpublished TDS study for boron
        others=[trap4_T, trap4_D],
        name="empty_trap4",
    )

    # empty_trap5 = F.ImplicitSpecies(
    #     n=1.800e-1*b_density, # from Johnathan Dufour's unpublished TDS study for boron
    #     others=[trap5_T, trap5_D],
    #     name="empty_trap5",
    # )

    my_model.species = [
        mobile_D,
        mobile_T,
        trap1_D,
        trap1_T,
        trap2_D,
        trap2_T,
        trap3_D,
        trap3_T,
        trap4_D,
        trap4_T,
        # trap5_D,
        # trap5_T,
    ]

    # hydrogen reactions - 1 per trap per species
    interstitial_distance = 8e-10  # m
    interstitial_sites_per_atom = 1

    my_model.reactions = [
        F.Reaction(
            k_0=1e13 / b_density,
            E_k=E_D,
            p_0=1e13,  # from Etienne Hodilles's unpublished TDS study for boron
            E_p=1.052,
            volume=b_subdomain,
            reactant=[mobile_D, empty_trap1],
            product=trap1_D,
        ),
        F.Reaction(
            k_0=1e13 / b_density,
            E_k=E_D,
            p_0=1e13,  # from Etienne Hodilles's unpublished TDS study for boron
            E_p=1.052,
            volume=b_subdomain,
            reactant=[mobile_T, empty_trap1],
            product=trap1_T,
        ),
        F.Reaction(
            k_0=1e13 / b_density,
            E_k=E_D,
            p_0=1e13,  # from Etienne Hodilles's unpublished TDS study for boron
            E_p=1.199,
            volume=b_subdomain,
            reactant=[mobile_D, empty_trap2],
            product=trap2_D,
        ),
        F.Reaction(
            k_0=1e13 / b_density,
            E_k=E_D,
            p_0=1e13,  # from Etienne Hodilles's unpublished TDS study for boron
            E_p=1.199,
            volume=b_subdomain,
            reactant=[mobile_T, empty_trap2],
            product=trap2_T,
        ),
        F.Reaction(
            k_0=1e13 / b_density,
            E_k=E_D,
            p_0=1e13,  # from Etienne Hodilles's unpublished TDS study for boron
            E_p=1.389,
            volume=b_subdomain,
            reactant=[mobile_D, empty_trap3],
            product=trap3_D,
        ),
        F.Reaction(
            k_0=1e13 / b_density,
            E_k=E_D,
            p_0=1e13,  # from Etienne Hodilles's unpublished TDS study for boron
            E_p=1.389,
            volume=b_subdomain,
            reactant=[mobile_T, empty_trap3],
            product=trap3_T,
        ),
        F.Reaction(
            k_0=1e13 / b_density,
            E_k=E_D,
            p_0=1e13,  # from Etienne Hodilles's unpublished TDS study for boron
            E_p=1.589,
            volume=b_subdomain,
            reactant=[mobile_D, empty_trap4],
            product=trap4_D,
        ),
        F.Reaction(
            k_0=1e13 / b_density,
            E_k=E_D,
            p_0=1e13,  # from Etienne Hodilles's unpublished TDS study for boron
            E_p=1.589,
            volume=b_subdomain,
            reactant=[mobile_T, empty_trap4],
            product=trap4_T,
        ),
        # F.Reaction(
        #     k_0=1e13 / b_density,
        #     E_k=E_D,
        #     p_0=1e13,  # from Johnathan Dufour's unpublished TDS study for boron
        #     E_p=1.776,
        #     volume=b_subdomain,
        #     reactant=[mobile_D, empty_trap5],
        #     product=trap5_D,
        # ),
        # F.Reaction(
        #     k_0=1e13 / b_density,
        #     E_k=E_D,
        #     p_0=1e13,  # from Johnathan Dufour's unpublished TDS study for boron
        #     E_p=1.776,
        #     volume=b_subdomain,
        #     reactant=[mobile_T, empty_trap5],
        #     product=trap5_T,
        # ),
    ]

    ############# Temperature Parameters (K) #############

    my_model.temperature = temperature

    ############# Flux Parameters #############

    def Gamma_D_total(t): 
        return float(deuterium_ion_flux(t)+deuterium_atom_flux(t))

    def Gamma_T_total(t): 
        return float(tritium_atom_flux(t)+tritium_ion_flux(t))

    # Build the two BC callables
    c_sD = make_surface_concentration_time_function_J(temperature, Gamma_D_total, D_0, E_D, implantation_range, surface_x=0.0)
    c_sT = make_surface_concentration_time_function_J(temperature, Gamma_T_total, D_0, E_D, implantation_range, surface_x=0.0)

    # Register as Dirichlet BCs at the inlet (replace existing BCs if desired)
    bc_D = F.FixedConcentrationBC(subdomain=inlet, value=c_sD, species="D")
    bc_T = F.FixedConcentrationBC(subdomain=inlet, value=c_sT, species="T")

    ############# Boundary Conditions #############
    my_model.boundary_conditions = [
        #F.FixedConcentrationBC(subdomain=inlet, value=0.0, species="D"),
        #F.FixedConcentrationBC(subdomain=inlet, value=0.0, species="T"),
        bc_D,
        bc_T,
        F.ParticleFluxBC(subdomain=outlet, value=0.0, species="D"),
        F.ParticleFluxBC(subdomain=outlet, value=0.0, species="T"),
    ]

    ############# Exports #############
    if exports:
        my_model.exports = [
            F.VTXSpeciesExport(f"{folder}/mobile_concentration_t.bp", field=mobile_T),
            F.VTXSpeciesExport(f"{folder}/mobile_concentration_d.bp", field=mobile_D),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_d1.bp", field=trap1_D),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_t1.bp", field=trap1_T),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_d2.bp", field=trap2_D),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_t2.bp", field=trap2_T),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_d3.bp", field=trap3_D),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_t3.bp", field=trap3_T),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_d4.bp", field=trap4_D),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_t4.bp", field=trap4_T),
            # F.VTXSpeciesExport(f"{folder}/trapped_concentration_d5.bp", field=trap5_D),
            # F.VTXSpeciesExport(f"{folder}/trapped_concentration_t5.bp", field=trap5_T),
        ]

    quantities = {}
    for species in my_model.species:
        quantity = F.TotalVolume(field=species, volume=b_subdomain)
        my_model.exports.append(quantity)
        quantities[species.name] = quantity
        if species.mobile:
            flux = F.SurfaceFlux(field=species, surface=inlet)
            my_model.exports.append(flux)
            quantities[species.name + "_surface_flux"] = flux

    ############# Settings #############
    my_model.settings = CustomSettings(
        atol=custom_atol,
        rtol=custom_rtol,
        max_iterations=100,
        final_time=final_time,
    )

    my_model.settings.stepsize = Stepsize(initial_value=1e-4)
    my_model.settings.linear_solver   = "preonly"  # one direct solve per Newton iteration
    my_model.settings.preconditioner  = "lu"       # LU factorization
    my_model._element_for_traps = "CG"
    return my_model, quantities


def make_DFW_mb_model(
    temperature: Callable | float | int,
    deuterium_ion_flux: Callable,
    tritium_ion_flux: Callable,
    deuterium_atom_flux: Callable,
    tritium_atom_flux: Callable,
    final_time: float,
    folder: str,
    L: float,
    custom_atol: Union[float, Callable] = 1e10,  # default atol unless otherwise specified, used for FP, ICWC, RISP in hisp-for-iter
    custom_rtol: Union[float, Callable] = 1e-10,  # default rtol unless otherwise specified, used for FP, ICWC, RISP in hisp-for-iter
    exports=False,
) -> Tuple[CustomProblem, Dict[str, F.TotalVolume]]:
    """Create a FESTIM model for the DFW MB scenario.

    Args:
        temperature: the temperature in K.
        deuterium_ion_flux: the deuterium ion flux in m^-2 s^-1.
        tritium_ion_flux: the tritium ion flux in m^-2 s^-1.
        deuterium_atom_flux: the deuterium atom flux in m^-2 s^-1.
        tritium_atom_flux: the tritium atom flux in m^-2 s^-1.
        final_time: the final time in s.
        folder: the folder to save the results.
        L: the length of the domain in m.

    Returns:
        the FESTIM model, the quantities to export.
    """

    my_model = CustomProblem()

    ############# Material Parameters #############

    vertices_graded = graded_vertices(L=L, h0=L/12e9, r=1.01)

    my_model.mesh = F.Mesh1D(vertices_graded)

    # TODO: pull DFW material parameters from HTM?

    # from ITER mean value parameters (FIXME: add DOI)
    ss_density = 8.45e28  # atoms/m3
    D_0 = 1.45e-6  # m^2/s
    E_D = 0.59  # eV
    ss = F.Material(
        D_0=D_0,
        E_D=E_D,
        name="ss",
    )

    # mb subdomains
    ss_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, L], material=ss)
    inlet = F.SurfaceSubdomain1D(id=1, x=0)
    outlet = F.SurfaceSubdomain1D(id=2, x=L)

    my_model.subdomains = [ss_subdomain, inlet, outlet]

    # hydrogen species
    mobile_D = F.Species("D")
    mobile_T = F.Species("T")

    trap1_D = F.Species("trap1_D", mobile=False)
    trap1_T = F.Species("trap1_T", mobile=False)

    # traps
    empty_trap1 = F.ImplicitSpecies(  # implicit trap 1
        n=8e-2 * ss_density,  # from Guillermain D 2016 ITER report T2YEND
        others=[trap1_T, trap1_D],
        name="empty_trap1",
    )

    my_model.species = [
        mobile_D,
        mobile_T,
        trap1_D,
        trap1_T,
    ]

    # hydrogen reactions - 1 per trap per species
    interstitial_distance = 2.545e-10  # m
    interstitial_sites_per_atom = 1

    my_model.reactions = [
        F.Reaction(
            k_0=D_0
            / (interstitial_distance * interstitial_sites_per_atom * ss_density),
            E_k=E_D,
            p_0=1e13,  # from Guillermain D 2016 ITER report T2YEND
            E_p=0.7,
            volume=ss_subdomain,
            reactant=[mobile_D, empty_trap1],
            product=trap1_D,
        ),
        F.Reaction(
            k_0=D_0
            / (interstitial_distance * interstitial_sites_per_atom * ss_density),
            E_k=E_D,
            p_0=1e13,  # from Guillermain D 2016 ITER report T2YEND
            E_p=0.7,
            volume=ss_subdomain,
            reactant=[mobile_T, empty_trap1],
            product=trap1_T,
        ),
    ]

    ############# Temperature Parameters (K) #############

    my_model.temperature = temperature

    ############# Flux Parameters #############

    def total_D_flux(t: float) -> float:
        t = float(t)
        # If you also have additional sources/sinks, sum them here.
        return float(deuterium_ion_flux(t)) + float(deuterium_atom_flux(t))

    def total_T_flux(t: float) -> float:
        t = float(t)
        return float(tritium_ion_flux(t)) + float(tritium_atom_flux(t))

    # Tritium fraction: T_frac(t) = total_T_flux(t) / (total_T_flux(t) + total_D_flux(t))
    # NOTE: If total flux can be exactly zero at some times, this will raise ZeroDivisionError.
    # If you want a robust behavior there, add a small guard (e.g., return 0.0 when denom == 0).
    def T_frac(t: float) -> float:
        t = float(t)
        Tt = total_T_flux(t)
        Dt = total_D_flux(t)
        denom = Tt + Dt
        if denom > 0:
            return float(Tt / denom)
        else:
            return 0.0

    ############# Boundary Conditions #############

    k_r0 = 1.75e-24
    E_kr = -0.594

    # Build the two BC callables
    c_sD = make_D_surface_concentration_SS(temperature, total_D_flux, T_frac, D_0, k_r0, E_D, E_kr, implantation_range, surface_x=0.0)
    c_sT = make_T_surface_concentration_SS(temperature, total_T_flux, T_frac, D_0, k_r0, E_D, E_kr, implantation_range, surface_x=0.0)

    # Register as Dirichlet BCs at the inlet (replace existing BCs if desired)
    bc_D = F.FixedConcentrationBC(subdomain=inlet, value=c_sD, species="D")
    bc_T = F.FixedConcentrationBC(subdomain=inlet, value=c_sT, species="T")



    my_model.boundary_conditions = [
        bc_D,
        bc_T,
    ]

    ############# Exports #############
    if exports:
        my_model.exports = [
            F.VTXSpeciesExport(f"{folder}/mobile_concentration_t.bp", field=mobile_T),
            F.VTXSpeciesExport(f"{folder}/mobile_concentration_d.bp", field=mobile_D),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_d1.bp", field=trap1_D),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_t1.bp", field=trap1_T),
        ]

    quantities = {}
    for species in my_model.species:
        quantity = F.TotalVolume(field=species, volume=ss_subdomain)
        my_model.exports.append(quantity)
        quantities[species.name] = quantity
        if species.mobile:
            flux = F.SurfaceFlux(field=species, surface=inlet)
            my_model.exports.append(flux)
            quantities[species.name + "_surface_flux"] = flux

    ############# Settings #############
    my_model.settings = CustomSettings(
        atol=custom_atol,
        rtol=custom_rtol,
        max_iterations=100,
        final_time=final_time,
    )

    my_model.settings.stepsize = Stepsize(initial_value=1e-3)
    my_model.settings.linear_solver   = "preonly"  # one direct solve per Newton iteration
    my_model.settings.preconditioner  = "lu"       # LU factorization
    my_model._element_for_traps = "CG"
    return my_model, quantities

def make_W_mb_model_oldBC(
    temperature: Callable | float | int,
    final_time: float,
    folder: str,
    L: float,
    occurrences: List[Dict],
    custom_rtol: Union[
        float, Callable
    ] = 1e-4,  # default rtol unless otherwise specified, used for everything but BAKE
    exports=False,
) -> Tuple[CustomProblem, Dict[str, F.TotalVolume]]:
    """Create a FESTIM model for the W MB scenario.

    Args:
        temperature: the temperature in K.
        final_time: the final time in s.
        folder: the folder to save the results.
        L: the length of the domain in m.

    Returns:
        the FESTIM model, the quantities to export.
    """
    my_model = CustomProblem()

    ############# Material Parameters #############
    
    vertices_graded = graded_vertices(L=L, h0=1e-10, r=1.01)
    my_model.mesh = F.Mesh1D(vertices_graded)

    # W material parameters
    w_density = 6.3382e28  # atoms/m3
    w_diffusivity = (
        htm.diffusivities.filter(material="tungsten")
        .filter(isotope="h")
        .filter(author="holzner")
    )
    w_diffusivity = w_diffusivity[0]
    D_0 = w_diffusivity.pre_exp.magnitude
    E_D = w_diffusivity.act_energy.magnitude
    tungsten = F.Material(
        D_0=D_0,
        E_D=E_D,
        name="tungsten",
    )

    # mb subdomains
    w_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, L], material=tungsten)
    inlet = F.SurfaceSubdomain1D(id=1, x=0)
    outlet = F.SurfaceSubdomain1D(id=2, x=L)

    my_model.subdomains = [w_subdomain, inlet, outlet]

    # hydrogen species
    mobile_D = F.Species("D")
    mobile_T = F.Species("T")

    trap1_D = F.Species("trap1_D", mobile=False)
    trap1_T = F.Species("trap1_T", mobile=False)
    trap2_D = F.Species("trap2_D", mobile=False)
    trap2_T = F.Species("trap2_T", mobile=False)
    # trap3_D = F.Species("trap3_D", mobile=False)
    # trap3_T = F.Species("trap3_T", mobile=False)

    # traps
    empty_trap1 = F.ImplicitSpecies(  # implicit trap 1
        n=6.338e24,  # 1e-4 at.fr.
        others=[trap1_T, trap1_D],
        name="empty_trap1",
    )

    empty_trap2 = F.ImplicitSpecies(  # implicit trap 2
        n=6.338e24,
        others=[trap2_T, trap2_D],
        name="empty_trap2",
    )

    # TODO: make trap space dependent (existing in only first 10nm). commenting out until implemented
    # density_func = lambda x: ufl.conditional(ufl.gt(x[0],10), 6.338e27, 0.0) #  small damanged zone in first 10nm, 1e-1 at.fr.
    # empty_trap3 = F.ImplicitSpecies(
    #     n=6.338e27,
    #     others=[trap3_T, trap3_D],
    #     name="empty_trap3",
    # )

    my_model.species = [
        mobile_D,
        mobile_T,
        trap1_D,
        trap1_T,
        trap2_D,
        trap2_T,
        # trap3_D,
        # trap3_T,
    ]

    interstitial_distance = 1.117e-10  # m
    interstitial_sites_per_atom = 6

    # hydrogen reactions - 1 per trap per species
    my_model.reactions = [
        F.Reaction(
            k_0=D_0
            / (interstitial_distance**2 * interstitial_sites_per_atom * w_density),
            E_k=E_D,
            p_0=1e13,
            E_p=0.85,
            volume=w_subdomain,
            reactant=[mobile_D, empty_trap1],
            product=trap1_D,
        ),
        F.Reaction(
            k_0=D_0
            / (interstitial_distance**2 * interstitial_sites_per_atom * w_density),
            E_k=E_D,
            p_0=1e13,
            E_p=0.85,
            volume=w_subdomain,
            reactant=[mobile_T, empty_trap1],
            product=trap1_T,
        ),
        F.Reaction(
            k_0=D_0
            / (interstitial_distance**2 * interstitial_sites_per_atom * w_density),
            E_k=E_D,
            p_0=1e13,
            E_p=1,
            volume=w_subdomain,
            reactant=[mobile_D, empty_trap2],
            product=trap2_D,
        ),
        F.Reaction(
            k_0=D_0
            / (interstitial_distance**2 * interstitial_sites_per_atom * w_density),
            E_k=E_D,
            p_0=1e13,
            E_p=1,
            volume=w_subdomain,
            reactant=[mobile_T, empty_trap2],
            product=trap2_T,
        ),
        # F.Reaction(
        #     k_0=D_0 / (interstitial_distance**2 * interstitial_sites_per_atom * w_density),
        #     E_k=E_D,
        #     p_0=1e13,
        #     E_p=1.5,
        #     volume=w_subdomain,
        #     reactant=[mobile_D, empty_trap3],
        #     product=trap3_D,
        # ),
        # F.Reaction(
        #     k_0=D_0 / (interstitial_distance**2 * interstitial_sites_per_atom * w_density),
        #     E_k=E_D,
        #     p_0=1e13,
        #     E_p=1.5,
        #     volume=w_subdomain,
        #     reactant=[mobile_T, empty_trap3],
        #     product=trap3_T,
        # ),
    ]

    #############   Parameters (K) #############

    my_model.temperature = temperature


    ############# Flux Parameters #############
    
    distribution = gaussian_implantation_ufl(implantation_range, width, thickness = L)
    deuterium_ion_flux, deuterium_atom_flux, tritium_ion_flux, tritium_atom_flux = build_ufl_flux_expression(occurrences)

    my_model.sources = [
    F.ParticleSource(
            value = lambda x,t: deuterium_ion_flux(t) * distribution(x),
            volume = w_subdomain,
            species = mobile_D
        ),
        F.ParticleSource(
            value = lambda x,t: deuterium_atom_flux(t) * distribution(x),
            volume = w_subdomain,
            species = mobile_D
        ),
        F.ParticleSource(
            value = lambda x,t: tritium_ion_flux(t) * distribution(x),
            volume = w_subdomain,
            species = mobile_T
        ),
        F.ParticleSource(
            value = lambda x,t: tritium_atom_flux(t) * distribution(x),
            volume = w_subdomain,
            species = mobile_T
        ),
    ]

    ############ Boundary Conditions #############
    surface_reaction_dd = F.SurfaceReactionBC(
        reactant=[mobile_D, mobile_D],
        gas_pressure=0,
        k_r0=7.94e-17,  # calculated from simplified surface kinetic model with Montupet-Leblond 10.1016/j.nme.2021.101062
        E_kr=-2.0,
        k_d0=0,
        E_kd=0,
        subdomain=inlet,
    )

    surface_reaction_tt = F.SurfaceReactionBC(
        reactant=[mobile_T, mobile_T],
        gas_pressure=0,
        k_r0=7.94e-17,
        E_kr=-2.0,
        k_d0=0,
        E_kd=0,
        subdomain=inlet,
    )

    surface_reaction_dt = F.SurfaceReactionBC(
        reactant=[mobile_D, mobile_T],
        gas_pressure=0,
        k_r0=7.94e-17,
        E_kr=-2.0,
        k_d0=0,
        E_kd=0,
        subdomain=inlet,
    )


    my_model.boundary_conditions = [
        surface_reaction_dd,
        surface_reaction_dt,
        surface_reaction_tt,
        #F.FixedConcentrationBC(subdomain=inlet, value=0.0, species="D"),
        #F.FixedConcentrationBC(subdomain=inlet, value=0.0, species="T"),
    ]

    ############# Exports #############
    if exports:
        my_model.exports = [
            F.VTXSpeciesExport(f"{folder}/mobile_concentration_t.bp", field=mobile_T),
            F.VTXSpeciesExport(f"{folder}/mobile_concentration_d.bp", field=mobile_D),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_d1.bp", field=trap1_D),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_t1.bp", field=trap1_T),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_d2.bp", field=trap2_D),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_t2.bp", field=trap2_T),
            F.VTXTemperatureExport(f"{folder}/temperature.bp"),
            # F.VTXSpeciesExport(f"{folder}/trapped_concentration_d3.bp", field=trap3_D),
            # F.VTXSpeciesExport(f"{folder}/trapped_concentration_t3.bp", field=trap3_T),
        ]

    quantities = {}
    for species in my_model.species:
        quantity = F.TotalVolume(field=species, volume=w_subdomain)
        my_model.exports.append(quantity)
        quantities[species.name] = quantity
        if species.mobile:
            flux = F.SurfaceFlux(field=species, surface=inlet)
            my_model.exports.append(flux)
            quantities[species.name + "_surface_flux"] = flux

    #surface_temperature = F.SurfaceTemperature(my_model.temperature, surface=inlet)
    #my_model.exports.append(surface_temperature)
    #quantities["surface_temperature"] = surface_temperature

    ############# Settings #############
    my_model.settings = CustomSettings(
        atol=1e14,
        rtol=custom_rtol,
        max_iterations=500,  # the first timestep needs about 66 iterations....
        final_time=final_time,
    )

    my_model.settings.stepsize = Stepsize(initial_value=1e-3)
    my_model.settings.linear_solver   = "preonly"  # one direct solve per Newton iteration
    my_model.settings.preconditioner  = "lu"       # LU factorization
    my_model._element_for_traps = "CG"

    return my_model, quantities


def make_B_mb_model_oldBC(
    temperature: Callable | float | int,
    final_time: float,
    folder: str,
    L: float,
    occurrences: List[Dict],
    custom_atol: Union[
        float, Callable
    ] = 1e8,  # default atol unless otherwise specified, used for FP, ICWC, RISP in hisp-for-iter
    custom_rtol: Union[
        float, Callable
    ] = 1e-10,  # default rtol unless otherwise specified, used for FP, ICWC, RISP in hisp-for-iter
    exports=False,
) -> Tuple[CustomProblem, Dict[str, F.TotalVolume]]:
    """Create a FESTIM model for the B MB scenario.

    Args:
        temperature: the temperature in K.
        final_time: the final time in s.
        folder: the folder to save the results.
        L: the length of the domain in m.

    Returns:
        the FESTIM model, the quantities to export.
    """
    my_model = CustomProblem()

    ############# Material Parameters #############

    vertices_graded = graded_vertices(L=L, h0=L/12e9, r=1.008)
    my_model.mesh = F.Mesh1D(vertices_graded)

    # B material parameters from Etienne Hodilles's unpublished TDS study for boron
    b_density = 1.34e29  # atoms/m3
    D_0 = 1.07e-6  # m^2/s
    E_D = 0.3  # eV
    boron = F.Material(
        D_0=D_0,
        E_D=E_D,
        name="boron",
    )

    # mb subdomains
    b_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, L], material=boron)
    inlet = F.SurfaceSubdomain1D(id=1, x=0)
    outlet = F.SurfaceSubdomain1D(id=2, x=L)

    my_model.subdomains = [b_subdomain, inlet, outlet]

    # hydrogen species
    mobile_D = F.Species("D")
    mobile_T = F.Species("T")

    trap1_D = F.Species("trap1_D", mobile=False)
    trap1_T = F.Species("trap1_T", mobile=False)
    trap2_D = F.Species("trap2_D", mobile=False)
    trap2_T = F.Species("trap2_T", mobile=False)
    trap3_D = F.Species("trap3_D", mobile=False)
    trap3_T = F.Species("trap3_T", mobile=False)
    trap4_D = F.Species("trap4_D", mobile=False)
    trap4_T = F.Species("trap4_T", mobile=False)
    # trap5_D = F.Species("trap5_D", mobile=False)
    # trap5_T = F.Species("trap5_T", mobile=False)

    # traps
    empty_trap1 = F.ImplicitSpecies(  # implicit trap 1
        n=6.867e-1
        * b_density,  # from Etienne Hodilles's unpublished TDS study for boron
        others=[trap1_T, trap1_D],
        name="empty_trap1",
    )

    empty_trap2 = F.ImplicitSpecies(  # implicit trap 2
        n=5.214e-1
        * b_density,  # from Etienne Hodilles's unpublished TDS study for boron
        others=[trap2_T, trap2_D],
        name="empty_trap2",
    )

    empty_trap3 = F.ImplicitSpecies(
        n=2.466e-1
        * b_density,  # from Etienne Hodilles's unpublished TDS study for boron
        others=[trap3_T, trap3_D],
        name="empty_trap3",
    )

    empty_trap4 = F.ImplicitSpecies(
        n=1.280e-1
        * b_density,  # from Etienne Hodilles's unpublished TDS study for boron
        others=[trap4_T, trap4_D],
        name="empty_trap4",
    )

    # empty_trap5 = F.ImplicitSpecies(
    #     n=1.800e-1*b_density, # from Johnathan Dufour's unpublished TDS study for boron
    #     others=[trap5_T, trap5_D],
    #     name="empty_trap5",
    # )

    my_model.species = [
        mobile_D,
        mobile_T,
        trap1_D,
        trap1_T,
        trap2_D,
        trap2_T,
        trap3_D,
        trap3_T,
        trap4_D,
        trap4_T,
        # trap5_D,
        # trap5_T,
    ]

    # hydrogen reactions - 1 per trap per species
    interstitial_distance = 8e-10  # m
    interstitial_sites_per_atom = 1

    my_model.reactions = [
        F.Reaction(
            k_0=1e13 / b_density,
            E_k=E_D,
            p_0=1e13,  # from Etienne Hodilles's unpublished TDS study for boron
            E_p=1.052,
            volume=b_subdomain,
            reactant=[mobile_D, empty_trap1],
            product=trap1_D,
        ),
        F.Reaction(
            k_0=1e13 / b_density,
            E_k=E_D,
            p_0=1e13,  # from Etienne Hodilles's unpublished TDS study for boron
            E_p=1.052,
            volume=b_subdomain,
            reactant=[mobile_T, empty_trap1],
            product=trap1_T,
        ),
        F.Reaction(
            k_0=1e13 / b_density,
            E_k=E_D,
            p_0=1e13,  # from Etienne Hodilles's unpublished TDS study for boron
            E_p=1.199,
            volume=b_subdomain,
            reactant=[mobile_D, empty_trap2],
            product=trap2_D,
        ),
        F.Reaction(
            k_0=1e13 / b_density,
            E_k=E_D,
            p_0=1e13,  # from Etienne Hodilles's unpublished TDS study for boron
            E_p=1.199,
            volume=b_subdomain,
            reactant=[mobile_T, empty_trap2],
            product=trap2_T,
        ),
        F.Reaction(
            k_0=1e13 / b_density,
            E_k=E_D,
            p_0=1e13,  # from Etienne Hodilles's unpublished TDS study for boron
            E_p=1.389,
            volume=b_subdomain,
            reactant=[mobile_D, empty_trap3],
            product=trap3_D,
        ),
        F.Reaction(
            k_0=1e13 / b_density,
            E_k=E_D,
            p_0=1e13,  # from Etienne Hodilles's unpublished TDS study for boron
            E_p=1.389,
            volume=b_subdomain,
            reactant=[mobile_T, empty_trap3],
            product=trap3_T,
        ),
        F.Reaction(
            k_0=1e13 / b_density,
            E_k=E_D,
            p_0=1e13,  # from Etienne Hodilles's unpublished TDS study for boron
            E_p=1.589,
            volume=b_subdomain,
            reactant=[mobile_D, empty_trap4],
            product=trap4_D,
        ),
        F.Reaction(
            k_0=1e13 / b_density,
            E_k=E_D,
            p_0=1e13,  # from Etienne Hodilles's unpublished TDS study for boron
            E_p=1.589,
            volume=b_subdomain,
            reactant=[mobile_T, empty_trap4],
            product=trap4_T,
        ),
        # F.Reaction(
        #     k_0=1e13 / b_density,
        #     E_k=E_D,
        #     p_0=1e13,  # from Johnathan Dufour's unpublished TDS study for boron
        #     E_p=1.776,
        #     volume=b_subdomain,
        #     reactant=[mobile_D, empty_trap5],
        #     product=trap5_D,
        # ),
        # F.Reaction(
        #     k_0=1e13 / b_density,
        #     E_k=E_D,
        #     p_0=1e13,  # from Johnathan Dufour's unpublished TDS study for boron
        #     E_p=1.776,
        #     volume=b_subdomain,
        #     reactant=[mobile_T, empty_trap5],
        #     product=trap5_T,
        # ),
    ]

    ############# Temperature Parameters (K) #############

    my_model.temperature = temperature

    ############# Flux Parameters #############

    distribution = gaussian_implantation_ufl(implantation_range, width, thickness = L)
    deuterium_ion_flux, deuterium_atom_flux, tritium_ion_flux, tritium_atom_flux = build_ufl_flux_expression(occurrences)

    my_model.sources = [
    F.ParticleSource(
            value = lambda x,t: deuterium_ion_flux(t) * distribution(x),
            volume = b_subdomain,
            species = mobile_D
        ),
        F.ParticleSource(
            value = lambda x,t: deuterium_atom_flux(t) * distribution(x),
            volume = b_subdomain,
            species = mobile_D
        ),
        F.ParticleSource(
            value = lambda x,t: tritium_ion_flux(t) * distribution(x),
            volume = b_subdomain,
            species = mobile_T
        ),
        F.ParticleSource(
            value = lambda x,t: tritium_atom_flux(t) * distribution(x),
            volume = b_subdomain,
            species = mobile_T
        ),
    ]

    ############# Boundary Conditions #############
    my_model.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=inlet, value=0.0, species="D"),
        F.FixedConcentrationBC(subdomain=inlet, value=0.0, species="T"),
        F.ParticleFluxBC(subdomain=outlet, value=0.0, species="D"),
        F.ParticleFluxBC(subdomain=outlet, value=0.0, species="T"),
    ]

    ############# Exports #############
    if exports:
        my_model.exports = [
            F.VTXSpeciesExport(f"{folder}/mobile_concentration_t.bp", field=mobile_T),
            F.VTXSpeciesExport(f"{folder}/mobile_concentration_d.bp", field=mobile_D),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_d1.bp", field=trap1_D),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_t1.bp", field=trap1_T),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_d2.bp", field=trap2_D),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_t2.bp", field=trap2_T),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_d3.bp", field=trap3_D),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_t3.bp", field=trap3_T),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_d4.bp", field=trap4_D),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_t4.bp", field=trap4_T),
            F.VTXTemperatureExport(f"{folder}/temperature.bp"),
            # F.VTXSpeciesExport(f"{folder}/trapped_concentration_d5.bp", field=trap5_D),
            # F.VTXSpeciesExport(f"{folder}/trapped_concentration_t5.bp", field=trap5_T),
        ]

    quantities = {}
    for species in my_model.species:
        quantity = F.TotalVolume(field=species, volume=b_subdomain)
        my_model.exports.append(quantity)
        quantities[species.name] = quantity
        if species.mobile:
            flux = F.SurfaceFlux(field=species, surface=inlet)
            my_model.exports.append(flux)
            quantities[species.name + "_surface_flux"] = flux

    #surface_temperature = F.SurfaceTemperature(my_model.temperature, surface=inlet)
    #my_model.exports.append(surface_temperature)
    #quantities["surface_temperature"] = surface_temperature

    ############# Settings #############
    my_model.settings = CustomSettings(
        atol=custom_atol,
        rtol=custom_rtol,
        max_iterations=100,
        final_time=final_time,
    )

    my_model.settings.stepsize = Stepsize(initial_value=1e-4)
    my_model.settings.linear_solver   = "preonly"  # one direct solve per Newton iteration
    my_model.settings.preconditioner  = "lu"       # LU factorization
    my_model._element_for_traps = "CG"
    return my_model, quantities


def make_DFW_mb_model_oldBC(
    temperature: Callable | float | int,
    final_time: float,
    folder: str,
    L: float,
    occurrences: List[Dict],
    exports=False,
) -> Tuple[CustomProblem, Dict[str, F.TotalVolume]]:
    """Create a FESTIM model for the DFW MB scenario.

    Args:
        temperature: the temperature in K.
        deuterium_ion_flux: the deuterium ion flux in m^-2 s^-1.
        tritium_ion_flux: the tritium ion flux in m^-2 s^-1.
        deuterium_atom_flux: the deuterium atom flux in m^-2 s^-1.
        tritium_atom_flux: the tritium atom flux in m^-2 s^-1.
        final_time: the final time in s.
        folder: the folder to save the results.
        L: the length of the domain in m.

    Returns:
        the FESTIM model, the quantities to export.
    """

    my_model = CustomProblem()


    ############# Material Parameters #############

    vertices = np.concatenate(  # 1D mesh with extra refinement
        [
            np.linspace(0, 30e-9, num=200),
            np.linspace(30e-9, 3e-6, num=300),
            np.linspace(3e-6, 30e-6, num=300),
            np.linspace(30e-6, 1e-4, num=300),
            np.linspace(1e-4, L, num=200),
        ]
    )
    my_model.mesh = F.Mesh1D(vertices)

    # TODO: pull DFW material parameters from HTM?

    # from ITER mean value parameters (FIXME: add DOI)
    ss_density = 8.45e28  # atoms/m3
    D_0 = 1.45e-6  # m^2/s
    E_D = 0.59  # eV
    ss = F.Material(
        D_0=D_0,
        E_D=E_D,
        name="ss",
    )

    # mb subdomains
    ss_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, L], material=ss)
    inlet = F.SurfaceSubdomain1D(id=1, x=0)
    outlet = F.SurfaceSubdomain1D(id=2, x=L)

    my_model.subdomains = [ss_subdomain, inlet, outlet]

    # hydrogen species
    mobile_D = F.Species("D")
    mobile_T = F.Species("T")

    trap1_D = F.Species("trap1_D", mobile=False)
    trap1_T = F.Species("trap1_T", mobile=False)

    # traps
    empty_trap1 = F.ImplicitSpecies(  # implicit trap 1
        n=8e-2 * ss_density,  # from Guillermain D 2016 ITER report T2YEND
        others=[trap1_T, trap1_D],
        name="empty_trap1",
    )

    my_model.species = [
        mobile_D,
        mobile_T,
        trap1_D,
        trap1_T,
    ]

    # hydrogen reactions - 1 per trap per species
    interstitial_distance = 2.545e-10  # m
    interstitial_sites_per_atom = 1

    my_model.reactions = [
        F.Reaction(
            k_0=D_0
            / (interstitial_distance * interstitial_sites_per_atom * ss_density),
            E_k=E_D,
            p_0=1e13,  # from Guillermain D 2016 ITER report T2YEND
            E_p=0.7,
            volume=ss_subdomain,
            reactant=[mobile_D, empty_trap1],
            product=trap1_D,
        ),
        F.Reaction(
            k_0=D_0
            / (interstitial_distance * interstitial_sites_per_atom * ss_density),
            E_k=E_D,
            p_0=1e13,  # from Guillermain D 2016 ITER report T2YEND
            E_p=0.7,
            volume=ss_subdomain,
            reactant=[mobile_T, empty_trap1],
            product=trap1_T,
        ),
    ]

    ############# Temperature Parameters (K) #############

    my_model.temperature = temperature

    ############# Flux Parameters #############

    distribution = gaussian_implantation_ufl(implantation_range, width, thickness = L)
    deuterium_ion_flux, deuterium_atom_flux, tritium_ion_flux, tritium_atom_flux = build_ufl_flux_expression(occurrences)

    my_model.sources = [
        PulsedSource(
            flux=deuterium_ion_flux,
            value = lambda x,t: deuterium_ion_flux(t) * distribution(x),
            species=mobile_D,
            volume=ss_subdomain,
        ),
        PulsedSource(
            flux=tritium_ion_flux,
            value = lambda x,t: tritium_ion_flux(t) * distribution(x),
            species=mobile_T,
            volume=ss_subdomain,
        ),
        PulsedSource(
            flux=deuterium_atom_flux,
            value=lambda x,t: deuterium_atom_flux(t) * distribution(x),
            species=mobile_D,
            volume=ss_subdomain,
        ),
        PulsedSource(
            flux=tritium_atom_flux,
            value=lambda x,t: tritium_atom_flux(t) * distribution(x),
            species=mobile_T,
            volume=ss_subdomain,
        ),
    ]

    ############# Boundary Conditions #############
    surface_reaction_dd = F.SurfaceReactionBC(
        reactant=[mobile_D, mobile_D],
        gas_pressure=0,
        k_r0=1.75e-24,
        E_kr=-0.594,
        k_d0=0,
        E_kd=0,
        subdomain=inlet,
    )

    surface_reaction_tt = F.SurfaceReactionBC(
        reactant=[mobile_T, mobile_T],
        gas_pressure=0,
        k_r0=1.75e-24,
        E_kr=-0.594,
        k_d0=0,
        E_kd=0,
        subdomain=inlet,
    )

    surface_reaction_dt = F.SurfaceReactionBC(
        reactant=[mobile_D, mobile_T],
        gas_pressure=0,
        k_r0=1.75e-24,
        E_kr=-0.594,
        k_d0=0,
        E_kd=0,
        subdomain=inlet,
    )

    k_r0=1.75e-24
    E_kr=-0.594


    my_model.boundary_conditions = [
        surface_reaction_dd,
        surface_reaction_dt,
        surface_reaction_tt,
    ]

    ############# Exports #############
    if exports:
        my_model.exports = [
            F.VTXSpeciesExport(f"{folder}/mobile_concentration_t.bp", field=mobile_T),
            F.VTXSpeciesExport(f"{folder}/mobile_concentration_d.bp", field=mobile_D),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_d1.bp", field=trap1_D),
            F.VTXSpeciesExport(f"{folder}/trapped_concentration_t1.bp", field=trap1_T),
            F.VTXTemperatureExport(f"{folder}/temperature.bp"),
        ]

    quantities = {}
    for species in my_model.species:
        quantity = F.TotalVolume(field=species, volume=ss_subdomain)
        my_model.exports.append(quantity)
        quantities[species.name] = quantity
        if species.mobile:
            flux = F.SurfaceFlux(field=species, surface=inlet)
            my_model.exports.append(flux)
            quantities[species.name + "_surface_flux"] = flux

    surface_temperature = F.SurfaceTemperature(my_model.temperature, surface=inlet)
    my_model.exports.append(surface_temperature)
    quantities["surface_temperature"] = surface_temperature

    ############# Settings #############
    my_model.settings = F.Settings(
        atol=1e5,
        rtol=1e-10,
        max_iterations=30,
        final_time=final_time,
    )

    my_model.settings.stepsize = Stepsize(initial_value=1e-3)
    my_model.settings.linear_solver   = "preonly"  # one direct solve per Newton iteration
    my_model.settings.preconditioner  = "lu"       # LU factorization
    my_model._element_for_traps = "CG"
    return my_model, quantities


# calculate how the rear temperature of the W layer evolves with the surface temperature
# data from E.A. Hodille et al 2021 Nucl. Fusion 61 126003 10.1088/1741-4326/ac2abc (Table I)
heat_fluxes_hodille = [10e6, 5e6, 1e6]  # W/m2
T_rears_hodille = [552, 436, 347]  # K

import scipy.stats

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
        heat_flux: heat flux in W/m2
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
    bin: hisp.bin.SubBin | hisp.bin.DivBin,
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
        assert isinstance(t, float), f"t should be a float, not {type(t)}"

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
            if (
                bin.material == "W" or bin.material == "SS"
            ):  # FIXME: update ss temp when gven data:
                value = calculate_temperature_W(
                    x[0], heat_flux, coolant_temp, bin.thickness, bin.copper_thickness
                )
            elif bin.material == "B":
                T_value = calculate_temperature_B(heat_flux, coolant_temp)
                value = np.full_like(x[0], T_value)
            else:
                raise ValueError(f"Unsupported material: {bin.material}")

        return value

    return T_function


def make_particle_flux_function(
    scenario: Scenario,
    plasma_data_handling: PlasmaDataHandling,
    bin: hisp.bin.SubBin | hisp.bin.DivBin,
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
        assert isinstance(t, float), f"t should be a float, not {type(t)}"

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



kB_J   = 1.380649e-23      # J/K
eV_to_J = 1.602176634e-19  # J/eV

def make_surface_concentration_time_function_J(T_fun, flux_fun, D0, E_eV, R_p, surface_x=0.0):
    x_surf = np.array([[float(surface_x)]])
    E_J    = float(E_eV) * eV_to_J

    def c_S(t):
        t = float(t)
        T_surf = float(T_fun(x_surf, t)[0])
        phi    = float(flux_fun(t))
        val    = (phi * float(R_p)) / (D0 * np.exp(-E_J / (kB_J * T_surf)))
        return float(val)
    return c_S

def _as_time_function(x):
    """Return a callable f(t) that yields a float, whether x is a scalar or a callable."""
    if callable(x):
        return lambda t: float(x(float(t)))
    else:
        # treat x as a constant fraction
        c = float(x)
        return lambda t: c

def make_D_surface_concentration_SS(T_fun, flux_fun, T_frac, D0, kr0, E_D, E_k, R_p, surface_x=0.0):
    x_surf = np.array([[float(surface_x)]])
    E_DJ = float(E_D)*eV_to_J
    E_kJ = float(E_k)*eV_to_J
    T_frac_t = _as_time_function(T_frac)
    def c_SD_SS(t):
        t = float(t)
        T_surf = float(T_fun(x_surf, t)[0])
        D_T   = D0 * np.exp(-E_DJ / (kB_J * T_surf))
        K_T   = kr0 * np.exp(-E_kJ / (kB_J * T_surf))
        f = 1-float(T_frac_t(t))
        z = (1-f)/f
        phi = float(flux_fun(t))    
        c_sD_SS_val = np.sqrt(phi*(7+z-np.sqrt(1 + 14*z + z**2))/12.0/K_T)
        return float(phi * R_p / D_T + c_sD_SS_val)
    return c_SD_SS

def make_T_surface_concentration_SS(T_fun, flux_fun, T_frac, D0, kr0, E_D, E_k, R_p, surface_x=0.0):
    x_surf = np.array([[float(surface_x)]])
    E_DJ = float(E_D)*eV_to_J
    E_kJ = float(E_k)*eV_to_J
    T_frac_t = _as_time_function(T_frac)
    def c_ST_SS(t):
        t = float(t)
        T_surf = float(T_fun(x_surf, t)[0])
        D_T   = D0 * np.exp(-E_DJ / (kB_J * T_surf))
        K_T   = kr0 * np.exp(-E_kJ / (kB_J * T_surf))
        f = 1-float(T_frac_t(t))
        z = f/(1-f)
        phi = float(flux_fun(t))    
        c_sT_SS_val = np.sqrt(phi*(7+z-np.sqrt(1 + 14*z + z**2))/12.0/K_T)
        return float( phi * R_p / D_T + c_sT_SS_val)
    return c_ST_SS


def build_pulse_info_array(scenario):
    """
    Returns an array of arrays with pulse info:
    [pulse_number, type, tritium_fraction, ramp_up, steady_state, ramp_down, waiting_time]
    """
    info_array = []
    pulse_number = 0
    for pulse in scenario.pulses:
        for _ in range(pulse.nb_pulses):
            info_array.append([
                pulse_number,
                pulse.pulse_type,
                pulse.tritium_fraction,
                pulse.ramp_up,
                pulse.steady_state,
                pulse.ramp_down,
                pulse.waiting
            ])
            pulse_number += 1
    return info_array


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

