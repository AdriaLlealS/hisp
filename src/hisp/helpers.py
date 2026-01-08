import festim as F
from dolfinx.fem.function import Constant
import ufl
import numpy as np
import numpy.typing as npt
from hisp.scenario import Pulse
from dolfinx import fem
import math
from festim import XDMFExport


class PulsedSource(F.ParticleSource):
    def __init__(self, flux, distribution, volume, species):
        """Initalizes flux and distribution for PulsedSource.

        Args:
            flux (callable): the input flux value from DINA data
            distribution (function of x): distribution of flux throughout mb
            volume (F.VolumeSubdomain1D): volume where this flux is imposed
            species (F.species): species of flux (e.g. D/T)

        Returns:
            flux and distribution of species.
        """
        self.flux = flux
        self.distribution = distribution
        super().__init__(None, volume, species)

    @property
    def time_dependent(self):
        return True

    def create_value_fenics(self, mesh, temperature, t: Constant):
        self.flux_fenics = F.as_fenics_constant(self.flux(t.value), mesh)
        x = ufl.SpatialCoordinate(mesh)
        self.distribution_fenics = self.distribution(x)

        self.value_fenics = self.flux_fenics * self.distribution_fenics

    def update(self, t: float):
        self.flux_fenics.value = self.flux(t)

# we override Stepsize to control the precision of milestones detection
# TODO remove this when https://github.com/festim-dev/FESTIM/issues/933 is fixed
class Stepsize(F.Stepsize):
    def modify_value(self, value, nb_iterations, t=None):
        if not self.is_adapt(t):
            return value

        if nb_iterations < self.target_nb_iterations:
            updated_value = value * self.growth_factor
        elif nb_iterations > self.target_nb_iterations:
            updated_value = value * self.cutback_factor
        else:
            updated_value = value

        if max_step := self.get_max_stepsize(t):
            if updated_value > max_step:
                updated_value = max_step

        next_milestone = self.next_milestone(t)
        if next_milestone is not None:
            time_to_milestone = next_milestone - t
            if updated_value > time_to_milestone and not np.isclose(
                t, next_milestone, atol=0.0001, rtol=0
            ):
                updated_value = time_to_milestone

        return updated_value

def gaussian_distribution(
    x: npt.NDArray, mean: float, width: float, mod=ufl
) -> ufl.core.expr.Expr:
    """Generates a gaussian distribution for particle sources.

    Args:
        x (npt.NDArray): x values along the length of given bin.
        mean (float): Mean of the distribution.
        width (float): Width of the gaussian distribution.
        mod (_type_, optional): Module used to express gaussian distribution. Defaults to ufl.

    Returns:
        ufl.core.expr.Expr: Gaussian distribution with area 1.  
    """
    return mod.exp(-((x[0] - mean) ** 2) / (2 * width**2)) / (
        np.sqrt(2 * np.pi * width**2)
    )


def periodic_step_function(x, period_on, period_total, value, value_off=0.0):
    """
    Creates a periodic step function with two periods.
    """

    if period_total < period_on:
        raise ValueError("period_total must be greater than period_on")

    if x % period_total < period_on:
        return value
    else:
        return value_off
    
def periodic_pulse_function(current_time: float, pulse: Pulse, value, value_off=343.0):
    """Creates bake function with ramp up rate and ramp down rate.

    Args:
        current_time (float): time within the pulse 
        pulse (Pulse): pulse of HISP Pulse class
        value (float): steady-state value 
        value_off (float): value at t=0 and t=final time. 
    """
    
    if current_time == pulse.total_duration:
        return value_off
    elif current_time % pulse.total_duration < pulse.ramp_up: # ramp up 
        return (value - value_off) / (pulse.ramp_up) * current_time + value_off # y = mx + b, slope is temp/ramp up time
    elif current_time % pulse.total_duration < pulse.ramp_up + pulse.steady_state: # steady state
        return value
    else: # ramp down, waiting
        lower_value = value - (value - value_off)/pulse.ramp_down * (current_time - (pulse.ramp_up + pulse.steady_state)) # y = mx + b, slope is temp/ramp down time
        if lower_value >= value_off: 
            return lower_value
        else: 
            return value_off

class XDMFExportEveryDt(XDMFExport):
    """
    Write to XDMF only if enough time has elapsed since the last write.
    Uses min_dt1 for t <= switch and min_dt2 for t > switch.

    Parameters
    ----------
    filename : str
        Path for the XDMF file(s).
    field : str or festim Field
        What to export (same as base XDMFExport).
    min_dt1 : float
        Minimum time spacing before `switch` (inclusive).
    min_dt2 : float
        Minimum time spacing after `switch` (strictly greater).
    switch : float
        Time at which the cadence changes from min_dt1 to min_dt2.
    atol : float, optional
        Small tolerance to account for floating point accumulation.
        Default is 0.0 (set e.g. 1e-12 if needed).
    """
    def __init__(self, filename, field, min_dt1: float, min_dt2: float, switch: float, atol: float = 0.0):
        super().__init__(filename, field)
        self._min_dt1 = float(min_dt1)
        self._min_dt2 = float(min_dt2)
        self._switch = float(switch)
        self._atol = float(atol)
        self._last_t = None

    def _current_min_dt(self, t: float) -> float:
        return self._min_dt2 if t > self._switch else self._min_dt1

    def write(self, t: float):
        t = float(t)
        min_dt = self._current_min_dt(t)

        if (self._last_t is None) or ((t - self._last_t) >= (min_dt - self._atol)):
            super().write(t)
            self._last_t = t

def gaussian_implantation_ufl(Rp, sigma, axis=0, thickness=None):
    """
    Returns callable value(x, t) -> UFL expression S(x,t) [m^-3 s^-1]
    - Rp, sigma in meters
    - axis in {0,1,2} selects x[axis] as depth coordinate
    - If thickness is not None (meters), renormalize over [0, thickness] to conserve J(t)
    """
    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
    if thickness is None:
        C = 0.5 * (1.0 + erf(Rp / (sigma * sqrt(2.0))))  # Gaussian mass in [0, +inf)
        C = max(C, 1e-12)  # numerical safeguard
        norm = inv_sqrt_2pi / sigma
        def value(x):
            xi = x[axis]
            z  = (xi - Rp) / sigma
            return norm * ufl.exp(-0.5 * z * z)
        return value
    else:
        # Renormalize over [0, thickness]
        from math import erf, sqrt
        a = (0.0 - Rp) / (sigma * sqrt(2.0))
        b = (thickness - Rp) / (sigma * sqrt(2.0))
        C = max(0.5 * (erf(b) - erf(a)), 1e-12)        # in-domain Gaussian mass
        norm = (inv_sqrt_2pi / sigma) / C
        def value(x):
            xi = x[axis]
            z  = (xi - Rp) / sigma
            return norm * ufl.exp(-0.5 * z * z)
        return value
    
def periodic_pulse_ufl(t, pulse, value, value_off=343.0):
    """
    UFL symbolic version of periodic_pulse_function.
    Args:
        t: UFL expression (time)
        pulse: Pulse object with ramp_up, steady_state, ramp_down, waiting
        value: UFL expression or Constant (steady-state value)
        value_off: float or UFL Constant (off value)
    Returns:
        UFL expression representing the piecewise ramp profile.
    """

    # Compute relative time within one pulse cycle (no modulo in UFL)
    tau = t  # Assume t is within [0, pulse.total_duration] for now

    # Conditions for each phase
    within_up = ufl.lt(tau, pulse.ramp_up)
    within_steady = ufl.And(ufl.ge(tau, pulse.ramp_up),
                             ufl.lt(tau, pulse.ramp_up + pulse.steady_state))
    within_down = ufl.And(ufl.ge(tau, pulse.ramp_up + pulse.steady_state),
                          ufl.lt(tau, pulse.ramp_up + pulse.steady_state + pulse.ramp_down))
    # Waiting phase: tau >= ramp_up + steady_state + ramp_down

    # Ramp-up: linear interpolation
    up_val = (value - value_off) / pulse.ramp_up * tau + value_off

    # Ramp-down: linear decrease
    down_val = value - (value - value_off) / pulse.ramp_down * (tau - (pulse.ramp_up + pulse.steady_state))
    down_val = ufl.conditional(ufl.ge(down_val, value_off), down_val, value_off)

    # Piecewise conditional chain
    shape = ufl.conditional(
        within_up, up_val,
        ufl.conditional(
            within_steady, value,
            ufl.conditional(within_down, down_val, value_off)
        )
    )

    return shape

def make_ufl_flux_function(scalar_flux_function):
    """
    Convert a scalar flux function to a UFL expression function.
    
    Args:
        scalar_flux_function: A function that takes time (float) and returns flux value (float)
        
    Returns:
        A function that takes time (UFL expression) and returns UFL expression
    """
    def ufl_flux(t):
        # For time-dependent behavior, we need to create conditional expressions
        # This is a simplified version - for complex time dependencies,
        # you might need to sample at specific times and interpolate
        return ufl.Constant(1.0)  # Placeholder - this needs proper implementation
    
    return ufl_flux

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