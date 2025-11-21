import festim as F
from dolfinx import fem
from dolfinx.fem import Constant
import ufl
import numpy as np
import math
from hisp.scenario import Pulse


class PulsedSource(F.ParticleSource):
    def __init__(self, flux, distribution, volume, species):
        """
        Initializes flux and distribution for PulsedSource.

        Args:
            flux (callable): time-dependent flux function (m^-2 s^-1)
            distribution (function): spatial distribution function returning UFL expr
            volume (F.VolumeSubdomain1D): volume where flux is imposed
            species (F.Species): species of flux (e.g., D/T)
        """
        self.flux = flux
        self.distribution = distribution
        super().__init__(None, volume, species)

    @property
    def time_dependent(self):
        return True

    def create_value_fenics(self, mesh, temperature, t: Constant):
        # Compute flux value safely
        flux_value = self.flux(float(t))
        if flux_value is None or math.isnan(flux_value):
            flux_value = 0.0

        # Create a proper DOLFINx Constant
        self.flux_fenics = fem.Constant(mesh, np.array(float(flux_value), dtype=fem.ScalarType))

        # Build UFL distribution
        x = ufl.SpatialCoordinate(mesh)
        self.distribution_fenics = self.distribution(x)
        if self.distribution_fenics is None:
            raise RuntimeError("distribution(x) returned None")

        # Combine flux and distribution into a UFL expression
        self.value_fenics = self.flux_fenics * self.distribution_fenics

    def update(self, t: float):
        flux_value = self.flux(t)
        if flux_value is None or math.isnan(flux_value):
            flux_value = 0.0
        self.flux_fenics.value = np.array(float(flux_value), dtype=fem.ScalarType)


# Override Stepsize for FESTIM milestone precision
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


# ✅ UFL-compliant Gaussian distribution
def gaussian_distribution(x: ufl.SpatialCoordinate, mean: float, width: float) -> ufl.core.expr.Expr:
    """Generates a normalized Gaussian distribution for particle sources."""
    normalization = 1.0 / ufl.sqrt(2 * ufl.pi * width**2)
    return normalization * ufl.exp(-((x[0] - mean)**2) / (2 * width**2))


def periodic_step_function(x, period_on, period_total, value, value_off=0.0):
    """Creates a periodic step function with two periods."""
    if period_total < period_on:
        raise ValueError("period_total must be greater than period_on")
    return value if (x % period_total) < period_on else value_off


def periodic_pulse_function(current_time: float, pulse: Pulse, value, value_off=343.0):
    """Creates bake function with ramp up/down and steady state."""
    if current_time == pulse.total_duration:
        return value_off
    elif current_time % pulse.total_duration < pulse.ramp_up:  # ramp up
        return (value - value_off) / pulse.ramp_up * current_time + value_off
    elif current_time % pulse.total_duration < pulse.ramp_up + pulse.steady_state:  # steady state
        return value
    else:  # ramp down
        lower_value = value - (value - value_off) / pulse.ramp_down * (
            current_time - (pulse.ramp_up + pulse.steady_state)
        )
        return lower_value if lower_value >= value_off else value_off