import festim as F

import dolfinx.fem as fem
import numpy as np


def is_it_time_to_export(current_time, times):
    """Local fallback for festim.helpers.is_it_time_to_export
    (not available in older FESTIM versions)."""
    for t in times:
        if np.isclose(t, current_time, atol=0, rtol=1.0e-5):
            return True
    return False
import ufl
import basix


class CustomProblem(F.HydrogenTransportProblem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._remaining_profile_export_times = None  # populated from export.times on first call

    def _is_it_time_to_export_profile(self, current_time, times):
        """Check whether current_time matches an outstanding profile export time.

        On first call, builds a sorted auxiliary list from *times* so that
        consuming entries does not mutate the original export.times list.
        When a match is found the entry is removed from the auxiliary list
        so it cannot trigger a second export.
        """
        # lazily build the auxiliary list (copy so original is untouched)
        if self._remaining_profile_export_times is None:
            self._remaining_profile_export_times = sorted(list(times)) if times else []

        # walk through remaining times and pop the first match
        for i, t_export in enumerate(self._remaining_profile_export_times):
            if np.isclose(t_export, current_time, atol=0, rtol=1.0e-5):
                self._remaining_profile_export_times.pop(i)
                return True

        return False

    # NOTE: post_processing override removed for older FESTIM compatibility.
    # The override was only needed for Profile1DExport custom timing.
    # When upgrading FESTIM, restore the override from git history.

    def define_temperature(self):
        # check if temperature is None
        if self.temperature is None:
            raise ValueError("the temperature attribute needs to be defined")

        # if temperature is a float or int, create a fem.Constant
        elif isinstance(self.temperature, (float, int)):
            self.temperature_fenics = F.as_fenics_constant(
                self.temperature, self.mesh.mesh
            )
        # if temperature is a fem.Constant or function, pass it to temperature_fenics
        elif isinstance(self.temperature, (fem.Constant, fem.Function)):
            self.temperature_fenics = self.temperature

        # if temperature is callable, process accordingly
        elif callable(self.temperature):
            arguments = self.temperature.__code__.co_varnames
            if "t" in arguments and "x" not in arguments:
                if not isinstance(self.temperature(t=float(self.t)), (float, int)):
                    raise ValueError(
                        f"self.temperature should return a float or an int, not {type(self.temperature(t=float(self.t)))} "
                    )
                # only t is an argument
                self.temperature_fenics = F.as_fenics_constant(
                    mesh=self.mesh.mesh, value=self.temperature(t=float(self.t))
                )
            else:
                degree = 1
                element_temperature = basix.ufl.element(
                    basix.ElementFamily.P,
                    self.mesh.mesh.basix_cell(),
                    degree,
                    basix.LagrangeVariant.equispaced,
                )
                function_space_temperature = fem.functionspace(
                    self.mesh.mesh, element_temperature
                )
                self.temperature_fenics = fem.Function(function_space_temperature)
                self.temperature_fenics.interpolate(
                    lambda x: self.temperature(x, float(self.t))
                )

    def update_time_dependent_values(self):

        # this is for the last time step, don't update the fluxes to avoid overshoot in the scenario file
        if float(self.t) > self.settings.final_time:
            return

        F.ProblemBase.update_time_dependent_values(self)

        if not self.temperature_time_dependent:
            return

        t = float(self.t)

        if isinstance(self.temperature_fenics, fem.Constant):
            self.temperature_fenics.value = self.temperature(t=t)
        elif isinstance(self.temperature_fenics, fem.Function):
            self.temperature_fenics.interpolate(
                lambda x: self.temperature(x, float(self.t))
            )

        for bc in self.boundary_conditions:
            if isinstance(bc, (F.FixedConcentrationBC, F.ParticleFluxBC, F.SurfaceReactionBC)):
                if hasattr(bc, 'temperature_dependent') and bc.temperature_dependent:
                    bc.update(t=t)

        for source in self.sources:
            if hasattr(source, 'temperature_dependent') and source.temperature_dependent:
                source.update(t=t)
