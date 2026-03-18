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

    def post_processing(self):
        """Override post_processing to use custom export timing for Profile1DExport."""
        self.update_post_processing_solutions()

        if self.temperature_time_dependent:
            species_not_updated = self.species.copy()
            for export in self.exports:
                if isinstance(export, F.SurfaceFlux):
                    if export.field in species_not_updated:
                        export.D.interpolate(export.D_expr)
                        species_not_updated.remove(export.field)

        # Determine once per timestep whether profiles should be exported.
        # Using the first Profile1DExport with times as the representative check.
        # This ensures ALL species profiles are exported together (or not at all).
        _export_profiles_this_step = False
        for export in self.exports:
            if isinstance(export, F.Profile1DExport) and hasattr(export, "times"):
                if self._is_it_time_to_export_profile(
                    current_time=float(self.t), times=export.times
                ):
                    _export_profiles_this_step = True
                break  # one check is enough; all profiles share the same timing

        for export in self.exports:
            # For Profile1DExport, use the pre-computed per-step decision
            if isinstance(export, F.Profile1DExport):
                if hasattr(export, "times"):
                    if not _export_profiles_this_step:
                        continue
                # compute profile
                if export._dofs is None:
                    index = self.species.index(export.field)
                    V0, export._dofs = self.u.function_space.sub(index).collapse()
                    coords = V0.tabulate_dof_coordinates()[:, 0]
                    export._sort_coords = np.argsort(coords)
                    export.x = coords[export._sort_coords]

                c = self.u.x.array[export._dofs][export._sort_coords]
                export.data.append(c)
                export.t.append(float(self.t))
                continue

            # For all other exports, use standard is_it_time_to_export
            if hasattr(export, "times"):
                if not is_it_time_to_export(
                    current_time=float(self.t), times=export.times
                ):
                    continue

            if isinstance(export, F.exports.ExportBaseClass):
                if isinstance(export, F.exports.VTXSpeciesExport):
                    if export._checkpoint:
                        import adios4dolfinx
                        for field in export.field:
                            adios4dolfinx.write_function(
                                export.filename,
                                field.post_processing_solution,
                                time=float(self.t),
                                name=field.name,
                            )
                    else:
                        export.writer.write(float(self.t))
                elif (
                    isinstance(export, F.VTXTemperatureExport)
                    and self.temperature_time_dependent
                ):
                    self._temperature_as_function.interpolate(
                        self._get_temperature_field_as_function()
                    )
                    export.writer.write(float(self.t))

            if isinstance(export, F.exports.SurfaceQuantity):
                if isinstance(
                    export,
                    F.exports.SurfaceFlux | F.exports.TotalSurface | F.exports.AverageSurface,
                ):
                    export.compute(export.field.solution, self.ds)
                else:
                    export.compute()
                export.t.append(float(self.t))
                if export.filename is not None:
                    export.write(t=float(self.t))
            elif isinstance(export, F.exports.VolumeQuantity):
                if isinstance(export, F.exports.TotalVolume | F.exports.AverageVolume):
                    export.compute(u=export.field.solution, dx=self.dx)
                else:
                    export.compute()
                export.t.append(float(self.t))
                if export.filename is not None:
                    export.write(t=float(self.t))
            if isinstance(export, F.exports.XDMFExport):
                export.write(float(self.t))

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
