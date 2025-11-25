
"""
Simplified HISP Model for PFC runs
- Pass only two r_tol values: one for B and one for W (both = 10e-10)
- Cap the adaptive stepsize to a constant 100000.0 seconds for every case

This file is designed to be a drop-in replacement for model(2).txt.
"""
from typing import List
import numpy as np
import ufl

# NOTE: fixed import name (plasma_data_handling)
from hisp.plamsa_data_handling import PlasmaDataHandling
from hisp.scenario import Scenario
from hisp.bin import Reactor, SubBin, DivBin
from hisp.helpers import periodic_step_function
from hisp.festim_models import (
    make_W_mb_model,
    make_B_mb_model,
    make_DFW_mb_model,
    make_temperature_function,
    make_particle_flux_function,
)


class Model:
    """
    HISP main model wrapper used by the PFC driver.
    Modifications vs. the original model(2).txt:
      • custom_rtol is a constant numeric value, depending on material
        - W → 10e-10
        - B → 10e-10
        - SS → 10e-10 (for completeness)
      • stepsize cap is a constant 100000.0 s for all bins and times
    """

    def __init__(
        self,
        reactor: Reactor,
        scenario: Scenario,
        plasma_data_handling: PlasmaDataHandling,
        coolant_temp: float = 343.0,
    ) -> None:
        self.reactor = reactor
        self.scenario = scenario
        self.plasma_data_handling = plasma_data_handling
        self.coolant_temp = coolant_temp

    # ----------------------- public API used by the runner -----------------------
    def run_bin(self, bin: SubBin | DivBin):
        """Build and run a FESTIM model for a given bin or subbin."""
        # Build FESTIM model from HISP inputs
        my_model, quantities = self.which_model(bin)

        # ---- Adaptivity & milestones ----
        milestones = self.make_milestones(
            initial_stepsize_value=my_model.settings.stepsize.initial_value
        )
        milestones.append(my_model.settings.final_time)
        my_model.settings.stepsize.milestones = milestones

        # Adaptivity knobs (unchanged)
        my_model.settings.stepsize.growth_factor = 1.2
        my_model.settings.stepsize.cutback_factor = 0.9
        my_model.settings.stepsize.target_nb_iterations = 4

        # ---- Constant stepsize cap: 100000 s everywhere ----
        my_model.settings.stepsize.max_stepsize = self.constant_max_stepsize

        # Run
        my_model.initialise()
        my_model.run()
        return my_model, quantities

    def run_all_bins(self):
        """(Optional) iterate over all bins of the reactor if needed."""
        raise NotImplementedError

    # ----------------------- model construction -----------------------
    def which_model(self, bin: SubBin | DivBin):
        """Return a (FESTIM model, quantities) pair for the provided bin/subbin."""
        # Temperature & flux functions from HISP + scenario
        temperature_function = make_temperature_function(
            scenario=self.scenario,
            plasma_data_handling=self.plasma_data_handling,
            bin=bin,
            coolant_temp=self.coolant_temp,
        )
        d_ion_incident_flux = make_particle_flux_function(
            scenario=self.scenario,
            plasma_data_handling=self.plasma_data_handling,
            bin=bin,
            ion=True,
            tritium=False,
        )
        tritium_ion_flux = make_particle_flux_function(
            scenario=self.scenario,
            plasma_data_handling=self.plasma_data_handling,
            bin=bin,
            ion=True,
            tritium=True,
        )
        deuterium_atom_flux = make_particle_flux_function(
            scenario=self.scenario,
            plasma_data_handling=self.plasma_data_handling,
            bin=bin,
            ion=False,
            tritium=False,
        )
        tritium_atom_flux = make_particle_flux_function(
            scenario=self.scenario,
            plasma_data_handling=self.plasma_data_handling,
            bin=bin,
            ion=False,
            tritium=True,
        )

        common_args = {
            "deuterium_ion_flux": d_ion_incident_flux,
            "tritium_ion_flux": tritium_ion_flux,
            "deuterium_atom_flux": deuterium_atom_flux,
            "tritium_atom_flux": tritium_atom_flux,
            "final_time": self.scenario.get_maximum_time() - 1,
            "temperature": temperature_function,
            "L": bin.thickness,
        }

        # Pick parent index for folder naming (compatible with existing tooling)
        if isinstance(bin, DivBin):
            parent_bin_index = bin.index
        elif isinstance(bin, SubBin):
            parent_bin_index = bin.parent_bin_index
        else:
            parent_bin_index = getattr(bin, "index", -1)

        # ---------------- r_tol policy ----------------
        # Both B and W (and SS for completeness) use the same numeric value: 10e-10
        rtol_value = float(10e-10)  # 1e-9

        if bin.material == "W":
            return make_W_mb_model(
                **common_args,
                custom_rtol=rtol_value,
                folder=f"mb{parent_bin_index}_{getattr(bin, 'mode', 'NA')}_results",
            )
        elif bin.material == "B":
            return make_B_mb_model(
                **common_args,
                custom_rtol=rtol_value,
                folder=f"mb{parent_bin_index}_{getattr(bin, 'mode', 'NA')}_results",
            )
        elif bin.material == "SS":
            # Treat SS like W for tolerance purposes
            return make_DFW_mb_model(
                **common_args,
                # Some DFW paths accept custom_rtol; if not, the kw is ignored safely
                custom_rtol=rtol_value,
                folder=f"mb{parent_bin_index}_dfw_results",
            )
        else:
            raise ValueError(f"Unknown material: {bin.material} for bin {getattr(bin, 'index', '?')}")

    # ----------------------- helpers -----------------------
    def constant_max_stepsize(self, t: float) -> float:
        """Constant stepsize cap (s) = 100000.0 for every t and every case."""
        return 1000.0

    def make_milestones(self, initial_stepsize_value: float) -> List[float]:
        """
        Build stepsize/adaptivity milestones from scenario pulses.
        (Same logic as original file, preserved for stability of runs.)
        """
        milestones: List[float] = []
        current_time = 0.0

        for pulse in self.scenario.pulses:
            start_of_pulse = self.scenario.get_time_start_current_pulse(current_time)
            for i in range(pulse.nb_pulses):
                # small milestone right after each sub-pulse start
                milestones.append(start_of_pulse + pulse.total_duration * i + initial_stepsize_value)

                # ramp-up / ramp-down edges
                if i == 0:
                    milestones.append(start_of_pulse + pulse.ramp_up)
                    milestones.append(start_of_pulse + pulse.ramp_up + pulse.steady_state)
                else:
                    milestones.append(start_of_pulse + pulse.total_duration * (i - 1) + pulse.ramp_up)
                    milestones.append(start_of_pulse + pulse.total_duration * (i - 1) + pulse.ramp_up + pulse.steady_state)

                # start of next sub-pulse
                milestones.append(start_of_pulse + pulse.total_duration * (i + 1))

                # before the end of waiting period
                assert pulse.total_duration - pulse.duration_no_waiting >= 10
                milestones.append(start_of_pulse + pulse.total_duration * (i + 1) - 10)
                milestones.append(start_of_pulse + pulse.total_duration * (i + 1) - 2)

                # start of waiting for this sub-pulse
                milestones.append(start_of_pulse + pulse.total_duration * i + pulse.duration_no_waiting)

                # RISP special anchor
                if getattr(pulse, "pulse_type", None) == "RISP":
                    t_begin_real_pulse = start_of_pulse + 95
                    milestones.append(t_begin_real_pulse + pulse.total_duration * i)
                    milestones.append(t_begin_real_pulse + pulse.total_duration * i + 0.001)

            current_time = start_of_pulse + pulse.total_duration * pulse.nb_pulses

        return sorted(np.unique(milestones).tolist())
