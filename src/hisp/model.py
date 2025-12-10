
"""
Simplified HISP Model for PFC runs using CSV-driven bin configuration
- Uses CSV-specific parameters for each bin (rtol, atol, stepsize limits)
- Integrates with local CSV bin classes from            if bin.material == "W":
                return make_W_mb_model_oldBC(
                    **common_args,
                    custom_rtol=rtol_value,
                    folder=f"mb{bin_index}_{bin.mode}_results",
                )
            elif bin.material == "B":
                return make_B_mb_model_oldBC(
                    **common_args,
                    custom_rtol=rtol_value,
                    folder=f"mb{bin_index}_{bin.mode}_results",
                )
            elif bin.material == "SS":
                # Treat SS like W for tolerance purposes
                return make_DFW_mb_model_oldBC(
                    **common_args,
                    # Some DFW paths accept custom_rtol; if not, the kw is ignored safely
                    custom_rtol=rtol_value,
                    folder=f"mb{bin_index}_dfw_results",
                )esigned to be a drop-in replacement for model(2).txt.
"""
from typing import List, Literal
import numpy as np
import sys
import os

# Add PFC directory to path for local imports
sys.path.insert(0, '/home/ITER/llealsa/AdriaLlealS/PFC-Tritium-Transport')

# Import CSV bin classes from local PFC
from csv_bin import CSVBin, CSVReactor, BinConfiguration

# NOTE: fixed import name (plasma_data_handling)
from hisp.plamsa_data_handling import PlasmaDataHandling
from hisp.scenario import Scenario
from hisp.helpers import periodic_step_function
from hisp.festim_models import (
    make_W_mb_model,
    make_W_mb_model_oldBC,
    make_B_mb_model,
    make_B_mb_model_oldBC,
    make_DFW_mb_model,
    make_DFW_mb_model_oldBC,
    make_temperature_function,
    make_particle_flux_function,
    compute_flux_values,
)


class Model:
    """
    HISP main model wrapper for CSV-driven bin configuration.
    Uses CSV-specific parameters for each bin:
      • rtol: relative tolerance from CSV (applied to all materials)
      • atol: absolute tolerance from CSV  
      • FP max stepsize: maximum stepsize during FP pulses from CSV
      • Max stepsize no FP: maximum stepsize during non-FP pulses from CSV
      • BC logic: based on plasma facing surface BC from CSV
        - "Robin - Surf. Rec. + Implantation" → Old BC
        - "Dirichlet Implantation approx." → New BC
    """

    def __init__(
        self,
        reactor: CSVReactor,
        scenario: Scenario,
        plasma_data_handling: PlasmaDataHandling,
        coolant_temp: float = 343.0,
    ) -> None:
        self.reactor = reactor
        self.scenario = scenario
        self.plasma_data_handling = plasma_data_handling
        self.coolant_temp = coolant_temp
        
    def get_bin_csv_params(self, bin: CSVBin) -> BinConfiguration:
        """Get CSV configuration parameters for a specific bin."""
        return bin.bin_configuration

    # ----------------------- public API used by the runner -----------------------
    def run_bin(self, bin: CSVBin):
        """Build and run a FESTIM model for a given CSV bin."""
        # Build FESTIM model from HISP inputs
        my_model, quantities = self.which_model(bin)

        # ---- Adaptivity & milestones ----
        milestones = self.make_milestones(
            initial_stepsize_value=my_model.settings.stepsize.initial_value
        )
        milestones.append(my_model.settings.final_time)
        my_model.settings.stepsize.milestones = milestones

        # Adaptivity knobs (same as model.py)
        my_model.settings.stepsize.growth_factor = 1.1
        my_model.settings.stepsize.cutback_factor = 0.3
        my_model.settings.stepsize.target_nb_iterations = 4

        # ---- Use unified max_stepsize function ----
        my_model.settings.stepsize.max_stepsize = self.max_stepsize

        # Store current bin for stepsize function access
        self.current_bin = bin

        # Run
        my_model.initialise()
        my_model.run()
        return my_model, quantities

    def run_all_bins(self):
        """(Optional) iterate over all bins of the reactor if needed."""
        raise NotImplementedError

    # ----------------------- model construction -----------------------
    def which_model(self, bin: CSVBin):
        """Return a (FESTIM model, quantities) pair for the provided CSV bin."""
        # Temperature & flux functions from HISP + scenario
        temperature_function = make_temperature_function(
            scenario=self.scenario,
            plasma_data_handling=self.plasma_data_handling,
            bin=bin,
            coolant_temp=self.coolant_temp,
        )

        # Use bin_number for folder naming (matches CSV bin number)
        bin_number = bin.bin_number

        # ---------------- CSV-specific parameters ----------------
        bin_config = self.get_bin_csv_params(bin)
        rtol_value = float(bin_config.rtol)
        atol_value = float(bin_config.atol) if bin_config.atol != float('inf') else None
        cu_thickness = float(bin.cu_thickness)
        
        print(f"Using CSV parameters for bin {bin_number}: rtol={rtol_value}, atol={atol_value}, Cu thickness={cu_thickness}m")

        # ---------------- BC logic based on plasma facing surface ----------------
        # Old BC: "Robin - Surf. Rec. + Implantation"
        # New BC: "Dirichlet Implantation approx."
        bc_plasma_facing = bin_config.bc_plasma_facing_surface
        use_old_bc = (bc_plasma_facing == "Robin - Surf. Rec. + Implantation")
        
        print(f"BC plasma facing: {bc_plasma_facing}, Using Old BC: {use_old_bc}")

        #---BC branching based on plasma facing surface---
        if not use_old_bc:  # New BC (Dirichlet Implantation approx.)

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

            if bin.material == "W":
                return make_W_mb_model(
                    **common_args,
                    custom_atol=atol_value,
                    custom_rtol=rtol_value,
                    folder=f"mb{bin_number}_{bin.mode}_results",
                )
            elif bin.material == "B":
                return make_B_mb_model(
                    **common_args,
                    custom_atol=atol_value,
                    custom_rtol=rtol_value,
                    folder=f"mb{bin_number}_{bin.mode}_results",
                )
            elif bin.material == "SS":
                # Treat SS like W for tolerance purposes
                return make_DFW_mb_model(
                    **common_args,
                    custom_atol=atol_value,
                    custom_rtol=rtol_value,
                    folder=f"mb{bin_number}_dfw_results",
                )
            else:
                raise ValueError(f"Unknown material: {bin.material} for bin {bin.bin_number}")
            
        else:  # Old BC (Robin - Surf. Rec. + Implantation)

            common_args = {
                "final_time": self.scenario.get_maximum_time() - 1,
                "temperature": temperature_function,
                "L": bin.thickness,
                "occurrences": compute_flux_values(self.scenario, self.plasma_data_handling, bin),
            }
    
            if bin.material == "W":
                return make_W_mb_model_oldBC(
                    **common_args,
                    custom_atol=atol_value,
                    custom_rtol=rtol_value,
                    folder=f"mb{bin_number}_{bin.mode}_results",
                )
            elif bin.material == "B":
                return make_B_mb_model_oldBC(
                    **common_args,
                    custom_atol=atol_value,
                    custom_rtol=rtol_value,
                    folder=f"mb{bin_number}_{bin.mode}_results",
                )
            elif bin.material == "SS":
                # Treat SS like W for tolerance purposes
                return make_DFW_mb_model_oldBC(
                    **common_args,
                    custom_atol=atol_value,
                    custom_rtol=rtol_value,
                    folder=f"mb{bin_number}_dfw_results",
                )
            else:
                raise ValueError(f"Unknown material: {bin.material} for bin {bin.bin_number}")

    # ----------------------- helpers -----------------------
    def max_stepsize(self, t: float) -> float:
        """Unified stepsize function using CSV bin configuration values."""
        if not hasattr(self, 'current_bin'):
            return 100.0  # fallback
            
        bin_config = self.get_bin_csv_params(self.current_bin)
        pulse = self.scenario.get_pulse(t)
        relative_time = t - self.scenario.get_time_start_current_pulse(t)
        
        if pulse.pulse_type == "RISP":
            relative_time_within_sub_pulse = relative_time % pulse.total_duration
            # RISP has a special treatment (same as model.py)
            time_real_risp_starts = 100  # (s) relative time at which the real RISP starts
            
            if relative_time_within_sub_pulse < time_real_risp_starts - 11:
                value = None  # s
            elif relative_time_within_sub_pulse < time_real_risp_starts + 160:
                value = float(bin_config.fp_max_stepsize)
            else:
                value = float(bin_config.max_stepsize_no_fp)
        else:
            relative_time_within_sub_pulse = relative_time % pulse.total_duration
            
            # Use CSV-specific stepsize values based on pulse type and timing
            if pulse.pulse_type == "FP":
                if relative_time_within_sub_pulse < pulse.duration_no_waiting:
                    # During FP pulse: use FP max stepsize from bin config
                    value = float(bin_config.fp_max_stepsize)
                else:
                    # Outside FP pulse: use max stepsize no FP from bin config  
                    value = float(bin_config.max_stepsize_no_fp)
            else:
                # For non-FP pulses: use max stepsize no FP from bin config
                value = float(bin_config.max_stepsize_no_fp)
        
        return periodic_step_function(
            relative_time,
            period_on=pulse.duration_no_waiting,
            period_total=pulse.total_duration,
            value=value,
            value_off=None,
        )

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
