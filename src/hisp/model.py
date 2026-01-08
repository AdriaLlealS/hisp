
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

# Import CSV bin classes from hisp package
from hisp.bin import Bin, Reactor, BinConfiguration

# NOTE: fixed import name (plasma_data_handling)
from hisp.plasma_data_handling import PlasmaDataHandling
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

import festim as F
from hisp.h_transport_class import CustomProblem
from hisp.settings import CustomSettings
from hisp.helpers import Stepsize, XDMFExportEveryDt


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
        reactor: Reactor,
        scenario: Scenario,
        plasma_data_handling: PlasmaDataHandling,
        coolant_temp: float = 343.0,
    ) -> None:
        self.reactor = reactor
        self.scenario = scenario
        self.plasma_data_handling = plasma_data_handling
        self.coolant_temp = coolant_temp
        
    def get_bin_csv_params(self, bin: Bin) -> BinConfiguration:
        """Get CSV configuration parameters for a specific bin."""
        return bin.bin_configuration

    # ----------------------- public API used by the runner -----------------------
    def run_bin(self, bin: Bin):
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
    def which_model(self, bin: Bin):
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

            # Handle both string materials and Material objects
            material_name = bin.material.name if hasattr(bin.material, 'name') else bin.material
            
            if material_name == "W":
                return make_W_mb_model(
                    **common_args,
                    custom_atol=atol_value,
                    custom_rtol=rtol_value,
                    folder=f"mb{bin_number}_{bin.mode}_results",
                )
            elif material_name == "B":
                return make_B_mb_model(
                    **common_args,
                    custom_atol=atol_value,
                    custom_rtol=rtol_value,
                    folder=f"mb{bin_number}_{bin.mode}_results",
                )
            elif material_name == "SS":
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
    
            # Handle both string materials and Material objects
            material_name = bin.material.name if hasattr(bin.material, 'name') else bin.material
            
            if material_name == "W":
                return make_W_mb_model_oldBC(
                    **common_args,
                    custom_atol=atol_value,
                    custom_rtol=rtol_value,
                    folder=f"mb{bin_number}_{bin.mode}_results",
                )
            elif material_name == "B":
                return make_B_mb_model_oldBC(
                    **common_args,
                    custom_atol=atol_value,
                    custom_rtol=rtol_value,
                    folder=f"mb{bin_number}_{bin.mode}_results",
                )
            elif material_name == "SS":
                # Treat SS like W for tolerance purposes
                return make_DFW_mb_model_oldBC(
                    **common_args,
                    custom_atol=atol_value,
                    custom_rtol=rtol_value,
                    folder=f"mb{bin_number}_dfw_results",
                )
            else:
                raise ValueError(f"Unknown material: {bin.material} for bin {bin.bin_number}")

    # ----------------------- unified builder -----------------------
    def mb_model_new(
        self,
        bin: Bin,
        material: dict,
        temperature: callable,
        final_time: float,
        folder: str,
        L: float,
        custom_atol: float | None = None,
        custom_rtol: float | None = None,
        exports: bool = False,
    ):
        """Unified MB model builder that consumes a `bin` and a `material` dict.

        Expected `material` dict keys (recommended):
          - name: material name (str)
          - Mat_density: float (atoms/m3)
          - D0: float (m2/s)
          - E_D: float (eV)
          - N_traps: int
          - traps: list of dicts with keys: Trap_density, k_0, E_k, p_0, E_p

        The function builds species, trap implicit species and reactions from
        the provided material dictionary. Boundary conditions (old/new)
        are both available and the one attached depends on the bin
        configuration `bin.bin_configuration.bc_plasma_facing_surface`.
        """
        my_model = CustomProblem()

        # Mesh
        try:
            n_points = max(50, int(min(1000, max(50, L / 1e-9))))
        except Exception:
            n_points = 200
        vertices = np.linspace(0.0, L, n_points)
        my_model.mesh = F.Mesh1D(vertices)

        # Material properties
        mat_name = str(material.get("name", "material")).lower()
        mat_density = float(material.get("Mat_density", material.get("mat_density", 1.0)))
        D_0 = float(material.get("D0", material.get("D_0", 1.0)))
        E_D = float(material.get("E_D", material.get("E_D_eV", 0.5)))

        festim_mat = F.Material(D_0=D_0, E_D=E_D, name=mat_name)
        vol = F.VolumeSubdomain1D(id=1, borders=[0, L], material=festim_mat)
        inlet = F.SurfaceSubdomain1D(id=1, x=0)
        outlet = F.SurfaceSubdomain1D(id=2, x=L)
        my_model.subdomains = [vol, inlet, outlet]

        # Species
        mobile_D = F.Species("D")
        mobile_T = F.Species("T")

        # Traps
        traps_list = material.get("traps")
        N_traps = int(material.get("N_traps", material.get("n_traps", 0)))
        trap_species = []
        implicit_traps = []

        # If traps list not provided, try to construct from repeated keys in dict
        if traps_list is None:
            traps_list = []
            for i in range(1, N_traps + 1):
                key_d = f"Trap_density_{i}"
                if key_d in material:
                    traps_list.append({
                        "Trap_density": material.get(key_d),
                        "k_0": material.get(f"k_0_{i}", None),
                        "E_k": material.get(f"E_k_{i}", None),
                        "p_0": material.get(f"p_0_{i}", None),
                        "E_p": material.get(f"E_p_{i}", None),
                    })
                else:
                    break

        # create trap species and implicit traps
        for idx, t in enumerate(traps_list[:N_traps] if traps_list else range(N_traps)):
            if isinstance(t, dict):
                n_val = float(t.get("Trap_density", t.get("n", mat_density * 1e-4)))
                k0 = t.get("k_0", None)
                E_k = t.get("E_k", None)
                p0 = t.get("p_0", None)
                E_p = t.get("E_p", None)
            else:
                n_val = float(material.get("Trap_density", mat_density * 1e-4))
                k0 = None
                E_k = None
                p0 = None
                E_p = None

            sp_d = F.Species(f"trap{idx+1}_D", mobile=False)
            sp_t = F.Species(f"trap{idx+1}_T", mobile=False)
            trap_species.extend([sp_d, sp_t])

            impl = F.ImplicitSpecies(n=n_val, others=[sp_t, sp_d], name=f"empty_trap{idx+1}")
            implicit_traps.append(impl)

        my_model.species = [mobile_D, mobile_T] + trap_species

        # Reactions
        interstitial_distance = float(material.get("interstitial_distance", 1.117e-10))
        interstitial_sites_per_atom = int(material.get("interstitial_sites_per_atom", 6))

        reactions = []
        for idx, impl in enumerate(implicit_traps):
            trap_params = traps_list[idx] if isinstance(traps_list, list) and idx < len(traps_list) else {}
            k_0_val = trap_params.get("k_0")
            if k_0_val is None:
                k_0_val = D_0 / (interstitial_distance**2 * interstitial_sites_per_atom * mat_density)
            E_k_val = trap_params.get("E_k", E_D)
            p_0_val = trap_params.get("p_0", 1e13)
            E_p_val = trap_params.get("E_p", 1.0)

            trap_d = F.Species(f"trap{idx+1}_D", mobile=False)
            trap_t = F.Species(f"trap{idx+1}_T", mobile=False)

            reactions.append(
                F.Reaction(
                    k_0=k_0_val,
                    E_k=E_k_val,
                    p_0=p_0_val,
                    E_p=E_p_val,
                    volume=vol,
                    reactant=[mobile_D, impl],
                    product=trap_d,
                )
            )
            reactions.append(
                F.Reaction(
                    k_0=k_0_val,
                    E_k=E_k_val,
                    p_0=p_0_val,
                    E_p=E_p_val,
                    volume=vol,
                    reactant=[mobile_T, impl],
                    product=trap_t,
                )
            )

        my_model.reactions = reactions

        # Temperature
        my_model.temperature = temperature

        # Boundary conditions selection
        bin_config = self.get_bin_csv_params(bin)
        bc_choice = getattr(bin_config, "bc_plasma_facing_surface", None)
        use_old_bc = (bc_choice == "Robin - Surf. Rec. + Implantation")

        if use_old_bc:
            occurrences = compute_flux_values(self.scenario, self.plasma_data_handling, bin)
            deuterium_ion_flux, deuterium_atom_flux, tritium_ion_flux, tritium_atom_flux = build_ufl_flux_expression(occurrences)
            c_sD = make_surface_concentration_time_function_J(temperature, lambda t: float(deuterium_ion_flux(t) + deuterium_atom_flux(t)), D_0, E_D, 3e-9, surface_x=0.0)
            c_sT = make_surface_concentration_time_function_J(temperature, lambda t: float(tritium_ion_flux(t) + tritium_atom_flux(t)), D_0, E_D, 3e-9, surface_x=0.0)
        else:
            d_ion = make_particle_flux_function(self.scenario, self.plasma_data_handling, bin, ion=True, tritium=False)
            t_ion = make_particle_flux_function(self.scenario, self.plasma_data_handling, bin, ion=True, tritium=True)
            d_atom = make_particle_flux_function(self.scenario, self.plasma_data_handling, bin, ion=False, tritium=False)
            t_atom = make_particle_flux_function(self.scenario, self.plasma_data_handling, bin, ion=False, tritium=True)
            Gamma_D_total = lambda t: float(d_ion(t) + d_atom(t))
            Gamma_T_total = lambda t: float(t_ion(t) + t_atom(t))
            c_sD = make_surface_concentration_time_function_J(temperature, Gamma_D_total, D_0, E_D, 3e-9, surface_x=0.0)
            c_sT = make_surface_concentration_time_function_J(temperature, Gamma_T_total, D_0, E_D, 3e-9, surface_x=0.0)

        bc_D = F.FixedConcentrationBC(subdomain=inlet, value=c_sD, species="D")
        bc_T = F.FixedConcentrationBC(subdomain=inlet, value=c_sT, species="T")
        my_model.boundary_conditions = [bc_D, bc_T]

        # Exports and quantities
        quantities = {}
        my_model.exports = []
        for species in my_model.species:
            q = F.TotalVolume(field=species, volume=vol)
            my_model.exports.append(q)
            quantities[species.name] = q
            if species.mobile:
                flux = F.SurfaceFlux(field=species, surface=inlet)
                my_model.exports.append(flux)
                quantities[species.name + "_surface_flux"] = flux

        # Settings
        my_model.settings = CustomSettings(
            atol=custom_atol if custom_atol is not None else 1e11,
            rtol=custom_rtol if custom_rtol is not None else 1e-9,
            max_iterations=100,
            final_time=final_time,
        )
        my_model.settings.stepsize = Stepsize(initial_value=1e-3)
        my_model._element_for_traps = "CG"

        return my_model, quantities
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
