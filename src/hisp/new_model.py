"""
New model class for CSV-driven bin simulations using new_mb_model.

This module provides a NewModel class that:
1. Uses new_mb_model.py for dynamic FESTIM model creation
2. Manages bin simulations with CSV-based configuration
3. Handles adaptive timestepping based on scenario pulses
"""

from typing import Dict, Tuple
import numpy as np

from hisp.plasma_data_handling import PlasmaDataHandling
from hisp.scenario import Scenario
from hisp.festim_models.new_mb_model import make_model_with_scenario

import festim as F


class NewModel:
    """
    Model runner that uses new_mb_model for dynamic FESTIM model creation.
    """
    
    def __init__(
        self,
        reactor,  # Reactor type from csv_bin
        scenario: Scenario,
        plasma_data_handling: PlasmaDataHandling,
        coolant_temp: float = 343.0,
    ):
        """
        Initialize the model runner.
        
        Args:
            reactor: Reactor object containing all bins
            scenario: Scenario with pulse sequence
            plasma_data_handling: Plasma data handler for flux/heat
            coolant_temp: Coolant temperature (K)
        """
        self.reactor = reactor
        self.scenario = scenario
        self.plasma_data_handling = plasma_data_handling
        self.coolant_temp = coolant_temp
        
    def run_bin(self, bin, exports: bool = False) -> Tuple[F.HydrogenTransportProblem, Dict]:
        """
        Run a FESTIM simulation for a single bin.
        
        Args:
            bin: Bin object to simulate
            exports: Whether to export XDMF files
            
        Returns:
            Tuple of (festim_model, quantities_dict)
        """
        print(f"\n{'='*60}")
        print(f"Running bin {bin.bin_number} (id={bin.bin_id})")
        print(f"  Material: {bin.material.name}")
        print(f"  Mode: {bin.mode}")
        print(f"  Location: {bin.location}")
        print(f"  Thickness: {bin.thickness*1e3:.2f} mm")
        print(f"  Surface area: {bin.surface_area:.4f} m²")
        print(f"{'='*60}")
        
        # Create FESTIM model using new_mb_model
        try:
            my_model, quantities = make_model_with_scenario(
                bin=bin,
                scenario=self.scenario,
                plasma_data_handling=self.plasma_data_handling,
                coolant_temp=self.coolant_temp,
                exports=exports,
            )
        except Exception as e:
            print(f"ERROR: Failed to create model for bin {bin.bin_number}: {e}")
            raise
        
        # Add derived quantities to model
        my_model.exports = my_model.exports if hasattr(my_model, 'exports') and my_model.exports else []
        for qty_name, qty in quantities.items():
            my_model.exports.append(qty)
        
        # Set up milestones for adaptive timestepping
        bin_config = bin.bin_configuration
        milestones = self._make_milestones(bin_config)
        milestones.append(my_model.settings.final_time)
        my_model.settings.stepsize.milestones = milestones
        
        # Adaptivity settings
        my_model.settings.stepsize.growth_factor = 1.1
        my_model.settings.stepsize.cutback_factor = 0.3
        
        # Initialize and run
        print(f"Initializing FESTIM model...")
        my_model.initialise()
        
        print(f"Running simulation (final_time={self.scenario.total_time:.0f} s)...")
        my_model.run()
        
        print(f"✓ Simulation complete for bin {bin.bin_number}")
        
        return my_model, quantities
    
    def _make_milestones(self, bin_config):
        """
        Create milestone times for adaptive timestepping based on scenario pulses.
        
        Args:
            bin_config: BinConfiguration with stepsize limits
            
        Returns:
            List of milestone times
        """
        milestones = []
        current_time = 0.0
        
        for pulse in self.scenario.pulses:
            pulse_start = current_time
            pulse_end = current_time + pulse.total_duration
            
            # Add milestone at pulse start
            if pulse_start > 0:
                milestones.append(pulse_start)
            
            # Add milestones during pulse based on pulse type
            if pulse.pulse_type in ["FP", "ICWC", "GDC"]:
                # For plasma pulses, add more frequent milestones
                max_step = bin_config.fp_max_stepsize
                n_steps = int(np.ceil(pulse.total_duration / max_step))
                for i in range(1, n_steps):
                    t = pulse_start + i * pulse.total_duration / n_steps
                    milestones.append(t)
            else:
                # For non-plasma pulses, use larger steps
                max_step = bin_config.max_stepsize_no_fp
                n_steps = int(np.ceil(pulse.total_duration / max_step))
                for i in range(1, n_steps):
                    t = pulse_start + i * pulse.total_duration / n_steps
                    milestones.append(t)
            
            # Add milestone at pulse end
            milestones.append(pulse_end)
            
            current_time = pulse_end
        
        # Remove duplicates and sort
        milestones = sorted(set(milestones))
        
        return milestones
