from typing import List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np


class SubBin:
    thickness: float
    copper_thickness: float
    material: str
    mode: str
    dfw: bool
    parent_bin_index: int
    low_wetted_area: float
    high_wetted_area: float
    total_area: float
    f: float

    def __init__(
        self,
        mode: str,
        thickness: float = None,
        material: str = None,
    ):
        """
        A SubBin object represents a sub-bin of a FW bin in a reactor.


        Args:
            thickness: The thickness of the subbin (in m).
            material: The material of the subbin.
            mode: The mode of the subbin (shadowed, wetted, low_wetted, high_wetted).

        Attributes:
            thickness: The thickness of the subbin (in m).
            copper_thickness: The thickness of the copper layer behind W bins (m).
            material: The material of the subbin.
            mode: The mode of the subbin (shadowed, wetted, low_wetted, high_wetted).
            dfw: A boolean indicating if the subbin is a Divertor First Wall (DFW) subbin.
            parent_bin_index: The index of the parent bin.
            low_wetted_area: The low wetted area of the parent bin (in m^2).
            high_wetted_area: The high wetted area of the parent bin (in m^2).
            total_area: The total area of the parent bin (in m^2).
            f: The fraction of heat values in the low wetted area. Calculated from SMITER as:
                f = H_low * low_wetted_area / (H_low * low_wetted_area + H_high * high_wetted_area)
        """
        self.thickness = thickness
        self.material = material
        self.mode = mode
        self.dfw = False
        self.copper_thickness = None
        self.parent_bin_index = None
        self.low_wetted_area = None
        self.high_wetted_area = None
        self.total_area = None
        self.f = None

    @property
    def shadowed(self) -> bool:
        return self.mode == "shadowed" or self.dfw

    @property
    def wetted_frac(self):
        if self.shadowed:
            return 0.0
        elif self.mode == "wetted":
            return self.total_area / self.low_wetted_area

        elif self.mode == "low_wetted":
            return self.f * self.total_area / self.low_wetted_area

        elif self.mode == "high_wetted":
            return (1 - self.f) * self.total_area / self.high_wetted_area

    @property
    def surface_area(self) -> float:
        """Calculates the surface area of the subbin (in m^2).

        Returns:
            The surface area of the subbin (in m^2).
        """
        if self.shadowed:
            low_wetted_area = self.low_wetted_area
            high_wetted_area = self.high_wetted_area
            if (
                isinstance(self.low_wetted_area, type(np.nan))
                or self.low_wetted_area is None
            ):
                low_wetted_area = 0
            if (
                isinstance(self.high_wetted_area, type(np.nan))
                or self.high_wetted_area is None
            ):
                high_wetted_area = 0
            return self.total_area - low_wetted_area - high_wetted_area
        elif self.mode in ["wetted", "low_wetted"]:
            return self.low_wetted_area
        elif self.mode == "high_wetted":
            return self.high_wetted_area


class FWBin:
    index: int
    sub_bins: List[SubBin]
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]

    def __init__(self, sub_bins: List[SubBin] = None):
        self.sub_bins = sub_bins or []
        self.index = None
        self.start_point = None
        self.end_point = None

    @property
    def shadowed_subbin(self) -> SubBin:
        for subbin in self.sub_bins:
            if subbin.shadowed:
                return subbin

        raise ValueError(f"No shadowed subbin found in bin {self.index}")

    @property
    def length(self) -> float:
        """Calculates the poloidal length of the bin (in m).

        Returns:
            The poloidal length of the bin (in m).
        """
        return (
            (self.end_point[0] - self.start_point[0]) ** 2
            + (self.end_point[1] - self.start_point[1]) ** 2
        ) ** 0.5

    def add_dfw_bin(self, **kwargs):
        dfw_bin = SubBin(mode="shadowed", **kwargs)
        dfw_bin.dfw = True  # TODO do we need this?
        self.sub_bins.append(dfw_bin)


class FWBin3Subs(FWBin):
    def __init__(self):
        subbins = [
            SubBin(mode="shadowed"),
            SubBin(mode="low_wetted"),
            SubBin(mode="high_wetted"),
        ]
        super().__init__(subbins)

    @property
    def low_wetted_subbin(self) -> SubBin:
        return self.sub_bins[1]

    @property
    def high_wetted_subbin(self) -> SubBin:
        return self.sub_bins[2]


class FWBin2Subs(FWBin):
    def __init__(self):
        subbins = [
            SubBin(mode="shadowed"),
            SubBin(mode="wetted"),
        ]
        super().__init__(subbins)

    @property
    def wetted_subbin(self) -> SubBin:
        return self.sub_bins[1]


class DivBin:
    index: int
    thickness: float
    material: str
    mode = "wetted"
    inner_bin = bool
    outer_bin = bool
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]

    def __init__(self):
        self.index = None
        self.thickness = None
        self.material = None
        self.inner_bin = False
        self.outer_bin = False
        self.start_point = None
        self.end_point = None

    def set_inner_and_outer_bins(self) -> bool:
        """Flags if a DivBin is an inner target or outer target bin.

        Returns:
            inner_bin: True if inner bin
            outer_bin: True if outer bin
        """
        inner_swept_bins = list(range(45, 64))
        outer_swept_bins = list(range(18, 32))

        if self.index in inner_swept_bins:
            self.inner_bin = True
        elif self.index in outer_swept_bins:
            self.outer_bin = True

    @property
    def length(self) -> float:
        """Calculates the poloidal length of the bin (in m).

        Returns:
            The poloidal length of the bin (in m).
        """
        return (
            (self.end_point[0] - self.start_point[0]) ** 2
            + (self.end_point[1] - self.start_point[1]) ** 2
        ) ** 0.5


class BinCollection:
    def __init__(self, bins: List[FWBin | DivBin] = None):
        """Initializes a BinCollection object containing several bins.

        Args:
            bins: The list of bins in the collection. Each bin is a Bin object.
        """
        self.bins = bins if bins is not None else []

    def get_bin(self, index: int) -> FWBin | DivBin:
        for bin in self.bins:
            if bin.index == index:
                return bin
        raise ValueError(f"No bin found with index {index}")

    def arc_length(self, middle: bool = False):
        """Returns the cumulative length of all bins in the collection.

        Args:
            middle: If True, computes from the middle of each bin.
                If False, computes from the start of each bin.
        """
        if middle:
            middle_of_bins = []
            cumulative_lengths = [0]
            for bin in self.bins:
                middle_of_bins.append(cumulative_lengths[-1] + bin.length / 2)
                cumulative_lengths.append(cumulative_lengths[-1] + bin.length)
            return middle_of_bins
        else:
            return np.cumsum([bin.length for bin in self.bins])


class Reactor:
    first_wall: BinCollection
    divertor: BinCollection

    def __init__(
        self, first_wall: BinCollection = None, divertor: BinCollection = None
    ):
        self.first_wall = first_wall
        self.divertor = divertor
        all_bins = first_wall.bins + divertor.bins
        for i, bin in enumerate(all_bins):
            bin.index = i

        for i, bin in enumerate(first_wall.bins):
            for subbin in bin.sub_bins:
                subbin.parent_bin_index = i

    def get_bin(self, index: int) -> FWBin | DivBin:
        for bin in self.first_wall.bins + self.divertor.bins:
            if bin.index == index:
                return bin
        raise ValueError(f"No bin found with index {index}")

    def read_wetted_data(self, filename: str):
        data = pd.read_csv(filename)

        for fw_bin in self.first_wall.bins:
            for subbin in fw_bin.sub_bins:
                subbin.low_wetted_area = data.iloc[fw_bin.index]["Slow"]
                subbin.high_wetted_area = data.iloc[fw_bin.index]["Shigh"]
                subbin.total_area = data.iloc[fw_bin.index]["Stot"]
                subbin.f = data.iloc[fw_bin.index]["f"]


# =============================================================================
# CSV-driven bin classes for HISP reactor modeling
# =============================================================================


@dataclass
class BinConfiguration:
    """Configuration parameters for HISP simulation."""
    rtol: float
    atol: float
    fp_max_stepsize: float  # FP max. stepsize (s)
    max_stepsize_no_fp: float  # Max. stepsize no FP (s)
    bc_plasma_facing_surface: str  # BC Plasma Facing Surface
    bc_rear_surface: str  # BC rear surface
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.rtol <= 0:
            raise ValueError(f"rtol must be positive, got {self.rtol}")
        if self.atol <= 0:
            raise ValueError(f"atol must be positive, got {self.atol}")
        if self.fp_max_stepsize <= 0:
            raise ValueError(f"fp_max_stepsize must be positive, got {self.fp_max_stepsize}")
        if self.max_stepsize_no_fp <= 0:
            raise ValueError(f"max_stepsize_no_fp must be positive, got {self.max_stepsize_no_fp}")


class CSVBin:
    """
    A bin class representing one row from the CSV configuration table.
    Each bin contains all geometric, material, and simulation properties.
    """
    
    def __init__(
        self,
        bin_number: int,
        z_start: float,
        r_start: float, 
        z_end: float,
        r_end: float,
        material: str,
        thickness: float,
        cu_thickness: float,
        mode: str,
        parent_bin_surf_area: float,
        surface_area: float,
        f_ion_flux_fraction: float,
        location: str,
        coolant_temp: float = 343.0,
        bin_configuration: Optional[BinConfiguration] = None,
        bin_id: Optional[int] = None
    ):
        """
        Initialize a CSV-based bin.
        
        Args:
            bin_number: Bin number from CSV
            z_start: Z coordinate start position (m)
            r_start: R coordinate start position (m)
            z_end: Z coordinate end position (m)
            r_end: R coordinate end position (m)
            material: Material type (W, B, SS, etc.)
            thickness: Bin thickness (m)
            cu_thickness: Copper thickness (m)
            mode: Operating mode (hw, lw, shadowed, wetted, etc.)
            parent_bin_surf_area: Surface area of parent bin (m^2)
            surface_area: Surface area of this specific bin/mode (m^2)
            f_ion_flux_fraction: Ion flux fraction
            location: Location identifier (FW, DIV, etc.)
            coolant_temp: Coolant temperature (K)
            bin_configuration: BinConfiguration object with simulation parameters
            bin_id: Row number from CSV table (optional)
            
        Calculated Properties:
            ion_scaling_factor: Calculated as f_ion_flux_fraction * parent_bin_surf_area / surface_area
        """
        # Geometric properties
        self.bin_number = bin_number
        self.z_start = z_start
        self.r_start = r_start
        self.z_end = z_end
        self.r_end = r_end
        
        # Material properties
        self.material = material
        self.thickness = thickness
        self.cu_thickness = cu_thickness
        
        # Operating properties
        self.mode = mode
        self.parent_bin_surf_area = parent_bin_surf_area
        self.surface_area = surface_area
        self.f_ion_flux_fraction = f_ion_flux_fraction
        self.location = location
        self.coolant_temp = coolant_temp
        
        # CSV row identifier
        self.bin_id = bin_id if bin_id is not None else bin_number
        
        # Calculate ion scaling factor
        self.ion_scaling_factor = self.f_ion_flux_fraction * self.parent_bin_surf_area / self.surface_area
        
        # Simulation configuration (use provided or create default)
        self.bin_configuration = bin_configuration if bin_configuration is not None else BinConfiguration(
            rtol=1e-10,
            atol=1e10,
            fp_max_stepsize=5.0,
            max_stepsize_no_fp=100.0,
            bc_plasma_facing_surface="Robin - Surf. Rec. + Implantation",
            bc_rear_surface="Neumann - no flux"
        )
    
    @property
    def copper_thickness(self) -> float:
        """Compatibility property for HISP temperature functions."""
        return self.cu_thickness
    
    @property
    def start_point(self) -> tuple[float, float]:
        """Get start point as (Z, R) tuple."""
        return (self.z_start, self.r_start)
    
    @property
    def end_point(self) -> tuple[float, float]:
        """Get end point as (Z, R) tuple."""
        return (self.z_end, self.r_end)
    
    @property
    def length(self) -> float:
        """Calculate the poloidal length of the bin (m)."""
        return (
            (self.z_end - self.z_start) ** 2 + 
            (self.r_end - self.r_start) ** 2
        ) ** 0.5
    
    @property
    def is_first_wall(self) -> bool:
        """Check if this is a first wall bin."""
        return self.location.upper() == "FW"
    
    @property
    def is_divertor(self) -> bool:
        """Check if this is a divertor bin."""
        return self.location.upper() in ["DIV", "DIVERTOR"]
    
    @property
    def is_shadowed(self) -> bool:
        """Check if this bin is in shadowed mode."""
        return self.mode.lower() in ["shadowed", "shadow"]
    
    @property
    def is_wetted(self) -> bool:
        """Check if this bin is in any wetted mode."""
        return self.mode.lower() in ["wetted", "wet", "hw", "lw", "high_wetted", "low_wetted"]
    
    def __str__(self) -> str:
        """String representation of the bin."""
        return (
            f"CSVBin(id={self.bin_id}, bin_num={self.bin_number}, "
            f"material={self.material}, mode={self.mode}, "
            f"location={self.location}, thickness={self.thickness*1000:.1f}mm)"
        )
    
    def __repr__(self) -> str:
        """Detailed representation of the bin."""
        return self.__str__()


class CSVBinCollection:
    """Collection of CSV-based bins."""
    
    def __init__(self, bins: list[CSVBin] = None):
        """Initialize collection with list of CSVBin objects."""
        self.bins = bins if bins is not None else []
    
    def add_bin(self, bin: CSVBin):
        """Add a bin to the collection."""
        self.bins.append(bin)
    
    def get_bin_by_id(self, bin_id: int) -> CSVBin:
        """Get bin by its CSV row ID."""
        for bin in self.bins:
            if bin.bin_id == bin_id:
                return bin
        raise ValueError(f"No bin found with ID {bin_id}")
    
    def get_bin_by_number(self, bin_number: int) -> CSVBin:
        """Get bin by its bin number."""
        for bin in self.bins:
            if bin.bin_number == bin_number:
                return bin
        raise ValueError(f"No bin found with number {bin_number}")
    
    def get_bins_by_material(self, material: str) -> list[CSVBin]:
        """Get all bins with specified material."""
        return [bin for bin in self.bins if bin.material.upper() == material.upper()]
    
    def get_bins_by_location(self, location: str) -> list[CSVBin]:
        """Get all bins at specified location (FW, DIV, etc.)."""
        return [bin for bin in self.bins if bin.location.upper() == location.upper()]
    
    def get_bins_by_mode(self, mode: str) -> list[CSVBin]:
        """Get all bins with specified mode."""
        return [bin for bin in self.bins if bin.mode.lower() == mode.lower()]
    
    @property
    def first_wall_bins(self) -> list[CSVBin]:
        """Get all first wall bins."""
        return [bin for bin in self.bins if bin.is_first_wall]
    
    @property
    def divertor_bins(self) -> list[CSVBin]:
        """Get all divertor bins."""
        return [bin for bin in self.bins if bin.is_divertor]
    
    def __len__(self) -> int:
        """Return number of bins in collection."""
        return len(self.bins)
    
    def __iter__(self):
        """Make collection iterable."""
        return iter(self.bins)
    
    def __str__(self) -> str:
        """String representation of the collection."""
        fw_count = len(self.first_wall_bins)
        div_count = len(self.divertor_bins)
        return f"CSVBinCollection({len(self.bins)} bins: {fw_count} FW, {div_count} DIV)"


class CSVReactor(CSVBinCollection):
    """
    A reactor representing the complete collection of all bins from a CSV table.
    This is the main class for representing the entire ITER reactor configuration.
    """
    
    def __init__(self, bins: list[CSVBin] = None):
        """
        Initialize reactor with list of CSVBin objects.
        
        Args:
            bins: List of CSVBin objects representing all reactor bins
        """
        super().__init__(bins)
    
    @property
    def total_bins(self) -> int:
        """Get total number of bins in the reactor."""
        return len(self.bins)
    
    @property
    def materials_summary(self) -> dict[str, int]:
        """Get summary of materials used in the reactor."""
        materials = {}
        for bin in self.bins:
            material = bin.material.upper()
            materials[material] = materials.get(material, 0) + 1
        return materials
    
    @property
    def locations_summary(self) -> dict[str, int]:
        """Get summary of bin locations in the reactor."""
        locations = {}
        for bin in self.bins:
            location = bin.location.upper()
            locations[location] = locations.get(location, 0) + 1
        return locations
    
    def get_reactor_summary(self) -> str:
        """Get comprehensive summary of the reactor configuration."""
        summary = [
            f"CSVReactor Summary:",
            f"  Total bins: {self.total_bins}",
            f"  First Wall bins: {len(self.first_wall_bins)}",
            f"  Divertor bins: {len(self.divertor_bins)}",
            f"  Materials: {self.materials_summary}",
            f"  Locations: {self.locations_summary}"
        ]
        return "\n".join(summary)
    
    def __str__(self) -> str:
        """String representation of the reactor."""
        return f"CSVReactor({self.total_bins} total bins: {len(self.first_wall_bins)} FW, {len(self.divertor_bins)} DIV)"
