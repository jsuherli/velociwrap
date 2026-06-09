### velociwrap
### 
### A lightweight python package to create velocity-consistent channel maps 
### and position-velocity (PV) diagrams from astronomical datacube. 
###
### Janette Suherli (c)2026. 
### jsuherli@gmail.com

from .cube import Cube
from .velocity import c_kms, wavelength_to_velocity, velocity_to_wavelength, wavenumber_to_wavelength, wavelength_to_wavenumber, frequency_to_velocity, velocity_to_frequency, velocity_to_kms, make_velocity_bins
from .channel_maps import make_channel_maps, make_rgb_channel_maps, make_integrated_map, velocity_bin_table
from .plotting import plot_channel_grid, plot_rgb_channel_grid
from .io import save_channel_maps_fits, save_rgb_components_fits
from .animation import save_channel_animation, save_rgb_channel_animation
from .pv import PVDiagram, extract_pv, plot_pv, add_pv_slit
from .demo import load_demo_cube, list_demo_cubes, get_demo_region_file

__version__ = "1.0.0b0"

__all__ = [
    "Cube",
    "c_kms",
    "wavelength_to_velocity",
    "velocity_to_wavelength",
    "wavenumber_to_wavelength",
    "wavelength_to_wavenumber",
    "frequency_to_velocity",
    "velocity_to_frequency",
    "velocity_to_kms",
    "make_velocity_bins",
    "make_channel_maps",
    "make_rgb_channel_maps",
    "make_integrated_map",
    "velocity_bin_table",
    "plot_channel_grid",
    "plot_rgb_channel_grid",
    "save_channel_maps_fits",
    "save_rgb_components_fits",
    "save_channel_animation",
    "save_rgb_channel_animation",
    "extract_pv",
    "plot_pv",
    "add_pv_slit",
    "load_demo_cube",
    "list_demo_cubes",
    "get_demo_region_file",
    "PVDiagram",
]

