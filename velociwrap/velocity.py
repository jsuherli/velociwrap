### velociwrap
### 
### A lightweight python package to create velocity-consistent channel maps 
### and position-velocity (PV) diagrams from astronomical datacube. 
###
### Janette Suherli (c)2026. 
### jsuherli@gmail.com
###
### Velocity, wavelength, wavenumber, and frequency conversion utilities.


from __future__ import annotations

import numpy as np

c_kms = 299792.458

def wavelength_to_velocity(wavelength, rest_wavelength, convention: str = "optical"):
    """
    To convert wavelength to velocity in km/s.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    
    if convention == "optical":
        return c_kms * (wavelength - rest_wavelength) / rest_wavelength
    if convention == "radio":
        return c_kms * (rest_wavelength - wavelength) / wavelength
    if convention == "relativistic":
        r = wavelength / rest_wavelength
        return c_kms * (r**2 - 1.0) / (r**2 + 1.0)
    
    raise ValueError("convention must be 'optical', 'radio', or 'relativistic'.")


def velocity_to_wavelength(velocity, rest_wavelength, convention: str = "optical"):
    """
    To convert velocity in km/s to wavelength.
    """
    velocity = np.asarray(velocity, dtype=float)
    
    beta = velocity / c_kms
    if convention == "optical":
        return rest_wavelength * (1.0 + beta)
    if convention == "radio":
        return rest_wavelength / (1.0 - beta)
    if convention == "relativistic":
        return rest_wavelength * np.sqrt((1.0 + beta) / (1.0 - beta))
    raise ValueError("convention must be 'optical', 'radio', or 'relativistic'.")


def wavenumber_to_wavelength(wavenumber):
    """
    To convert wavenumber in cm^-1 to wavelength in Angstrom.
    """
    return 1e8 / np.asarray(wavenumber, dtype=float)


def wavelength_to_wavenumber(wavelength):
    """
    To convert wavelength in Angstrom to wavenumber in cm^-1.
    """
    return 1e8 / np.asarray(wavelength, dtype=float)


def frequency_to_velocity(frequency, rest_frequency, convention: str = "radio"):
    """
    To convert frequency to velocity in km/s.
    """
    frequency = np.asarray(frequency, dtype=float)
    if convention == "radio":
        return c_kms * (rest_frequency - frequency) / rest_frequency
    if convention == "optical":
        return c_kms * (rest_frequency - frequency) / frequency
    if convention == "relativistic":
        r = rest_frequency / frequency
        return c_kms * (r**2 - 1.0) / (r**2 + 1.0)
    
    raise ValueError("convention must be 'optical', 'radio', or 'relativistic'.")


def velocity_to_frequency(velocity, rest_frequency, convention: str = "radio"):
    """
    To convert velocity in km/s to frequency in Hz.
    """
    velocity = np.asarray(velocity, dtype=float)
    beta = velocity / c_kms

    if convention == "radio":
        return rest_frequency * (1.0 - beta)
    if convention == "optical":
        return rest_frequency / (1.0 + beta)
    if convention == "relativistic":
        return rest_frequency * np.sqrt((1.0 - beta) / (1.0 + beta))
    
    raise ValueError("convention must be 'optical', 'radio', or 'relativistic'.")


def velocity_to_kms(velocity, unit: str | None = None):
    """
    To convert other velocity-like values to km/s.
    """
    velocity = np.asarray(velocity, dtype=float)
    if unit is None:
        return velocity
    unit = unit.strip().lower().replace(" ", "")
    if unit in {"km/s", "kms-1", "km.s-1", "km/s-1"}:
        return velocity
    if unit in {"m/s", "ms-1", "m.s-1"}:
        return velocity / 1000.0
    raise ValueError(f"Unsupported velocity unit: {unit}")


def make_velocity_bins(vmin, vmax, dv):
    """
    To create velocity bin edges and centers in km/s.
    """
    edges = np.arange(vmin, vmax + 0.5 * dv, dv, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


