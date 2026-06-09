### velociwrap
### 
### A lightweight python package to create velocity-consistent channel maps 
### and position-velocity (PV) diagrams from astronomical datacube. 
###
### Janette Suherli (c)2026. 
### jsuherli@gmail.com
###
### FITS datacube reader.

from __future__ import annotations

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u

from .velocity import velocity_to_kms, wavenumber_to_wavelength


class Cube:
    """
    Lightweight FITS datacube reader.

    Parameters
    ----------
    filename : str
        FITS cube path.
    ext : int or str, default=0
        HDU containing data. For many MUSE products this may be 1.
    spectral_axis_type : str, optional
        User override: 'wavelength', 'wavenumber', 'frequency', or 'velocity'.
    spectral_unit : str, optional
        User override for spectral unit, e.g. 'Angstrom', 'cm-1', 'Hz', 'm/s', 'km/s'.
    squeeze : bool
        Remove singleton axes, e.g. Stokes axis with length 1.
    """

    def __init__(
        self,
        filename: str,
        ext=None,
        spectral_axis_type: str | None = None,
        spectral_unit: str | None = None,
        squeeze: bool = True,
    ):
        self.filename = filename
        self.hdul = fits.open(filename)

        if ext is None:
            ext = self._find_data_hdu()
        self.ext = ext

        self.header = self.hdul[ext].header.copy()
        data = self.hdul[ext].data
        if data is None:
            raise ValueError(f"No data found in HDU {ext}.")
        data = data.astype(float)

        if squeeze:
            data = np.squeeze(data)
        if data.ndim != 3:
            raise ValueError(f"Velociwrap expects a 3D cube after squeeze; got shape {data.shape}.")
        self.data = data
        
        self.wcs = WCS(self.header).celestial
        self.spectral_axis_type = spectral_axis_type or self._detect_spectral_axis_type()
        self.spectral_unit = spectral_unit or self._detect_spectral_unit()
        self.spectral_axis_native = self._build_spectral_axis_native()
        self.spectral_axis = self._standardize_spectral_axis()

    @property
    def shape(self):
        return self.data.shape

    @property
    def unit(self):
        return self.header.get("BUNIT", "")

    @property
    def velocity_axis_kms(self):
        if self.spectral_axis_type == "velocity":
            if self.spectral_unit in ["m/s", "ms-1", "m s-1"]:
                return self.spectral_axis / 1000.0
            return self.spectral_axis

        raise ValueError(
            "Cube spectral axis is not already velocity. "
            "For optical cubes, make a velocity cube first or pass a rest wavelength in a future version."
        )

    def _detect_spectral_axis_type(self):
        ctype = str(self.header.get("CTYPE3", "")).upper()
        cunit = str(self.header.get("CUNIT3", "")).lower()
        if "VELO" in ctype or "VRAD" in ctype or "VOPT" in ctype:
            return "velocity"
        if "FREQ" in ctype or "HZ" in cunit:
            return "frequency"
        if "WAVE" in ctype or "AWAV" in ctype or "WAVE" in ctype:
            return "wavelength"
        if "WAVN" in ctype or "CM-1" in cunit or "1/CM" in cunit:
            return "wavenumber"
        return "unknown"

    def _detect_spectral_unit(self):
        cunit = str(self.header.get("CUNIT3", "")).strip()
        ctype = str(self.header.get("CTYPE3", "")).upper()
        if cunit:
            return cunit
        if self.spectral_axis_type == "velocity":
            # Legacy radio cubes often omit CUNIT3 but store VELO-LSR in m/s.
            # Prefer user override when ambiguous.
            if "VELO" in ctype:
                return "m/s"
            return "km/s"
        if self.spectral_axis_type == "frequency":
            return "Hz"
        if self.spectral_axis_type == "wavenumber":
            return "cm-1"
        if self.spectral_axis_type == "wavelength":
            return "Angstrom"
        return ""

    def _build_spectral_axis_native(self):
        n_spec = self.data.shape[0]
        crval = self.header.get("CRVAL3")
        crpix = self.header.get("CRPIX3", 1.0)
        cdelt = self.header.get("CDELT3", self.header.get("CD3_3"))
        if crval is None or cdelt is None:
            raise ValueError("Could not find spectral WCS keywords CRVAL3 and CDELT3/CD3_3.")
        pix = np.arange(n_spec) + 1.0
        return crval + (pix - crpix) * cdelt

    def _standardize_spectral_axis(self):
        axis = np.asarray(self.spectral_axis_native, dtype=float)
        kind = self.spectral_axis_type
        unit = self.spectral_unit
        if kind == "velocity":
            return velocity_to_kms(axis, unit)
        if kind == "wavenumber":
            return wavenumber_to_wavelength(axis)
        if kind == "frequency":
            if unit and unit.lower() in {"ghz"}:
                return axis * 1e9
            if unit and unit.lower() in {"mhz"}:
                return axis * 1e6
            return axis
        if kind == "wavelength":
            try:
                q = axis * u.Unit(unit)
                return q.to(u.AA).value
            except Exception:
                return axis
        return axis

    def _find_data_hdu(self):
        for i, hdu in enumerate(self.hdul):
            if hdu.data is not None:
                data = hdu.data
                if getattr(data, "ndim", 0) >= 3:
                    return i

        raise ValueError("No 3D data cube found in this FITS file.")


