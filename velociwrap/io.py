### velociwrap
### 
### A lightweight python package to create velocity-consistent channel maps 
### and position-velocity (PV) diagrams from astronomical datacube. 
###
### Janette Suherli (c)2026. 
### jsuherli@gmail.com
###
### Export helpers.

from __future__ import annotations

import numpy as np

from astropy.io import fits


def _add_velociwrap_header(header, metadata=None):
    header = header.copy()
    header["VELOCIWR"] = (True, "Created/modified by Velociwrap")
    header["VW_VER"] = ("1.0.0b0", "Velociwrap version")

    if metadata:
        for key, value in metadata.items():
            hkey = ("VW_" + key.upper())[:8]
            if np.isscalar(value) and not isinstance(value, (dict, list, tuple)):
                try:
                    header[hkey] = value
                except Exception:
                    pass

    return header


def save_channel_maps_fits(maps, filename, header=None, velocity_edges=None, metadata=None, overwrite=True):
    hdr = header.copy() if header is not None else fits.Header()
    hdr = _add_velociwrap_header(hdr, metadata)
    hdr["NAXIS"] = 3
    hdr["NAXIS3"] = maps.shape[0]
    hdr["CTYPE3"] = "VELO-BIN"
    hdr["CUNIT3"] = "km/s"
    
    if velocity_edges is not None and len(velocity_edges) > 1:
        hdr["CRVAL3"] = float(0.5 * (velocity_edges[0] + velocity_edges[1]))
        hdr["CDELT3"] = float(velocity_edges[1] - velocity_edges[0])
        hdr["CRPIX3"] = 1.0
        hdr["VW_VMIN"] = float(velocity_edges[0])
        hdr["VW_VMAX"] = float(velocity_edges[-1])
    
    fits.PrimaryHDU(data=np.asarray(maps, dtype=float), header=hdr).writeto(filename, overwrite=overwrite)


def save_rgb_components_fits(components, filename, header=None, velocity_edges=None, metadata=None, overwrite=True):
    hdus = [fits.PrimaryHDU(header=_add_velociwrap_header(header or fits.Header(), metadata))]
    
    for label, maps in components.items():
        hdr = _add_velociwrap_header(header or fits.Header(), metadata)
        hdr["EXTNAME"] = label
        
        if velocity_edges is not None and len(velocity_edges) > 1:
            hdr["CTYPE3"] = "VELO-BIN"
            hdr["CUNIT3"] = "km/s"
            hdr["CRVAL3"] = float(0.5 * (velocity_edges[0] + velocity_edges[1]))
            hdr["CDELT3"] = float(velocity_edges[1] - velocity_edges[0])
            hdr["CRPIX3"] = 1.0
        hdus.append(fits.ImageHDU(data=np.asarray(maps, dtype=float), header=hdr, name=label))
    
    fits.HDUList(hdus).writeto(filename, overwrite=overwrite)


