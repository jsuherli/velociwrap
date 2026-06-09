### velociwrap
### 
### A lightweight python package to create velocity-consistent channel maps 
### and position-velocity (PV) diagrams from astronomical datacube. 
###
### Janette Suherli (c)2026. 
### jsuherli@gmail.com
###
### Image stretching, normalization, and masking helpers.

from __future__ import annotations

import numpy as np


def robust_sigma(data):
    """
    Robust sigma estimate using 1.4826 * MAD. (Median Absolute Deviation)
    """
    arr = np.asarray(data, dtype=float)
    med = np.nanmedian(arr)
    mad = np.nanmedian(np.abs(arr - med))
    return 1.4826 * mad


def apply_mask(image, mask_below=None, noise=None, mask_sigma=None):
    """
    To mask values below an absolute threshold or sigma threshold.
    """
    out = np.array(image, dtype=float, copy=True)
    threshold = mask_below
    
    if mask_sigma is not None:
        sigma = noise if noise is not None else robust_sigma(out)
        threshold = mask_sigma * sigma
    
    if threshold is not None:
        out[out < threshold] = np.nan
    
    return out


def stretch_image(image, stretch="asinh", asinh_a=0.1):
    """
    To apply a display stretch to an already normalized image in [0, 1].
    """
    x = np.clip(np.asarray(image, dtype=float), 0, 1)
    
    if stretch in {None, "linear"}:
        return x
    if stretch == "sqrt":
        return np.sqrt(x)
    if stretch == "log":
        return np.log10(1 + 1000 * x) / np.log10(1001)
    if stretch == "asinh":
        return np.arcsinh(x / asinh_a) / np.arcsinh(1 / asinh_a)
    
    raise ValueError("stretch must be 'linear', 'sqrt', 'log', or 'asinh'.")


def normalize_image(
    image,
    vmin=None,
    vmax=None,
    percentile=(1, 99.5),
    stretch="asinh",
    asinh_a=0.1,
    mask_below=None,
    noise=None,
    mask_sigma=None,
):
    """
    To normalize one image for display.
    """
    image = apply_mask(image, mask_below=mask_below, noise=noise, mask_sigma=mask_sigma)
    
    if vmin is None or vmax is None:
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            return np.zeros_like(image, dtype=float)
        p0, p1 = percentile
        if vmin is None:
            vmin = np.nanpercentile(finite, p0)
        if vmax is None:
            vmax = np.nanpercentile(finite, p1)
    
    if not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(image, dtype=float)
    
    norm = (image - vmin) / (vmax - vmin)
    
    return stretch_image(norm, stretch=stretch, asinh_a=asinh_a)


