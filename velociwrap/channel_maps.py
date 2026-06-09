### velociwrap
### 
### A lightweight python package to create velocity-consistent channel maps 
### and position-velocity (PV) diagrams from astronomical datacube. 
###
### Janette Suherli (c)2026. 
### jsuherli@gmail.com
###
### Velocity channel-map and RGB channel-map generation.

from __future__ import annotations

import numpy as np

from scipy.interpolate import interp1d

from matplotlib.colors import to_rgb

from .velocity import wavelength_to_velocity, velocity_to_wavelength, frequency_to_velocity, velocity_to_frequency, make_velocity_bins
from .scaling import normalize_image


def _spectral_axis_as_velocity(cube, rest_value=None, convention="optical"):
    if cube.spectral_axis_type == "velocity":
        return cube.spectral_axis
    if rest_value is None:
        raise ValueError("rest_wavelength/rest_frequency is required for non-velocity cubes.")
    if cube.spectral_axis_type in {"wavelength", "wavenumber"}:
        return wavelength_to_velocity(cube.spectral_axis, rest_value, convention=convention)
    if cube.spectral_axis_type == "frequency":
        return frequency_to_velocity(cube.spectral_axis, rest_value, convention=convention)
    raise ValueError(f"Unsupported spectral axis type: {cube.spectral_axis_type}")


def _target_axis_from_velocity(cube, velocity, rest_value=None, convention="optical"):
    if cube.spectral_axis_type == "velocity":
        return np.asarray(velocity, dtype=float)
    if cube.spectral_axis_type in {"wavelength", "wavenumber"}:
        return velocity_to_wavelength(velocity, rest_value, convention=convention)
    if cube.spectral_axis_type == "frequency":
        return velocity_to_frequency(velocity, rest_value, convention=convention)
    raise ValueError(f"Unsupported spectral axis type: {cube.spectral_axis_type}")


def _local_cutout(cube, rest_value, vmin, vmax, padding_channels=3, convention="optical"):
    target1 = _target_axis_from_velocity(cube, vmin, rest_value, convention)
    target2 = _target_axis_from_velocity(cube, vmax, rest_value, convention)
    amin, amax = min(target1, target2), max(target1, target2)

    axis = cube.spectral_axis
    order = np.argsort(axis)
    axis_sorted = axis[order]
    i0 = np.searchsorted(axis_sorted, amin) - padding_channels
    i1 = np.searchsorted(axis_sorted, amax) + padding_channels + 1
    i0 = max(i0, 0)
    i1 = min(i1, len(axis_sorted))
    use = order[i0:i1]
    axis_cut = axis[use]
    data_cut = cube.data[use, :, :]
    sort_cut = np.argsort(axis_cut)
    axis_cut = axis_cut[sort_cut]
    data_cut = data_cut[sort_cut]
    if len(axis_cut) < 2:
        raise ValueError("Not enough spectral channels in the requested velocity range.")
    return axis_cut, data_cut


def make_channel_maps(
    cube,
    rest_value=None,
    vmin=-300,
    vmax=300,
    dv=50,
    mode="resample",
    samples_per_bin=5,
    statistic="integrated",
    padding_channels=3,
    convention="optical",
    return_metadata=True,
):
    """
    To create single-line velocity channel maps.

    For wavelength/wavenumber cubes, rest_value is rest wavelength in Angstrom.
    For frequency cubes, rest_value is rest frequency in Hz.
    For velocity cubes, rest_value can be omitted.
    """
    edges, centers = make_velocity_bins(vmin, vmax, dv)
    maps = []

    if mode == "native":
        velocity = _spectral_axis_as_velocity(cube, rest_value, convention)
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (velocity >= lo) & (velocity < hi)
            if not np.any(mask):
                maps.append(np.full(cube.data.shape[1:], np.nan))
            else:
                if statistic == "mean":
                    maps.append(np.nanmean(cube.data[mask], axis=0))
                else:
                    maps.append(np.nansum(cube.data[mask], axis=0))
        maps = np.asarray(maps)

    elif mode == "resample":
        axis_cut, data_cut = _local_cutout(
            cube, rest_value, vmin, vmax, padding_channels=padding_channels, convention=convention
        )
        interpolator = interp1d(axis_cut, data_cut, axis=0, bounds_error=False, fill_value=np.nan)
        for lo, hi in zip(edges[:-1], edges[1:]):
            v_samples = np.linspace(lo, hi, samples_per_bin)
            axis_samples = _target_axis_from_velocity(cube, v_samples, rest_value, convention)
            sampled = interpolator(axis_samples)
            if statistic == "mean":
                channel_map = np.nanmean(sampled, axis=0)
            elif statistic == "integrated":
                channel_map = np.trapz(sampled, axis_samples, axis=0)
            else:
                raise ValueError("For mode='resample', statistic must be 'integrated' or 'mean'.")
            maps.append(channel_map)
        maps = np.asarray(maps)
    else:
        raise ValueError("mode must be 'native' or 'resample'.")

    meta = channel_metadata(
        edges=edges,
        centers=centers,
        rest_value=rest_value,
        mode=mode,
        statistic=statistic,
        samples_per_bin=samples_per_bin,
        spectral_axis_type=cube.spectral_axis_type,
        spectral_unit=cube.spectral_unit,
        convention=convention,
    )
    return (maps, edges, centers, meta) if return_metadata else (maps, edges, centers)


def make_integrated_map(
    cube,
    rest_value=None,
    vmin=None,
    vmax=None,
    native_channels=None,
    mode="resample",
    samples_per_bin=25,
    statistic="integrated",
    convention="optical",
):
    """
    To create one integrated map either from velocity limits or native channels.
    """
    if native_channels is not None:
        velocity = _spectral_axis_as_velocity(cube, rest_value, convention)
        center_index = int(np.argmin(np.abs(velocity)))
        half = native_channels // 2
        i0 = max(center_index - half, 0)
        i1 = min(i0 + native_channels, cube.data.shape[0])
        image = np.nansum(cube.data[i0:i1], axis=0)
        return image, {"mode": "native_channels", "channels": (i0, i1), "velocity_range": (velocity[i0], velocity[i1-1])}
    
    if vmin is None or vmax is None:
        raise ValueError("Provide either native_channels or both vmin and vmax.")
    
    maps, edges, centers, meta = make_channel_maps(
        cube,
        rest_value=rest_value,
        vmin=vmin,
        vmax=vmax,
        dv=vmax - vmin,
        mode=mode,
        samples_per_bin=samples_per_bin,
        statistic=statistic,
        convention=convention,
    )
    return maps[0], meta


def make_rgb_channel_maps(
    cube,
    lines,
    colors=None,
    vmin=-300,
    vmax=300,
    dv=50,
    mode="resample",
    samples_per_bin=5,
    statistic="integrated",
    percentile=(1, 99.5),
    stretch="asinh",
    asinh_a=0.1,
    rgb_scale="line",
    padding_channels=3,
    convention="optical",
    return_components=False,
):
    """
    To create 2-line or 3-line velocity-matched color channel maps.
    """
    if len(lines) not in {2, 3}:
        raise ValueError("lines must contain either 2 or 3 entries.")
    
    if colors is None:
        defaults = ["red", "green", "blue"]
        colors = {label: defaults[i] for i, label in enumerate(lines)}
    
    components = {}
    edges = centers = meta = None
    
    for label, rest in lines.items():
        maps, edges, centers, meta = make_channel_maps(
            cube,
            rest_value=rest,
            vmin=vmin,
            vmax=vmax,
            dv=dv,
            mode=mode,
            samples_per_bin=samples_per_bin,
            statistic=statistic,
            padding_channels=padding_channels,
            convention=convention,
            return_metadata=True,
        )
        components[label] = maps

    nchan, ny, nx = next(iter(components.values())).shape
    rgb = np.zeros((nchan, ny, nx, 3), dtype=float)

    if rgb_scale == "global":
        all_values = np.concatenate([m[np.isfinite(m)].ravel() for m in components.values()])
        gvmin, gvmax = np.nanpercentile(all_values, percentile)

    for label, maps in components.items():
        color = np.array(to_rgb(colors[label]))
        if rgb_scale == "line":
            finite = maps[np.isfinite(maps)]
            lvmin, lvmax = np.nanpercentile(finite, percentile)
        
        for i in range(nchan):
            if rgb_scale == "channel":
                image = normalize_image(maps[i], percentile=percentile, stretch=stretch, asinh_a=asinh_a)
            elif rgb_scale == "line":
                image = normalize_image(maps[i], vmin=lvmin, vmax=lvmax, stretch=stretch, asinh_a=asinh_a)
            elif rgb_scale == "global":
                image = normalize_image(maps[i], vmin=gvmin, vmax=gvmax, stretch=stretch, asinh_a=asinh_a)
            else:
                raise ValueError("rgb_scale must be 'line', 'global', or 'channel'.")
            rgb[i] += image[:, :, None] * color[None, None, :]
    
    rgb = np.clip(rgb, 0, 1)
    rgb_meta = meta or {}
    rgb_meta.update({"lines": lines, "colors": colors, "rgb_scale": rgb_scale, "stretch": stretch})
    
    if return_components:
        return rgb, edges, centers, rgb_meta, components
    return rgb, edges, centers, rgb_meta


def channel_metadata(**kwargs):
    """
    To return a metadata dictionary for velocity products.
    """
    return dict(kwargs)


def velocity_bin_table(edges):
    """
    To return a simple list of velocity-bin table rows.
    """
    return [
        {"channel": i, "vmin": float(lo), "vmax": float(hi), "label": f"{lo:.0f} to {hi:.0f} km/s"}
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))
    ]


