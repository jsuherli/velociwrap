### velociwrap
### 
### A lightweight python package to create velocity-consistent channel maps 
### and position-velocity (PV) diagrams from astronomical datacube. 
###
### Janette Suherli (c)2026. 
### jsuherli@gmail.com
###
### PV diagram helpers.

import numpy as np

import matplotlib.pyplot as plt

from dataclasses import dataclass

from scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d

from .velocity import wavelength_to_velocity
from .plotting import normalize_image


@dataclass
class PVDiagram:
    data: np.ndarray
    offset: np.ndarray

    spectral_axis: np.ndarray
    spectral_axis_type: str
    velocity_axis: np.ndarray | None

    start: tuple
    end: tuple
    width: float

    metadata: dict


def _sample_line(start, end, step=1.0):
    x0, y0 = start
    x1, y1 = end
    length = np.hypot(x1 - x0, y1 - y0)

    n = int(np.ceil(length / step)) + 1
    x = np.linspace(x0, x1, n)
    y = np.linspace(y0, y1, n)

    offset = np.linspace(0, length, n)

    return x, y, offset


def _perpendicular_offsets(start, end, width):
    if width <= 1:
        return np.array([0.0]), np.array([0.0])

    x0, y0 = start
    x1, y1 = end

    dx = x1 - x0
    dy = y1 - y0
    length = np.hypot(dx, dy)

    px = -dy / length
    py = dx / length
    offsets = np.linspace(-0.5 * width, 0.5 * width, int(np.ceil(width)))

    return (px * offsets, py * offsets)


def _extract_native_pv(
    cube,
    start,
    end,
    width=1,
    step=1.0,
    statistic="mean",
):

    data = cube.data

    x, y, offset = _sample_line(start, end, step=step)
    dx_perp, dy_perp = _perpendicular_offsets(start, end, width)

    samples = []
    for ox, oy in zip(dx_perp, dy_perp):
        xs = x + ox
        ys = y + oy

        slit = []
        for k in range(data.shape[0]):
            coords = np.vstack([np.full_like(xs, k), ys, xs])
            vals = map_coordinates(data, coords, order=1, mode="nearest")
            slit.append(vals)
        samples.append(np.asarray(slit))
    samples = np.asarray(samples)

    if statistic == "mean":
        pv_data = np.nanmean(samples, axis=0)
    elif statistic == "sum":
        pv_data = np.nansum(samples, axis=0)
    elif statistic == "median":
        pv_data = np.nanmedian(samples, axis=0)
    else:
        raise ValueError("statistic must be mean/sum/median")

    return pv_data, offset


def extract_pv(
    cube,
    start,
    end,
    width=1,
    step=1,
    rest_value=None,
    vmin=None,
    vmax=None,
    dv=None,
    mode="native",
    statistic="mean",
):
    pv_data, offset = _extract_native_pv(cube, start, end, width=width, step=step, statistic=statistic)
    spectral_axis = cube.spectral_axis
    velocity_axis = None

    if mode == "native":
        if cube.spectral_axis_type == "velocity":
            velocity_axis = spectral_axis
        elif (cube.spectral_axis_type == "wavelength" and rest_value is not None):
            velocity_axis = wavelength_to_velocity(spectral_axis, rest_value)
    elif mode == "interpolated":
        if rest_value is None:
            raise ValueError("rest_value required for interpolated mode.")
        velocity_native = wavelength_to_velocity(spectral_axis, rest_value)
        velocity_axis = np.arange(vmin, vmax + dv, dv)
        f = interp1d(velocity_native, pv_data, axis=0, bounds_error=False, fill_value=np.nan)
        pv_data = f(velocity_axis)
    else:
        raise ValueError("mode must be native or interpolated")

    return PVDiagram(
        data=pv_data,
        offset=offset,
        spectral_axis=spectral_axis,
        spectral_axis_type=cube.spectral_axis_type,
        velocity_axis=velocity_axis,
        start=start,
        end=end,
        width=width,
        metadata={
            "mode": mode,
            "rest_value": rest_value,
        },
    )


def plot_pv(
    pv,
    cmap="gray_r",
    stretch="asinh",
    percentile=(1, 99.5),
    colorbar=True,
    colorbar_label=None,
    figsize=(7, 5),
):

    fig, ax = plt.subplots(figsize=figsize)

    finite = pv.data[np.isfinite(pv.data)]
    vmin = np.nanpercentile(finite, percentile[0])
    vmax = np.nanpercentile(finite, percentile[1])

    image = normalize_image(pv.data, vmin=vmin, vmax=vmax, stretch=stretch)

    if pv.velocity_axis is not None:
        extent = [
            pv.offset.min(),
            pv.offset.max(),
            pv.velocity_axis.min(),
            pv.velocity_axis.max(),
        ]
        ylabel = r"Velocity (km s$^{-1}$)"
    else:
        extent = [
            pv.offset.min(),
            pv.offset.max(),
            pv.spectral_axis.min(),
            pv.spectral_axis.max(),
        ]
        ylabel = pv.spectral_axis_type

    im = ax.imshow(image, origin="lower", aspect="auto", extent=extent, cmap=cmap, vmin=0, vmax=1)
    ax.set_xlabel("Offset (pixel)")
    ax.set_ylabel(ylabel)
    ax.tick_params(direction="in", top=True, right=True)

    if colorbar:
        cbar = fig.colorbar(im, ax=ax)
        
        if colorbar_label:
            cbar.set_label(colorbar_label)

    return fig, ax


def add_pv_slit(
    ax,
    start,
    end,
    width=None,
    color="cyan",
    linewidth=1.5,
):

    x0, y0 = start
    x1, y1 = end

    ax.plot([x0, x1], [y0, y1], color=color, linewidth=linewidth)

    if width is None:
        return

    dx = x1 - x0
    dy = y1 - y0
    length = np.hypot(dx, dy)
    px = -dy / length
    py = dx / length

    for sign in [-1, 1]:
        ox = sign * width * px / 2
        oy = sign * width * py / 2

        ax.plot([x0 + ox, x1 + ox], [y0 + oy, y1 + oy], color=color, alpha=0.5, linewidth=linewidth)




