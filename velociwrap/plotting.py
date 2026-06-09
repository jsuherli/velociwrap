### velociwrap
###
### A lightweight python package to create velocity-consistent channel maps
### and position-velocity (PV) diagrams from astronomical datacubes.
###
### Janette Suherli (c) 2026.
### jsuherli@gmail.com
###
### Publication-style plotting functions.

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import FancyArrowPatch
from matplotlib.offsetbox import HPacker, TextArea, AnchoredOffsetbox
from matplotlib.colors import Normalize, AsinhNorm, LogNorm, PowerNorm
from mpl_toolkits.axes_grid1 import ImageGrid

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs.utils import proj_plane_pixel_scales

from .scaling import normalize_image


def _arcsec_per_pixel(wcs):
    if wcs is None:
        raise ValueError("A celestial WCS is required.")

    w = wcs.celestial if hasattr(wcs, "celestial") else wcs
    scales_deg = proj_plane_pixel_scales(w)
    return float(np.nanmean(np.abs(scales_deg)) * 3600.0)


def add_scalebar(
    ax,
    wcs,
    length_arcsec=10,
    loc="lower right",
    color="white",
    lw=3,
    fontsize=10,
):
    """
    To add a scale bar using the celestial WCS.
    """
    if wcs is None:
        raise ValueError("add_scalebar requires a celestial WCS.")

    pixscale = _arcsec_per_pixel(wcs)
    length_pix = length_arcsec / pixscale

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    dx = abs(xlim[1] - xlim[0])
    dy = abs(ylim[1] - ylim[0])

    margin_x = 0.08 * dx
    margin_y = 0.08 * dy

    if "right" in loc:
        x1 = max(xlim) - margin_x
        x0 = x1 - length_pix
    else:
        x0 = min(xlim) + margin_x
        x1 = x0 + length_pix

    if "lower" in loc:
        y = min(ylim) + margin_y
        va = "bottom"
        y_text = y + 0.03 * dy
    else:
        y = max(ylim) - margin_y
        va = "top"
        y_text = y - 0.03 * dy

    ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt")
    ax.text(0.5 * (x0 + x1), y_text, f'{length_arcsec:g}"', color=color, ha="center", va=va, fontsize=fontsize)


def add_north_east(
    ax,
    wcs,
    loc=(0.86, 0.78),
    size=0.10,
    color="white",
    fontsize=10,
    lw=1.8,
    mutation_scale=12,
):
    """
    To draw gap-free North/East arrows.
    This function requires WCS. 
    """
    if wcs is None:
        raise ValueError("add_north_east requires a celestial WCS.")

    if len(ax.images) == 0:
        raise ValueError("add_north_east requires an image already plotted on the axis.")

    w = wcs.celestial if hasattr(wcs, "celestial") else wcs

    x_ax, y_ax = loc

    arr = ax.images[0].get_array()
    ny, nx = arr.shape[:2]

    x_pix = x_ax * nx
    y_pix = y_ax * ny

    sky0 = w.pixel_to_world(x_pix, y_pix)

    if not isinstance(sky0, SkyCoord):
        sky0 = SkyCoord(sky0)

    sky_n = SkyCoord(ra=sky0.ra, dec=sky0.dec + 1 * u.arcsec, frame=sky0.frame)
    sky_e = SkyCoord(ra=sky0.ra + (1 * u.arcsec) / np.cos(sky0.dec), dec=sky0.dec, frame=sky0.frame)

    x_n, y_n = w.world_to_pixel(sky_n)
    x_e, y_e = w.world_to_pixel(sky_e)

    dx_n = (x_n - x_pix) / nx
    dy_n = (y_n - y_pix) / ny
    dx_e = (x_e - x_pix) / nx
    dy_e = (y_e - y_pix) / ny

    norm_n = np.hypot(dx_n, dy_n)
    norm_e = np.hypot(dx_e, dy_e)

    if norm_n == 0 or norm_e == 0:
        raise ValueError("Could not determine WCS north/east direction.")

    dx_n = dx_n / norm_n * size
    dy_n = dy_n / norm_n * size
    dx_e = dx_e / norm_e * size
    dy_e = dy_e / norm_e * size

    arr_n = FancyArrowPatch(
        (x_ax, y_ax),
        (x_ax + dx_n, y_ax + dy_n),
        transform=ax.transAxes,
        arrowstyle="-|>",
        color=color,
        lw=lw,
        shrinkA=0,
        shrinkB=0,
        mutation_scale=mutation_scale,
    )
    ax.add_patch(arr_n)

    arr_e = FancyArrowPatch(
        (x_ax, y_ax),
        (x_ax + dx_e, y_ax + dy_e),
        transform=ax.transAxes,
        arrowstyle="-|>",
        color=color,
        lw=lw,
        shrinkA=0,
        shrinkB=0,
        mutation_scale=mutation_scale,
    )
    ax.add_patch(arr_e)

    ax.text(x_ax + dx_n * 1.12, y_ax + dy_n * 1.12, "N", transform=ax.transAxes, color=color, ha="center", va="center", ontsize=fontsize)
    ax.text(x_ax + dx_e * 1.18, y_ax + dy_e * 1.18, "E", transform=ax.transAxes, color=color, ha="center", va="center", ontsize=fontsize)


def _add_velocity_label(
    ax,
    lo,
    hi,
    loc="upper left",
    color="white",
    label_box=True,
    fontsize=10,
):
    """
    To add velocity-bin label inside a panel.
    """
    bbox = None
    if label_box:
        bbox = dict(facecolor="black", alpha=0.45, edgecolor="none", pad=2)

    if loc == "upper left":
        x, y, ha, va = 0.03, 0.97, "left", "top"
    elif loc == "upper right":
        x, y, ha, va = 0.97, 0.97, "right", "top"
    elif loc == "lower left":
        x, y, ha, va = 0.03, 0.03, "left", "bottom"
    elif loc == "lower right":
        x, y, ha, va = 0.97, 0.03, "right", "bottom"
    else:
        raise ValueError("label_loc must be upper/lower left/right.")

    label = rf"{lo:.0f} to {hi:.0f} km s$^{{-1}}$"
    ax.text(x, y, label, transform=ax.transAxes, color=color, ha=ha, va=va, fontsize=fontsize, bbox=bbox)


def _add_contours(ax, contours, index):
    if contours is None:
        return

    data = contours["data"]
    channels = contours.get("channels", "all")

    if channels != "all" and index not in channels:
        return

    image = data if np.ndim(data) == 2 else data[index]
    kwargs = {k: v for k, v in contours.items() if k not in {"data", "channels"}}

    ax.contour(image, **kwargs)


def _add_regions(ax, regions, index, wcs=None):
    if regions is None:
        return

    try:
        from regions import Regions
    except Exception as exc:
        raise ImportError("DS9 region overlay requires the 'regions' package.") from exc

    if isinstance(regions, str):
        regfile = regions
        channels = "all"
        kwargs = {}
    else:
        regfile = regions.get("file")
        channels = regions.get("channels", "all")
        kwargs = regions.get("kwargs", {})

    if channels != "all" and index not in channels:
        return

    regs = Regions.read(regfile, format="ds9")

    for reg in regs:
        try:
            pixreg = reg.to_pixel(wcs) if getattr(reg, "coord_format", None) == "fk5" and wcs else reg
            artist = pixreg.as_artist(**kwargs)
            ax.add_artist(artist)
        except Exception:
            continue


def _draw_panel_extras(
    ax,
    index,
    wcs=None,
    scalebar=False,
    scalebar_mode="all",
    scalebar_kwargs=None,
    north_east=False,
    ne_mode="all",
    ne_kwargs=None,
):
    """
    To draw optional panel extras consistently.
    scalebar_mode, ne_mode : {"first", "all", "none"}.
    """
    if scalebar_mode not in {"first", "all", "none"}:
        raise ValueError("scalebar_mode must be 'first', 'all', or 'none'.")

    if ne_mode not in {"first", "all", "none"}:
        raise ValueError("ne_mode must be 'first', 'all', or 'none'.")

    scalebar_kwargs = scalebar_kwargs or {}
    ne_kwargs = ne_kwargs or {}

    draw_scalebar = (
        scalebar
        and scalebar_mode != "none"
        and (scalebar_mode == "all" or index == 0)
    )

    if draw_scalebar:
        if wcs is None:
            raise ValueError("scalebar=True requires wcs.")
        add_scalebar(ax, wcs=wcs, **scalebar_kwargs)

    draw_ne = (
        north_east
        and ne_mode != "none"
        and (ne_mode == "all" or index == 0)
    )

    if draw_ne:
        if wcs is None:
            raise ValueError("north_east=True requires wcs.")
        add_north_east(ax, wcs=wcs, **ne_kwargs)


def _hide_wcs_coord(coord):
    coord.set_ticks_visible(False)
    coord.set_ticklabel_visible(False)
    coord.set_axislabel("")


def _apply_axes(ax, axes_mode, row, col, nrows, ncols, use_wcs=False):
    """
    To apply x-y axis. axes_mode options:
        "none"  : no ticks/labels
        "pixel" : pixel axes on all panels
        "wcs"   : WCS axes on all panels
        "outer" : only bottom row and left column show labels
    """
    if axes_mode in {False, "none", None}:
        if use_wcs and hasattr(ax, "coords"):
            for coord in ax.coords:
                _hide_wcs_coord(coord)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
        return

    if axes_mode == "pixel":
        ax.tick_params(direction="in", top=True, right=True)
        return

    if axes_mode == "wcs":
        if use_wcs and hasattr(ax, "coords"):
            ax.coords[0].set_ticks(direction="in")
            ax.coords[1].set_ticks(direction="in")
        else:
            ax.tick_params(direction="in", top=True, right=True)
        return

    if axes_mode == "outer":
        is_bottom = row == nrows - 1
        is_left = col == 0

        if use_wcs and hasattr(ax, "coords"):
            lon = ax.coords[0]
            lat = ax.coords[1]

            lon.set_ticks(direction="in")
            lat.set_ticks(direction="in")

            if not is_bottom:
                _hide_wcs_coord(lon)

            if not is_left:
                _hide_wcs_coord(lat)

        else:
            ax.tick_params(direction="in", top=True, right=True)

            if not is_bottom:
                ax.set_xticklabels([])
                ax.set_xlabel("")

            if not is_left:
                ax.set_yticklabels([])
                ax.set_ylabel("")

        return

    raise ValueError("axes must be 'none', 'pixel', 'wcs', or 'outer'.")


def _rgb_line_legend(fig, lines, colors, y=0.98):
    boxes = []

    for label, rest in lines.items():
        text = TextArea(
            f"{label} {rest:g}  ",
            textprops=dict(color=colors[label], size=13),
        )
        boxes.append(text)

    packed = HPacker(
        children=boxes,
        align="center",
        pad=0,
        sep=6,
    )

    anchored = AnchoredOffsetbox(
        loc="upper center",
        child=packed,
        pad=0,
        frameon=False,
        bbox_to_anchor=(0.5, y),
        bbox_transform=fig.transFigure,
        borderpad=0,
    )

    fig.add_artist(anchored)


def _stretch_rgb(rgb, stretch="asinh", clip=True, asinh_a=0.15):
    """
    To apply stretch to an RGB image already scaled to roughly 0-1.
    """
    rgb = np.asarray(rgb, dtype=float)

    if clip:
        rgb = np.clip(rgb, 0, 1)

    if stretch in {None, "linear"}:
        out = rgb
    elif stretch == "sqrt":
        out = np.sqrt(np.clip(rgb, 0, None))
    elif stretch == "asinh":
        out = np.arcsinh(rgb / asinh_a) / np.arcsinh(1.0 / asinh_a)
    else:
        raise ValueError("stretch must be 'linear', 'sqrt', or 'asinh'.")

    if clip:
        out = np.clip(out, 0, 1)

    return out


def plot_channel_grid(
    maps,
    velocity_edges,
    ncols=4,
    origin="lower",
    cmap="gray_r",
    wcs=None,
    axes="none",
    percentile=(1, 99.5),
    scale="global",
    stretch="asinh",
    asinh_a=0.1,
    colorbar=True,
    colorbar_label=None,
    panel_spacing=0.03,
    cbar_size="3%",
    cbar_pad=0.08,
    figsize=None,
    velocity_label=True,
    label_loc="upper left",
    label_color="white",
    label_box=True,
    scalebar=False,
    scalebar_mode="all",
    scalebar_kwargs=None,
    north_east=False,
    ne_mode="all",
    ne_kwargs=None,
    contours=None,
    regions=None,
    annotate_func=None,
    savepath=None,
    dpi=300,
):
    """
    To plot single-line velocity channel maps.
    """
    if maps.ndim != 3:
        raise ValueError("maps must have shape (n_channels, ny, nx).")

    if ncols not in {3, 4, 5}:
        raise ValueError("ncols must be 3, 4, or 5.")

    nchan = maps.shape[0]

    if len(velocity_edges) != nchan + 1:
        raise ValueError("velocity_edges must have length n_channels + 1.")

    ncols = min(ncols, nchan)
    nrows = int(np.ceil(nchan / ncols))

    use_wcs = axes in {"wcs", "outer"} and wcs is not None

    if figsize is None:
        figsize = (3.0 * ncols + (0.45 if colorbar else 0), 3.0 * nrows)

    fig = plt.figure(figsize=figsize)

    if use_wcs:
        from astropy.visualization.wcsaxes import WCSAxes
        axes_class = (WCSAxes, {"wcs": wcs})
    else:
        axes_class = None

    grid_kwargs = dict(
        nrows_ncols=(nrows, ncols),
        axes_pad=panel_spacing,
        share_all=True,
        cbar_mode="single" if colorbar else None,
        cbar_location="right",
        cbar_pad=cbar_pad,
        cbar_size=cbar_size,
    )

    if axes_class is not None:
        grid_kwargs["axes_class"] = axes_class

    grid = ImageGrid(fig, 111, **grid_kwargs)

    if scale == "global":
        finite = maps[np.isfinite(maps)]
        if finite.size == 0:
            gvmin, gvmax = 0.0, 1.0
        else:
            gvmin = np.nanpercentile(finite, percentile[0])
            gvmax = np.nanpercentile(finite, percentile[1])
    elif scale != "individual":
        raise ValueError("scale must be 'global' or 'individual'.")

    last_im = None

    for i, ax in enumerate(grid):
        row, col = divmod(i, ncols)

        if i >= nchan:
            ax.axis("off")
            continue

        image = maps[i]

        if scale == "individual":
            finite = image[np.isfinite(image)]
            if finite.size == 0:
                vmin, vmax = 0.0, 1.0
            else:
                vmin = np.nanpercentile(finite, percentile[0])
                vmax = np.nanpercentile(finite, percentile[1])
        else:
            vmin, vmax = gvmin, gvmax

        if stretch in {None, "linear"}:
            norm = Normalize(vmin=vmin, vmax=vmax)
        elif stretch == "asinh":
            norm = AsinhNorm(linear_width=asinh_a, vmin=vmin, vmax=vmax)
        elif stretch == "sqrt":
            norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
        elif stretch == "log":
            positive = image[np.isfinite(image) & (image > 0)]
            if positive.size == 0:
                norm = Normalize(vmin=vmin, vmax=vmax)
            else:
                safe_vmin = max(
                    np.nanpercentile(positive, percentile[0]),
                    np.nanmin(positive),
                )
                norm = LogNorm(vmin=safe_vmin, vmax=vmax)
        else:
            raise ValueError("stretch must be 'linear', 'asinh', 'sqrt', or 'log'.")

        last_im = ax.imshow(image, origin=origin, cmap=cmap, norm=norm)

        lo = velocity_edges[i]
        hi = velocity_edges[i + 1]

        if velocity_label:
            _add_velocity_label(ax, lo, hi, loc=label_loc, color=label_color, label_box=label_box)

        _add_contours(ax, contours, i)
        _add_regions(ax, regions, i, wcs=wcs)

        _draw_panel_extras(ax, index=i, wcs=wcs,
                            scalebar=scalebar, scalebar_mode=scalebar_mode, scalebar_kwargs=scalebar_kwargs,
                            north_east=north_east, ne_mode=ne_mode, ne_kwargs=ne_kwargs)

        if annotate_func is not None:
            annotate_func(ax=ax, index=i, vmin=lo, vmax=hi)

        _apply_axes(ax, axes, row, col, nrows, ncols, use_wcs=use_wcs)

    if colorbar and last_im is not None:
        cbar = grid.cbar_axes[0].colorbar(last_im)
        cbar.set_label(colorbar_label or "Intensity")

    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return fig, grid


def plot_rgb_channel_grid(
    rgb_maps,
    velocity_edges,
    lines=None,
    colors=None,
    ncols=4,
    origin="lower",
    wcs=None,
    axes="outer",
    stretch="asinh",
    asinh_a=0.15,
    velocity_label=True,
    label_loc="upper left",
    label_color="white",
    label_box=True,
    show_legend=True,
    scalebar=False,
    scalebar_mode="first",
    scalebar_kwargs=None,
    north_east=False,
    ne_mode="first",
    ne_kwargs=None,
    contours=None,
    regions=None,
    annotate_func=None,
    wspace=0.01,
    hspace=0.01,
    figsize=None,
    savepath=None,
    dpi=300,
):
    """
    To plot RGB velocity channel maps.
    """
    if rgb_maps.ndim != 4 or rgb_maps.shape[-1] != 3:
        raise ValueError("rgb_maps must have shape (n_channels, ny, nx, 3).")

    if ncols not in {3, 4, 5}:
        raise ValueError("ncols must be 3, 4, or 5.")

    if scalebar_mode not in {"first", "all", "none"}:
        raise ValueError("scalebar_mode must be 'first', 'all', or 'none'.")

    if ne_mode not in {"first", "all", "none"}:
        raise ValueError("ne_mode must be 'first', 'all', or 'none'.")

    nchan = rgb_maps.shape[0]
    ncols = min(ncols, nchan)
    nrows = int(np.ceil(nchan / ncols))

    use_wcs = axes in {"wcs", "outer"} and wcs is not None
    subplot_kw = {"projection": wcs} if use_wcs else {}

    if figsize is None:
        figsize = (3.0 * ncols, 3.0 * nrows)

    fig, axes_arr = plt.subplots(nrows, ncols, figsize=figsize, subplot_kw=subplot_kw, squeeze=False)
    axes_flat = axes_arr.ravel()

    for i, ax in enumerate(axes_flat):
        row, col = divmod(i, ncols)

        if i >= nchan:
            ax.axis("off")
            continue

        rgb_display = _stretch_rgb(rgb_maps[i], stretch=stretch, asinh_a=asinh_a)

        ax.imshow(rgb_display, origin=origin)

        lo = velocity_edges[i]
        hi = velocity_edges[i + 1]

        if velocity_label:
            _add_velocity_label(ax, lo, hi, loc=label_loc, color=label_color, label_box=label_box)

        _add_contours(ax, contours, i)
        _add_regions(ax, regions, i, wcs=wcs)
        _draw_panel_extras(ax, index=i, wcs=wcs, 
                            scalebar=scalebar, scalebar_mode=scalebar_mode, scalebar_kwargs=scalebar_kwargs,
                            north_east=north_east, ne_mode=ne_mode, ne_kwargs=ne_kwargs)

        if annotate_func is not None:
            annotate_func(ax=ax, index=i, vmin=lo, vmax=hi)

        _apply_axes(ax, axes, row, col, nrows, ncols, use_wcs=use_wcs)

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.93 if show_legend and lines and colors else 0.98, wspace=wspace, hspace=hspace)

    if show_legend and lines and colors:
        _rgb_line_legend(fig, lines, colors)

    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return fig, axes_flat



