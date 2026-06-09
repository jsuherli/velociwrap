### velociwrap
### 
### A lightweight python package to create velocity-consistent channel maps 
### and position-velocity (PV) diagrams from astronomical datacube. 
###
### Janette Suherli (c)2026. 
### jsuherli@gmail.com
###
### Animation export helpers.

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

from .scaling import normalize_image


def save_channel_animation(maps, velocity_edges, savepath, cmap="gray_r", fps=2, stretch="asinh", percentile=(1, 99.5)):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_axis_off()

    finite = maps[maps == maps]
    vmin, vmax = (None, None) if finite.size == 0 else __import__("numpy").nanpercentile(finite, percentile)

    img0 = normalize_image(maps[0], vmin=vmin, vmax=vmax, stretch=stretch)
    im = ax.imshow(img0, origin="lower", cmap=cmap, vmin=0, vmax=1)
    txt = ax.text(0.03, 0.95, "", transform=ax.transAxes, color="white", va="top")
    ax.set_xticks([]); ax.set_yticks([])

    def update(i):
        im.set_data(normalize_image(maps[i], vmin=vmin, vmax=vmax, stretch=stretch))
        txt.set_text(rf"{velocity_edges[i]:.0f} to {velocity_edges[i+1]:.0f} km s$^{{-1}}$")
        return im, txt

    ani = FuncAnimation(fig, update, frames=maps.shape[0], blit=True)
    writer = PillowWriter(fps=fps) if savepath.lower().endswith(".gif") else FFMpegWriter(fps=fps)

    ani.save(savepath, writer=writer, savefig_kwargs={"bbox_inches": "tight", "pad_inches": 0})
    plt.close(fig)


def save_rgb_channel_animation(rgb_maps, velocity_edges, savepath, fps=2):
    fig, ax = plt.subplots(figsize=(6, 6), frameon=False)
    ax.set_axis_off()
    
    im = ax.imshow(rgb_maps[0], origin="lower")
    txt = ax.text(0.03, 0.95, "", transform=ax.transAxes, color="white", va="top")
    ax.set_xticks([]); ax.set_yticks([])

    def update(i):
        im.set_data(rgb_maps[i])
        txt.set_text(rf"{velocity_edges[i]:.0f} to {velocity_edges[i+1]:.0f} km s$^{{-1}}$")
        return im, txt

    ani = FuncAnimation(fig, update, frames=rgb_maps.shape[0], blit=True)
    writer = PillowWriter(fps=fps) if savepath.lower().endswith(".gif") else FFMpegWriter(fps=fps)

    ani.save(savepath, writer=writer, savefig_kwargs={"bbox_inches": "tight", "pad_inches": 0})
    plt.close(fig)

