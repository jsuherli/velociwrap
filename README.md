# velociwrap
A lightweight python package to create velocity-consistent channel maps and position-velocity (PV) diagrams from astronomical datacube. 
Instead of relying on whatever wavelength/channel spacing a datacube happens to have, user may define the velocity bins and generate reproducible products from any astronomical datacubes (instrument-agnostic).

## Current Features
- Wavelength, wavenumber, frequency, and velocity datacube support
- Native and resampled velocity channel maps
- Velocity-matched 2-line and 3-line pseudo-RGB channel maps
- Publication-style grids with WCS/pixel axes, contours, DS9 regions, scalebars, and NE arrows options
- FITS and animation export
- Simple PV diagram (straight-line slice)


## Installation
From a local clone:
```bash
git clone https://github.com/jsuherli/velociwrap.git
cd velociwrap
pip install -e .
```

Install from GitHub:
```bash
pip install git+https://github.com/jsuherli/velociwrap.git
```

## Quick example

```python
import matplotlib.pyplot as plt

import velociwrap as vw

cube = vw.Cube("your_datacube.fits")

# Create [N II]6583 velocity channel maps
maps, edges, centers, meta = vw.make_channel_maps(cube, rest_value=6583.45, vmin=-200, vmax=200, dv=50)
vw.plot_channel_grid(maps, edges)
plt.show()

# Save the channel maps into a FITS file and a gif file. 
vw.save_channel_maps_fits(maps, "n2_demo_channels.fits", header=cube.header, velocity_edges=edges, metadata=meta)
vw.save_channel_animation(maps, edges, "n2_demo_channels.gif", fps=2)

# Create pseudo-RGB channel maps
rgb_maps, edges, centers, meta = vw.make_rgb_channel_maps(cube,
                                                            lines={"NII": 6583.45, "Ha": 6562.80, "SII": 6716.44},
                                                            colors={"NII": "magenta", "Ha": "cyan", "SII": "yellow"},
                                                            vmin=-200, vmax=200, dv=50)
fig, axes = vw.plot_rgb_channel_grid(rgb_maps, edges, lines=meta["lines"], colors=meta["colors"],
                                        ncols=5, wcs=cube.wcs.celestial, axes="none",
                                        stretch="asinh", asinh_a=0.12,
                                        north_east=True, ne_mode="all", scalebar=True, scalebar_mode="first")

plt.show()
```

More example with detailed explanations of all `velociwrap` capabilities will be available in the upcoming documentation. 


## Citation

If you use Velociwrap, please consider citing the GitHub/Zenodo release. A `CITATION.cff` file is included in this repo.

'''
Suherli, J. (2026). velociwrap: velocity-consistent (pseudo-RGB) channel maps for astronomical datacubes (v1.0.0). Zenodo. [https://doi.org/10.5281/zenodo.20604383](https://doi.org/10.5281/zenodo.20604383)
'''
