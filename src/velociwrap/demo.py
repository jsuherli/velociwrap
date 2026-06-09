### velociwrap
### 
### A lightweight python package to create velocity-consistent channel maps 
### and position-velocity (PV) diagrams from astronomical datacube. 
###
### Janette Suherli (c)2026. 
### jsuherli@gmail.com
###
### To handle the demo data.


from pathlib import Path

from .cube import Cube


_DEMO_CUBES = {
    "muse": {
        "filename": "muse_demo.fits",
        "ext": 1,
    },
    "sitelle": {
        "filename": "sitelle_demo.fits",
        "ext": 0,
    },
    "cgps_hi": {
        "filename": "cgps_hi_demo.fits",
        "ext": 0,
    },
}


def list_demo_cubes():
    return list(_DEMO_CUBES.keys())


def load_demo_cube(name):
    name = name.lower()

    if name not in _DEMO_CUBES:
        raise ValueError(
            f"Unknown demo cube '{name}'. "
            f"Available cubes: {list_demo_cubes()}"
        )

    info = _DEMO_CUBES[name]

    filename = (
        Path(__file__).parent
        / "data"
        / info["filename"]
    )

    return Cube(
        filename,
        ext=info["ext"],
    )


def get_demo_region_file(name="muse"):
    demo_regions = {
        "muse": "muse_demo_regions.reg",
    }

    if name not in demo_regions:
        raise ValueError(
            f"Unknown demo region file '{name}'. "
            f"Available options: {list(demo_regions.keys())}"
        )

    return Path(__file__).parent / "data" / demo_regions[name]

