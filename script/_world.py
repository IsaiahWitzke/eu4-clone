"""
Shared world-config constants for the asset-pipeline scripts.

All `gen-*` scripts in this directory import from here so the bbox /
resolution stays consistent across heightmap, water mask, biome mask,
province mask, and the border SDF.

Coordinate system: EPSG:3857 (Web / Pseudo-Mercator). Mercator units
are metres, but distorted away from the equator — at lat 53° one
mercator-metre is about 0.6 real-metres east-west.

These values are mirrored on the GPU side as `WORLD_BOUNDS_HALF` /
`WORLD_HEIGHTMAP_SIZE` in `crates/app/src/shaders/world.wgsl` and
`WORLD_TEX_SIZE` / `WORLD_BOUNDS_HALF_KM` in
`crates/app/src/renderer.rs`. Keep them in sync if you change them
here.
"""
from pathlib import Path

# Resolve assets/ relative to the repo root (this file is in script/).
REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "crates" / "app" / "assets"

# ---- World bbox (EPSG:3857) -----------------------------------------------
#
# 5500 km square centred roughly on (10°E, 53°N) — covers Western and
# Central Europe end-to-end:
#   * Western edge (-13.5°E): just past Ireland's west coast
#   * Eastern edge (35.9°E):  just past Donetsk / east of Caspian
#   * Southern edge (35.4°N): Crete / Tunisian coast
#   * Northern edge (64.1°N): mid-Norway / N. Sweden
#
# Square in mercator units (5,500,000 m on each side). Pseudo-Mercator
# distortion at this latitude squashes the real east-west distance to
# ~3700 km; that's geometrically correct for Mercator.
WORLD_BBOX = (-1_500_000.0, 4_250_000.0, 4_000_000.0, 9_750_000.0)
"""(xmin, ymin, xmax, ymax) in EPSG:3857 metres."""


def bbox_size_m() -> float:
    """Width = height of the bbox in metres."""
    xmin, ymin, xmax, ymax = WORLD_BBOX
    w = xmax - xmin
    h = ymax - ymin
    assert abs(w - h) < 1.0, f"bbox must be square; got {w} \u00d7 {h}"
    return w


# ---- Texture resolution ---------------------------------------------------
WORLD_TEX_SIZE = 4096
"""Side length of every world-anchored mask texture (heightmap, water, biome,
province, border SDF). 4096\u00b2 is 4\u00d7 fewer pixels than the previous 8192\u00b2
build but plenty for NUTS-3 polygons (\u224830 km wide \u2192 ~22 px each at our
~1.34 km/px scale)."""


def world_to_pixel(coords, *, width=None, height=None):
    """
    Map (x, y) world coords (Pseudo-Mercator metres) to (px, py) image
    pixels for a `width \u00d7 height` raster covering `WORLD_BBOX`. The image
    convention is +X right, +Y down, while world is +X right, +Y up \u2014
    so we flip Y here.

    Accepts a (N, 2) numpy array or a sequence of (x, y) pairs and returns
    the same shape with floats.
    """
    import numpy as np

    width = width if width is not None else WORLD_TEX_SIZE
    height = height if height is not None else WORLD_TEX_SIZE
    coords = np.asarray(coords, dtype=np.float64)
    xmin, ymin, xmax, ymax = WORLD_BBOX
    px = (coords[..., 0] - xmin) / (xmax - xmin) * width
    py = height - (coords[..., 1] - ymin) / (ymax - ymin) * height
    return np.stack([px, py], axis=-1)
