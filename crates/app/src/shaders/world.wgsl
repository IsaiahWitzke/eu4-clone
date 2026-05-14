// ============================================================================
// World-scale constants + helpers shared between the heightmap and image
// passes. The Europe heightmap (assets/heightmap.png) is 4096² covering
// 5500 km × 5500 km in Web Mercator, centred ~(10°E, 53°N).
//
// Conventions:
//   * 1 world XZ unit = 1 km (mercator units, distorted at high latitudes)
//   * Heights normalised to [0, 1] = elevation in [0, HEIGHT_SCALE_M] meters
//
// Mirrors `WORLD_BBOX` / `WORLD_TEX_SIZE` in `script/_world.py`. Keep these
// in sync if the bbox changes.
// ============================================================================

const HEIGHT_SCALE_M: f32 = 5000.0;
const WORLD_HEIGHTMAP_SIZE: f32 = 4096.0;

// Half-extents of the world heightmap rectangle, in world units (km).
// Bbox is 5,500,000 m = 5500 km on each side → half-extent 2750 km.
const WORLD_BOUNDS_HALF: vec2<f32> = vec2<f32>(2750.0, 2750.0);

// Map a world XZ coordinate into the [0, 1] UV of the world heightmap +
// water mask textures.
//
// World convention: world Z = +north. PNG row 0 (uv.y = 0) sits at the
// highest Mercator northing, i.e. geographic north. So world_z = +HALF
// (north) must map to uv.y = 0 (top of texture) — that's why we flip Y.
fn world_to_world_uv(world_xz: vec2<f32>) -> vec2<f32> {
    let uv = (world_xz + WORLD_BOUNDS_HALF) / (2.0 * WORLD_BOUNDS_HALF);
    return vec2<f32>(uv.x, 1.0 - uv.y);
}
