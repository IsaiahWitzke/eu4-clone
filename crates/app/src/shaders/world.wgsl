// ============================================================================
// World-scale constants + helpers shared between the heightmap and image
// passes. The Switzerland heightmap (assets/heightmap.png) is 8192² covering
// 626.074 km × 626.074 km in Web Mercator, anchored at world origin.
//
// Conventions:
//   * 1 world XZ unit = 1 km
//   * Heights normalised to [0, 1] = elevation in [0, HEIGHT_SCALE_M] meters
// ============================================================================

const HEIGHT_SCALE_M: f32 = 5000.0;
const WORLD_HEIGHTMAP_SIZE: f32 = 8192.0;

// Half-extents of the world heightmap rectangle, in world units (km).
const WORLD_BOUNDS_HALF: vec2<f32> = vec2<f32>(313.037, 313.037);

// Map a world XZ coordinate into the [0, 1] UV of the world heightmap +
// water mask textures.
fn world_to_world_uv(world_xz: vec2<f32>) -> vec2<f32> {
    return (world_xz + WORLD_BOUNDS_HALF) / (2.0 * WORLD_BOUNDS_HALF);
}
