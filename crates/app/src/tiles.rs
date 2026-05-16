//! Tile-pyramid geometry. Shared constants + helpers for the bake pass
//! and the per-frame world-mesh pass.
//!
//! World convention (matches the heightmap PNG):
//!   * world X = +east, Z = +north
//!   * world bounds = ±2750 km on each axis (5500 km square)
//!   * Atlas row 0 = +north (Y-flipped relative to world Z), so the same
//!     `world_to_world_uv` mapping works for both atlas sampling and the
//!     source heightmap / water / biome PNGs.
//!
//! LoD pyramid: 4 levels, doubling tile-density per step. LoD 3 sits at
//! the WebGPU 8192² 2D-texture limit; any finer requires switching to
//! progressive baking + LRU eviction (out of scope for the first cut).

/// Atlas texels per side per tile. Same at every LoD.
pub const TILE_SIZE: u32 = 256;

/// Number of LoD levels. Indexed 0 (coarsest) to LOD_COUNT-1 (finest).
pub const LOD_COUNT: usize = 4;

/// Tiles per side at each LoD. LoD i has `LOD_TILES[i]²` tiles total.
/// Doubling per step would give 1, 2, 4, 8 — but we use the wider
/// 1, 4, 16, 32 cadence so LoD 3 hits manor-scale (~0.67 km/px) within
/// the WebGPU 2D-max budget.
pub const LOD_TILES: [u32; LOD_COUNT] = [1, 4, 16, 32];

/// Half-extent of the world in km along each axis. Mirrors
/// `WORLD_BOUNDS_HALF` in the WGSL shaders.
pub const WORLD_HALF_KM: f32 = 2750.0;

/// Atlas dimension (width = height) for LoD `lod`, in texels.
/// LoD 0 → 256, LoD 1 → 1024, LoD 2 → 4096, LoD 3 → 8192.
pub const fn atlas_dim(lod: usize) -> u32 {
    LOD_TILES[lod] * TILE_SIZE
}

/// Total tile count across the whole pyramid. With (1, 4, 16, 32) =
/// 1 + 16 + 256 + 1024 = 1297.
pub const fn total_tile_count() -> u32 {
    let mut sum: u32 = 0;
    let mut i: usize = 0;
    while i < LOD_COUNT {
        sum += LOD_TILES[i] * LOD_TILES[i];
        i += 1;
    }
    sum
}

/// World km covered by one atlas texel at this LoD.
/// LoD 0 → 21.48, LoD 1 → 5.37, LoD 2 → 1.34, LoD 3 → 0.67.
#[allow(dead_code)]
pub fn lod_km_per_texel(lod: usize) -> f32 {
    (2.0 * WORLD_HALF_KM) / atlas_dim(lod) as f32
}
