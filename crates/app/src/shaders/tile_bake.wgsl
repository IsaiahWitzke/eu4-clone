// ============================================================================
// tile_bake — bake the world into one LoD atlas. Runs once per LoD, with
// `instance_count = LOD_TILES[lod]²`. Each instance's vertex shader places
// that tile's quad inside the atlas (in NDC); the fragment shader does the
// shading math (height palette + water + lambert hillshade) at the
// corresponding world XZ.
//
// Reads:
//   * world heightmap (Rg8Unorm, 16-bit grayscale packed)
//   * biome mask (R8Unorm)
//   * BakeUniforms (per-LoD: tiles_per_side)
//
// Water blending is NOT done here. The atlas stores *only* terrain
// colour (biome palette × hillshade). The per-frame world_mesh shader
// composites water on top at screen-pixel resolution using the SDF —
// that's the only way to get a one-screen-pixel-wide coastline AA
// regardless of zoom. Baking a water blend at atlas resolution
// (8192² at the finest LoD) locks the coastline AA to a 0.67 km grid,
// which is what produced the visible staircase in earlier iterations.
//
// Writes:
//   * One LoD atlas (RGBA8Unorm).
// ============================================================================

const WORLD_HALF_KM:   f32 = 2750.0;
const HEIGHT_SCALE_M:  f32 = 5000.0;
const WORLD_TEX_SIZE:  f32 = 4096.0;

struct BakeUniforms {
    tiles_per_side: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(0) var<uniform>           bake:           BakeUniforms;
@group(0) @binding(1) var                    world_heightmap: texture_2d<f32>;
@group(0) @binding(2) var                    biome_mask:      texture_2d<f32>;
@group(0) @binding(3) var                    samp:            sampler;

struct VsOut {
    @builtin(position) pos:      vec4<f32>,
    @location(0)       world_xz: vec2<f32>,
}

// World XZ → UV in the source PNGs (Y-flipped so atlas/PNG row 0 = +north).
fn world_to_world_uv(xz: vec2<f32>) -> vec2<f32> {
    let uv = (xz + vec2<f32>(WORLD_HALF_KM, WORLD_HALF_KM))
             / (2.0 * WORLD_HALF_KM);
    return vec2<f32>(uv.x, 1.0 - uv.y);
}

// Decode the 16-bit elevation packed into Rg8Unorm into meters.
fn sample_height_m(xz: vec2<f32>) -> f32 {
    let uv = world_to_world_uv(xz);
    let rg = textureSampleLevel(world_heightmap, samp, uv, 0.0).rg;
    // PNG ships high byte first → R holds high byte, G holds low byte.
    let h_norm = rg.r + rg.g / 256.0;
    return h_norm * HEIGHT_SCALE_M;
}

// 6 vertices form one quad (two triangles). Each (s, t) ∈ {0, 1}² is the
// corner this vertex sits at within a tile.
fn quad_corner(vid: u32) -> vec2<f32> {
    var quad = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0),
    );
    return quad[vid];
}

@vertex
fn vs_main(
    @builtin(vertex_index)   vid: u32,
    @builtin(instance_index) iid: u32,
) -> VsOut {
    let corner = quad_corner(vid);
    let n = f32(bake.tiles_per_side);
    let tile_x = f32(iid % bake.tiles_per_side);
    let tile_y = f32(iid / bake.tiles_per_side);

    // Atlas-fractional coords of this vertex ∈ [0, 1]².
    let fx = (tile_x + corner.x) / n;
    let fy = (tile_y + corner.y) / n;

    // NDC: atlas top-left is (-1, +1). Flip Y so tile_y = 0 lands at
    // the top of the atlas.
    let ndc = vec2<f32>(fx * 2.0 - 1.0, 1.0 - fy * 2.0);

    // World XZ: tile_y = 0 → world_z = +HALF (north), matching PNG row 0.
    let world_size_km = 2.0 * WORLD_HALF_KM;
    let world_x = -WORLD_HALF_KM + fx * world_size_km;
    let world_z =  WORLD_HALF_KM - fy * world_size_km;

    return VsOut(vec4<f32>(ndc, 0.0, 1.0), vec2<f32>(world_x, world_z));
}

// Per-biome “lowland” palette. Biome IDs come from the WWF / RESOLVE
// Ecoregions 2017 BIOME_NUM taxonomy (see `script/gen-biome-mask`):
//   0  ocean / none / outside
//   4  Temperate Broadleaf       (deep deciduous green)
//   5  Temperate Conifer         (cooler blue-tinted green)
//   6  Boreal                    (muted grey-green w/ blue cast)
//   8  Temperate Grassland       (pale ochre-green steppe)
//   11 Tundra                    (desaturated grey-mauve)
//   12 Mediterranean             (dusty olive-tan)
//   13 Desert                    (warm sand)
fn biome_lowland(id: u32) -> vec3<f32> {
    switch id {
        // 4 Temperate Broadleaf — the baseline green; everything
        // else reads relative to this.
        case 4u:  { return vec3<f32>(0.30, 0.46, 0.24); }
        // 5 Temperate Conifer — slightly cooler, slightly darker.
        case 5u:  { return vec3<f32>(0.26, 0.42, 0.26); }
        // 6 Boreal — desaturated mossy green with a cold cast.
        case 6u:  { return vec3<f32>(0.34, 0.44, 0.34); }
        // 8 Temperate Grassland — dry steppe khaki. Pulled toward
        // green from straight ochre so the boundary with broadleaf
        // is less of a stamp.
        case 8u:  { return vec3<f32>(0.48, 0.52, 0.30); }
        // 11 Tundra — cool desaturated grey-green. Was warm/orange
        // in iter1; the warm “tundra” feel was actually the highland
        // brown bleeding through Norway.
        case 11u: { return vec3<f32>(0.42, 0.46, 0.42); }
        // 12 Mediterranean — dusty olive. Iter1 was too
        // golden/saturated and stamped on hard against neighbours.
        case 12u: { return vec3<f32>(0.48, 0.48, 0.30); }
        // 13 Desert — warm sand, slightly desaturated.
        case 13u: { return vec3<f32>(0.72, 0.62, 0.42); }
        // Ocean / no-coverage fallback. Real ocean fragments get
        // overwritten by the water blend anyway; this colour shows up
        // only for land tiles outside the Ecoregions polygon
        // coverage (Iceland edge cases, etc).
        default:  { return vec3<f32>(0.32, 0.46, 0.26); }
    }
}

// Mountain-rock colour per biome. The shared “warm sandstone” brown
// from iter1 made Scandinavian uplands look Mediterranean. Cold
// biomes get a grey-slate highland; warm biomes get the
// dry-sandstone highland; everything else lands in between.
fn biome_highland(id: u32) -> vec3<f32> {
    switch id {
        case 6u, 11u:        { return vec3<f32>(0.42, 0.42, 0.42); }  // cold grey
        case 12u, 13u:       { return vec3<f32>(0.58, 0.48, 0.34); }  // warm sandstone
        default:             { return vec3<f32>(0.48, 0.44, 0.36); }  // neutral
    }
}

fn terrain_color(h_m: f32, biome_id: u32) -> vec3<f32> {
    let lowland  = biome_lowland(biome_id);
    let highland = biome_highland(biome_id);
    let snow     = vec3<f32>(0.95, 0.95, 0.95);
    let t = clamp(h_m / 3000.0, 0.0, 1.0);
    var c = mix(lowland, highland, smoothstep(0.0, 0.6, t));
    c = mix(c, snow, smoothstep(0.75, 1.0, t));
    return c;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let xz = in.world_xz;

    // Central-difference normal estimate in world km. eps is in km; the
    // step is sized to be a few atlas texels at the finest LoD so the
    // normal still has detail at close zoom.
    let eps_km = 1.0;
    let hc    = sample_height_m(xz);
    let hx_p  = sample_height_m(xz + vec2<f32>(eps_km, 0.0));
    let hx_m  = sample_height_m(xz - vec2<f32>(eps_km, 0.0));
    let hz_p  = sample_height_m(xz + vec2<f32>(0.0, eps_km));
    let hz_m  = sample_height_m(xz - vec2<f32>(0.0, eps_km));
    // dh/dx in meters per km → dimensionless slope.
    let slope_x = (hx_p - hx_m) / (2.0 * 1000.0);
    let slope_z = (hz_p - hz_m) / (2.0 * 1000.0);
    let normal  = normalize(vec3<f32>(-slope_x, 1.0, slope_z));

    // Sun pointing roughly south-east, well above horizon.
    let sun     = normalize(vec3<f32>(0.4, 0.85, -0.35));
    let lambert = max(dot(normal, sun), 0.0);
    let light   = 0.45 + 0.55 * lambert;

    // Biome IDs are categorical, so we *can’t* bilinear-filter them.
    // But we can blend the resulting *colours* from a couple of
    // nearby taps to soften the worst stamped-edge artefacts where
    // two biomes meet. 5-tap (centre + 4 cardinal neighbours one
    // texel away) is a cheap compromise.
    let uv = world_to_world_uv(xz);
    let dims_i = vec2<i32>(textureDimensions(biome_mask));
    let dims_f = vec2<f32>(dims_i);
    let center_px = clamp(
        vec2<i32>(uv * dims_f),
        vec2<i32>(0),
        dims_i - vec2<i32>(1),
    );
    var color_sum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    // Centre tap is weighted heavier so the dominant biome wins; the
    // neighbours only smooth the seam.
    let offs = array<vec2<i32>, 5>(
        vec2<i32>( 0,  0),
        vec2<i32>( 1,  0),
        vec2<i32>(-1,  0),
        vec2<i32>( 0,  1),
        vec2<i32>( 0, -1),
    );
    let weights = array<f32, 5>(2.0, 1.0, 1.0, 1.0, 1.0);
    for (var i = 0u; i < 5u; i = i + 1u) {
        let p = clamp(center_px + offs[i], vec2<i32>(0), dims_i - vec2<i32>(1));
        let id = u32(round(textureLoad(biome_mask, p, 0).r * 255.0));
        color_sum  = color_sum + terrain_color(hc, id) * weights[i];
        weight_sum = weight_sum + weights[i];
    }
    let color = (color_sum / weight_sum) * light;

    // No water blend here — see the file header. The atlas stores
    // *only* terrain colour; `world_mesh.wgsl` composites water on
    // top per-frame using the SDF for screen-pixel-accurate AA.

    return vec4<f32>(color, 1.0);
}
