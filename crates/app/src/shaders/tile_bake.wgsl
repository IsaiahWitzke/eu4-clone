// ============================================================================
// tile_bake — bake the world into one LoD atlas. Runs once per LoD, with
// `instance_count = LOD_TILES[lod]²`. Each instance's vertex shader places
// that tile's quad inside the atlas (in NDC); the fragment shader does the
// shading math (height palette + water + lambert hillshade) at the
// corresponding world XZ.
//
// Reads:
//   * world heightmap (Rg8Unorm, 16-bit grayscale packed)
//   * water mask (R8Unorm)
//   * biome mask (R8Unorm) — held for future biome tinting; currently unused
//   * BakeUniforms (per-LoD: tiles_per_side)
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
@group(0) @binding(2) var                    water_mask:      texture_2d<f32>;
@group(0) @binding(3) var                    biome_mask:      texture_2d<f32>;
@group(0) @binding(4) var                    samp:            sampler;

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

// Smooth elevation → base colour. Tuned to look reasonable from sea
// level (0 m) up to alpine snow (~3500 m).
fn elevation_color(h_m: f32) -> vec3<f32> {
    let lowland  = vec3<f32>(0.30, 0.55, 0.25);  // green
    let highland = vec3<f32>(0.55, 0.45, 0.30);  // brown
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

    var color = elevation_color(hc) * light;

    // Water blend. The mask is soft-edged (8-bit greyscale), so we
    // ramp through it for a clean shoreline.
    let uv = world_to_world_uv(xz);
    let water = textureSampleLevel(water_mask, samp, uv, 0.0).r;
    let water_color = vec3<f32>(0.12, 0.28, 0.48);
    color = mix(color, water_color, smoothstep(0.30, 0.70, water));

    // biome_mask is held but unused for now to keep the bake simple;
    // sampling it suppresses the "binding unused" validation warning
    // and lets us layer biome tints in without changing the bind group.
    let _biome_id = textureSampleLevel(biome_mask, samp, uv, 0.0).r;

    return vec4<f32>(color, 1.0);
}
