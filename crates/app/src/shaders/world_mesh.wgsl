// ============================================================================
// world_mesh — per-frame draw of the world disc. A heightmap-displaced
// grid mesh in world XZ; each fragment picks the appropriate LoD atlas
// via fwidth(world_xz) and applies a per-fragment realm-field tint.
// ============================================================================

const WORLD_HALF_KM:   f32 = 2750.0;
const HEIGHT_SCALE_M:  f32 = 5000.0;

// Grid size for the mesh. (MESH_GRID - 1)² quads; each quad = 6 verts.
// 256 picked as a balance: fine enough for the European Alps to read
// as 3D, coarse enough that vertex throughput isn't a concern.
const MESH_GRID:       u32 = 256u;

// LoD selection thresholds. The pyramid uses non-uniform steps
// ([1, 4, 16, 32] tiles per side), so we can't just `log2` a single
// base — explicit km/px per LoD instead. Mirrors `LOD_TILES` in
// `tiles.rs`.
//   LoD 0: 21.484 km/px (1×1)
//   LoD 1:  5.371 km/px (4×4)
//   LoD 2:  1.343 km/px (16×16)
//   LoD 3:  0.671 km/px (32×32, == source heightmap upscaled 2×)
const LOD_KM_PER_TEXEL_0: f32 = 21.484375;
const LOD_KM_PER_TEXEL_1: f32 = 5.37109375;
const LOD_KM_PER_TEXEL_2: f32 = 1.342773;
const LOD_KM_PER_TEXEL_3: f32 = 0.6713867;

// Realm field is a fixed 2048² texture; matches `FIELD_SIZE` in
// `passes/realm_field.rs`.
const REALM_FIELD_SIZE: f32 = 2048.0;

// ---- bindings --------------------------------------------------------------

// Mirrors `CameraUniforms` in `camera.rs`. The camera passes the *raw
// parameters* (eye_offset, world_center, FOV, canvas size) and lets the
// shader build view/projection on the fly — the matrix would also work
// but this keeps the CPU side dumb. 48 bytes.
struct CameraUniforms {
    i_resolution: vec3<f32>,
    i_time:       f32,
    world_center: vec2<f32>,
    hovered_pid:  u32,
    hovered_city: u32,
    eye_offset:   vec3<f32>,
    map_mode:     u32,
}
@group(0) @binding(0)  var<uniform>          camera:          CameraUniforms;
@group(0) @binding(1)  var                   world_heightmap: texture_2d<f32>;
@group(0) @binding(2)  var                   atlas_lod0:      texture_2d<f32>;
@group(0) @binding(3)  var                   atlas_lod1:      texture_2d<f32>;
@group(0) @binding(4)  var                   atlas_lod2:      texture_2d<f32>;
@group(0) @binding(5)  var                   atlas_lod3:      texture_2d<f32>;
@group(0) @binding(6)  var                   samp:            sampler;
@group(0) @binding(7)  var                   realm_field:     texture_2d<f32>;
@group(0) @binding(8)  var                   water_sdf:       texture_2d<f32>;

// Water SDF decode — mirrors `WATER_SDF_RANGE_KM` in `tile_bake.wgsl`
// and `SDF_RANGE_KM` in `script/gen-water-sdf`. The R8 byte maps
// linearly to the band [-RANGE, +RANGE]: byte=0 = deepest sea, byte=255
// = deepest inland, byte=128 ≈ the coast.
const WATER_SDF_RANGE_KM: f32 = 8.0;

fn sample_water_dist_km(uv: vec2<f32>) -> f32 {
    let byte = textureSampleLevel(water_sdf, samp, uv, 0.0).r;
    return byte * (2.0 * WATER_SDF_RANGE_KM) - WATER_SDF_RANGE_KM;
}

// ---- Coastline domain warping ---------------------------------------------
//
// The water SDF source is 1.34 km/texel — below that, the coast is
// piecewise-linear (one straight segment per source texel) and reads
// as stair-stepping at close zoom. Inigo Quilez’s domain-warping
// trick (https://iquilezles.org/articles/warp/) hallucinates
// sub-source detail by perturbing the SDF lookup coordinates with a
// stack of value-noise octaves. The macro shape stays put; only the
// exact sub-texel zero-crossing wiggles in a natural-looking way.
//
// Three octaves are added together (all amplitudes in km):
//   smooth     — always on; rounds the source-grid staircase into
//                gentle bay arcs.
//   med        — scales linearly with ruggedness; adds peninsular
//                shape variation everywhere, more in rugged regions.
//   fine       — scales as ruggedness²; only kicks in on rugged
//                coasts. Gives the Croatian / fjord fringing.
//
// `coast_ruggedness(world_xz)` is itself a slow-varying noise
// (~25 km wavelength) that says “this region of coast is smooth
// vs. rugged”; a smoothstep concentrates the distribution toward
// the extremes so each cohesive stretch gets a clear character.
//
// Total amplitude stays under one source texel so the macro shape —
// which is real geographic data — is preserved.

// ---- Noise primitives (2D value noise) ------------------------------------

// 2D hash → [0, 1]. Standard `fract(sin(dot(...)))` lookalike.
fn hash2(p_in: vec2<f32>) -> f32 {
    var p = fract(p_in * vec2<f32>(123.34, 456.21));
    p = p + dot(p, p + 78.233);
    return fract(p.x * p.y);
}

// Value noise with smoothstep interpolation. Output [0, 1].
fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash2(i + vec2<f32>(0.0, 0.0));
    let b = hash2(i + vec2<f32>(1.0, 0.0));
    let c = hash2(i + vec2<f32>(0.0, 1.0));
    let d = hash2(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Two decorrelated samples of signed value noise (range [-1, +1] per
// component). Used to draw a 2D warp displacement from a single base
// position with one offset for the y component.
fn vnoise2(p: vec2<f32>, offset_y: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        vnoise(p)            * 2.0 - 1.0,
        vnoise(p + offset_y) * 2.0 - 1.0,
    );
}

// ---- Coastline warp tunables (km) -----------------------------------------

// Wavelength of the region-scale ruggedness field. Larger = bigger
// stretches of coast share one character.
const COAST_RUGGEDNESS_WL_KM: f32 = 25.0;

// Per-octave wavelengths and amplitudes (km). `_AMP_MIN`/`_AMP_MAX`
// brackets bracket the linear ramp from smooth→rugged for the medium
// octave; the fine octave’s amplitude is scaled by ruggedness² so
// smooth regions are *clean*, not just less-rough.
const COAST_WARP_SMOOTH_WL_KM:  f32 = 3.0;
const COAST_WARP_SMOOTH_AMP_KM: f32 = 0.45;
const COAST_WARP_MED_WL_KM:     f32 = 0.9;
const COAST_WARP_MED_AMP_MIN:   f32 = 0.10;
const COAST_WARP_MED_AMP_MAX:   f32 = 0.30;
const COAST_WARP_FINE_WL_KM:    f32 = 0.3;
const COAST_WARP_FINE_AMP_KM:   f32 = 0.18;

// ---- Coastline warp -------------------------------------------------------

fn coast_ruggedness(world_xz: vec2<f32>) -> f32 {
    return smoothstep(0.30, 0.70, vnoise(world_xz / COAST_RUGGEDNESS_WL_KM));
}

fn warp_world_xz(world_xz: vec2<f32>) -> vec2<f32> {
    let rugged = coast_ruggedness(world_xz);
    let warp_smooth =
        vnoise2(world_xz / COAST_WARP_SMOOTH_WL_KM, vec2<f32>(7.3, 11.5))
        * COAST_WARP_SMOOTH_AMP_KM;
    let warp_med =
        vnoise2(world_xz / COAST_WARP_MED_WL_KM, vec2<f32>(3.1, 5.7))
        * mix(COAST_WARP_MED_AMP_MIN, COAST_WARP_MED_AMP_MAX, rugged);
    let warp_fine =
        vnoise2(world_xz / COAST_WARP_FINE_WL_KM, vec2<f32>(1.9, 4.1))
        * COAST_WARP_FINE_AMP_KM * rugged * rugged;
    return world_xz + warp_smooth + warp_med + warp_fine;
}

// ---- camera math -----------------------------------------------------------
// Matches `camera.rs`. FOV = 30°; target Y = 4 km (mid-altitude); near/far
// chosen wide so the depth buffer has decent resolution across the full
// camera-distance range (5..8000 km).
const CAMERA_FOV_Y_RAD:  f32 = 0.5235987755982988;
const CAMERA_TARGET_Y:   f32 = 4.0;
const CLIP_NEAR:         f32 = 1.0;
const CLIP_FAR:          f32 = 50000.0;

struct CamBasis {
    eye:     vec3<f32>,
    right:   vec3<f32>,
    up:      vec3<f32>,
    forward: vec3<f32>,
}

fn cam_basis() -> CamBasis {
    let look_at = vec3<f32>(camera.world_center.x, CAMERA_TARGET_Y, camera.world_center.y);
    let eye     = look_at + camera.eye_offset;
    let forward = normalize(look_at - eye);
    let world_up = vec3<f32>(0.0, 1.0, 0.0);
    let right   = normalize(cross(world_up, forward));
    let up      = cross(forward, right);
    return CamBasis(eye, right, up, forward);
}

// World-space point → clip space. Standard right-handed perspective with
// the depth term mapping `view_z = near → 0` and `view_z = far → 1`.
fn project(p: vec3<f32>) -> vec4<f32> {
    let cb = cam_basis();
    let v = p - cb.eye;
    let view_x = dot(v, cb.right);
    let view_y = dot(v, cb.up);
    let view_z = dot(v, cb.forward);   // +ve = in front of the camera

    let aspect      = camera.i_resolution.x / max(camera.i_resolution.y, 1.0);
    let tan_half_y  = tan(CAMERA_FOV_Y_RAD * 0.5);
    let tan_half_x  = tan_half_y * aspect;

    let a = CLIP_FAR / (CLIP_FAR - CLIP_NEAR);
    let b = -CLIP_NEAR * CLIP_FAR / (CLIP_FAR - CLIP_NEAR);

    let clip_x = view_x / tan_half_x;
    let clip_y = view_y / tan_half_y;
    let clip_z = a * view_z + b;
    let clip_w = view_z;
    return vec4<f32>(clip_x, clip_y, clip_z, clip_w);
}

// ---- helpers ---------------------------------------------------------------

fn world_to_world_uv(xz: vec2<f32>) -> vec2<f32> {
    let uv = (xz + vec2<f32>(WORLD_HALF_KM, WORLD_HALF_KM))
             / (2.0 * WORLD_HALF_KM);
    return vec2<f32>(uv.x, 1.0 - uv.y);
}

fn sample_height_km(uv: vec2<f32>) -> f32 {
    let rg = textureSampleLevel(world_heightmap, samp, uv, 0.0).rg;
    let h_norm = rg.r + rg.g / 256.0;
    return h_norm * HEIGHT_SCALE_M / 1000.0; // metres → km, so we stay
                                              // in world-XZ units.
}

// Stateless palette: HSV-ish ramp via the golden-ratio trick so adjacent
// realm IDs don't get visually-confusable colours.
fn realm_palette(id: u32) -> vec3<f32> {
    if (id == 0u) {
        return vec3<f32>(0.5, 0.5, 0.5);
    }
    let h = fract(f32(id) * 0.61803398875);
    let k = vec3<f32>(5.0, 3.0, 1.0);
    let c = abs(fract(vec3<f32>(h) + k / 6.0) * 6.0 - 3.0);
    return vec3<f32>(0.85) - clamp(c - 1.0, vec3<f32>(0.0), vec3<f32>(1.0)) * 0.5;
}

// ---- vertex shader: 6 verts per cell, MESH_GRID² grid -----------------------

struct VsOut {
    @builtin(position) clip:     vec4<f32>,
    @location(0)       world_xz: vec2<f32>,
}

fn quad_corner(corner: u32) -> vec2<u32> {
    var quad = array<vec2<u32>, 6>(
        vec2<u32>(0u, 0u),
        vec2<u32>(1u, 0u),
        vec2<u32>(1u, 1u),
        vec2<u32>(0u, 0u),
        vec2<u32>(1u, 1u),
        vec2<u32>(0u, 1u),
    );
    return quad[corner];
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let cells_per_side = MESH_GRID - 1u;
    let cell_idx       = vid / 6u;
    let corner         = vid % 6u;
    let cx             = cell_idx % cells_per_side;
    let cy             = cell_idx / cells_per_side;
    let off            = quad_corner(corner);

    let gx = cx + off.x;  // [0, MESH_GRID-1]
    let gy = cy + off.y;

    let fx = f32(gx) / f32(MESH_GRID - 1u);  // 0..1 east → west
    let fy = f32(gy) / f32(MESH_GRID - 1u);  // 0..1 north → south
    let world_size_km = 2.0 * WORLD_HALF_KM;
    let world_x =  -WORLD_HALF_KM + fx * world_size_km;
    let world_z =   WORLD_HALF_KM - fy * world_size_km;

    // Heightmap-displaced world Y. The PNG uses the same UV convention
    // (row 0 = north), so we feed it (fx, fy) directly.
    let uv      = vec2<f32>(fx, fy);
    let world_y = sample_height_km(uv);

    let clip = project(vec3<f32>(world_x, world_y, world_z));
    return VsOut(clip, vec2<f32>(world_x, world_z));
}

// ---- fragment shader -------------------------------------------------------

fn sample_atlas(lod: i32, uv: vec2<f32>) -> vec3<f32> {
    // `if` chain over textures — WebGPU doesn't support indexing into
    // an array of textures (no bindless), so this is the simplest path.
    if (lod <= 0) {
        return textureSampleLevel(atlas_lod0, samp, uv, 0.0).rgb;
    } else if (lod == 1) {
        return textureSampleLevel(atlas_lod1, samp, uv, 0.0).rgb;
    } else if (lod == 2) {
        return textureSampleLevel(atlas_lod2, samp, uv, 0.0).rgb;
    } else {
        return textureSampleLevel(atlas_lod3, samp, uv, 0.0).rgb;
    }
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let uv = world_to_world_uv(in.world_xz);

    // Warped UV used *only* for the water/coastline lookup — the atlas
    // still gets sampled at the true position so terrain features
    // (rivers, mountain ridges, biome boundaries) stay where they
    // actually are. The warp adds natural sub-source-resolution
    // wiggle to the coastline, hiding the source grid.
    let warped_uv = world_to_world_uv(warp_world_xz(in.world_xz));

    // LoD selection: pick the coarsest atlas whose km-per-texel is
    // still finer than the on-screen km-per-pixel. Thresholds are at
    // the geometric midpoint between adjacent LoD resolutions, so the
    // switch happens at the zoom level where atlas and screen sampling
    // density are matched.
    let pixel_world_km = max(fwidth(in.world_xz.x), fwidth(in.world_xz.y));
    var lod: i32 = 0;
    if (pixel_world_km < sqrt(LOD_KM_PER_TEXEL_0 * LOD_KM_PER_TEXEL_1)) { lod = 1; }
    if (pixel_world_km < sqrt(LOD_KM_PER_TEXEL_1 * LOD_KM_PER_TEXEL_2)) { lod = 2; }
    if (pixel_world_km < sqrt(LOD_KM_PER_TEXEL_2 * LOD_KM_PER_TEXEL_3)) { lod = 3; }

    // M-key debug overlay: replace the world colour with a per-LoD
    // tint so you can see which atlas is being sampled at each
    // fragment. Cycle: Terrain (0) → Political (1) → DebugLod (2) → ...
    if (camera.map_mode == 2u) {
        var lod_color = vec3<f32>(0.0);
        if (lod == 0) {
            lod_color = vec3<f32>(1.00, 0.25, 0.25); // red
        } else if (lod == 1) {
            lod_color = vec3<f32>(1.00, 0.70, 0.20); // orange
        } else if (lod == 2) {
            lod_color = vec3<f32>(0.45, 0.85, 0.30); // green
        } else {
            lod_color = vec3<f32>(0.30, 0.55, 1.00); // blue
        }
        return vec4<f32>(lod_color, 1.0);
    }

    var color = sample_atlas(lod, uv);

    // ---- Water blend, screen-pixel-accurate ------------------------------
    //
    // This used to live in `tile_bake.wgsl`, but the bake outputs at
    // atlas-texel resolution — so any AA we computed there was locked
    // to the atlas grid, leaving a visible staircase at close zoom.
    // Moving the SDF sample here lets us drive the transition by
    // `fwidth(dist_km)`, which is the screen-pixel-sized derivative.
    // Result: the land/water seam is always exactly one screen pixel
    // wide, no matter how zoomed in we are.
    let dist_km     = sample_water_dist_km(warped_uv);
    let dist_fwidth = max(fwidth(dist_km), 0.001);
    let half_band   = dist_fwidth * 0.5;
    let water_alpha = smoothstep(half_band, -half_band, dist_km);

    // Shallow-shelf gradient. Subtle by design so it doesn’t fight
    // for visual attention against the coastline itself.
    let deep_water  = vec3<f32>(0.08, 0.20, 0.40);
    let shelf_water = vec3<f32>(0.14, 0.28, 0.46);
    let shelf_inner = smoothstep(-5.0, 0.0, dist_km);
    let water_color = mix(deep_water, shelf_water, shelf_inner);
    color = mix(color, water_color, water_alpha);

    // TEMPORARY: country/realm tinting disabled while iterating on
    // map look. The realm_field texture binding is still present (the
    // bind-group layout still requires it) — we just don't sample it.
    // The realm_palette helper and hover-highlight branch are kept in
    // source so re-enabling is a single block uncomment.
    //
    // let field_px_f = uv * REALM_FIELD_SIZE;
    // let field_px   = clamp(vec2<i32>(field_px_f),
    //                        vec2<i32>(0),
    //                        vec2<i32>(i32(REALM_FIELD_SIZE) - 1));
    // let field      = textureLoad(realm_field, field_px, 0);
    // let realm_id   = u32(round(field.r));
    // let field_a    = field.g;
    // let realm_rgb  = realm_palette(realm_id);
    // var tint_strength = field_a * 0.35;
    // if (camera.hovered_pid != 0u && realm_id + 1u == camera.hovered_pid) {
    //     tint_strength = field_a * 0.55;
    // }
    // color = mix(color, color * 0.55 + realm_rgb * 0.55, tint_strength);

    return vec4<f32>(color, 1.0);
}
