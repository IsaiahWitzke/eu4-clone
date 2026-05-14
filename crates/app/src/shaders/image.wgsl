// ============================================================================
// Image pass: raymarched terrain rendering. Renders to the swapchain.
//
// Reads:
//   u              (binding 0) — camera uniforms (from camera.wgsl)
//   base_heightmap (binding 1) — world-anchored painted heightmap
//   detail_noise   (binding 2) — world-anchored detail noise
//   terrain        (binding 3) — world-anchored eroded heightmap (the terrain)
//   layer          (binding 4) — world AABB covered by the three layers above
//   water_mask     (binding 5) — binary mask, 1.0 where water, 0.0 elsewhere
// ============================================================================
@group(0) @binding(1) var base_heightmap: texture_2d<f32>;
@group(0) @binding(2) var detail_noise: texture_2d<f32>;
@group(0) @binding(3) var terrain: texture_2d<f32>;
@group(0) @binding(4) var<uniform> layer: LayerUniforms;
@group(0) @binding(5) var water_mask: texture_2d<f32>;
@group(0) @binding(6) var samp: sampler;
// Biome IDs are categorical — must NEVER be filtered. Always sampled with
// textureLoad at integer coords; interpolating between e.g. biome 4 and 12
// produces fictional 5..11 pixels at borders.
@group(0) @binding(7) var biome_mask: texture_2d<f32>;

// Settlement influence-field uniform block.
// `strength * exp(-distance/E_FOLD)` field for its realm; the shader takes
// the per-pixel argmax (`sample_realm_field`) to decide which realm owns
// the fragment. Mirrors the Rust `SettlementUniforms` in `settlements.rs`.
const MAX_SETTLEMENTS: u32 = 1024u;
struct GpuSettlement {
    world_xz: vec2<f32>,
    strength: f32,
    realm_id: u32,
}
struct Settlements {
    // Three scalar u32 pads (NOT a vec3<u32>) so the array starts at
    // offset 16. `vec3<u32>` has 16-byte alignment in the uniform
    // address space, which would force naga to insert 12 bytes of
    // implicit padding *before* the vec3 — pushing `items` to offset 32
    // and the buffer size beyond what we want. Three u32 scalars each
    // have 4-byte alignment so they sit cleanly at offsets 4/8/12, and
    // `items` lands at 16 to match `SettlementUniforms` in Rust.
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    items: array<GpuSettlement, 1024>,
}
@group(0) @binding(10) var<uniform> settlements: Settlements;

// Pre-baked realm influence field. 2048² Rgba16Float over the 5500 km
// world. The bake pass (`shaders/realm_field.wgsl`) ran the per-pixel
// argmax-realm loop once at startup (and any time the settlement list
// changes), so the per-fragment cost here is a single textureLoad
// instead of the old O(N) loop. Channels:
//   R: realm_id    (raw f32; f16 exact through 2048)
//   G: alpha       (saturating fade, 0 wilderness → ≈1 deep interior)
//   B: contested-1 (clamped to [0, 1]; iso-line at 0)
//   A: city_idx    (raw f32 of the dominant settlement; powers hover
//                   hinterland highlighting)
@group(0) @binding(11) var realm_field_tex: texture_2d<f32>;

// Full-resolution world heightmap (4096² Rg8Unorm covering the whole 5500 km
// world bbox). 16-bit elevation is split across R+G (big-endian). Sampled by
// `vs_mesh` so mesh vertices land on the source data's native ~1.34 km/px
// resolution instead of going through the 1024² cached `terrain` layer
// (which downsamples by ~5× at default zoom).
@group(0) @binding(13) var world_heightmap_tex: texture_2d<f32>;

// Reconstruct a [0, 1] normalised elevation from `world_heightmap_tex`'s
// split-byte encoding. Mirrors `sample_height` in `base_heightmap.wgsl`.
// Linear filtering of the two bytes is mathematically equivalent to linear
// filtering of the reconstructed 16-bit value, so bilinear sampling "just
// works" here.
//
// Out-of-bounds → sea level so the mesh's edge column/row, which can sit
// outside the texture footprint when the camera is at the world bbox edge,
// doesn't smear whatever happens to be at the edge across the void.
fn sample_world_height(uv: vec2<f32>) -> f32 {
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
        return 0.0;
    }
    let rg = textureSampleLevel(world_heightmap_tex, samp, uv, 0.0).xy;
    return (rg.x * 65280.0 + rg.y * 255.0) / 65535.0;
}

// Binding 12 used to be a Canvas2D-painted country-name overlay; replaced
// by the in-engine SDF glyph atlas pass (see `passes::realm_labels`).

// ----------------------------------------------------------------------------
// Constants
// ----------------------------------------------------------------------------
const PI: f32 = 3.14159265358979;
const DEG_TO_RAD: f32 = PI / 180.0;

// Camera: EU4-style top-down. Eye position relative to look_at comes from the
// `eye_offset` uniform (driven by camera_distance + camera_tilt on the Rust
// side). North = +Z, so north stays "up" on the screen.
const CAMERA_FOV_Y:    f32 = 30.0;  // Vertical FOV in degrees — must match CAMERA_FOV_Y_RAD in lib.rs.
const CAMERA_TARGET_Y: f32 = 4.0;   // Lookat altitude (post-exaggeration). Roughly mid-mountain.

// Material IDs.
const M_GROUND: i32 = 0;
const M_STRATA: i32 = 1;
const M_WATER:  i32 = 2;

// Color palette.
const CLIFF_COLOR       = vec3<f32>(0.22, 0.20, 0.20);
const DIRT_COLOR        = vec3<f32>(0.60, 0.50, 0.40);
const GRASS_COLOR1      = vec3<f32>(0.15, 0.30, 0.10);
const GRASS_COLOR2      = vec3<f32>(0.40, 0.50, 0.20);
const SAND_COLOR        = vec3<f32>(0.80, 0.70, 0.60);
const WATER_COLOR       = vec3<f32>(0.00, 0.05, 0.10);
const WATER_SHORE_COLOR = vec3<f32>(0.00, 0.25, 0.25);
const SUN_COLOR         = vec3<f32>(2.0, 1.96, 1.90);
const AMBIENT_COLOR     = vec3<f32>(0.03, 0.05, 0.07);

// Vertical exaggeration. The raw heightmap is normalised: sample value 1.0 =
// HEIGHT_SCALE_M (5000 m) of elevation. With XZ in km, that means Y is
// implicitly compressed 5× relative to XZ — mountains look flat. We multiply
// by VERTICAL_EXAGGERATION to give them visual presence:
//   * 5.0 → true 1:1 scale (1 unit Y = 1 km, same as XZ)
//   * 10.0 → 2× game-y exaggeration (Mt Rosa = ~9.3 units tall)
// Every height-related threshold below is multiplied by this constant so the
// snow line, water level, grass band etc. stay at the correct fraction of
// the (scaled) elevation range.
const VERTICAL_EXAGGERATION: f32 = 10.0;

// Heights are stored as fractions of HEIGHT_SCALE_M (= 5000 m), then
// multiplied by VERTICAL_EXAGGERATION. Each band uses the real Swiss
// elevation in metres on the comment for sanity:
//   WATER_HEIGHT  : 190 m (Lake Maggiore)
//   TREE_LINE     : 2000 m (transition grass → bare rock)
//   SNOW_LINE     : 3000 m (rock → snow)
//   PEAK_SNOW     : 4000 m (full snow)
const WATER_HEIGHT:    f32 = (190.0  / 5000.0) * VERTICAL_EXAGGERATION;
const TREE_LINE:       f32 = (2000.0 / 5000.0) * VERTICAL_EXAGGERATION;
const SNOW_LINE:       f32 = (3000.0 / 5000.0) * VERTICAL_EXAGGERATION;
const PEAK_SNOW:       f32 = (4000.0 / 5000.0) * VERTICAL_EXAGGERATION;
const RAYMARCH_QUALITY: f32 = 2.0;

// Atmosphere coefficients (Rayleigh + Mie).
const C_RAYLEIGH = vec3<f32>(5.802e-6, 13.558e-6, 33.100e-6);
const C_MIE      = vec3<f32>(3.996e-6,  3.996e-6,  3.996e-6);

// Half-extents of the playable box in world space. XZ comes from the cached
// world layer's covered AABB — the box matches the region for which the
// heightmap is valid. Y is sized to fit the exaggerated elevation range plus
// a little headroom for haze.
fn box_size() -> vec3<f32> {
    let half = (layer.covered_max - layer.covered_min) * 0.5;
    return vec3<f32>(half.x, VERTICAL_EXAGGERATION, half.y);
}

// Center of the playable box in world space (Y = 0, XZ at the layer center).
fn box_center() -> vec3<f32> {
    let mid = (layer.covered_min + layer.covered_max) * 0.5;
    return vec3<f32>(mid.x, 0.0, mid.y);
}

// ----------------------------------------------------------------------------
// Math helpers
// ----------------------------------------------------------------------------
fn clamp01(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }
fn sq(x: f32) -> f32 { return x * x; }
fn pow5(x: f32) -> f32 { let x2 = x * x; return x2 * x2 * x; }

// ----------------------------------------------------------------------------
// Box intersection (Inigo Quilez)
// ----------------------------------------------------------------------------
struct BoxHit {
    t_near: f32,
    t_far:  f32,
    normal: vec3<f32>,
}

fn box_intersection(ro: vec3<f32>, rd: vec3<f32>, box_size: vec3<f32>) -> BoxHit {
    let m = 1.0 / rd;
    let n = m * ro;
    let k = abs(m) * box_size;
    let t1 = -n - k;
    let t2 = -n + k;
    let t_near = max(max(t1.x, t1.y), t1.z);
    let t_far  = min(min(t2.x, t2.y), t2.z);
    if (t_near > t_far || t_far < 0.0) {
        return BoxHit(-1.0, -1.0, vec3<f32>(0.0));
    }
    let normal = -sign(rd) * step(t1.yzx, t1.xyz) * step(t1.zxy, t1.xyz);
    return BoxHit(t_near, t_far, normal);
}

// ----------------------------------------------------------------------------
// Camera: top-down look-at, panned via `u.world_center`.
//
// The framebuffer is Y-down but our world is Y-up. The single Y-flip needed
// to bridge the two lives in the `ndc_y` line below.
// ----------------------------------------------------------------------------
struct Ray { ro: vec3<f32>, rd: vec3<f32> }

fn get_ray(frag_pos: vec2<f32>) -> Ray {
    // Lookat point: pan via world_center.x/y (mapped to world XZ).
    // (Don't name this `target` — reserved keyword in WGSL.)
    let look_at = vec3<f32>(u.world_center.x, CAMERA_TARGET_Y, u.world_center.y);
    // Eye: look_at + uniform-driven offset (encodes both distance and tilt).
    let eye = look_at + u.eye_offset;

    // Orthonormal camera basis (right-handed, Y-up).
    let forward = normalize(look_at - eye);
    let world_up = vec3<f32>(0.0, 1.0, 0.0);
    let right = normalize(cross(world_up, forward));   // east
    let up = cross(forward, right);                    // perp to both

    // Pixel → ray direction. NDC in [-1, 1] both axes; aspect-correct on x.
    let aspect = u.i_resolution.x / u.i_resolution.y;
    let tan_half_fov_y = tan(CAMERA_FOV_Y * 0.5 * DEG_TO_RAD);
    let tan_half_fov_x = tan_half_fov_y * aspect;
    let ndc_x = (frag_pos.x / u.i_resolution.x) * 2.0 - 1.0;
    // Y-flip: framebuffer-Y is top-down, but we want top-of-screen pixels to
    // produce rays tilted in the +up direction.
    let ndc_y = 1.0 - (frag_pos.y / u.i_resolution.y) * 2.0;

    let rd = normalize(
        forward
        + right * (ndc_x * tan_half_fov_x)
        + up    * (ndc_y * tan_half_fov_y)
    );
    return Ray(eye, rd);
}

// ----------------------------------------------------------------------------
// Sky / atmosphere
// ----------------------------------------------------------------------------
fn sky_color(rd: vec3<f32>, sun: vec3<f32>) -> vec3<f32> {
    let costh = dot(rd, sun);
    return AMBIENT_COLOR * PI * (1.0 - abs(costh) * 0.8);
}

fn phase_rayleigh(costh: f32) -> f32 {
    return 3.0 * (1.0 + costh * costh) / (16.0 * PI);
}

fn phase_mie(costh: f32, g_in: f32) -> f32 {
    let g = min(g_in, 0.9381);
    let k = 1.55 * g - 0.55 * g * g * g;
    let kcosth = k * costh;
    return (1.0 - k * k) / ((4.0 * PI) * (1.0 - kcosth) * (1.0 - kcosth));
}

// ----------------------------------------------------------------------------
// Filament-style BRDF
// ----------------------------------------------------------------------------
fn d_ggx(linear_roughness: f32, n_o_h: f32) -> f32 {
    let one_minus_noh_squared = 1.0 - n_o_h * n_o_h;
    let a = n_o_h * linear_roughness;
    let k = linear_roughness / (one_minus_noh_squared + a * a);
    return k * k * (1.0 / PI);
}

fn v_smith(linear_roughness: f32, n_o_v: f32, n_o_l: f32) -> f32 {
    let a2 = linear_roughness * linear_roughness;
    let ggxv = n_o_l * sqrt((n_o_v - a2 * n_o_v) * n_o_v + a2);
    let ggxl = n_o_v * sqrt((n_o_l - a2 * n_o_l) * n_o_l + a2);
    return 0.5 / (ggxv + ggxl);
}

fn f_schlick(f0: vec3<f32>, v_o_h: f32) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow5(1.0 - v_o_h);
}

fn f_schlick_scalar(f0: f32, f90: f32, v_o_h: f32) -> f32 {
    return f0 + (f90 - f0) * pow5(1.0 - v_o_h);
}

fn fd_burley(linear_roughness: f32, n_o_v: f32, n_o_l: f32, l_o_h: f32) -> f32 {
    let f90 = 0.5 + 2.0 * linear_roughness * l_o_h * l_o_h;
    let light_scatter = f_schlick_scalar(1.0, f90, n_o_l);
    let view_scatter  = f_schlick_scalar(1.0, f90, n_o_v);
    return light_scatter * view_scatter * (1.0 / PI);
}

fn fd_lambert() -> f32 { return 1.0 / PI; }

fn shade(
    diffuse: vec3<f32>, f0: vec3<f32>, smoothness: f32,
    n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, lc: vec3<f32>,
) -> vec3<f32> {
    let h = normalize(v + l);
    let n_o_v = abs(dot(n, v)) + 1e-5;
    let n_o_l = clamp01(dot(n, l));
    let n_o_h = clamp01(dot(n, h));
    let l_o_h = clamp01(dot(l, h));

    let roughness = 1.0 - smoothness;
    let lr = roughness * roughness;
    let d  = d_ggx(lr, n_o_h);
    let vt = v_smith(lr, n_o_v, n_o_l);
    let fresnel = f_schlick(f0, l_o_h);
    let fr = (d * vt) * fresnel;
    let fd = diffuse * fd_burley(lr, n_o_v, n_o_l, l_o_h);
    return (fd + fr) * lc * n_o_l;
}

// ACES filmic tonemapping (Narkowicz 2015).
fn tonemap_aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return (x * (a * x + b)) / (x * (c * x + d) + e);
}

// Soft-light blend (W3C compositing spec). Tints `base` toward `blend`
// while preserving the base's tonal structure: highlights stay light,
// shadows stay dark, midtones shift most. Used for realm-colour shading
// over terrain without washing out detail. Both inputs must be in [0, 1];
// HDR values should be clamped before calling.
fn soft_light_channel(b: f32, s: f32) -> f32 {
    if (s < 0.5) {
        return b - (1.0 - 2.0 * s) * b * (1.0 - b);
    }
    var d: f32;
    if (b <= 0.25) {
        d = ((16.0 * b - 12.0) * b + 4.0) * b;
    } else {
        d = sqrt(b);
    }
    return b + (2.0 * s - 1.0) * (d - b);
}

fn soft_light(base: vec3<f32>, blend: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        soft_light_channel(base.x, blend.x),
        soft_light_channel(base.y, blend.y),
        soft_light_channel(base.z, blend.z),
    );
}

// ----------------------------------------------------------------------------
// Biome lookup. World-anchored, sampled with textureLoad (no filtering),
// reconstructed from R8Unorm to integer biome ID via *255 + round.
//
// IDs follow the WWF/RESOLVE Ecoregions taxonomy:
//   0 = no biome (water, outside-clip, etc — falls through to default color)
//   1 = Tropical Moist Broadleaf       2 = Tropical Dry Broadleaf
//   3 = Tropical Conifer               4 = Temperate Broadleaf & Mixed
//   5 = Temperate Conifer              6 = Boreal / Taiga
//   7 = Tropical Grassland             8 = Temperate Grassland
//   9 = Flooded Grasslands            10 = Montane Grassland
//  11 = Tundra                        12 = Mediterranean Forest/Scrub
//  13 = Desert / Xeric                14 = Mangroves
// ----------------------------------------------------------------------------
fn sample_biome_id(world_xz: vec2<f32>) -> i32 {
    // Stochastic / noise-perturbed border. The biome mask is categorical
    // (you can't lerp IDs), so we instead jitter the *lookup point* by
    // multi-scale gradient noise. The boundary between two biomes wiggles
    // organically along this perturbation field instead of running on the
    // rigid mask pixel grid.
    //
    // Two scales of noise added together so borders look natural at
    // multiple zoom levels:
    //   * 0.3 cycles/km → ~3 km wavelengths (big organic curves at country zoom)
    //   * 4.0 cycles/km → ~250 m wavelengths (fine wiggles at close zoom)
    // Two independent noise samples per scale for X / Y offsets (we shift
    // the input by an arbitrary constant to decorrelate the second axis).
    let off = vec2<f32>(53.7, 17.3);
    let big   = vec2<f32>(noised(world_xz * 0.3).x, noised(world_xz * 0.3 + off).x);
    let fine  = vec2<f32>(noised(world_xz * 4.0).x, noised(world_xz * 4.0 + off).x);
    let perturb = big * 1.0 + fine * 0.1;  // up to ~1.1 km of jitter

    let uv = world_to_world_uv(world_xz + perturb);
    let dim = vec2<f32>(textureDimensions(biome_mask));
    let dim_i = vec2<i32>(dim);
    let coord = clamp(
        vec2<i32>(uv * dim),
        vec2<i32>(0),
        dim_i - vec2<i32>(1),
    );
    return i32(round(textureLoad(biome_mask, coord, 0).x * 255.0));
}

// Per-biome diffuse color used for the "grass / vegetation cover" zone (i.e.
// below the tree line). Returns a color in linear-ish [0, 1] RGB.
//
// Anything not in this table falls back to the default GRASS gradient —
// caller decides via `biome_known`.
fn biome_grass_color(biome_id: i32) -> vec3<f32> {
    if (biome_id == 4) {
        // Temperate Broadleaf & Mixed Forests — most of central Europe.
        return vec3<f32>(0.30, 0.55, 0.20);
    }
    if (biome_id == 5) {
        // Temperate Conifer Forests — the Alps.
        return vec3<f32>(0.16, 0.36, 0.18);
    }
    if (biome_id == 6) {
        // Boreal Forests / Taiga.
        return vec3<f32>(0.22, 0.34, 0.22);
    }
    if (biome_id == 7 || biome_id == 8 || biome_id == 10) {
        // Tropical / Temperate / Montane grasslands.
        return vec3<f32>(0.55, 0.62, 0.30);
    }
    if (biome_id == 11) {
        // Tundra — mostly bare with patchy moss/lichen.
        return vec3<f32>(0.55, 0.58, 0.50);
    }
    if (biome_id == 12) {
        // Mediterranean — Italian/Provence coast.
        return vec3<f32>(0.74, 0.69, 0.34);
    }
    if (biome_id == 13) {
        // Desert / xeric.
        return vec3<f32>(0.85, 0.75, 0.55);
    }
    // Fallback (no real biome at this pixel). Caller usually overrides.
    return vec3<f32>(0.30, 0.55, 0.20);
}

fn biome_known(biome_id: i32) -> bool {
    return biome_id >= 1 && biome_id <= 14;
}

// ----------------------------------------------------------------------------
// Realm palette.
//
// Curated palette of 16 realm colours. Picked to be distinct across the
// hue wheel and saturated enough to read against grass / forest /
// mountain terrain without being garish. Indexed by
// `realm_id % REALM_PALETTE_SIZE`.
// ----------------------------------------------------------------------------
const REALM_PALETTE_SIZE: u32 = 16u;

fn realm_palette(idx: u32) -> vec3<f32> {
    let palette = array<vec3<f32>, 16>(
        vec3<f32>(0.78, 0.20, 0.20),  //  0 crimson
        vec3<f32>(0.85, 0.45, 0.15),  //  1 burnt orange
        vec3<f32>(0.90, 0.78, 0.20),  //  2 gold
        vec3<f32>(0.55, 0.65, 0.20),  //  3 olive
        vec3<f32>(0.20, 0.55, 0.30),  //  4 forest green
        vec3<f32>(0.10, 0.55, 0.55),  //  5 teal
        vec3<f32>(0.30, 0.55, 0.80),  //  6 sky blue
        vec3<f32>(0.20, 0.30, 0.65),  //  7 navy
        vec3<f32>(0.45, 0.25, 0.65),  //  8 royal purple
        vec3<f32>(0.80, 0.25, 0.55),  //  9 magenta
        vec3<f32>(0.60, 0.30, 0.30),  // 10 brick
        vec3<f32>(0.75, 0.65, 0.30),  // 11 mustard
        vec3<f32>(0.45, 0.60, 0.45),  // 12 sage
        vec3<f32>(0.55, 0.50, 0.75),  // 13 lavender
        vec3<f32>(0.85, 0.55, 0.65),  // 14 rose pink
        vec3<f32>(0.55, 0.40, 0.25),  // 15 brown
    );
    return palette[idx % REALM_PALETTE_SIZE];
}

// Strength of the soft-light tint applied across the *entire* realm
// interior. 0 = no region shading; 1 = full soft-light tint. Tune for taste.
const REALM_SHADE_STRENGTH: f32 = 0.55;

// Realm-edge tint applied in screen space to the influence-field iso-line
// (`field.contested` near 1.0). The band itself is sized by
// `BORDER_CONTEST_BAND` inside `fs_main`; these two control the colour of
// the pulse drawn within it.
const BORDER_TINT_STRENGTH: f32 = 0.65;  // peak mix amount on the iso-line
const BORDER_TINT_BRIGHTEN: f32 = 1.4;   // pre-mix brightness boost on the realm tint

// ----------------------------------------------------------------------------
// Settlement influence field.
//
// Each settlement projects a radial influence `strength * exp(-d/E_FOLD)`.
// The shader walks the (small) settlement list per pixel, tracking the
// strongest realm and the strongest *competing*-realm strength as a
// running second place. Output:
//   * `realm_id`  — the realm with the largest field at this pixel.
//   * `alpha`     — [0,1], saturates as `best_strength` grows; near 0 in
//                   wilderness (no settlement reaches here), 1 well inside
//                   any realm. Used to fade realm colouring out at the
//                   edge of civilisation.
//   * `contested` — best/second ratio. 1.0 right at the iso-line where two
//                   realms tie; >1 inside a realm. Drives the live border
//                   pulse via screen-space derivatives.
//
// Tracking second-place across realms (rather than across all settlements)
// preserves correct ratios when a realm has multiple cities: a sibling
// city's contribution is *not* a competitor, so the border between two
// realms still falls at the equal-strength iso-line of the *strongest*
// city in each.
// ----------------------------------------------------------------------------
// E-folding distance in km. Must match `E_FOLD_KM` in `settlements.rs`.
const SETTLEMENT_E_FOLD_KM: f32 = 30.0;

// Terrain-cost coefficients for the influence field. Each coefficient
// scales the multiplier on "effective km per real km" along the path:
//   * MOUNTAIN_COST: at the highest peak (h = 1.0 → 5000 m) the path
//     costs `1 + MOUNTAIN_COST` units per real km, so a city's reach
//     across a 5000 m peak is 1/(1 + MOUNTAIN_COST) of its plain reach.
//   * WATER_COST: similarly for water bodies (lakes / sea).
// Plains (low h, no water) stay at cost 1.0, the original euclidean
// behaviour.
const FIELD_MOUNTAIN_COST: f32 = 4.0;
const FIELD_WATER_COST:    f32 = 3.0;

// Number of midpoint samples along each settlement→fragment line. More
// samples = more accurate path integral but more texture fetches per
// pixel per settlement. 3 catches a single mountain ridge between two
// valleys; bump to 5 if you want finer resolution at the cost of ~2/3
// more samples in this hot loop.
const FIELD_PATH_SAMPLES: u32 = 3u;

// Per-point travel cost: how many "effective km" one real km of path
// traversal costs at this world XZ. Samples the *base* heightmap (so
// erosion bumps don't randomly spike the cost) and the world-anchored
// water mask. Returns >= 1.0 always; 1.0 = plains.
fn travel_cost(world_xz: vec2<f32>) -> f32 {
    let layer_uv = clamp(
        world_to_layer_uv(world_xz, layer),
        vec2<f32>(0.0), vec2<f32>(1.0),
    );
    let h = textureSampleLevel(base_heightmap, samp, layer_uv, 0.0).x;
    let w = textureSampleLevel(
        water_mask, samp, world_to_world_uv(world_xz), 0.0,
    ).x;
    let h_n = clamp(h, 0.0, 1.0);
    let w_n = clamp(w, 0.0, 1.0);
    return 1.0 + h_n * FIELD_MOUNTAIN_COST + w_n * FIELD_WATER_COST;
}

// Effective km between two world points, integrating travel_cost across
// the path with a fixed number of midpoint samples (Simpson-ish, but
// for our purposes a plain mean is fine). Reduces to euclidean distance
// in flat / dry terrain (cost ≡ 1.0).
fn effective_distance(a: vec2<f32>, b: vec2<f32>) -> f32 {
    let d_km = distance(a, b);
    if (d_km < 1e-3) {
        return d_km;
    }
    var cost_sum: f32 = 0.0;
    let n = FIELD_PATH_SAMPLES;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        // Centred sample positions: t ∈ {1/2N, 3/2N, …, (2N-1)/2N}
        let t = (f32(i) + 0.5) / f32(n);
        cost_sum += travel_cost(mix(a, b, t));
    }
    let avg_cost = cost_sum / f32(n);
    return d_km * avg_cost;
}

struct RealmField {
    realm_id: u32,
    // Index of the *specific* settlement whose contribution dominates at
    // this pixel. Used by the hover path to highlight only the dominant
    // city's hinterland (the cells where this city is the strongest), in
    // contrast to `realm_id` which covers every same-realm city's cells.
    city_idx: u32,
    alpha: f32,
    contested: f32,
}

// Jitter scales for the cultural-noise perturbation (see
// `sample_realm_field`). Cycles-per-km on input × amplitude in km.
//   * Big scale: ~17 km wavelength, up to ±8 km of jitter — country-scale
//     meander where one realm wiggles a long arc into another.
//   * Fine scale: ~2 km wavelength, up to ±1.5 km — ragged village-scale
//     edges so the line never reads as perfectly smooth.
// Tune `REALM_PERTURB_BIG_AMP` for taste: larger = more dramatic curves,
// smaller = closer to the bake's clean iso-line.
const REALM_PERTURB_BIG_FREQ:  f32 = 0.06;
const REALM_PERTURB_BIG_AMP:   f32 = 8.0;
const REALM_PERTURB_FINE_FREQ: f32 = 0.5;
const REALM_PERTURB_FINE_AMP:  f32 = 1.5;

fn sample_realm_field(world_xz: vec2<f32>) -> RealmField {
    // Cultural-noise perturbation. The bake's argmax is sharp and the
    // resulting iso-line traces straight-ish equidistance arcs between
    // neighbouring realms; that reads as "painted on" once the eye gets
    // used to it. We jitter the *lookup point* by multi-scale gradient
    // noise so the boundary wiggles organically along this perturbation
    // field instead of along the rigid argmax line. Two independent
    // noise samples per scale (input shifted by an arbitrary constant)
    // give independent X / Y jitter components.
    let off = vec2<f32>(73.1, 41.9);
    let big  = vec2<f32>(
        noised(world_xz * REALM_PERTURB_BIG_FREQ).x,
        noised(world_xz * REALM_PERTURB_BIG_FREQ + off).x,
    );
    let fine = vec2<f32>(
        noised(world_xz * REALM_PERTURB_FINE_FREQ).x,
        noised(world_xz * REALM_PERTURB_FINE_FREQ + off).x,
    );
    let perturb = big * REALM_PERTURB_BIG_AMP + fine * REALM_PERTURB_FINE_AMP;
    let perturbed = world_xz + perturb;

    // The bake pass already evaluated the per-pixel argmax-realm at every
    // cell of `realm_field_tex`. Just look up the (perturbed) texel and
    // unpack it. NEAREST is fine: realm_id and city_idx are categorical
    // (we'd get fictional intermediate IDs from filtering), and the smooth
    // GB channels are over a 2048² grid — plenty of resolution at this
    // 5500 km world bbox.
    let dim = vec2<f32>(textureDimensions(realm_field_tex));
    let uv = world_to_world_uv(perturbed);
    let coord_f = clamp(uv * dim, vec2<f32>(0.0), dim - vec2<f32>(1.0));
    let coord = vec2<i32>(coord_f);
    let s = textureLoad(realm_field_tex, coord, 0);
    // R / A are stored as raw f32 ids (Rgba16Float, exact through 2048).
    let realm_id = u32(round(s.x));
    let city_idx = u32(round(s.w));
    let alpha = s.y;
    // The bake stored `clamp(contested - 1, 0, 1)`; recover by adding 1.
    // Deep interiors saturate at contested = 2 (was unbounded), but
    // that's only used near the iso-line where contested is near 1
    // anyway and the gradient is preserved.
    let contested = 1.0 + s.z;
    return RealmField(realm_id, city_idx, alpha, contested);
}

// ----------------------------------------------------------------------------
// Water classification.
//
// Real DEM data encodes lakes as flat plateaus at the water surface
// elevation — every pixel inside a lake reads the same height value. We
// exploit that here: instead of treating the binary water_mask as the
// source of truth (which produced raster-aligned, painted-on coasts),
// we use it as a *first guess* ("a water body lives near here") and let
// the heightmap itself decide where the water surface ends.
//
// Algorithm:
//   1. Mask gate. Bilinear-sample the binary mask and `smoothstep` it.
//      Where the gate is zero we're far from any water body → land,
//      cheap early-out.
//   2. Lake-surface reference. Take the *minimum* of `base_heightmap`
//      across a small "+" stencil. Inside a lake the min equals the
//      lake surface (flat plateau); on shore pixels just outside the
//      mask, the stencil reaches into the lake → still finds the lake
//      surface. So we get a stable per-pixel water-level reference
//      that smoothly extends a couple of texels past the binary mask.
//   3. Height gate. The center pixel's own base height minus the
//      reference is "how far above lake level am I?". Smoothstep that
//      over a small relief band (~10 m post-VE) for a soft alpha
//      — lake interior → 1.0, shore rising out of the water → fades
//      to 0 over the band, vertical cliffs cut to land immediately.
// Dilation radius for the water mask, in mask texels. The mask is 8192²
// over ~626 km → ~76 m/texel, so 4 texels ≈ 300 m of outward growth.
// This is the upper bound on how far water can "leak" past the binary
// mask boundary; the height gate decides which of those dilated pixels
// are actually flat enough to be water.
const WATER_DILATE_PX: f32 = 4.0;

fn sample_water_blend(world_xz: vec2<f32>) -> f32 {
    // 1. Dilated mask gate. The binary mask alone gives a 1-texel
    //    transition (≈76 m), too sharp at this zoom. We grow each lake
    //    outward by WATER_DILATE_PX mask texels via an 8-tap max kernel
    //    (Chebyshev-radius); the height gate (step 3) then carves the
    //    actual coastline out of that dilated region wherever the
    //    terrain is still at lake level.
    let mask_uv = world_to_world_uv(world_xz);
    let mask_dim = vec2<f32>(textureDimensions(water_mask));
    let off = vec2<f32>(WATER_DILATE_PX) / mask_dim;
    let diag = off * 0.7071;  // sqrt(2)/2 — same Chebyshev radius diagonally.
    var mask = textureSampleLevel(water_mask, samp, mask_uv, 0.0).x;
    mask = max(mask, textureSampleLevel(water_mask, samp, mask_uv + vec2<f32>( off.x,  0.0  ), 0.0).x);
    mask = max(mask, textureSampleLevel(water_mask, samp, mask_uv + vec2<f32>(-off.x,  0.0  ), 0.0).x);
    mask = max(mask, textureSampleLevel(water_mask, samp, mask_uv + vec2<f32>( 0.0  ,  off.y), 0.0).x);
    mask = max(mask, textureSampleLevel(water_mask, samp, mask_uv + vec2<f32>( 0.0  , -off.y), 0.0).x);
    mask = max(mask, textureSampleLevel(water_mask, samp, mask_uv + vec2<f32>( diag.x,  diag.y), 0.0).x);
    mask = max(mask, textureSampleLevel(water_mask, samp, mask_uv + vec2<f32>(-diag.x,  diag.y), 0.0).x);
    mask = max(mask, textureSampleLevel(water_mask, samp, mask_uv + vec2<f32>( diag.x, -diag.y), 0.0).x);
    mask = max(mask, textureSampleLevel(water_mask, samp, mask_uv + vec2<f32>(-diag.x, -diag.y), 0.0).x);
    let mask_gate = smoothstep(0.05, 0.95, mask);
    if (mask_gate <= 0.0) {
        return 0.0;
    }

    // 2. Lake-surface reference: 4-tap min on the *base* heightmap
    //    (pre-erosion, so lake plateaus stay flat). Offset is a few
    //    layer-texels in each cardinal direction — wide enough that
    //    dilated shore pixels still see the lake in at least one tap,
    //    narrow enough not to leak across distinct water bodies.
    let layer_uv = clamp(
        world_to_layer_uv(world_xz, layer),
        vec2<f32>(0.0), vec2<f32>(1.0),
    );
    let texel = vec2<f32>(3.0) / vec2<f32>(LAYER_SIZE);
    let center_h = textureSampleLevel(base_heightmap, samp, layer_uv, 0.0).x;
    var lake_h = center_h;
    lake_h = min(lake_h, textureSampleLevel(base_heightmap, samp, layer_uv + vec2<f32>(texel.x, 0.0), 0.0).x);
    lake_h = min(lake_h, textureSampleLevel(base_heightmap, samp, layer_uv - vec2<f32>(texel.x, 0.0), 0.0).x);
    lake_h = min(lake_h, textureSampleLevel(base_heightmap, samp, layer_uv + vec2<f32>(0.0, texel.y), 0.0).x);
    lake_h = min(lake_h, textureSampleLevel(base_heightmap, samp, layer_uv - vec2<f32>(0.0, texel.y), 0.0).x);

    // 3. Height gate. `above` = how far this pixel's terrain rises
    //    above the local lake surface, in fractional 5000-m units; ×VE
    //    to share the same scaled-Y space as WATER_HEIGHT etc. The
    //    relief band sets how aggressively water spills onto shallow
    //    shores: too narrow → sharp coast, too wide → water creeps up
    //    grassy slopes. ~25 m post-VE works for Swiss alpine terrain.
    let above = max(center_h - lake_h, 0.0) * VERTICAL_EXAGGERATION;
    let band = 0.05 * VERTICAL_EXAGGERATION;  // 25 m post-VE
    let height_gate = 1.0 - smoothstep(0.0, band, above);

    return mask_gate * height_gate;
}

// ----------------------------------------------------------------------------
// Per-fragment procedural "texture" detail.
//
// The cached `detail_noise` layer caps its highest representable frequency
// at 1 cycle per layer-texel (= ~100 m at default zoom, ~1 km zoomed-out).
// Anything finer than that has to be evaluated *directly per fragment* —
// no cache, no filtering, just the analytic noise function.
//
// 4 octaves stacked, frequencies in cycles per km:
//   50  → ~20 m features
//   100 → ~10 m features
//   200 → ~5 m features
//   400 → ~2.5 m features
// Output is in roughly [-1.5, +1.5]; multiply by an amplitude < 1 when used
// as a color tint so the surface doesn't go negative.
fn texture_noise(world_xz: vec2<f32>) -> f32 {
    var v: f32 = 0.0;
    var a: f32 = 1.0;
    var f: f32 = 50.0;
    for (var i = 0; i < 4; i = i + 1) {
        v = v + noised(world_xz * f).x * a;
        a = a * 0.5;
        f = f * 2.0;
    }
    return v;
}

// ----------------------------------------------------------------------------
// Heightmap sampling
// ----------------------------------------------------------------------------
fn uv_to_coord(uv: vec2<f32>) -> vec2<i32> {
    let dim_f = vec2<f32>(LAYER_SIZE);
    let dim_i = vec2<i32>(i32(LAYER_SIZE));
    return clamp(vec2<i32>(uv * dim_f), vec2<i32>(0), dim_i - vec2<i32>(1));
}

// Map a world XZ to the [0,1] UV of the world layers (which all share the
// same covered AABB). Clamped one texel inside the edge so neighbour
// fetches in `map_full` stay within the valid region.
fn world_xz_to_uv(p: vec3<f32>) -> vec2<f32> {
    let pixel = 1.0 / vec2<f32>(LAYER_SIZE);
    let uv = world_to_layer_uv(p.xz, layer);
    return clamp(uv, pixel, vec2<f32>(1.0) - pixel);
}

fn map_height(uv: vec2<f32>) -> f32 {
    // Read the eroded heightmap (real Switzerland elevation + Phacelle
    // procedural gullies layered on top). Bilinear-sampled so the layer's
    // 1024² texels don't show up as visible blocks on screen. Multiplied by
    // VERTICAL_EXAGGERATION here so all callers (raymarch + normals + height
    // comparisons) work in the same scaled Y space.
    let h = textureSampleLevel(terrain, samp, uv, 0.0).x;
    return h * VERTICAL_EXAGGERATION;
}

// Pre-erosion heightmap sample. Used for computing *smooth* derivatives:
// the eroded `terrain` layer carries ~50 m of vertical detail at ~70 m
// horizontal scale, which (a) aliases hard into the layer's 600 m–ish
// texels at wide zooms and (b) would otherwise feed noisy per-texel
// gradients into the shadow march and `cliff_mask` slope test. So we keep
// `map_height` (eroded) for the actual surface position the raymarch hits,
// but route normal / shadow lookups through the macro-scale base heightmap.
fn map_height_base(uv: vec2<f32>) -> f32 {
    let h = textureSampleLevel(base_heightmap, samp, uv, 0.0).x;
    return h * VERTICAL_EXAGGERATION;
}

// height in .x, normal in .yzw (Y-up world space).
fn map_full(uv: vec2<f32>) -> vec4<f32> {
    // Surface height (returned in .x) tracks the eroded terrain, so the
    // visual surface still has erosion micro-detail.
    let height = map_height(uv);
    // Normal (returned in .yzw) is computed from the *base* heightmap. This
    // gives smooth macro-slopes for cliff/snow/lighting decisions — erosion
    // bumps don't masquerade as cliffs and don't speckle the lighting with
    // pixelated artifacts.
    let pixel = 1.0 / vec2<f32>(LAYER_SIZE);
    let uv1 = uv + vec2<f32>(pixel.x, 0.0);
    let uv2 = uv + vec2<f32>(0.0, pixel.y);
    let h0 = map_height_base(uv);
    let h1 = map_height_base(uv1);
    let h2 = map_height_base(uv2);
    // Convert the uv steps into world XZ steps (km per layer texel) so the
    // gradient is dimensionally consistent with the (already-VE-scaled)
    // height. Without this, the dhdx / dhdz scale mismatch would yield a
    // near-horizontal normal regardless of the actual slope.
    let world_step = (layer.covered_max - layer.covered_min) * pixel;
    let dhdx = (h1 - h0) / world_step.x;
    let dhdz = (h2 - h0) / world_step.y;
    let normal = normalize(vec3<f32>(-dhdx, 1.0, -dhdz));
    return vec4<f32>(height, normal);
}

// ----------------------------------------------------------------------------
// Raymarching
// ----------------------------------------------------------------------------
struct MarchResult {
    t: f32,             // distance along ray to hit, or -1 if miss
    normal: vec3<f32>,  // only valid for STRATA / WATER (GROUND filled by caller)
    material: i32,
    s_t: f32,           // soft-shadow factor
}

fn march(ro: vec3<f32>, rd: vec3<f32>) -> MarchResult {
    var s_t: f32 = 9999.0;
    let bs = box_size();
    let bc = box_center();
    // The box is axis-aligned but not centered at world origin (it tracks the
    // camera via the world layer's covered AABB). Translate the ray origin
    // into the box's local frame for the intersection; the returned
    // (t_near, t_far, normal) are in ray-space and need no further fix-up.
    let local_ro = ro - bc;
    let box = box_intersection(local_ro, rd, bs);

    let t_start = max(0.0, box.t_near) + 1e-2;
    let t_end   = box.t_far - 1e-2;

    var material: i32 = M_GROUND;
    var normal = vec3<f32>(0.0);
    var step_size:  f32 = 0.0;
    var step_scale: f32 = 1.0 / RAYMARCH_QUALITY;
    let samples: i32 = i32(48.0 * RAYMARCH_QUALITY);
    var t: f32 = t_start;
    var hit_strata: bool = false;

    for (var i: i32 = 0; i < samples; i = i + 1) {
        let pos = ro + rd * t;
        let h = map_height(world_xz_to_uv(pos));
        let altitude = pos.y - h;
        s_t = max(0.0, min(s_t, altitude / t));

        if (altitude < 0.0 && i < 1) {
            // Hit a side wall of the bounding box on first sample. Below sea
            // level (y < 0) we escape to sky; otherwise it's an exposed cliff
            // (strata material).
            if (pos.y < 0.0) {
                return MarchResult(-1.0, vec3<f32>(0.0), M_GROUND, 9999.0);
            }
            normal = box.normal;
            material = M_STRATA;
            hit_strata = true;
            break;
        }

        if (altitude < 0.0) {
            // Refine: undo last step and halve the step scale.
            step_scale *= 0.5;
            t -= step_size * step_scale;
        } else {
            step_size = abs(altitude) + min(1e-2, abs(altitude) * 0.01);
            t += step_size * step_scale;
        }
    }

    // The original Shadertoy capped the ray on a global water plane at
    // WATER_HEIGHT covering the full playable XZ — a holdover from when
    // terrain only ranged ~[0.45, 0.55] and that plane sat just below it. With
    // real Swiss elevations spanning [0, 1], the plane is far below most
    // ground and would flood every valley with "ocean". Real water bodies
    // come from `water_mask` instead (consulted in fs_main).

    if (box.t_far < 0.0) {
        return MarchResult(-1.0, vec3<f32>(0.0), M_GROUND, 9999.0);
    }
    if (t > t_end) {
        return MarchResult(-1.0, normal, material, s_t);
    }
    return MarchResult(t, normal, material, s_t);
}

fn get_reflection(p: vec3<f32>, r: vec3<f32>, sun: vec3<f32>, smoothness: f32) -> vec3<f32> {
    let refl = sky_color(r, sun) * 4.0;
    let m = march(p, r);
    return refl * (1.0 - exp(-m.s_t * 10.0 * sq(smoothness)));
}

// Lightweight shadow march: same algorithm as `march`, but samples the
// smooth `base_heightmap` (no erosion) and only returns the soft-shadow
// factor we actually need at the call site. Keeps shadow casting tied to
// macro mountain shape instead of per-texel erosion bumps, which produced
// the blocky pixelated shadow grid we were seeing at default zoom.
fn shadow_march(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    var s_t: f32 = 9999.0;
    let bs = box_size();
    let bc = box_center();
    let local_ro = ro - bc;
    let box = box_intersection(local_ro, rd, bs);

    let t_start = max(0.0, box.t_near) + 1e-2;
    var step_size:  f32 = 0.0;
    var step_scale: f32 = 1.0 / RAYMARCH_QUALITY;
    let samples: i32 = i32(48.0 * RAYMARCH_QUALITY);
    var t: f32 = t_start;

    for (var i: i32 = 0; i < samples; i = i + 1) {
        let pos = ro + rd * t;
        let h = map_height_base(world_xz_to_uv(pos));
        let altitude = pos.y - h;
        s_t = max(0.0, min(s_t, altitude / t));
        if (altitude < 0.0) {
            step_scale *= 0.5;
            t -= step_size * step_scale;
        } else {
            step_size = abs(altitude) + min(1e-2, abs(altitude) * 0.01);
            t += step_size * step_scale;
        }
    }
    return s_t;
}

// ----------------------------------------------------------------------------
// Settlement markers.
//
// Draw a small dot at every settlement so the influence-field plot is
// legible — "OK Zürich is here, that's why this corner is crimson". Each
// dot has:
//   * an outer realm-coloured ring (so you can read the city's allegiance
//     at a glance), and
//   * a bright cream core (so the dot reads against any terrain colour).
// Radius scales with strength so big cities are visibly bigger pinpricks.
// World-unit sized: at default zoom the dots are ~2 km wide — visible but
// not overwhelming. Out of `MAX_SETTLEMENTS` we only consider the *closest*
// one (smallest `distance / radius`) per fragment, so overlapping dots
// don't double-tint.
fn settlement_marker(world_xz: vec2<f32>) -> vec4<f32> {
    var best_d_norm: f32 = 1.0e9;
    var best_realm: u32 = 0u;
    let n = min(settlements.count, MAX_SETTLEMENTS);
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let s = settlements.items[i];
        let d_km = distance(world_xz, s.world_xz);
        // Marker radius in km: ~1.0 km baseline + scales with sqrt(strength)
        // so doubling population only ~1.4×s the dot, not 2×. Keeps small
        // towns from being invisible while big cities don't dominate.
        let radius = 1.0 + sqrt(max(s.strength, 0.0)) * 0.10;
        let d_norm = d_km / radius;
        if (d_norm < best_d_norm) {
            best_d_norm = d_norm;
            best_realm = s.realm_id;
        }
    }
    if (best_d_norm > 1.0) {
        return vec4<f32>(0.0);
    }
    let realm_col = realm_palette(best_realm);
    // Inside-out structure: bright cream core, realm-coloured ring, soft
    // anti-aliased outer edge.
    let core = 1.0 - smoothstep(0.45, 0.55, best_d_norm);
    let ring_outer = 1.0 - smoothstep(0.92, 1.0, best_d_norm);
    let marker_rgb = mix(realm_col * 0.85, vec3<f32>(0.98, 0.95, 0.86), core);
    return vec4<f32>(marker_rgb, ring_outer);
}

// ----------------------------------------------------------------------------
// Main fragment shader
// ----------------------------------------------------------------------------
@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let frag_pos = frag.xy;

    // Camera setup
    let ray = get_ray(frag_pos);
    let ro = ray.ro;
    let rd = ray.rd;

    // Fixed sun direction (Y-up world).
    let sun = normalize(vec3<f32>(-1.0, 0.4, 0.05));

    // Trace the ray.
    let m = march(ro, rd);

    let fog_color = vec3<f32>(1.0) - exp(-sky_color(rd, sun) * 2.0);

    var color = vec3<f32>(0.0);

    if (m.t < 0.0) {
        // Sky. Brighter at top of screen — Y-down screen so flip the gradient term.
        let sky_t = 1.0 - frag_pos.y / u.i_resolution.y;
        color = fog_color * (1.0 + pow(sky_t, 3.0) * 3.0) * 0.5;
    } else {
        let pos = ro + rd * m.t;
        var normal = m.normal;
        var material = m.material;

        if (material == M_GROUND) {
            normal = map_full(world_xz_to_uv(pos)).yzw;
        }

        // Settlement influence field, evaluated once per fragment. Drives
        // the M_GROUND realm tint *and* the post-lighting border pulse;
        // both share this single eval to keep the per-fragment
        // settlement-loop cost down to one pass.
        let field = sample_realm_field(pos.xz);

        // Water classification. The water_mask is treated as a *first
        // guess* only — the actual decision uses the heightmap (lakes
        // are flat plateaus in the DEM, so the local height tells us
        // whether we're inside a lake basin). See `sample_water_blend`.
        // Above 0.5 we classify the hit as water; the partial range
        // below 0.5 feeds the wet-shore tint on land.
        var water_blend: f32 = 0.0;
        if (material != M_STRATA) {
            water_blend = sample_water_blend(pos.xz);
            if (water_blend > 0.5) {
                material = M_WATER;
                normal = vec3<f32>(0.0, 1.0, 0.0);
            }
        }

        // Detail breakup texture (bilinear so the layer's coarse texels
        // don't show through as visible breakup-pattern boundaries).
        // The raw `noised` octave sum has magnitude ~3.4 (= 0.5 * Σ
        // 0.95^i for 8 octaves); divide by that so `breakup` stays in
        // roughly [-1, 1] and downstream multipliers (e.g. `breakup * 0.3`
        // for surface tint) behave like the comments say.
        let breakup_tex = textureSampleLevel(
            detail_noise, samp, world_xz_to_uv(pos), 0.0,
        );
        let breakup = breakup_tex.x / 3.4;
        if (material == M_WATER) {
            normal = normalize(normal + vec3<f32>(breakup_tex.z, 0.0, breakup_tex.y) * 0.1);
        }

        var diffuse_color = vec3<f32>(0.5);
        var f0 = vec3<f32>(0.04);
        var smoothness: f32 = 0.0;
        // Hard-coded since we deferred packed-data extraction.
        let occlusion: f32 = 1.0;
        let trees_v: f32 = -1.0;

        let r_dir = reflect(rd, normal);

        if (material == M_GROUND) {
            // The biome is the *base* surface colour everywhere on land.
            // Rock, snow, and sand override it in specific cases (high
            // elevation, steep cliffs, water line). This is the inverse of
            // the original Shadertoy logic, which started from rock and
            // mixed grass on top — but that approach fought the heightmap's
            // exaggerated slopes and left most of the terrain as bare rock.
            // Use the *honest* surface height for biome/material decisions.
            // Adding noise here makes random pixels jump above/below band
            // thresholds (e.g. snow line, tree line), which paints noise-shaped
            // blobs of the wrong material across an otherwise uniform region.
            let h = pos.y;

            // 1. Base colour: biome (or default green if outside the
            //    classified region).
            let biome_id = sample_biome_id(pos.xz);
            let default_grass = mix(
                GRASS_COLOR1,
                GRASS_COLOR2,
                smoothstep(WATER_HEIGHT, TREE_LINE, h),
            );
            diffuse_color = select(
                default_grass,
                biome_grass_color(biome_id),
                biome_known(biome_id),
            );

            // 2. Above the tree line, fade biome colour into bare rock.
            //    Pure rock between SNOW_LINE and PEAK_SNOW; pure biome
            //    below TREE_LINE.
            let rock_alt_mask = smoothstep(TREE_LINE, SNOW_LINE, h);
            diffuse_color = mix(diffuse_color, CLIFF_COLOR, rock_alt_mask);

            // 3. Cliffs (anywhere): override with rock on steep slopes.
            //    `normal.y` is the post-VE world-space normal (so it sees
            //    the mountains as 10× steeper than reality), which is
            //    what we want for visual purposes. Calibration:
            //      normal.y > 0.7 → 0% cliff (slope < ~45° post-VE)
            //      normal.y < 0.4 → 100% cliff (slope > ~65° post-VE)
            //    Switzerland has alpine pasture on slopes mortals would
            //    call "steep", so we're generous on the upper bound.
            //    Slope analysis uses the bare normal — don't perturb it
            //    with surface-tint noise (that's what produced the
            //    grid-of-grey-splotches artifact).
            let cliff_mask = 1.0 - smoothstep(0.4, 0.7, normal.y);
            diffuse_color = mix(diffuse_color, CLIFF_COLOR, cliff_mask);

            // 4. Snow above the snow line on shallow-enough slopes
            //    (steep faces shed snow — avalanches). Calibrated like
            //    cliff_mask but a bit stricter: snow needs flatter
            //    ground than grass to accumulate. No breakup in the
            //    slope test (same reasoning as cliff_mask).
            let snow_alt_mask = smoothstep(SNOW_LINE, PEAK_SNOW, h);
            let snow_slope_mask = smoothstep(0.55, 0.85, normal.y);
            diffuse_color = mix(
                diffuse_color,
                vec3<f32>(0.95, 0.97, 1.0),
                snow_alt_mask * snow_slope_mask,
            );

            // 5. Sand / shore at the water line.
            let sand_mask = 1.0 - smoothstep(
                WATER_HEIGHT,
                WATER_HEIGHT + 0.005 * VERTICAL_EXAGGERATION,
                h,
            );
            diffuse_color = mix(diffuse_color, SAND_COLOR, sand_mask);

            // 5b. Wet-shore tint. Land pixels still inside the soft
            //     edge of the water classifier (water_blend ∈ [0, 0.5])
            //     get nudged toward the water-shore colour so the coast
            //     dissolves rather than ending in a hard line.
            let wet_shore = smoothstep(0.05, 0.5, water_blend);
            diffuse_color = mix(diffuse_color, WATER_SHORE_COLOR, wet_shore * 0.45);

            // Slight breakup tint so the surface isn't perfectly uniform.
            diffuse_color *= 1.0 + breakup * 0.3;

            // Per-fragment procedural texture detail. Stronger contrast on
            // exposed rock / cliff faces (roughly the snow-line altitude or
            // anywhere the cliff_mask kicks in), softer on grass / forest,
            // and even softer on snow (snow reads visually as smooth).
            let tex = texture_noise(pos.xz);
            let tex_strength = mix(
                mix(0.10, 0.20, cliff_mask),  // forest 0.10  → cliff 0.20
                0.05,                          // snow flattens to 0.05
                snow_alt_mask * snow_slope_mask,
            );
            diffuse_color *= 1.0 + tex * tex_strength;

            // 6. Realm shading driven by the settlement influence field.
            //    `field.alpha` fades the realm tint to 0
            //    in wilderness so unclaimed land keeps its honest biome /
            //    cliff colour, and saturates near 1 well inside any
            //    realm.
            //    * Terrain mode: soft-light blend, scaled by alpha. Within
            //      a realm core, full realm wash; at the field edge, the
            //      tint smoothly disappears.
            //    * Political mode: pale-grey wilderness fades to opaque
            //      realm colour as alpha rises.
            if (field.alpha > 0.0) {
                let realm_col = realm_palette(field.realm_id);
                if (u.map_mode == MAP_MODE_POLITICAL) {
                    let wilderness = vec3<f32>(0.55, 0.55, 0.50);
                    diffuse_color = mix(wilderness, realm_col, field.alpha);
                } else {
                    let safe_diffuse = clamp(
                        diffuse_color, vec3<f32>(0.0), vec3<f32>(1.0),
                    );
                    diffuse_color = mix(
                        diffuse_color,
                        soft_light(safe_diffuse, realm_col),
                        REALM_SHADE_STRENGTH * field.alpha,
                    );
                }
            }

            // 7. Country-name labels are now drawn by a separate SDF
            //    glyph-atlas pass on top of the swapchain (see
            //    `passes::realm_labels`); previously we composited a
            //    Canvas2D-baked overlay here.
        } else if (material == M_STRATA) {
            let diff = pos.y - map_height(world_xz_to_uv(pos));
            let strata = smoothstep(
                vec3<f32>(0.0), vec3<f32>(1.0),
                cos(diff * vec3<f32>(130.0, 190.0, 250.0)),
            );
            diffuse_color = vec3<f32>(0.3);
            diffuse_color = mix(diffuse_color, vec3<f32>(0.50), strata.x);
            diffuse_color = mix(diffuse_color, vec3<f32>(0.55), strata.y);
            diffuse_color = mix(diffuse_color, vec3<f32>(0.60), strata.z);
            diffuse_color *= exp(diff * 10.0) * vec3<f32>(1.0, 0.9, 0.7);
        } else { // M_WATER
            // Water comes from the binary mask, so we don't have a
            // flat-water-plane vs ground delta to compute foam/shore from
            // (the original Shadertoy assumed one). Instead lerp between
            // open-water and shore colour by `water_blend` — the soft
            // mask edge gives a natural shoreline tint, and breakup adds
            // a subtle wave variation.
            let shore_strength = clamp(1.0 - (water_blend - 0.5) * 2.0, 0.0, 1.0);
            diffuse_color = mix(
                WATER_COLOR,
                WATER_SHORE_COLOR,
                shore_strength,
            );
            diffuse_color *= 1.0 + breakup * 0.15;
            smoothness = 0.95;
        }

        // Shadow ray (skip for the inside-of-strata case). Routed through
        // `shadow_march`, which samples the smooth base heightmap so
        // erosion's per-texel bumps don't paint a blocky shadow grid over
        // the surface.
        var shadow: f32 = 1.0;
        if (material != M_STRATA) {
            let s_t = shadow_march(pos + vec3<f32>(0.0, 1.0, 0.0) * 1e-4, sun);
            shadow = 1.0 - exp(-s_t * 20.0);
        }

        // Lighting decomposition.
        color = diffuse_color * sky_color(normal, sun) * fd_lambert();
        color *= occlusion;
        color += shade(diffuse_color, f0, smoothness, normal, -rd, sun, SUN_COLOR * shadow);
        // Fake bounce.
        color += diffuse_color * SUN_COLOR
            * (dot(normal, sun * vec3<f32>(1.0, -1.0, 1.0)) * 0.5 + 0.5)
            * fd_lambert() / PI;
        // Reflection.
        color += get_reflection(pos, r_dir, sun, smoothness)
            * f_schlick(f0, dot(-rd, normal));

        // Live influence-field borders — the iso-line where two realms'
        // fields tie. `field.contested = best/second_best` equals 1.0
        // right on that line and grows inside a realm; we draw a band
        // covering `contested ∈ [1.0, 1.0 + BORDER_CONTEST_BAND]`.
        //
        // We deliberately *don't* use `fwidth` here: WGSL strict mode
        // disallows derivative builtins under non-uniform control flow
        // (the surrounding `if (m.t < 0)` branch is non-uniform), and
        // hoisting the derivative out of that branch requires evaluating
        // the field at a screen-anchored pseudo-position. Constant-band
        // is simpler and predictable; the trade-off is the on-screen
        // border width changes with zoom (thicker at deep zoom, thinner
        // when zoomed out). Tune `BORDER_CONTEST_BAND` for taste — with
        // 30 km e-fold and ~equal-strength cities, 0.15 gives roughly a
        // 2 km wide ground band.
        const BORDER_CONTEST_BAND: f32 = 0.15;
        let edge = 1.0 - smoothstep(
            0.0, BORDER_CONTEST_BAND, field.contested - 1.0,
        );
        if (material == M_GROUND && field.alpha > 0.05) {
            let realm_col = realm_palette(field.realm_id);
            let tint = clamp(
                realm_col * BORDER_TINT_BRIGHTEN,
                vec3<f32>(0.0), vec3<f32>(1.0),
            );
            color = mix(color, tint, edge * BORDER_TINT_STRENGTH * field.alpha);

            // Hover highlights, layered:
            //   * Realm wash: gentle white tint on every cell of the
            //     hovered realm — shows the realm's overall extent.
            //   * Hinterland wash: stronger tint on top, restricted to
            //     cells whose dominant city matches the hovered city.
            //     Both indices are `value + 1` on the CPU side so 0 is a
            //     reserved "nothing hovered" sentinel.
            if (u.hovered_pid != 0u && (field.realm_id + 1u) == u.hovered_pid) {
                color = mix(color, vec3<f32>(1.0), 0.10);
            }
            if (u.hovered_city != 0u && (field.city_idx + 1u) == u.hovered_city) {
                color = mix(color, vec3<f32>(1.0), 0.18);
            }
        }

        // Settlement markers. Dots that show *where the cities are*; the
        // influence field is otherwise an invisible math object you have
        // to infer from the colours. Drawn last so they sit on top of
        // everything else (border tint, realm wash, hover, terrain).
        if (material == M_GROUND) {
            let marker = settlement_marker(pos.xz);
            color = mix(color, marker.rgb, marker.a);
        }
    }

    // ---- Atmospheric scattering integral --------------------------------
    let bc_atm = box_center();
    let box = box_intersection(ro - bc_atm, rd, box_size());
    let costh = dot(rd, sun);
    let phase_r = phase_rayleigh(costh);
    let phase_m = phase_mie(costh, 0.6);

    var od = vec2<f32>(0.0);
    var tsm = vec3<f32>(1.0);
    var sct = vec3<f32>(0.0);
    let ray_length = select(box.t_far, m.t, m.t > 0.0) - box.t_near;
    let stepsize = ray_length / 16.0;
    for (var i: f32 = 0.0; i < 16.0; i = i + 1.0) {
        let p = ro + rd * (box.t_near + (i + 0.5) * stepsize);
        // Stylised low-altitude haze: full density at sea level (y = 0),
        // fading to nothing roughly 1000 m up (post-exaggeration that's
        // 0.2 * VERTICAL_EXAGGERATION in scaled units). Below sea level we
        // have no atmosphere (ray is underground inside the box).
        var d = 1.0 - clamp01(max(0.0, p.y) / (0.2 * VERTICAL_EXAGGERATION));
        if (p.y < 0.0) { d = 0.0; }
        // Divide by VERTICAL_EXAGGERATION so the integrated optical depth
        // (∫ density ds) stays calibrated for vertical rays. Oblique rays
        // at high tilt still traverse a long horizontal stretch through
        // the haze band, so we additionally cut the density base by 5×
        // so distant mountains don't wash out under a low camera angle.
        let density_r = d * 2e4 / VERTICAL_EXAGGERATION;
        let density_m = d * 2e4 / VERTICAL_EXAGGERATION;
        od += stepsize * vec2<f32>(density_r, density_m);
        tsm = exp(-(od.x * C_RAYLEIGH + od.y * C_MIE));
        sct += tsm * C_RAYLEIGH * phase_r * density_r * stepsize;
        sct += tsm * C_MIE      * phase_m * density_m * stepsize;
    }
    color = color * tsm + sct * 10.0;

    // Tonemap + gamma.
    color = tonemap_aces(color);
    color = pow(color, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(color, 1.0);
}

// ============================================================================
// Mesh path (Option 3 from the perf investigation).
//
// `vs_mesh` rasterizes a tessellated grid over the cached layer AABB,
// displacing each vertex by sampling the terrain heightmap. `fs_mesh` then
// runs the same biome/water/realm/lighting code as `fs_main` but skips
// everything that depends on the per-fragment ray:
//   * No `march` / `box_intersection` — the rasterizer + depth buffer
//     handle surface visibility.
//   * No `shadow_march` — fixed `shadow = 1.0` (precompute later via a
//     world-space sun-visibility bake).
//   * No `get_reflection` — sky-only stand-in (it would call `march`).
//   * No atmospheric scattering integral — replaced with a cheap
//     exponential distance fog.
//
// Toggle from the `T` key in `lib.rs` to A/B against `fs_main`.
// ============================================================================

// Cells per side; vertex count = 6 * (MESH_GRID - 1)^2. Must match
// `MESH_GRID` in `passes/image.rs` (used for the `draw(0..N, 0..1)`
// count there).
const MESH_GRID: u32 = 1024u;
const MESH_NEAR: f32 = 0.1;
const MESH_FAR: f32 = 12000.0;

struct VsMeshOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
}

// Build (proj * view) from the existing camera uniforms. Same eye / look_at
// / basis math as `get_ray()` in `fs_main` so the two paths align pixel-for-
// pixel under matching cameras.
fn make_view_proj() -> mat4x4<f32> {
    let look_at = vec3<f32>(u.world_center.x, CAMERA_TARGET_Y, u.world_center.y);
    let eye = look_at + u.eye_offset;
    let forward = normalize(look_at - eye);
    let world_up = vec3<f32>(0.0, 1.0, 0.0);
    let right = normalize(cross(world_up, forward));
    let up = cross(forward, right);

    // View matrix: camera looks down -Z in view space. WGSL mat4x4 is
    // column-major, so each `vec4` here is a column.
    let view = mat4x4<f32>(
        vec4<f32>(right.x, up.x, -forward.x, 0.0),
        vec4<f32>(right.y, up.y, -forward.y, 0.0),
        vec4<f32>(right.z, up.z, -forward.z, 0.0),
        vec4<f32>(-dot(right, eye), -dot(up, eye), dot(forward, eye), 1.0),
    );

    // Reverse-Z would be more depth-precise, but [0,1] perspective is
    // standard and matches WebGPU's default depth range. Right-handed,
    // looking down -Z; near maps to 0, far maps to 1 after w-divide.
    let aspect = u.i_resolution.x / u.i_resolution.y;
    let f = 1.0 / tan(CAMERA_FOV_Y * 0.5 * DEG_TO_RAD);
    let near = MESH_NEAR;
    let far = MESH_FAR;
    let proj = mat4x4<f32>(
        vec4<f32>(f / aspect, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, f, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, far / (near - far), -1.0),
        vec4<f32>(0.0, 0.0, near * far / (near - far), 0.0),
    );
    return proj * view;
}

// Decode a linear vertex index into a (u, v) ∈ [0, 1]^2 grid coordinate.
// Six verts per cell encode two triangles in CCW order from above:
//   tri A: BL  BR  TL
//   tri B: BR  TR  TL
// We use `cull_mode: None` on the pipeline so winding direction is
// irrelevant; this layout is just convention.
fn grid_uv(vi: u32) -> vec2<f32> {
    let cells = MESH_GRID - 1u;
    let cell = vi / 6u;
    let corner = vi % 6u;
    let cx = cell % cells;
    let cy = cell / cells;
    var ox: u32 = 0u;
    var oy: u32 = 0u;
    switch (corner) {
        case 0u: { ox = 0u; oy = 0u; }
        case 1u: { ox = 1u; oy = 0u; }
        case 2u: { ox = 0u; oy = 1u; }
        case 3u: { ox = 1u; oy = 0u; }
        case 4u: { ox = 1u; oy = 1u; }
        case 5u: { ox = 0u; oy = 1u; }
        default: {}
    }
    return vec2<f32>(f32(cx + ox), f32(cy + oy)) / f32(cells);
}

@vertex
fn vs_mesh(@builtin(vertex_index) vi: u32) -> VsMeshOut {
    let uv = grid_uv(vi);
    // Map uv ∈ [0,1] to the cached layer's world AABB. The mesh covers
    // the camera-anchored layer rectangle, but its *displacement* comes
    // from the world-anchored heightmap (4096², ~1.34 km/px) instead of
    // the layer cache (1024², ~3 km/px at default zoom). That's the
    // difference between visibly-faceted ridges and smooth ones.
    let world_xz = mix(layer.covered_min, layer.covered_max, uv);
    let world_uv = world_to_world_uv(world_xz);
    let h_norm = sample_world_height(world_uv);
    let h_terrain = h_norm * VERTICAL_EXAGGERATION;

    // Water flattening. The heightmap stores bathymetry (below-sea-level
    // seabed elevation), so leaving the mesh at h_terrain would draw
    // seabed contours through the water shading. Snap submarine
    // vertices to a flat sea level (WATER_HEIGHT, post-VE) so the ocean
    // reads as a flat plane.
    let w = textureSampleLevel(water_mask, samp, world_uv, 0.0).x;
    let world_y = select(h_terrain, WATER_HEIGHT, w > 0.5);

    let world_pos = vec3<f32>(world_xz.x, world_y, world_xz.y);
    let vp = make_view_proj();
    let clip_pos = vp * vec4<f32>(world_pos, 1.0);
    return VsMeshOut(clip_pos, world_pos);
}

@fragment
fn fs_mesh(in: VsMeshOut) -> @location(0) vec4<f32> {
    let pos = in.world_pos;
    let layer_uv = clamp(
        world_to_layer_uv(pos.xz, layer),
        vec2<f32>(0.0), vec2<f32>(1.0),
    );
    var normal = map_full(layer_uv).yzw;

    let field = sample_realm_field(pos.xz);

    // Water classification — same as fs_main but on a fixed surface.
    var water_blend: f32 = sample_water_blend(pos.xz);
    var material: i32 = M_GROUND;
    if (water_blend > 0.5) {
        material = M_WATER;
        normal = vec3<f32>(0.0, 1.0, 0.0);
    }

    let breakup_tex = textureSampleLevel(detail_noise, samp, layer_uv, 0.0);
    let breakup = breakup_tex.x / 3.4;
    if (material == M_WATER) {
        normal = normalize(
            normal + vec3<f32>(breakup_tex.z, 0.0, breakup_tex.y) * 0.1,
        );
    }

    // Eye-to-fragment direction for view-dependent shading. Mirrors
    // `rd` in fs_main.
    let look_at = vec3<f32>(u.world_center.x, CAMERA_TARGET_Y, u.world_center.y);
    let eye = look_at + u.eye_offset;
    let rd = normalize(pos - eye);

    var diffuse_color = vec3<f32>(0.5);
    var f0 = vec3<f32>(0.04);
    var smoothness: f32 = 0.0;
    let occlusion: f32 = 1.0;

    if (material == M_GROUND) {
        let h = pos.y;
        let biome_id = sample_biome_id(pos.xz);
        let default_grass = mix(
            GRASS_COLOR1,
            GRASS_COLOR2,
            smoothstep(WATER_HEIGHT, TREE_LINE, h),
        );
        diffuse_color = select(
            default_grass,
            biome_grass_color(biome_id),
            biome_known(biome_id),
        );
        let rock_alt_mask = smoothstep(TREE_LINE, SNOW_LINE, h);
        diffuse_color = mix(diffuse_color, CLIFF_COLOR, rock_alt_mask);
        let cliff_mask = 1.0 - smoothstep(0.4, 0.7, normal.y);
        diffuse_color = mix(diffuse_color, CLIFF_COLOR, cliff_mask);
        let snow_alt_mask = smoothstep(SNOW_LINE, PEAK_SNOW, h);
        let snow_slope_mask = smoothstep(0.55, 0.85, normal.y);
        diffuse_color = mix(
            diffuse_color,
            vec3<f32>(0.95, 0.97, 1.0),
            snow_alt_mask * snow_slope_mask,
        );
        let sand_mask = 1.0 - smoothstep(
            WATER_HEIGHT,
            WATER_HEIGHT + 0.005 * VERTICAL_EXAGGERATION,
            h,
        );
        diffuse_color = mix(diffuse_color, SAND_COLOR, sand_mask);
        let wet_shore = smoothstep(0.05, 0.5, water_blend);
        diffuse_color = mix(diffuse_color, WATER_SHORE_COLOR, wet_shore * 0.45);
        diffuse_color *= 1.0 + breakup * 0.3;
        let tex = texture_noise(pos.xz);
        let tex_strength = mix(
            mix(0.10, 0.20, cliff_mask),
            0.05,
            snow_alt_mask * snow_slope_mask,
        );
        diffuse_color *= 1.0 + tex * tex_strength;

        if (field.alpha > 0.0) {
            let realm_col = realm_palette(field.realm_id);
            if (u.map_mode == MAP_MODE_POLITICAL) {
                let wilderness = vec3<f32>(0.55, 0.55, 0.50);
                diffuse_color = mix(wilderness, realm_col, field.alpha);
            } else {
                let safe_diffuse = clamp(
                    diffuse_color, vec3<f32>(0.0), vec3<f32>(1.0),
                );
                diffuse_color = mix(
                    diffuse_color,
                    soft_light(safe_diffuse, realm_col),
                    REALM_SHADE_STRENGTH * field.alpha,
                );
            }
        }
    } else { // M_WATER
        let shore_strength = clamp(1.0 - (water_blend - 0.5) * 2.0, 0.0, 1.0);
        diffuse_color = mix(WATER_COLOR, WATER_SHORE_COLOR, shore_strength);
        diffuse_color *= 1.0 + breakup * 0.15;
        smoothness = 0.95;
    }

    // Lighting. No shadow march in this path (precompute via a sun-
    // visibility bake later) and no recursive `get_reflection`. The
    // sky-color stand-in keeps water glints from going flat black.
    let sun = normalize(vec3<f32>(-1.0, 0.4, 0.05));
    let shadow: f32 = 1.0;
    var color = diffuse_color * sky_color(normal, sun) * fd_lambert();
    color *= occlusion;
    color += shade(
        diffuse_color, f0, smoothness, normal, -rd, sun, SUN_COLOR * shadow,
    );
    color += diffuse_color * SUN_COLOR
        * (dot(normal, sun * vec3<f32>(1.0, -1.0, 1.0)) * 0.5 + 0.5)
        * fd_lambert() / PI;
    let r_dir = reflect(rd, normal);
    color += sky_color(r_dir, sun) * 4.0
        * f_schlick(f0, dot(-rd, normal)) * 0.3;

    // Influence-field borders + hover wash (same as fs_main).
    const BORDER_CONTEST_BAND: f32 = 0.15;
    let edge = 1.0 - smoothstep(
        0.0, BORDER_CONTEST_BAND, field.contested - 1.0,
    );
    if (material == M_GROUND && field.alpha > 0.05) {
        let realm_col = realm_palette(field.realm_id);
        let tint = clamp(
            realm_col * BORDER_TINT_BRIGHTEN,
            vec3<f32>(0.0), vec3<f32>(1.0),
        );
        color = mix(color, tint, edge * BORDER_TINT_STRENGTH * field.alpha);
        if (u.hovered_pid != 0u && (field.realm_id + 1u) == u.hovered_pid) {
            color = mix(color, vec3<f32>(1.0), 0.10);
        }
        if (u.hovered_city != 0u && (field.city_idx + 1u) == u.hovered_city) {
            color = mix(color, vec3<f32>(1.0), 0.18);
        }
    }

    if (material == M_GROUND) {
        let marker = settlement_marker(pos.xz);
        color = mix(color, marker.rgb, marker.a);
    }

    // (No atmospheric fog in this path. The previous distance-fog
    // stand-in washed the entire map to white at EU4 camera scale
    // — default eye is 3000 km out, so every fragment sat at the fog
    // limit. A proper height-fog can be added later, but a missing fog
    // is much closer to correct than an over-applied one.)

    color = tonemap_aces(color);
    color = pow(color, vec3<f32>(1.0 / 2.2));
    return vec4<f32>(color, 1.0);
}
