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
// Province IDs (16-bit big-endian unsigned, packed into a Rg8Unorm texture
// with R = high byte, G = low byte). Categorical; sampled with textureLoad.
@group(0) @binding(8) var province_mask: texture_2d<f32>;
// Pre-baked border signed-distance-field. Pixel value (R, after unorm → [0,1])
// encodes `min(distance_to_nearest_border, BORDER_MAX_DIST_PX) / BORDER_MAX_DIST_PX`.
// Source: `script/gen-border-sdf` (Chaikin-smoothed NUTS-3 boundaries, then
// Euclidean distance transform). Bilinearly sampled in the shader.
@group(0) @binding(9) var border_sdf: texture_2d<f32>;

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
// Province lookup. Same world-anchored sampling pattern as the biome mask
// (textureLoad at integer pixel coords, no filtering), but the value is a
// 16-bit big-endian unsigned integer split across the R + G channels of an
// Rg8Unorm texture: `id = R * 256 + G` after rescaling unorm → 0…255.
//
// IDs are arbitrary opaque tokens; 0 is reserved for "no province" (water,
// outside the source shapefile, etc).
// ----------------------------------------------------------------------------
fn sample_province_id(world_xz: vec2<f32>) -> u32 {
    let uv = world_to_world_uv(world_xz);
    let dim = vec2<f32>(textureDimensions(province_mask));
    let dim_i = vec2<i32>(dim);
    let coord = clamp(
        vec2<i32>(uv * dim),
        vec2<i32>(0),
        dim_i - vec2<i32>(1),
    );
    let s = textureLoad(province_mask, coord, 0);
    let hi = u32(round(s.x * 255.0));
    let lo = u32(round(s.y * 255.0));
    return hi * 256u + lo;
}

// Cheap hash → distinct color per opaque integer ID. Saturated values would
// clash with snow / sun highlights, so we shift into a mid-bright range.
fn hash_color(id: u32) -> vec3<f32> {
    let f = f32(id);
    let r = fract(sin(f * 12.9898) * 43758.5453);
    let g = fract(sin(f * 78.233)  * 43758.5453);
    let b = fract(sin(f * 39.346)  * 43758.5453);
    return vec3<f32>(r, g, b) * 0.55 + 0.35;
}

fn province_color(id: u32) -> vec3<f32> {
    return hash_color(id);
}

// Faked "ownership" / realm assignment: integer-divide the province ID into
// chunks of `REALM_SIZE`. Province IDs are issued in source-shapefile order
// (FID + 1), so consecutive IDs aren't strictly geographically clustered but
// the result is randomised-feeling — each realm covers a handful of provinces.
//
// When real ownership data lands later, replace this with a province→owner
// LUT (e.g. an additional R8 texture indexed by pid) and keep callers using
// `realm_color(pid)` unchanged.
const REALM_SIZE: u32 = 7u;

fn realm_id(pid: u32) -> u32 {
    return pid / REALM_SIZE;
}

// Curated palette of 16 realm colours. Picked to be distinct across the
// hue wheel and saturated enough to read against grass / forest /
// mountain terrain without being garish. Indexed by
// `realm_id(pid) % REALM_PALETTE_SIZE`. When real ownership data lands,
// the palette can stay; only the realm → idx mapping changes.
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

fn realm_color(pid: u32) -> vec3<f32> {
    return realm_palette(realm_id(pid));
}

// Border SDF parameters. `BORDER_MAX_DIST_PX` must match `MAX_DIST_PX` in
// `script/gen-border-sdf` — the SDF stores `min(d, MAX_DIST_PX) / MAX_DIST_PX`,
// so multiplying by it recovers a distance in SDF-texture pixels.
//
// CK3-style border styling: each province paints the inward side of its
// own edges with a soft fade tinted by its realm colour. The fade is a
// *pulse* — zero on the border line itself, peaks ~PEAK_PX inside, then
// fades smoothly to zero at FADE_END_PX. Because the pulse goes back to
// zero at d = 0, two provinces sharing a border end up looking like *two*
// parallel coloured bands separated by a thin uncoloured line, instead of
// one bicoloured stripe. Provinces in the same realm share a colour, so
// within-realm borders disappear visually.
const BORDER_MAX_DIST_PX: f32     = 48.0;
const BORDER_PEAK_PX: f32         = 5.0;   // where the colour band peaks (px from edge)
const BORDER_FADE_END_PX: f32     = 28.0;  // where the fade returns to 0
const BORDER_TINT_STRENGTH: f32   = 0.65;  // peak mix amount at d = PEAK_PX
const BORDER_TINT_BRIGHTEN: f32   = 1.4;   // pre-mix brightness boost

// Strength of the soft-light tint applied across the *entire* province
// interior (in addition to the stronger border pulse near the edges).
// 0 = no region shading; 1 = full soft-light tint. Tune for taste.
const REALM_SHADE_STRENGTH: f32 = 0.55;

// Sample the border SDF and return the inward-fade pulse strength at a
// world position: 0 right on the border, peaks at BORDER_PEAK_PX inside,
// fades back to 0 at BORDER_FADE_END_PX. Tri-points stay sharp because
// the underlying SDF kinks (is non-differentiable) where multiple borders
// meet.
fn sample_border_fade(world_xz: vec2<f32>) -> f32 {
    let uv = world_to_world_uv(world_xz);
    let d_norm = textureSampleLevel(border_sdf, samp, uv, 0.0).x;
    let d_px = d_norm * BORDER_MAX_DIST_PX;
    let rise = smoothstep(0.0, BORDER_PEAK_PX, d_px);
    let fall = 1.0 - smoothstep(BORDER_PEAK_PX, BORDER_FADE_END_PX, d_px);
    return rise * fall;
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

// height in .x, normal in .yzw (Y-up world space).
fn map_full(uv: vec2<f32>) -> vec4<f32> {
    let height = map_height(uv);
    let pixel = 1.0 / vec2<f32>(LAYER_SIZE);
    let uv1 = uv + vec2<f32>(pixel.x, 0.0);
    let uv2 = uv + vec2<f32>(0.0, pixel.y);
    let h1 = map_height(uv1);
    let h2 = map_height(uv2);
    // Convert the uv steps into world XZ steps (km per layer texel) so the
    // gradient is dimensionally consistent with the (already-VE-scaled)
    // height. Without this, v1 / v2 would mix tiny uv units (~1e-3) with
    // height units that can be ≫ 1, and the resulting normal would be
    // almost horizontal regardless of the actual slope.
    let world_step = (layer.covered_max - layer.covered_min) * pixel;
    let dhdx = (h1 - height) / world_step.x;
    let dhdz = (h2 - height) / world_step.y;
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

            // 6. Realm shading. Two-stage tinting depending on map mode:
            //    * Terrain mode: soft-light blend at the diffuse stage, so
            //      the entire province interior gets a gentle realm-colour
            //      wash that follows the lighting (dimmer in shadow,
            //      brighter in sun) and preserves terrain tonal detail
            //      (highlights stay light, shadows stay dark). The
            //      stronger coloured border pulse is added later, post-
            //      lighting.
            //    * Political mode: full opaque overwrite — flat realm fill
            //      with a crisp shape, snow/cliff/sand discarded.
            let pid_for_shade = sample_province_id(pos.xz);
            if (pid_for_shade != 0u) {
                if (u.map_mode == MAP_MODE_POLITICAL) {
                    diffuse_color = realm_color(pid_for_shade);
                } else {
                    let safe_diffuse = clamp(
                        diffuse_color, vec3<f32>(0.0), vec3<f32>(1.0),
                    );
                    diffuse_color = mix(
                        diffuse_color,
                        soft_light(safe_diffuse, realm_color(pid_for_shade)),
                        REALM_SHADE_STRENGTH,
                    );
                }
            }
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

        // Shadow ray (skip for the inside-of-strata case).
        var shadow: f32 = 1.0;
        if (material != M_STRATA) {
            let sh = march(pos + vec3<f32>(0.0, 1.0, 0.0) * 1e-4, sun);
            shadow = 1.0 - exp(-sh.s_t * 20.0);
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

        // Province borders — applied to the *fully lit* color so their
        // strength is independent of local lighting / shadow. Soft colored
        // pulse: the local province's realm colour peaks ~5 px inside the
        // edge and fades smoothly to 0 deeper into the province. The pulse
        // also goes to 0 right at the edge itself, so neighbouring
        // provinces from different realms show as *two* parallel coloured
        // bands separated by a thin uncoloured line. Within-realm
        // boundaries vanish (both sides tint with the same realm colour
        // and there's no contrast).
        //
        // Mouse-hover highlight is also stitched in here so we only sample
        // the province ID once per fragment.
        if (material == M_GROUND) {
            let pid = sample_province_id(pos.xz);
            if (pid != 0u) {
                let fade = sample_border_fade(pos.xz);
                let tint = clamp(
                    realm_color(pid) * BORDER_TINT_BRIGHTEN,
                    vec3<f32>(0.0),
                    vec3<f32>(1.0),
                );
                color = mix(color, tint, fade * BORDER_TINT_STRENGTH);

                // Hover highlight: noticeable brightening + slight white
                // wash on the entire hovered province. Only when something
                // is actually being hovered (pid 0 is reserved for "no
                // province" — sea, oob, etc.).
                if (u.hovered_pid != 0u && pid == u.hovered_pid) {
                    color = mix(color, vec3<f32>(1.0), 0.30);
                }
            }
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
