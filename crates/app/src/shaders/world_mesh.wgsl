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
@group(0) @binding(9)  var                   bathymetry:      texture_2d<f32>;

// Water SDF decode — mirrors `WATER_SDF_RANGE_KM` in `tile_bake.wgsl`
// and `SDF_RANGE_KM` in `script/gen-water-sdf`. The R8 byte maps
// linearly to the band [-RANGE, +RANGE]: byte=0 = deepest sea, byte=255
// = deepest inland, byte=128 ≈ the coast.
const WATER_SDF_RANGE_KM: f32 = 8.0;

// Bathymetry decode — mirrors `MAX_DEPTH_M` in `script/gen-bathymetry`.
// The R8 byte maps linearly to [0, MAX_DEPTH_M]: byte=0 = surface or
// land, byte=255 = at or past MAX_DEPTH_M below sea level.
const BATHY_MAX_DEPTH_M: f32 = 6000.0;

fn sample_water_dist_km(uv: vec2<f32>) -> f32 {
    let byte = textureSampleLevel(water_sdf, samp, uv, 0.0).r;
    return byte * (2.0 * WATER_SDF_RANGE_KM) - WATER_SDF_RANGE_KM;
}

fn sample_bathymetry_m(uv: vec2<f32>) -> f32 {
    let byte = textureSampleLevel(bathymetry, samp, uv, 0.0).r;
    return byte * BATHY_MAX_DEPTH_M;
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

// Tent-shaped band centred at `peak_km` (negative = offshore) with
// half-width `half_width_km`. Returns 0 outside the band, 1 at the
// peak, smoothstepped in between. Handy for foam rings, surf bands,
// shelf stripes, etc.
fn coast_band(dist_km: f32, peak_km: f32, half_width_km: f32) -> f32 {
    let inner = smoothstep(peak_km - half_width_km, peak_km, dist_km);
    let outer = 1.0 - smoothstep(peak_km, peak_km + half_width_km, dist_km);
    return inner * outer;
}

// ---- Water surface: TDM Seascape adapted for top-down map view ------------
//
// Faithful adaptation of Alexander Alekseev's "Seascape" (Shadertoy
// Ms2SD1, 2014) — the same `sea_octave` heightfield, color model and
// Fresnel/specular formulas, scaled to operate on world coordinates
// in km. We can't reproduce TDM's silhouetted 3D wave forms (those
// come from oblique-camera raymarching of the heightfield, which we
// don't do) but we can reproduce its *surface* look: sharp choppy
// crests, height-modulated water color, sky reflection, sun glint.
//
// Key TDM mechanics preserved:
//   * `sea_octave`: domain-warped `(1 - |sin|) · |cos|` raised to
//     `choppy`, giving trochoidal sharp-crest profile.
//   * 5 octaves stacked through the rotate-and-scale
//     `mat2(1.6, 1.2, -1.2, 1.6)`, frequency ×1.9, amplitude ×0.22
//     per octave — the standard FBM ratios from GPU Gems.
//   * Two phase-offset taps per octave (`uv + SEA_TIME` and
//     `uv - SEA_TIME`) so the interference pattern between them
//     moves over time — this is what gives TDM's "wave fronts
//     propagating" look. Critical: when time = 0 the two taps
//     collapse and you get a degenerate static lattice. Animation
//     must be on.
//   * Color formula `mix(refracted, reflected, fresnel) +
//     SEA_WATER_COLOR · (h - SEA_HEIGHT) · 0.18` — the height-mod
//     term is the single most important visual element. It
//     brightens crests and darkens troughs *directly from height*,
//     bypassing the normal/light geometry. This is what makes the
//     surface read as 3D from a top-down view.
//
// Tuning:
//   * SEA_FREQ_PER_KM = 0.10 makes the dominant octave wavelength
//     2π/0.10 ≈ 63 km, fining down to 63/1.9^4 ≈ 4.8 km on the
//     5th octave. Matches the visible ocean scale at typical map
//     zoom levels.
//   * SEA_HEIGHT_KM = 0.6: amplitude such that h ranges ~±1.5 km
//     when octaves stack — well below mountains so heightmap
//     displacement still wins, but enough to brighten/darken
//     visibly via the height-mod term.
//   * SEA_CHOPPY = 4.0: TDM default. Higher = sharper crests,
//     flatter troughs.

// Sun direction (world space). Y dominates so the sun is high in
// the sky (~75° elevation) — important for top-down map views.
//
// A low sun (small Y) concentrates the glint into a narrow strip of
// the screen because the half-vector `normalize(sun + view)` only
// aligns with mostly-up wave normals where the camera's `view`
// vector happens to compensate for the sun's off-axis position.
// With Y ≈ 0.45 the glint piles up at whichever screen edge the
// camera pan happens to put on the right side of the half-vector.
// Y ≈ 0.92 means the half-vector is nearly straight up across the
// whole frame, so any sufficiently-tilted wave fires specular,
// distributing the glint over the whole visible ocean.
//
// X and Z give it a *little* directionality so reflections aren't
// perfectly axisymmetric — useful for the sky_color sun-glow term,
// which needs a definite azimuth.
// Sun direction (world space). Camera looks NORTH (+Z); for the sun
// glint to appear in the UPPER-LEFT of the screen the sun needs:
//   * -X (west)  — so half-vector tilts west on flat water
//   * +Z (north) — so half-vector tilts toward the far side of frame
//   * Y of ~0.7  — high enough to be a clean lobe, low enough to be
//                  localized (not the everywhere-glint of Y=0.92).
const WATER_SUN_DIR: vec3<f32> = vec3<f32>(-0.30, 0.65, 0.70);

const TAU: f32 = 6.283185307;

const SEA_HEIGHT_KM:    f32 = 0.14;
const SEA_CHOPPY:       f32 = 1.6;
const SEA_SPEED:        f32 = 0.5;
// Bulk wave-field translation. The two-tap interference inside each
// `sea_octave` call evolves wave crests *in place*; this constant
// translates the whole pattern in a fixed direction so the waves
// also visibly *propagate* across the surface. Units: km/s.
// 3 km/s is roughly one screen-width per minute at a typical regional
// zoom — fast enough to read as motion, slow enough not to look like
// the screen is sliding.
const WAVE_FLOW_SPEED_KM_PER_S: f32 = 3.0;
// Direction the wave field advects. Arbitrary; chosen so it isn't
// axis-aligned (otherwise the propagation reads as a horizontal
// scroll). 0.8/-0.6 ≈ from the NE.
const WAVE_FLOW_DIR: vec2<f32> = vec2<f32>(0.8, -0.6);
// Start frequency for the dominant (lowest-frequency) octave. With
// the 7-octave stack and ×1.9 per-octave frequency scaling, this
// gives wavelengths from ~300 km down to ~6 km. The big end
// covers "continent-scale ocean swell" visible at zoom-out; the
// small end is the wave-crest texture visible at close zoom. Having
// the full range means there's always *some* scale of variation
// matched to whatever zoom you're looking at — no more single-scale
// tile pattern when you pull back.
const SEA_FREQ_PER_KM:  f32 = 0.0105;  // 2π / 600 km — longer dominant
                                       // wavelength so the macro octave
                                       // doesn't read as repeating tiles.
// Base color lifted from TDM's (0, 0.09, 0.18) — TDM's dark base
// works because their oblique view has Fresnel ≈ 0.3–0.5 so most
// of the visible color is sky reflection. Our top-down Fresnel is
// tiny (≈ 0.02–0.05), so the base dominates and needs to be a
// usable ocean color on its own.
// Lighter, more desaturated mid-blue base — matches the CK3 reference
// look: water as a clean mid-blue surface, not a dark navy.
const SEA_BASE_COLOR:   vec3<f32> = vec3<f32>(0.28, 0.42, 0.55);
// Crest-tint color. Much closer to neutral white-blue than TDM's
// yellow-green; we want crests to read as foam/whitewater, not as
// a saturated color shift.
const SEA_WATER_COLOR:  vec3<f32> = vec3<f32>(0.70, 0.80, 0.85);

// Global time-scale on wave phase. Multiplied into TDM's SEA_TIME.
// Set 0 to freeze the surface; bump to taste. Animation is essential
// for the look — the two-tap interference inside each octave
// collapses to a degenerate static lattice at time = 0.
const WAVE_TIME_SCALE: f32 = 1.0;

// SEA_TIME equivalent. TDM's `1.0 + iTime * SEA_SPEED`; the +1 gives
// non-zero initial phase so first-frame doesn't look identical to a
// time=0 frozen pattern.
fn sea_time() -> f32 {
    return 1.0 + camera.i_time * SEA_SPEED * WAVE_TIME_SCALE;
}

// Global wave-strength fade. Now that the FBM stack uses per-octave
// aliasing fade (see `wave_height`), each individual octave drops
// out gracefully as it becomes too small to resolve. So this can be
// kept on at all zooms — the macro octaves (300 km wavelength) stay
// visible at world view, the micro octaves automatically fade.
//
// Returning 1.0 unconditionally; kept as a function so callers don't
// need to change, and so we can re-introduce a global gate later if
// we want to disable waves entirely at extreme zooms.
fn wave_strength(pixel_world_km: f32) -> f32 {
    return 1.0;
}

// Signed value noise in [-1, +1]. TDM uses `-1.0 + 2.0 * mix(...)`;
// we already have `vnoise` returning [0, 1] so just scale-shift.
fn vnoise_signed(p: vec2<f32>) -> f32 {
    return -1.0 + 2.0 * vnoise(p);
}

// TDM `sea_octave` primitive — produces a sharp-peak / wide-trough
// scalar heightfield in [0, 1] from one warped sin/cos. The scalar
// warp `uv += noise(uv)` works here because the rotate-and-scale
// matrix between octaves takes care of de-aligning each octave's
// crest direction — we just have to actually use it (5 octaves) and
// keep the animation on.
fn sea_octave(uv_in: vec2<f32>, choppy: f32) -> f32 {
    let n  = vnoise_signed(uv_in);
    let uv = uv_in + vec2<f32>(n, n);
    let wv  = vec2<f32>(1.0, 1.0) - abs(sin(uv));
    let swv = abs(cos(uv));
    let bl  = mix(wv, swv, wv);
    return pow(1.0 - pow(bl.x * bl.y, 0.65), choppy);
}

// Stacked wave heightfield in km. 7 octaves, FBM-style, spanning
// ~300 km down to ~6 km wavelength. Returns the signed height
// relative to mean sea level (0 km).
//
// Per-octave amplitude fade based on `pixel_world_km` prevents the
// fine octaves from aliasing at coarse zooms while keeping the macro
// octaves visible — so the wave field always has *some* visible
// scale appropriate to the current zoom.
//
// The fade kicks in when the octave's wavelength gets close to a
// few pixels: an octave contributes fully when there are >= 8
// pixels per wave, fades out as that drops, gone below 2 px/wave.
fn wave_height(world_xz: vec2<f32>, pixel_world_km: f32) -> f32 {
    let m = mat2x2<f32>(1.6, 1.2, -1.2, 1.6);
    // Per-octave large random offsets break the FBM's coherent
    // self-similarity — without them the same rotate-and-scale
    // matrix applied to every octave produces a visible spiral
    // marble pattern as you zoom out. With them, each octave's
    // phase is decorrelated from its neighbours, so the stack
    // reads as non-periodic noise.
    var octave_offsets = array<vec2<f32>, 7>(
        vec2<f32>(  0.0,   0.0),
        vec2<f32>(173.1, -89.7),
        vec2<f32>(-47.3, 211.5),
        vec2<f32>(312.7,  61.9),
        vec2<f32>( -201.5, -154.3),
        vec2<f32>(  88.9, 297.1),
        vec2<f32>(-265.4,  -33.7),
    );
    // Bulk translation: shift the whole wave field along WAVE_FLOW_DIR
    // over time. This is what makes waves visibly *propagate* (the
    // per-octave two-tap interference below makes them *evolve* in
    // place, but you can't see actual motion without a directional
    // translation on top).
    var uv = world_xz
        + WAVE_FLOW_DIR * (camera.i_time * WAVE_FLOW_SPEED_KM_PER_S);
    uv.x = uv.x * 0.75; // TDM's directional stretch
    var freq:   f32 = SEA_FREQ_PER_KM;
    var amp:    f32 = SEA_HEIGHT_KM;
    var choppy: f32 = SEA_CHOPPY;
    let t = sea_time();
    var h: f32 = 0.0;
    for (var i: i32 = 0; i < 7; i = i + 1) {
        // Per-octave aliasing fade. wavelength_km = TAU / freq.
        // We want full strength when pixel_world_km <= wavelength/8,
        // zero when pixel_world_km >= wavelength/2.
        let wavelength_km = TAU / freq;
        let px_per_wave   = wavelength_km / pixel_world_km;
        let octave_str    = smoothstep(2.0, 8.0, px_per_wave);
        let p = uv + octave_offsets[i];
        var d = sea_octave((p + vec2<f32>(t)) * freq, choppy);
        d   = d + sea_octave((p - vec2<f32>(t)) * freq, choppy);
        h   = h + d * amp * octave_str;
        uv     = m * uv;
        freq   = freq * 1.9;
        // Steeper amp falloff (0.45 vs TDM's 0.22) — still gentler
        // than vanilla TDM so mid-band octaves contribute, but
        // sharper than 0.55 so the high-frequency "scaly" detail
        // doesn't dominate at very-close zoom. The aliasing fade
        // already handles the *too-fine* case; this handles the
        // *too-visible-when-resolved* case.
        amp    = amp * 0.45;
        choppy = mix(choppy, 1.0, 0.2);
    }
    return h;
}

// Surface normal from central differences on `wave_height`. Epsilon
// scales with `pixel_world_km` so the slope sample matches the
// screen-pixel scale — sub-pixel wave detail gets averaged out
// instead of aliasing.
fn wave_normal(world_xz: vec2<f32>, pixel_world_km: f32, strength: f32) -> vec3<f32> {
    let eps = max(0.4, pixel_world_km * 2.0);
    let h0 = wave_height(world_xz,                          pixel_world_km);
    let hx = wave_height(world_xz + vec2<f32>(eps, 0.0),    pixel_world_km);
    let hz = wave_height(world_xz + vec2<f32>(0.0, eps),    pixel_world_km);
    let dx = (hx - h0) / eps * strength;
    let dz = (hz - h0) / eps * strength;
    return normalize(vec3<f32>(-dx, 1.0, -dz));
}

// Synthetic sky lookup. TDM's `getSkyColor` mapped to our color
// vibe — zenith blue, brighter horizon, mild sun glow.
fn sky_color(refl: vec3<f32>) -> vec3<f32> {
    var e = refl;
    e.y = (max(e.y, 0.0) * 0.8 + 0.2) * 0.8;
    var col = vec3<f32>(
        pow(1.0 - e.y, 2.0),
        1.0 - e.y,
        0.6 + (1.0 - e.y) * 0.4,
    ) * 1.1;
    // Soft, broad sun glow on top — widens the bright reflection
    // patch beyond the tight specular lobe.
    let sun_dot = max(dot(normalize(refl), normalize(WATER_SUN_DIR)), 0.0);
    col = col + vec3<f32>(1.0, 0.95, 0.82) * pow(sun_dot, 8.0) * 0.4;
    return col;
}

// TDM's diffuse: `pow(dot(n, l) * 0.4 + 0.6, p)`. The 0.4/0.6 split
// keeps it bright everywhere; the power sharpens it.
fn water_diffuse(n: vec3<f32>, l: vec3<f32>, p: f32) -> f32 {
    return pow(max(dot(n, l) * 0.4 + 0.6, 0.0), p);
}

// TDM's specular with built-in normalisation factor.
fn water_specular_tdm(n: vec3<f32>, l: vec3<f32>, e: vec3<f32>, s: f32) -> f32 {
    let nrm = (s + 8.0) / (TAU / 2.0 * 8.0); // (s+8) / (PI*8)
    return pow(max(dot(reflect(-e, n), l), 0.0), s) * nrm;
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

// ---- vertex shaders --------------------------------------------------------
//
// Two entry points, two pipelines:
//
//   * `vs_water` — a 6-vert (2-triangle) flat quad covering the world disc
//     at y=0. The water pass uses this. The water surface is structurally
//     decoupled from the heightmap so coastal-cliff mesh triangles can't
//     drag the rendered water up into the air anymore.
//
//   * `vs_main` — the heightmap-displaced land mesh (`MESH_GRID²` grid,
//     6 verts per cell). The land pass uses this. Same vertex math as
//     before; only the fragment side changed (now outputs alpha for
//     screen-pixel-accurate coast AA against the water plane underneath).

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

// Water plane vertex shader. Six verts forming two triangles — a single
// quad covering the whole world disc in XZ at y=0. Frags interpolate
// world_xz so the fragment shader can compute waves / depth color /
// foam exactly as before, just without inheriting any height from the
// heightmap.
@vertex
fn vs_water(@builtin(vertex_index) vid: u32) -> VsOut {
    var quad = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );
    let xy = quad[vid] * WORLD_HALF_KM;
    let world_x = xy.x;
    let world_z = xy.y;
    let clip = project(vec3<f32>(world_x, 0.0, world_z));
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

// ---- Procedural ground ----------------------------------------------------
//
// Replaces the baked atlas terrain colour (which was a flat per-biome
// palette × baked hillshade) with a procedural ground colour evaluated
// per fragment in world space.
//
// Two halves:
//   * `terrain_light` — the lambert hillshade. Was baked into the atlas
//     by `tile_bake.wgsl`; recomputed here per-frame from the heightmap
//     so it can multiply whatever procedural colour we choose.
//   * `terrain_color_proc` — the ground colour. Iter 1 is just a flat
//     green to validate the pipeline before adding noise-driven variation.

fn terrain_light(world_xz: vec2<f32>) -> f32 {
    // Central-difference normal estimate. eps in km; same recipe as
    // tile_bake.wgsl, just operating on the world-mesh's `sample_height_km`
    // (which already returns km). Slope is dimensionless either way.
    let eps_km = 1.0;
    let h0 = sample_height_km(world_to_world_uv(world_xz));
    let hx = sample_height_km(world_to_world_uv(world_xz + vec2<f32>(eps_km, 0.0)));
    let hz = sample_height_km(world_to_world_uv(world_xz + vec2<f32>(0.0, eps_km)));
    let dx = (hx - h0) / eps_km;
    let dz = (hz - h0) / eps_km;
    let n  = normalize(vec3<f32>(-dx, 1.0, dz));

    // Same sun direction tile_bake.wgsl used — keeps the lighting
    // continuous if we ever fall back to atlas sampling.
    let sun     = normalize(vec3<f32>(0.4, 0.85, -0.35));
    let lambert = max(dot(n, sun), 0.0);
    return 0.45 + 0.55 * lambert;
}

// 6-octave FBM in world km with per-octave aliasing fade. Returns
// roughly [0, 1]. Each octave fades out smoothly as its wavelength
// approaches the screen-pixel scale — same recipe `wave_height` uses,
// preventing the high-frequency octaves from devolving into noise at
// continent-scale zoom.
//
// Wavelengths chosen to land at "meaningful" geographic scales,
// extending well below 1 km so zoom-in reveals progressively finer
// detail until you're seeing field-scale variation:
//   * macro (~120 km): continent / large region. Always visible.
//   * meso  (~25 km):  country-side variation, valley vs ridge.
//   * micro (~5 km):   forest patches / agricultural belts.
//   * fine  (~1 km):   village / small-field scale.
//   * vfine (~0.2 km): individual field / forest stand.
//   * grain (~0.04 km ≈ 40 m): ground texture at extreme close zoom.
//
// Amplitude ratios use a *gentle* FBM falloff (×0.75 per octave) so
// the fine octaves still contribute visibly when resolved — the
// conventional ×0.5 makes the smallest octave contribute only ~1.6%
// of the total signal, which is invisible. Normalised so the sum
// stays in [0, 1] regardless of how many octaves are currently active.
//
// Aliasing fade uses smoothstep(1.5, 4.0, px_per_wave) — looser than
// the wave_height shader's 2-8 because we *want* the fine octaves
// to fight to be seen at close zoom. Risk: very slight aliasing on
// the smallest octave at borderline zoom levels, which is acceptable
// for ground colour (a few pixel-scale noise dots) but would be
// visible on the wave surface (specular highlights amplify it).
// The perf cost of all 6 octaves is one vnoise per fragment per
// octave — cheap (~5 hash + lerp ops each).
fn ground_fbm_octave(p: vec2<f32>, wavelength_km: f32, pixel_world_km: f32) -> f32 {
    let px_per_wave = wavelength_km / pixel_world_km;
    let str = smoothstep(1.5, 4.0, px_per_wave);
    return vnoise(p / wavelength_km) * str;
}

fn ground_fbm(world_xz: vec2<f32>, pixel_world_km: f32) -> f32 {
    // Plain FBM, no domain warping — we want splotchy regions, not
    // swirled brush strokes. Per-octave large random offsets break the
    // self-similarity of plain FBM so spirals/marbling don't appear.
    let n0 = ground_fbm_octave(world_xz + vec2<f32>(  0.0,     0.0),   120.0,  pixel_world_km);
    let n1 = ground_fbm_octave(world_xz + vec2<f32>(173.1,   -89.7),    25.0,  pixel_world_km);
    let n2 = ground_fbm_octave(world_xz + vec2<f32>(-47.3,   211.5),     5.0,  pixel_world_km);
    let n3 = ground_fbm_octave(world_xz + vec2<f32>(312.7,    61.9),     1.0,  pixel_world_km);
    let n4 = ground_fbm_octave(world_xz + vec2<f32>(-201.5, -154.3),     0.2,  pixel_world_km);
    let n5 = ground_fbm_octave(world_xz + vec2<f32>(  88.9,  297.1),     0.04, pixel_world_km);
    // Gentle 0.75 falloff: amplitudes 1.0, 0.75, 0.56, 0.42, 0.32, 0.24.
    // Normalised across the geometric sum so result ∈ [0, 1].
    let total = 1.0 + 0.75 + 0.5625 + 0.421875 + 0.31640625 + 0.2373046875;
    return (1.0 * n0 + 0.75 * n1 + 0.5625 * n2
          + 0.421875 * n3 + 0.31640625 * n4 + 0.2373046875 * n5) / total;
}

fn terrain_color_proc(world_xz: vec2<f32>, pixel_world_km: f32) -> vec3<f32> {
    // Three-stop noise-driven palette for the lowland "grass" tones,
    // then two height-driven over-mixes for elevation: rocky brown
    // at mid-altitude (mountain slopes) and snow at high altitude
    // (peaks).
    //
    // Noise-driven base (splotchy regional grass variation):
    //   t ∈ [0.0, 0.5]:  forest → grass
    //   t ∈ [0.5, 1.0]:  grass  → dry
    let t = clamp(ground_fbm(world_xz, pixel_world_km), 0.0, 1.0);
    // Slightly brighter lowland palette so the splotches read clearly
    // without the whole map feeling shaded.
    let forest = vec3<f32>(0.22, 0.34, 0.18);
    let grass  = vec3<f32>(0.34, 0.48, 0.22);
    let dry    = vec3<f32>(0.52, 0.52, 0.30);
    let low_t  = smoothstep(0.0, 0.5, t);
    let high_t = smoothstep(0.5, 1.0, t);
    var c = forest;
    c = mix(c, grass, low_t);
    c = mix(c, dry,   high_t);

    // Height-driven over-mix. Foothills stay green longer (rock band
    // pushed to 1.3–2.4 km) and snow only caps the tallest peaks
    // (start 2.6 km, full at 3.4 km). Rock tone is a touch warmer so
    // mountain slopes feel sun-lit rather than slate-grey.
    let h_km = sample_height_km(world_to_world_uv(world_xz));
    let rock = vec3<f32>(0.48, 0.42, 0.32); // warm sandstone-ish stone
    let snow = vec3<f32>(0.95, 0.95, 0.95);
    let rock_t = smoothstep(1.3, 2.4, h_km);
    let snow_t = smoothstep(2.6, 3.4, h_km);
    c = mix(c, rock, rock_t);
    c = mix(c, snow, snow_t);
    return c;
}

// ---- Fragment shaders ------------------------------------------------------
//
// fs_main = LAND. Procedural ground + beaches + shoreline alpha.
// fs_water = WATER. TDM seascape + depth ramp + foam.
//
// Both share the SDF + helper functions above. The land pass renders
// on top of the water pass with alpha blending so the coast AA happens
// via the shoreline alpha falloff rather than an inline land/water mix.

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let uv = world_to_world_uv(in.world_xz);

    // Warped UV for the SDF lookup — same sub-source coastline wiggle
    // the water pass uses, so the visible coast lines up.
    let warped_uv = world_to_world_uv(warp_world_xz(in.world_xz));

    // LoD selection (kept for the debug overlay; the procedural ground
    // doesn't actually consume `lod` since the atlas is no longer the
    // source of ground colour).
    let pixel_world_km = max(fwidth(in.world_xz.x), fwidth(in.world_xz.y));
    var lod: i32 = 0;
    if (pixel_world_km < sqrt(LOD_KM_PER_TEXEL_0 * LOD_KM_PER_TEXEL_1)) { lod = 1; }
    if (pixel_world_km < sqrt(LOD_KM_PER_TEXEL_1 * LOD_KM_PER_TEXEL_2)) { lod = 2; }
    if (pixel_world_km < sqrt(LOD_KM_PER_TEXEL_2 * LOD_KM_PER_TEXEL_3)) { lod = 3; }

    // M-key debug overlay: per-LoD tint over the land mesh. Water still
    // renders normally underneath.
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

    // Coast distance. CRITICAL: we use the UNWARPED uv here. The warp
    // adds up to ~1 km of UV displacement, which at close zoom is a
    // sub-source-texel wiggle (nice for the visible coast). At world
    // zoom one screen pixel covers several km, and that 1 km warp can
    // shift land fragments to sample water (rivers, lakes, narrow
    // inland features that the SDF records as local minima). Without
    // this fix, large swathes of inland land discard incorrectly at
    // world zoom and you see the water plane through them.
    //
    // Discard policy: HARD discard at the SDF zero-crossing. Any soft
    // alpha-blended AA band ends up tangling with the coastal-cliff
    // mesh triangles at zoomed-out views, so we accept slightly
    // aliased coasts at extreme close zoom in exchange for clean
    // rendering everywhere else.
    let dist_km = sample_water_dist_km(uv);
    if (dist_km < 0.0) {
        discard;
    }

    // Procedural ground colour. The atlas binding is kept (bind-group
    // layout still references it) but no longer the source of colour.
    var color = terrain_color_proc(in.world_xz, pixel_world_km) * terrain_light(in.world_xz);

    // ---- Coast geometry for beach band ---------------------------------
    //
    // The beach band gates need the same SDF gradient / coast_point /
    // big_coast probe that the water pass's foam uses. Duplicated here
    // rather than passed via varyings because the two passes can't
    // share fragment data.
    let eps_km = 2.0;
    let d_xp = sample_water_dist_km(world_to_world_uv(in.world_xz + vec2<f32>( eps_km, 0.0)));
    let d_xm = sample_water_dist_km(world_to_world_uv(in.world_xz + vec2<f32>(-eps_km, 0.0)));
    let d_zp = sample_water_dist_km(world_to_world_uv(in.world_xz + vec2<f32>(0.0,  eps_km)));
    let d_zm = sample_water_dist_km(world_to_world_uv(in.world_xz + vec2<f32>(0.0, -eps_km)));
    let grad_x = (d_xp - d_xm) / (2.0 * eps_km);
    let grad_z = (d_zp - d_zm) / (2.0 * eps_km);
    let grad_len = max(length(vec2<f32>(grad_x, grad_z)), 1e-6);
    let coast_normal = vec2<f32>(grad_x, grad_z) / grad_len;
    let coast_point = in.world_xz + (-dist_km) * coast_normal;
    let coast_size_probe_uv =
        world_to_world_uv(coast_point - coast_normal * 6.0);
    let coast_size_dist = sample_water_dist_km(coast_size_probe_uv);
    let big_coast = smoothstep(-1.5, -4.0, coast_size_dist);

    // ---- Beaches -------------------------------------------------------
    //
    // Same gates as before; see the original land/water-combined shader
    // for the full rationale. Sand mixed into the land colour wherever
    // the gates align.
    let BEACH_BAND_KM: f32 = 0.6;
    let land_dist_km = max(0.0, dist_km);
    let beach_proximity = smoothstep(BEACH_BAND_KM, 0.0, land_dist_km);

    let beach_run_lo = vnoise(coast_point / 12.0);
    let beach_run_hi = vnoise(coast_point / 1.5 + vec2<f32>(17.1, 23.3));
    let beach_seed = smoothstep(0.40, 0.70,
        beach_run_lo * 0.6 + beach_run_hi * 0.4);

    let slope_eps_km = 0.5;
    let h_xp = sample_height_km(world_to_world_uv(in.world_xz + vec2<f32>( slope_eps_km, 0.0)));
    let h_xm = sample_height_km(world_to_world_uv(in.world_xz + vec2<f32>(-slope_eps_km, 0.0)));
    let h_zp = sample_height_km(world_to_world_uv(in.world_xz + vec2<f32>(0.0,  slope_eps_km)));
    let h_zm = sample_height_km(world_to_world_uv(in.world_xz + vec2<f32>(0.0, -slope_eps_km)));
    let beach_slope =
        length(vec2<f32>(h_xp - h_xm, h_zp - h_zm)) / (2.0 * slope_eps_km);
    let flat_gate = 1.0 - smoothstep(0.06, 0.20, beach_slope);

    let beach_zoom_fade = smoothstep(0.80, 0.20, pixel_world_km);

    let beach =
        beach_proximity * beach_seed * flat_gate * big_coast * beach_zoom_fade;
    let sand = vec3<f32>(0.92, 0.84, 0.62);
    color = mix(color, sand, beach * 0.90);

    // Solid land. The hard discard above means we never run on water
    // fragments; coast AA will come back via a tighter mechanism in a
    // follow-up iteration.
    return vec4<f32>(color, 1.0);
}

// Water plane fragment shader. All the TDM seascape + depth ramp +
// foam logic that used to live in `fs_main` lives here now; opaque
// everywhere (alpha=1), so the land pass on top drives the coast AA.
@fragment
fn fs_water(in: VsOut) -> @location(0) vec4<f32> {
    let uv = world_to_world_uv(in.world_xz);
    let warped_uv = world_to_world_uv(warp_world_xz(in.world_xz));

    let pixel_world_km = max(fwidth(in.world_xz.x), fwidth(in.world_xz.y));
    let dist_km = sample_water_dist_km(warped_uv);

    // ---- TDM Seascape shading ---------------------------------------
    let coast_calm = smoothstep(-1.0, -6.0, dist_km);
    let wave_str = wave_strength(pixel_world_km);
    let wave_h_raw = wave_height(in.world_xz, pixel_world_km);
    let wave_h = wave_h_raw * coast_calm;
    let normal = wave_normal(in.world_xz, pixel_world_km, wave_str);

    let cb_water = cam_basis();
    let frag_pos = vec3<f32>(in.world_xz.x, 0.0, in.world_xz.y);
    let view_dir = normalize(cb_water.eye - frag_pos);
    let sun = normalize(WATER_SUN_DIR);

    let fresnel_raw = clamp(1.0 - dot(normal, view_dir), 0.0, 1.0);
    let fresnel = min(fresnel_raw * fresnel_raw * fresnel_raw, 0.5);
    let reflected = sky_color(reflect(-view_dir, normal));
    let refracted = SEA_BASE_COLOR
                    + water_diffuse(normal, sun, 80.0) * SEA_WATER_COLOR * 0.12;

    var water_color = mix(refracted, reflected, fresnel);
    water_color = water_color
                  + SEA_WATER_COLOR * (wave_h - SEA_HEIGHT_KM) * 0.04 * wave_str;
    let spec = water_specular_tdm(normal, sun, view_dir, 50.0) * wave_str * 0.25;
    water_color = water_color + vec3<f32>(spec);

    // ---- Depth-driven colour ramp -----------------------------------
    let offshore_km_for_color = max(0.0, -dist_km);
    let bathy_m = sample_bathymetry_m(uv);
    let shallow_c = vec3<f32>(0.42, 0.72, 0.70);
    let mid_c     = vec3<f32>(0.24, 0.46, 0.58);
    let deep_c    = vec3<f32>(0.06, 0.16, 0.34);
    let t_low  = smoothstep(0.0, 5.0, offshore_km_for_color);
    let t_high = smoothstep(200.0, 2500.0, bathy_m);
    var depth_color = shallow_c;
    depth_color = mix(depth_color, mid_c,  t_low);
    depth_color = mix(depth_color, deep_c, t_high);
    water_color = mix(water_color, depth_color, 0.75);

    let foam_color = vec3<f32>(0.78, 0.86, 0.90);

    // ---- SDF derivatives + coast geometry --------------------------
    let eps_km = 2.0;
    let d_xp = sample_water_dist_km(world_to_world_uv(in.world_xz + vec2<f32>( eps_km, 0.0)));
    let d_xm = sample_water_dist_km(world_to_world_uv(in.world_xz + vec2<f32>(-eps_km, 0.0)));
    let d_zp = sample_water_dist_km(world_to_world_uv(in.world_xz + vec2<f32>(0.0,  eps_km)));
    let d_zm = sample_water_dist_km(world_to_world_uv(in.world_xz + vec2<f32>(0.0, -eps_km)));
    let laplacian = (d_xp + d_xm + d_zp + d_zm - 4.0 * dist_km) / (eps_km * eps_km);
    let grad_x = (d_xp - d_xm) / (2.0 * eps_km);
    let grad_z = (d_zp - d_zm) / (2.0 * eps_km);
    let grad_len = max(length(vec2<f32>(grad_x, grad_z)), 1e-6);
    let coast_normal = vec2<f32>(grad_x, grad_z) / grad_len;
    let curvature = clamp(abs(laplacian) * 5.0, 0.0, 1.0);
    let foam_modulator =
        (0.30 + 0.70 * curvature) * (0.5 + 0.5 * coast_ruggedness(in.world_xz));

    // ---- Foam: anchored emission + propagating wave fronts ----------
    let FOAM_PERIOD_S       = 50.0;
    let FOAM_SPEED_KM_PER_S = 0.25;
    let FOAM_PEAK_OFFSHORE_KM = 3.50;
    let FOAM_BAND_INNER_KM    = 0.80;
    let FOAM_BAND_OUTER_KM    = 1.20;
    let FOAM_EMISSION_WL_KM   = 4.0;
    let FOAM_PULSE_DUTY: f32 = 0.10;
    let FOAM_COAST_SIZE_PROBE_KM: f32 = 6.0;

    let coast_point = in.world_xz + (-dist_km) * coast_normal;
    let coast_size_probe_uv =
        world_to_world_uv(coast_point - coast_normal * FOAM_COAST_SIZE_PROBE_KM);
    let coast_size_dist = sample_water_dist_km(coast_size_probe_uv);
    let big_coast = smoothstep(-1.5, -4.0, coast_size_dist);

    let emit_lo = vnoise(coast_point / FOAM_EMISSION_WL_KM);
    let emit_hi = vnoise(coast_point / (FOAM_EMISSION_WL_KM * 0.35)
                         + vec2<f32>(31.7, 17.3));
    let emission = smoothstep(0.35, 0.75, emit_lo * 0.70 + emit_hi * 0.30);

    let offshore_km = max(0.0, -dist_km);
    let foam_t = camera.i_time;
    let phase_offset = vnoise(coast_point / 150.0) * 0.6;
    let local_T = FOAM_PERIOD_S
        * (0.97 + 0.06 * vnoise(coast_point / 250.0 + vec2<f32>(5.1, 8.4)));
    let phase = (foam_t + offshore_km / FOAM_SPEED_KM_PER_S) / local_T
                + phase_offset;
    let phi = fract(phase);
    let dist_to_peak = min(phi, 1.0 - phi);
    let foam_brightness =
        1.0 - smoothstep(0.0, FOAM_PULSE_DUTY * 0.5, dist_to_peak);

    let inner_fade = smoothstep(
        FOAM_PEAK_OFFSHORE_KM - FOAM_BAND_INNER_KM,
        FOAM_PEAK_OFFSHORE_KM,
        offshore_km,
    );
    let outer_fade = 1.0 - smoothstep(
        FOAM_PEAK_OFFSHORE_KM,
        FOAM_PEAK_OFFSHORE_KM + FOAM_BAND_OUTER_KM,
        offshore_km,
    );
    let dissipation = inner_fade * outer_fade;

    let surf_signal = clamp(
        emission * foam_brightness * dissipation * foam_modulator * big_coast,
        0.0, 1.0,
    );
    water_color = water_color + SEA_WATER_COLOR * (surf_signal * 0.22);
    water_color = mix(water_color, foam_color, clamp(surf_signal * 0.30, 0.0, 0.18));

    return vec4<f32>(water_color, 1.0);
}
