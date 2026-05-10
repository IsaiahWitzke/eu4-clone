// ============================================================================
// World-anchored erosion layer. Reads the base heightmap layer, runs a Phacelle
// erosion filter, writes the eroded height to its own texture.
//
// Output: .x = eroded height. Other channels reserved for ridge map / packed
// metadata when the image pass needs them.
//
// Translated from the Buffer B tab of https://www.shadertoy.com/view/sf23W1.
// PhacelleNoise and ErosionFilter are © Rune Skovbo Johansen, MPL-2.0.
// ============================================================================
@group(0) @binding(0) var<uniform> layer: LayerUniforms;
@group(0) @binding(1) var base_heightmap: texture_2d<f32>;

const TAU: f32 = 6.28318530717959;
const DEFAULT_HEIGHT: f32 = 0.45;

// ---- Small math helpers (defined inline in the original Common tab) --------

fn clamp01(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

fn pow_inv(t: f32, power: f32) -> f32 {
    return 1.0 - pow(1.0 - clamp01(t), power);
}

fn ease_out(t: f32) -> f32 {
    let v = 1.0 - clamp01(t);
    return 1.0 - v * v;
}

fn smooth_start(t: f32, smoothing: f32) -> f32 {
    if (t >= smoothing) {
        return t - 0.5 * smoothing;
    }
    return 0.5 * t * t / smoothing;
}

fn safe_normalize2(n: vec2<f32>) -> vec2<f32> {
    let l = length(n);
    return select(n, n / l, abs(l) > 1e-10);
}

// ---- Phacelle Noise --------------------------------------------------------
// Produces a stripe pattern aligned with `norm_dir`, blended over a 4×4 cell
// grid centered on `p`. Returns:
//   .xy = normalized (cos, sin) waves at p
//   .zw = direction vector for derivative recovery (= -∂/∂cos × ∂/∂sin)
fn phacelle_noise(
    p: vec2<f32>,
    norm_dir: vec2<f32>,
    freq: f32,
    offset_in: f32,
    normalization: f32,
) -> vec4<f32> {
    // Direction orthogonal to `norm_dir`, scaled by frequency × τ so that
    // dot(v, side_dir) increments by τ over one cell width.
    let side_dir = norm_dir.yx * vec2<f32>(-1.0, 1.0) * freq * TAU;
    let offset = offset_in * TAU;

    let p_int = floor(p);
    let p_frac = fract(p);
    var phase_dir = vec2<f32>(0.0);
    var weight_sum: f32 = 0.0;

    for (var i: i32 = -1; i <= 2; i = i + 1) {
        for (var j: i32 = -1; j <= 2; j = j + 1) {
            let grid_offset = vec2<f32>(f32(i), f32(j));
            let grid_point = p_int + grid_offset;
            let random_offset = hash(grid_point) * 0.5;
            let v = p_frac - grid_offset - random_offset;

            // Bell-shaped weight; truly zero at sqr_dist ≈ 1.5² to kill grid lines.
            let sqr_dist = dot(v, v);
            var weight = exp(-sqr_dist * 2.0);
            weight = max(0.0, weight - 0.01111);
            weight_sum += weight;

            let wave_input = dot(v, side_dir) + offset;
            phase_dir += vec2<f32>(cos(wave_input), sin(wave_input)) * weight;
        }
    }

    let interpolated = phase_dir / weight_sum;
    let raw_mag = sqrt(dot(interpolated, interpolated));
    // Lower bound on the magnitude so values below `1 − normalization` get
    // boosted to unit length while stronger values stay un-stretched.
    let magnitude = max(1.0 - normalization, raw_mag);
    return vec4<f32>(interpolated / magnitude, side_dir);
}

// ---- Erosion Filter --------------------------------------------------------
// Stacked, faded gullies layered onto the input height-and-slope.
struct ErosionResult {
    // .xyz = (height delta, slope delta x, slope delta y), .w = magnitude sum
    delta: vec4<f32>,
    ridge_map: f32,
    debug: f32,
}

fn erosion_filter(
    p: vec2<f32>,
    height_and_slope_in: vec3<f32>,
    fade_target_in: f32,
    strength_in: f32,
    gully_weight: f32,
    detail: f32,
    rounding: vec4<f32>,
    onset: vec4<f32>,
    assumed_slope: vec2<f32>,
    scale: f32,
    octaves: i32,
    lacunarity: f32,
    gain: f32,
    cell_scale: f32,
    normalization: f32,
) -> ErosionResult {
    var strength = strength_in * scale;
    var fade_target = clamp(fade_target_in, -1.0, 1.0);

    let input_has = height_and_slope_in;
    var has = height_and_slope_in;
    var freq = 1.0 / (scale * cell_scale);
    let slope_length = max(length(has.yz), 1e-10);
    var magnitude: f32 = 0.0;
    var rounding_mult: f32 = 1.0;

    let rounding_for_input =
        mix(rounding.y, rounding.x, clamp01(fade_target + 0.5)) * rounding.z;
    var combi_mask =
        ease_out(smooth_start(slope_length * onset.x, rounding_for_input * onset.x));

    var ridge_combi_mask = ease_out(slope_length * onset.z);
    var ridge_fade_target = fade_target;

    // Optionally override the actual slope with an assumed slope of fixed length.
    var gully_slope = mix(has.yz, has.yz / slope_length * assumed_slope.x, assumed_slope.y);

    for (var i: i32 = 0; i < octaves; i = i + 1) {
        var ph = phacelle_noise(
            p * freq,
            safe_normalize2(gully_slope),
            cell_scale,
            0.25,
            normalization,
        );
        // Multiply derivative by freq (since p was scaled by freq) and negate
        // (slope directions point downhill).
        ph = vec4<f32>(ph.xy, ph.zw * -freq);
        let sloping = abs(ph.y);

        // Steer subsequent octaves' gully direction by the current octave's slope.
        gully_slope += sign(ph.y) * ph.zw * strength * gully_weight;

        let gullies = vec3<f32>(ph.x, ph.y * ph.zw);
        let faded = mix(vec3<f32>(fade_target, 0.0, 0.0), gullies * gully_weight, combi_mask);
        has += faded * strength;
        magnitude += strength;
        fade_target = faded.x;

        // Update masks for the next octave.
        let r_oct = mix(rounding.y, rounding.x, clamp01(ph.x + 0.5)) * rounding_mult;
        let new_mask = ease_out(smooth_start(sloping * onset.y, r_oct * onset.y));
        combi_mask = pow_inv(combi_mask, detail) * new_mask;

        ridge_fade_target = mix(ridge_fade_target, gullies.x, ridge_combi_mask);
        let new_ridge = ease_out(sloping * onset.w);
        ridge_combi_mask = ridge_combi_mask * new_ridge;

        strength *= gain;
        freq *= lacunarity;
        rounding_mult *= rounding.w;
    }

    return ErosionResult(
        vec4<f32>(has - input_has, magnitude),
        ridge_fade_target * (1.0 - ridge_combi_mask),
        fade_target,
    );
}

// ---- Heightmap pipeline ----------------------------------------------------
// Fetches the painted heightmap from the base layer at the same texel and
// runs it through the erosion filter. Returns the eroded height in .x.
fn heightmap(frag_xy: vec2<f32>, world_pos: vec2<f32>) -> vec4<f32> {
    // "Subtle" preset: ~±50 m of added detail, gentle ridges. Tuned for
    // real Swiss elevation data — the original Shadertoy values were
    // calibrated for a procedural [0.45, 0.55] band, which is 10× narrower
    // than the [0, 1] range we get from the actual heightmap.
    let erosion_scale: f32 = 0.05;
    let erosion_strength: f32 = 0.01;
    let erosion_gully_weight: f32 = 0.4;
    let erosion_detail: f32 = 1.2;
    let erosion_rounding = vec4<f32>(0.1, 0.0, 0.1, 2.0);
    let erosion_onset = vec4<f32>(0.7, 1.25, 2.8, 1.5);
    let erosion_assumed_slope = vec2<f32>(0.7, 1.0);
    let erosion_cell_scale: f32 = 0.7;
    let erosion_normalization: f32 = 0.5;
    let erosion_octaves: i32 = 4;
    let erosion_lacunarity: f32 = 2.0;
    let erosion_gain: f32 = 0.5;

    // The base heightmap layer and this erosion layer are the same size and
    // cover the same world AABB, so the corresponding texel for our fragment
    // is at the same integer coord.
    let coord = vec2<i32>(frag_xy);
    let n = textureLoad(base_heightmap, coord, 0).xyz;

    // Fade target: -1 at Swiss valleys, +1 at Swiss peaks. Calibrated around
    // Switzerland's actual normalised elevation distribution (mean ≈ 0.27,
    // half-range ≈ 0.45 in [0, 1] units = [0, 5000 m]) instead of the
    // Shadertoy's procedural plain at 0.45.
    let fade_target = clamp((n.x - 0.27) / 0.45, -1.0, 1.0);

    let r = erosion_filter(
        world_pos, n, fade_target,
        erosion_strength, erosion_gully_weight, erosion_detail,
        erosion_rounding, erosion_onset, erosion_assumed_slope,
        erosion_scale, erosion_octaves, erosion_lacunarity,
        erosion_gain, erosion_cell_scale, erosion_normalization,
    );

    let eroded = n.x + r.delta.x;
    return vec4<f32>(eroded, 0.0, 0.0, 0.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let world_pos = layer_frag_to_world(frag.xy, layer);
    return heightmap(frag.xy, world_pos);
}
