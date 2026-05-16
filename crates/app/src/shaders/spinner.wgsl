// ============================================================================
// Spinner pass — draws a small rotating arc in the top-right corner of the
// swapchain. Purely a frame-pacing diagnostic: with continuous rAF, smooth
// rotation = frame() is keeping up, stutter = frame() is lagging.
//
// Premultiplied-alpha output to match realm_labels' blend setup. Uses a
// fullscreen triangle + `discard` outside the corner ring — the corner box
// is tiny so the fragment cost is negligible.
// ============================================================================

struct SpinnerUniforms {
    time_s: f32,
    res_x: f32,
    res_y: f32,
    _pad: f32,
}
@group(0) @binding(0) var<uniform> u: SpinnerUniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
}

// Standard fullscreen-triangle setup; same trick as realm_field.wgsl.
@vertex
fn vs_main(@builtin(vertex_index) vidx: u32) -> VsOut {
    let x = f32((vidx << 1u) & 2u);
    let y = f32(vidx & 2u);
    let pos_ndc = vec2<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0);
    return VsOut(vec4<f32>(pos_ndc, 0.0, 1.0));
}

const TWO_PI:        f32 = 6.283185307;
// Pixel offsets are in physical (framebuffer) pixels — i.e. they shrink in
// CSS-pixel size on hi-DPI displays. That's fine for a debug overlay.
const CENTER_OFFSET: f32 = 60.0;   // distance from top-right corner
const RADIUS_OUTER:  f32 = 26.0;
const RADIUS_INNER:  f32 = 18.0;
const EDGE_AA:       f32 = 1.25;   // px of smoothstep on each edge of the ring
const SPIN_SPEED:    f32 = 2.5;    // rad/s — slow enough to read individual stutters

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    // Spinner center: inset from top-right corner. `frag_pos.xy` is in
    // framebuffer pixels, origin top-left.
    let center = vec2<f32>(u.res_x - CENTER_OFFSET, CENTER_OFFSET);
    let d = frag_pos.xy - center;
    let r = length(d);

    // Bail early on the vast majority of the screen so the rest of the
    // shader doesn't run for nothing.
    if (r > RADIUS_OUTER + EDGE_AA) {
        discard;
    }
    if (r < RADIUS_INNER - EDGE_AA) {
        discard;
    }

    // Smooth ring mask.
    let outer = smoothstep(RADIUS_OUTER + EDGE_AA, RADIUS_OUTER - EDGE_AA, r);
    let inner = smoothstep(RADIUS_INNER - EDGE_AA, RADIUS_INNER + EDGE_AA, r);
    let ring  = outer * inner;

    // Phase in [0, 1): 0 right at the leading edge, 1 just behind it.
    // `atan2(d.y, d.x)` is in [-pi, pi]; subtracting the time-driven
    // leader angle and wrapping to [0, 1) gives a clean tail-fade.
    let angle  = atan2(d.y, d.x);
    let leader = u.time_s * SPIN_SPEED;
    var phase  = (leader - angle) / TWO_PI;
    phase = phase - floor(phase);

    // Quadratic tail-fade. Pow 3.0 makes the bright leader compact and
    // the rest of the ring readable as a fading trail.
    let tail  = pow(1.0 - phase, 3.0);

    // Warm yellow so it pops against most terrain tints.
    let color = vec3<f32>(1.0, 0.85, 0.30);
    let alpha = ring * tail * 0.95;
    // Premultiplied alpha — see realm_labels.rs for the matching blend state.
    return vec4<f32>(color * alpha, alpha);
}
