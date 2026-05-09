// World-anchored heightmap layer.
//
// Renders into a fixed-size (LAYER_SIZE²) world-tile texture covering the AABB
// in `layer.covered_*`. Re-rendered only when the camera leaves the cached
// region (see WorldLayer::ensure_covers on the Rust side).
//
// Output:
//   .x  = height
//   .yz = slope derivative (∂h/∂x, ∂h/∂y in world XZ)
//   .w  = unused

@group(0) @binding(0) var<uniform> layer: LayerUniforms;

const DEFAULT_HEIGHT: f32 = 0.45;

// Returns a smoothstep "brush" centered on `cursor_pos`: a smooth radial bump
// with falloff = `brush_size`. The returned vec3 packs the value (.x) and its
// 2D derivative (.yz) so the heightmap and its analytic slope are kept in sync.
fn get_brush_delta(map_pos: vec2<f32>, cursor_pos: vec2<f32>, brush_size: f32) -> vec3<f32> {
    let to_cursor = cursor_pos - map_pos;
    let dist = length(to_cursor);
    let freq = 1.0 / brush_size;
    let x = clamp(1.0 - freq * dist, 0.0, 1.0);
    // dir is undefined at dist == 0 — but the slope multiplier (1 - x) is also 0
    // there, so we guard against NaN by zeroing the direction at the singular point.
    let dir = select(vec2<f32>(0.0), to_cursor / dist, dist > 1e-10);
    return vec3<f32>(
        // Smoothstep value: 3x² − 2x³ (= GLSL smoothstep with this remapping).
        x * x * (3.0 - 2.0 * x),
        // Derivative of the above × falloff direction × frequency.
        dir * 6.0 * x * (1.0 - x) * freq,
    );
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let world_pos = layer_frag_to_world(frag.xy, layer);

    // Start at default height, zero slope.
    var v = vec3<f32>(DEFAULT_HEIGHT, 0.0, 0.0);
    // Centered radial bump at world origin, sized in world units (radius 0.35
    // matches the original shader's footprint).
    v += get_brush_delta(world_pos, vec2<f32>(0.0, 0.0), 0.35) * 0.1;

    return vec4<f32>(v, 0.0);
}
