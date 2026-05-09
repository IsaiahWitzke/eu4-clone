// ============================================================================
// Camera uniforms + helpers. Included only by the screen-space `image` pass —
// the world-anchored layer shaders are camera-independent and use layer.wgsl
// instead.
// ============================================================================

struct Uniforms {
    i_resolution: vec3<f32>,
    i_time: f32,
    world_center: vec2<f32>,
    _pad0: vec2<f32>,
    // Camera position relative to look_at, in world units. Encodes both
    // distance and tilt; computed on the Rust side.
    eye_offset: vec3<f32>,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
