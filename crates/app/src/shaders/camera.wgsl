// ============================================================================
// Camera uniforms + helpers. Included only by the screen-space `image` pass —
// the world-anchored layer shaders are camera-independent and use layer.wgsl
// instead.
// ============================================================================

struct Uniforms {
    i_resolution: vec3<f32>,
    i_time: f32,
    world_center: vec2<f32>,
    // Currently-hovered realm encoded as `realm_id + 1` (0 = nothing).
    // Drives the realm-wide hover wash.
    hovered_pid: u32,
    // Currently-hovered settlement encoded as `settlement_index + 1`
    // (0 = nothing). Drives the stronger hinterland hover wash.
    hovered_city: u32,
    // Camera position relative to look_at, in world units. Encodes both
    // distance and tilt; computed on the Rust side.
    eye_offset: vec3<f32>,
    // Currently-active map mode. Discriminants must match `MapMode` in
    // `camera.rs`. The image pass branches rendering on this value.
    map_mode: u32,
}

// Map-mode discriminant constants, kept in sync with the Rust enum.
const MAP_MODE_TERRAIN: u32 = 0u;
const MAP_MODE_POLITICAL: u32 = 1u;

@group(0) @binding(0) var<uniform> u: Uniforms;
