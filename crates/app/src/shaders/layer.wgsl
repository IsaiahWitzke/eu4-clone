// ============================================================================
// LayerUniforms — declares the world AABB covered by a `WorldLayer` texture
// and its texel size (world units / texel). The binding index varies per
// shader (layer shaders bind it at 0; image binds it at 4), so we declare the
// `var<uniform>` inline in each shader rather than here.
// ============================================================================

const LAYER_SIZE: f32 = 1024.0;

struct LayerUniforms {
    covered_min: vec2<f32>,
    covered_max: vec2<f32>,
    texel_size: f32,
    _pad: f32,
}

// Map a world-layer fragment's pixel coordinate to its world-space XZ position.
// The layer's texel grid spans `[0, LAYER_SIZE]` and corresponds to the world
// rectangle `[covered_min, covered_max]`.
fn layer_frag_to_world(frag_xy: vec2<f32>, layer: LayerUniforms) -> vec2<f32> {
    let uv = frag_xy / vec2<f32>(LAYER_SIZE);
    return mix(layer.covered_min, layer.covered_max, uv);
}

// Inverse of `layer_frag_to_world` — map a world XZ position to the [0,1] UV
// of the layer texture covering it.
fn world_to_layer_uv(world_xz: vec2<f32>, layer: LayerUniforms) -> vec2<f32> {
    return (world_xz - layer.covered_min) / (layer.covered_max - layer.covered_min);
}
