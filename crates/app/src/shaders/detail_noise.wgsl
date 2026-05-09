// World-anchored detail texture, used by the image pass for surface breakup /
// dirt. 8 octaves of analytic gradient `noised`, summed with decreasing
// amplitude and doubling frequency.
//
// World-anchored: the noise pattern at any given world XZ stays the same as
// the camera pans/zooms (since this layer caches a world AABB).
//
// Output layout:
//   .x = noise value (sum of octaves)
//   .y = ∂value/∂worldX derivative (sum of octaves)
//   .z = ∂value/∂worldZ derivative
//   .w = 1.0
//
// Base frequency `f0 = 4.0` cycles/world-unit; with 8 octaves the smallest
// feature is ~0.002 world units.

@group(0) @binding(0) var<uniform> layer: LayerUniforms;

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let world = layer_frag_to_world(frag.xy, layer);

    var color = vec3<f32>(0.0);
    var a = 0.5;
    var f = 4.0;
    for (var i = 0; i < 8; i = i + 1) {
        color += noised(world * f) * a;
        a *= 0.95;
        f *= 2.0;
    }

    return vec4<f32>(color, 1.0);
}
