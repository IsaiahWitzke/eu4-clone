// World-anchored heightmap layer.
//
// Samples the real Switzerland heightmap (assets/heightmap.png, 8192²,
// R16Unorm) into a fixed-size (LAYER_SIZE²) world tile covering the AABB in
// `layer.covered_*`. The procedural smoothstep bump from the original
// Shadertoy is gone — erosion is layered on top of real elevation data by the
// downstream `terrain` (erosion) pass.
//
// Output:
//   .x  = height in [0, 1] (= elevation / HEIGHT_SCALE_M)
//   .yz = slope derivative (∂h/∂x, ∂h/∂z in world XZ, units of 1/km)
//   .w  = unused

@group(0) @binding(0) var<uniform> layer: LayerUniforms;
@group(0) @binding(1) var world_heightmap: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let world_pos = layer_frag_to_world(frag.xy, layer);
    let uv = world_to_world_uv(world_pos);

    // Bilinear sample of the real heightmap at this world position. Returns
    // h ∈ [0, 1]; the rest of the pipeline interprets that as elevation in
    // fractions of HEIGHT_SCALE_M (= 5000 m).
    let h = textureSampleLevel(world_heightmap, samp, uv, 0.0).x;

    // Slope via forward differences against neighbours one source-texel away.
    // Source texel size in world units = 2 * WORLD_BOUNDS_HALF / 8192.
    let texel = (2.0 * WORLD_BOUNDS_HALF) / vec2<f32>(WORLD_HEIGHTMAP_SIZE);
    let h_dx = textureSampleLevel(
        world_heightmap, samp,
        world_to_world_uv(world_pos + vec2<f32>(texel.x, 0.0)),
        0.0,
    ).x;
    let h_dz = textureSampleLevel(
        world_heightmap, samp,
        world_to_world_uv(world_pos + vec2<f32>(0.0, texel.y)),
        0.0,
    ).x;
    let dhdx = (h_dx - h) / texel.x;
    let dhdz = (h_dz - h) / texel.y;

    return vec4<f32>(h, dhdx, dhdz, 0.0);
}
