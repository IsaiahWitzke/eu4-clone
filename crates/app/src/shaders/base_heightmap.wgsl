// World-anchored heightmap layer.
//
// Samples the real Switzerland heightmap (assets/heightmap.png, 8192²,
// 16-bit grayscale) into a fixed-size (LAYER_SIZE²) world tile covering the
// AABB in `layer.covered_*`. The procedural smoothstep bump from the original
// Shadertoy is gone — erosion is layered on top of real elevation data by the
// downstream `terrain` (erosion) pass.
//
// Storage trick: WebGPU baseline doesn't expose `R16Unorm`, so the heightmap
// is uploaded as `Rg8Unorm` instead — the PNG's big-endian high/low bytes
// land directly in the .r / .g channels and we reassemble them here.
//
// Output:
//   .x  = height in [0, 1] (= elevation / HEIGHT_SCALE_M)
//   .yz = slope derivative (∂h/∂x, ∂h/∂z in world XZ, units of 1/km)
//   .w  = unused

@group(0) @binding(0) var<uniform> layer: LayerUniforms;
@group(0) @binding(1) var world_heightmap: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

// Sample the world heightmap (Rg8Unorm) at uv and reassemble high/low bytes
// into a [0, 1] elevation value. R = high byte / 255, G = low byte / 255.
//   u16 value = R*255 * 256 + G*255 = R * 65280 + G * 255
//   normalised  = (R * 65280 + G * 255) / 65535
//
// Outside the texture's [0, 1]² footprint we return 0 (sea level) instead of
// letting the sampler smear the edge column/row across the void. This makes
// far zoom-out look like flat ocean around Switzerland rather than a streaky
// extrapolation of whatever happened to be at the heightmap's border.
fn sample_height(uv: vec2<f32>) -> f32 {
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
        return 0.0;
    }
    let rg = textureSampleLevel(world_heightmap, samp, uv, 0.0).xy;
    return (rg.x * 65280.0 + rg.y * 255.0) / 65535.0;
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let world_pos = layer_frag_to_world(frag.xy, layer);
    let uv = world_to_world_uv(world_pos);

    let h = sample_height(uv);

    // Slope via forward differences against neighbours one source-texel away.
    // Source texel size in world units = 2 * WORLD_BOUNDS_HALF / 8192.
    let texel = (2.0 * WORLD_BOUNDS_HALF) / vec2<f32>(WORLD_HEIGHTMAP_SIZE);
    let h_dx = sample_height(world_to_world_uv(world_pos + vec2<f32>(texel.x, 0.0)));
    let h_dz = sample_height(world_to_world_uv(world_pos + vec2<f32>(0.0, texel.y)));
    let dhdx = (h_dx - h) / texel.x;
    let dhdz = (h_dz - h) / texel.y;

    return vec4<f32>(h, dhdx, dhdz, 0.0);
}
