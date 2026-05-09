// ============================================================================
// Truly shared declarations — included at the top of every shader.
//
// Pass-specific uniforms live in camera.wgsl (image pass) and layer.wgsl
// (world-anchored heightmap / detail-noise / erosion passes).
// ============================================================================

// Fullscreen "big triangle" — three vertices that cover the viewport with no
// vertex buffer. Cheaper than a quad because the GPU rejects the off-screen
// portion and avoids a redundant diagonal seam.
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(pos[idx], 0.0, 1.0);
}
