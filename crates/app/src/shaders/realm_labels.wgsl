// ============================================================================
// Realm-label render pass.
//
// Vertex stage: takes a per-vertex world XZ + atlas UV, projects through the
// shared camera basis, and forwards the UV to the fragment.
// Fragment stage: samples the SDF atlas (R channel), uses screen-space
// derivatives for AA, and outputs the realm-name "ink" colour.
//
// Bindings (group 0):
//   0  Uniforms        (camera/eye/projection state — same as image.wgsl)
//   1  glyph_atlas     (Rgba8Unorm; SDF in R, 0.5 = glyph edge)
//   2  glyph_sampler   (linear-filtering)
//   3  LabelUniforms   (sepia ink color, outline color, ground-y constant,
//                       AA scale knob)
// ============================================================================

@group(0) @binding(1) var glyph_atlas: texture_2d<f32>;
@group(0) @binding(2) var glyph_sampler: sampler;

struct LabelUniforms {
    /// Body fill colour (RGB), with `.a` repurposed as the body opacity
    /// multiplier (0..1).
    ink_color: vec4<f32>,
    /// Halo / outline colour. Drawn as a wider smoothstep band slightly
    /// outside the body so the label reads on any terrain colour.
    outline_color: vec4<f32>,
    /// Fixed Y altitude (post-VE) used to pin the label quads to a flat
    /// "label plane" above the ground. Picked roughly at average
    /// elevation — same constant the hover code uses for ground picks.
    ground_y: f32,
    /// Multiplier on `fwidth(sdf)` used to size the smoothstep band.
    /// 1.0 = pixel-tight edge; bigger values = softer.
    aa_scale: f32,
    /// Padding to keep the struct 16-byte aligned on the GPU side.
    _pad0: f32,
    _pad1: f32,
}
@group(0) @binding(3) var<uniform> labels_u: LabelUniforms;

// `Uniforms` (binding 0) and the projection helper come from camera.wgsl
// (concatenated by the host before the shader is compiled).

// The mesh path in image.wgsl uses the same projection. We reproduce a
// minimal version here so the labels module is self-contained — its own
// shader text doesn't depend on `make_view_proj` from image.wgsl.
const LABEL_FOV_Y_DEG: f32 = 30.0;
const LABEL_TARGET_Y:  f32 = 4.0;
const LABEL_NEAR: f32 = 0.1;
const LABEL_FAR:  f32 = 12000.0;
const LABEL_PI: f32 = 3.14159265358979;

fn label_view_proj() -> mat4x4<f32> {
    let look_at = vec3<f32>(u.world_center.x, LABEL_TARGET_Y, u.world_center.y);
    let eye = look_at + u.eye_offset;
    let forward = normalize(look_at - eye);
    let world_up = vec3<f32>(0.0, 1.0, 0.0);
    let right = normalize(cross(world_up, forward));
    let up = cross(forward, right);

    let view = mat4x4<f32>(
        vec4<f32>(right.x, up.x, -forward.x, 0.0),
        vec4<f32>(right.y, up.y, -forward.y, 0.0),
        vec4<f32>(right.z, up.z, -forward.z, 0.0),
        vec4<f32>(-dot(right, eye), -dot(up, eye), dot(forward, eye), 1.0),
    );

    let aspect = u.i_resolution.x / u.i_resolution.y;
    let f = 1.0 / tan(LABEL_FOV_Y_DEG * 0.5 * (LABEL_PI / 180.0));
    let near = LABEL_NEAR;
    let far  = LABEL_FAR;
    let proj = mat4x4<f32>(
        vec4<f32>(f / aspect, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, f, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, far / (near - far), -1.0),
        vec4<f32>(0.0, 0.0, near * far / (near - far), 0.0),
    );
    return proj * view;
}

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) atlas_uv: vec2<f32>,
}

@vertex
fn vs_label(
    @location(0) world_xz: vec2<f32>,
    @location(1) atlas_uv: vec2<f32>,
) -> VsOut {
    let world_pos = vec3<f32>(world_xz.x, labels_u.ground_y, world_xz.y);
    let vp = label_view_proj();
    let clip_pos = vp * vec4<f32>(world_pos, 1.0);
    return VsOut(clip_pos, atlas_uv);
}

@fragment
fn fs_label(in: VsOut) -> @location(0) vec4<f32> {
    let sample = textureSample(glyph_atlas, glyph_sampler, in.atlas_uv);
    let sdf = sample.r;

    // Screen-space derivative of the SDF gives us the per-fragment
    // change in the field, which is the natural width for the AA band:
    // ~1 pixel wide at every zoom, automatically.
    let aa = max(fwidth(sdf), 0.001) * labels_u.aa_scale;

    // Body of the glyph: above 0.5 → solid ink. Smoothstep across `aa`
    // gives ~1 fragment of soft edge.
    let body = smoothstep(0.5 - aa, 0.5 + aa, sdf);

    // Outline / halo: a wider band drawn slightly outside the body. We
    // re-use the same SDF and shift the threshold to `0.5 - OUTLINE`,
    // so the halo sits in the SDF range "just outside the glyph".
    let outline_thresh: f32 = 0.42;  // ~8% of SDF spread outside the edge
    let outline = smoothstep(outline_thresh - aa, outline_thresh + aa, sdf);
    // Halo strength = (outline mask) - (body mask), so we don't double
    // up the halo on top of body pixels.
    let halo = clamp(outline - body, 0.0, 1.0);

    let body_alpha    = body * labels_u.ink_color.a;
    let outline_alpha = halo * labels_u.outline_color.a;

    // Composite halo under body so the body's colour wins on the glyph
    // itself but the halo "bleeds" outside it.
    let rgb = labels_u.ink_color.rgb * body_alpha
            + labels_u.outline_color.rgb * outline_alpha * (1.0 - body_alpha);
    let alpha = body_alpha + outline_alpha * (1.0 - body_alpha);

    // Premultiplied output. The host pipeline blends with
    // `(One, OneMinusSrcAlpha)`.
    return vec4<f32>(rgb, alpha);
}
