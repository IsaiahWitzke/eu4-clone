//! Realm-name overlay pass — renders SDF glyph quads in world space.
//!
//! Reads:
//!   * camera Uniforms (binding 0; same struct as the image pass)
//!   * glyph_atlas texture + sampler (bindings 1 + 2)
//!   * LabelUniforms (binding 3; ink + outline colours, ground-y plane)
//!   * a per-vertex buffer of `LabelVertex` (world XZ + atlas UV)
//!
//! Writes:
//!   * the swapchain (LoadOp::Load) with alpha-blended ink/halo
//!
//! Drawn after the image pass so labels always sit on top of the
//! terrain. Depth test is disabled — labels are conceptually a 2D
//! overlay locked to a flat world-Y plane (`LabelUniforms.ground_y`),
//! so depth-fighting between adjacent realms isn't a concern.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::camera::HOVER_PICK_Y;
use crate::gpu::GpuContext;
use crate::labels::LabelVertex;

/// Uniforms for the SDF shader: per-frame stylistic knobs that don't
/// change per-glyph. Mirrors `LabelUniforms` in `realm_labels.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LabelUniforms {
    pub ink_color: [f32; 4],
    pub outline_color: [f32; 4],
    pub ground_y: f32,
    pub aa_scale: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

impl Default for LabelUniforms {
    fn default() -> Self {
        Self {
            // Dark sepia, fully opaque. Bumping `.a` past 1.0 doesn't
            // help — the shader clamps it via the smoothstep mask.
            ink_color: [0.18, 0.10, 0.04, 1.0],
            // Cream halo at moderate strength so the label stands out
            // against forest / snow / rock without overwhelming the
            // map.
            outline_color: [0.96, 0.93, 0.84, 0.85],
            ground_y: HOVER_PICK_Y,
            aa_scale: 1.0,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }
}

/// Maximum number of `LabelVertex` entries we'll ever upload. The
/// vertex buffer is sized to this once at startup and reused; the
/// renderer just rewrites the prefix when the layout changes.
///
/// One realm name = ~12 chars × 6 verts = ~72 verts; 8192 vertex slots
/// fits ~110 such labels comfortably. Bump if cities + realms together
/// ever push past that.
pub const MAX_LABEL_VERTICES: u32 = 32_768;

pub struct RealmLabelsPass {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
    /// Cached BGL so `set_glyph_atlas` can rebuild the bind group
    /// against a fresh atlas texture without the caller plumbing it.
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub vertex_buffer: wgpu::Buffer,
    pub uniform_buffer: wgpu::Buffer,
    /// Number of vertices currently uploaded. The render call draws
    /// `0..vertex_count`. 0 means "no labels yet" — skip the draw.
    pub vertex_count: u32,
}

impl RealmLabelsPass {
    pub fn build(
        gpu: &GpuContext,
        camera_uniform_buf: &wgpu::Buffer,
        atlas_view: &wgpu::TextureView,
        atlas_sampler: &wgpu::Sampler,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        let device = &gpu.device;

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("realm_labels bgl"),
            entries: &[
                // 0: camera uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: glyph atlas texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 2: glyph atlas sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // 3: label-uniforms (vertex stage uses ground_y; fragment
                // uses ink/outline + aa_scale).
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Shader source: camera.wgsl provides the `Uniforms` struct +
        // `@binding(0)` declaration; realm_labels.wgsl is everything
        // else.
        let camera_wgsl: &str = include_str!("../shaders/camera.wgsl");
        let label_wgsl: &str = include_str!("../shaders/realm_labels.wgsl");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("realm_labels"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{camera_wgsl}\n{label_wgsl}").into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("realm_labels pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        // Vertex buffer layout: matches `LabelVertex { world_xz, atlas_uv }`.
        let vertex_attrs = wgpu::vertex_attr_array![
            0 => Float32x2,
            1 => Float32x2,
        ];
        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LabelVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &vertex_attrs,
        };

        // Premultiplied alpha blending. The fragment shader emits
        // `(rgb * alpha, alpha)` so the GPU just adds the source on
        // top of the existing swapchain colour with a `1 - src_a`
        // suppression of the destination.
        let blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("realm_labels pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_label"),
                compilation_options: Default::default(),
                buffers: &[vertex_buffer_layout],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_label"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            // No face culling — labels can be viewed from any tilt;
            // the perpendicular vector points "above the baseline" but
            // the camera might be on either side.
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None, // overlay pass; we don't depth-test
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("realm_labels vb"),
            size: (MAX_LABEL_VERTICES as u64) * std::mem::size_of::<LabelVertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("realm_labels ub"),
            contents: bytemuck::bytes_of(&LabelUniforms::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("realm_labels bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(atlas_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            pipeline,
            bind_group,
            bind_group_layout: bgl,
            vertex_buffer,
            uniform_buffer,
            vertex_count: 0,
        }
    }

    /// Replace the bind group when the atlas texture or sampler change
    /// (e.g. when `set_glyph_atlas` lands the real PNG after startup
    /// placeholder).
    pub fn rebuild_bind_group(
        &mut self,
        gpu: &GpuContext,
        camera_uniform_buf: &wgpu::Buffer,
        atlas_view: &wgpu::TextureView,
        atlas_sampler: &wgpu::Sampler,
    ) {
        self.bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("realm_labels bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(atlas_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });
    }

    /// Upload a fresh batch of vertices. Truncates to `MAX_LABEL_VERTICES`
    /// with a console warning rather than refusing to render — better
    /// to drop a label or two than to drop the whole frame.
    pub fn set_vertices(&mut self, gpu: &GpuContext, verts: &[LabelVertex]) {
        let n = verts.len().min(MAX_LABEL_VERTICES as usize);
        if n < verts.len() {
            web_sys::console::warn_1(
                &format!(
                    "realm_labels: vertex count {} exceeds MAX_LABEL_VERTICES {}; \
                     truncating",
                    verts.len(),
                    MAX_LABEL_VERTICES
                )
                .into(),
            );
        }
        if n == 0 {
            self.vertex_count = 0;
            return;
        }
        gpu.queue.write_buffer(
            &self.vertex_buffer,
            0,
            bytemuck::cast_slice(&verts[..n]),
        );
        self.vertex_count = n as u32;
    }

    /// Draw one frame onto `target_view` (LoadOp::Load — the image pass
    /// already populated the swapchain). No-op when no labels have
    /// been uploaded yet.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
    ) {
        if self.vertex_count == 0 {
            return;
        }
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("realm_labels"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Load the existing swapchain pixels so we draw on
                    // top of the image pass's output.
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        rpass.draw(0..self.vertex_count, 0..1);
    }
}

/// Build a 1×1 dummy `Rgba8Unorm` texture + linear sampler. Used as
/// the placeholder atlas before `glyph_atlas.png` arrives over the
/// network — keeps the bind group valid so the realm-labels pass can
/// be constructed at startup. The pass is a no-op until vertices land
/// (`vertex_count == 0`), so the placeholder content doesn't matter.
pub fn placeholder_atlas(gpu: &GpuContext) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("glyph_atlas (placeholder)"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[0u8, 0u8, 0u8, 0u8],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&Default::default());
    let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("glyph_atlas sampler (placeholder)"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    (texture, view, sampler)
}
