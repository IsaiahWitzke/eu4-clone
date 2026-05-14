//! Bake pass for the settlement influence field.
//!
//! Once per settlement-list change we evaluate the per-pixel argmax-realm
//! at every cell of a fixed-size RGBA8Unorm texture, indexed by world_xz.
//! The image pass then `textureLoad`s the result instead of looping over
//! the full settlement array, which drops fragment cost from O(N
//! settlements) to O(1).
//!
//! Output texture format (Rgba16Float — same baseline format used for the
//! cached world layers, so we know it works without extra wgpu features):
//!   R: realm_id (stored as `f32(id)` — exact for ids ≤ 2048 in f16)
//!   G: alpha    (saturating fade, 0 = wilderness, ≈ 1 deep interior)
//!   B: contested - 1 (clamped to [0, 1] before storage)
//!   A: city_idx (stored as `f32(idx)` — exact for indices ≤ 2048; used
//!                by hover hinterland highlighting to single out the
//!                dominant city's cells, vs all same-realm cities)
//!
//! See `crates/app/src/shaders/realm_field.wgsl` for the per-pixel logic.
//! Same shader as the old in-line `sample_realm_field` in `image.wgsl`,
//! just hoisted into a separate offscreen render pass.

use crate::gpu::{GpuContext, make_pipeline};

const SHADER_WGSL: &str = include_str!("../shaders/realm_field.wgsl");

/// Bake-target side length. 2048\u00b2 over a 5500 km world ≈ 2.7 km/px,
/// well-resolved for a 30 km e-fold influence field. Storage = 16 MB
/// at RGBA8Unorm.
pub const FIELD_SIZE: u32 = 2048;

/// 16-bit float per channel. We need exact integer storage for `realm_id`
/// and `city_idx` (up to 1024 settlements), and f16 has 11 bits of
/// mantissa, so everything below 2048 round-trips exactly. The smooth
/// channels (alpha, contested) just use the float range directly.
/// Rgba16Float is the same format the cached world layers use, so
/// there's no concern about the wgpu backend supporting it.
pub const FIELD_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub struct RealmFieldPass {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
}

pub fn build(gpu: &GpuContext, settlements_uniform_buf: &wgpu::Buffer) -> RealmFieldPass {
    let device = &gpu.device;

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("realm_field bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("realm_field"),
        source: wgpu::ShaderSource::Wgsl(SHADER_WGSL.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("realm_field pl"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    let pipeline = make_pipeline(
        device,
        "realm_field pipeline",
        &pipeline_layout,
        &module,
        FIELD_FORMAT,
    );

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("realm_field tex"),
        size: wgpu::Extent3d {
            width: FIELD_SIZE,
            height: FIELD_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FIELD_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("realm_field bg"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: settlements_uniform_buf.as_entire_binding(),
        }],
    });

    RealmFieldPass {
        texture,
        view,
        pipeline,
        bind_group,
    }
}

impl RealmFieldPass {
    /// Run the bake into our owned texture. Caller queues the encoder.
    /// `timestamp_writes` brackets the bake for GPU profiling.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("realm_field bake"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }
}
