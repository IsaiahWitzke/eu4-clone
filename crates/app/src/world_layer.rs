//! World-anchored layer: a fixed-size offscreen texture rendered by a
//! dedicated pipeline whose shader reads `LayerUniforms` to know which world
//! AABB to paint into.
//!
//! All world layers in a frame share a single `LayerUniforms` buffer + a
//! single covered-AABB cache (managed by `Renderer`). When the camera leaves
//! the cached region, the renderer updates the buffer and asks every world
//! layer to render. This keeps the three textures in sync so the image pass's
//! `world_to_layer_uv` math produces correct UVs.

use bytemuck::{Pod, Zeroable};

use crate::camera::Aabb2;
use crate::gpu::{GpuContext, make_offscreen};

/// Width/height of every world-layer texture. Must match `LAYER_SIZE` in
/// `shaders/layer.wgsl`.
pub const LAYER_SIZE: u32 = 1024;

/// Mirrors the WGSL `LayerUniforms` struct in `shaders/layer.wgsl`.
/// Layout: vec2 + vec2 + f32 + f32 pad → 24 bytes, 8-byte aligned.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct LayerUniforms {
    pub covered_min: [f32; 2],
    pub covered_max: [f32; 2],
    pub texel_size: f32,
    pub _pad: f32,
}

impl LayerUniforms {
    pub fn from_aabb(aabb: Aabb2) -> Self {
        let texel_size = (aabb.max[0] - aabb.min[0]) / LAYER_SIZE as f32;
        Self {
            covered_min: aabb.min,
            covered_max: aabb.max,
            texel_size,
            _pad: 0.0,
        }
    }
}

/// Allocate a uniform buffer sized for `LayerUniforms`. Re-used by every
/// world layer's bind group plus the image pass's bind group.
pub fn make_layer_uniform_buf(gpu: &GpuContext, label: &str) -> wgpu::Buffer {
    gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: std::mem::size_of::<LayerUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// A `WorldLayer` is a "renderable" — it owns its offscreen texture +
/// pipeline + bind group, and can be asked to render into its texture.
/// It does *not* own caching state; that lives in `Renderer`.
pub struct WorldLayer {
    pub label: &'static str,
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
}

impl WorldLayer {
    pub fn new(
        gpu: &GpuContext,
        label: &'static str,
        pipeline: wgpu::RenderPipeline,
        bind_group: wgpu::BindGroup,
    ) -> Self {
        let (texture, view) = make_offscreen(&gpu.device, label, LAYER_SIZE, LAYER_SIZE);
        Self {
            label,
            texture,
            view,
            pipeline,
            bind_group,
        }
    }

    /// Render this layer's pipeline (a fullscreen triangle) into its texture.
    /// The caller is responsible for ensuring the shared `LayerUniforms`
    /// buffer is up-to-date before issuing this.
    ///
    /// `timestamp_writes` lets the caller bracket this render pass with
    /// `TIMESTAMP_QUERY` writes for GPU profiling; pass `None` for the
    /// normal path.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(self.label),
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
