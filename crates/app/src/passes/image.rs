//! Image pass — renders the final raymarched terrain to the swapchain.
//!
//! Reads the camera uniforms (binding 0), the three world-layer textures
//! (bindings 1–3), and the shared `LayerUniforms` (binding 4) so the shader
//! knows the world AABB covered by those textures. The image pass itself is
//! always run every frame; the per-frame work is dominated by the raymarcher.

use crate::gpu::{GpuContext, make_pipeline};

const COMMON_WGSL: &str = include_str!("../shaders/common.wgsl");
const CAMERA_WGSL: &str = include_str!("../shaders/camera.wgsl");
const LAYER_WGSL: &str = include_str!("../shaders/layer.wgsl");
const FS_WGSL: &str = include_str!("../shaders/image.wgsl");

pub struct ImagePass {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
}

impl ImagePass {
    pub fn new(
        gpu: &GpuContext,
        camera_uniform_buf: &wgpu::Buffer,
        base_heightmap_view: &wgpu::TextureView,
        detail_noise_view: &wgpu::TextureView,
        erosion_view: &wgpu::TextureView,
        layer_uniform_buf: &wgpu::Buffer,
    ) -> Self {
        let device = &gpu.device;

        let uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("image bgl"),
            entries: &[
                uniform_entry(0),
                texture_entry(1),
                texture_entry(2),
                texture_entry(3),
                uniform_entry(4),
            ],
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("image"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{COMMON_WGSL}\n{CAMERA_WGSL}\n{LAYER_WGSL}\n{FS_WGSL}").into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("image pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = make_pipeline(
            device,
            "image pipeline",
            &pipeline_layout,
            &module,
            gpu.swapchain_format,
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("image bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(base_heightmap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(detail_noise_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(erosion_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: layer_uniform_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            pipeline,
            bind_group,
        }
    }

    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, target_view: &wgpu::TextureView) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("image pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
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
        rpass.draw(0..3, 0..1);
    }
}
