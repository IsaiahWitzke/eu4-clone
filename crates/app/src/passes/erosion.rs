//! World-anchored erosion layer. Reads the heightmap layer's texture and
//! runs a Phacelle erosion filter on it.

use crate::gpu::{GpuContext, LAYER_FORMAT, make_pipeline};
use crate::world_layer::WorldLayer;

const COMMON_WGSL: &str = include_str!("../shaders/common.wgsl");
const LAYER_WGSL: &str = include_str!("../shaders/layer.wgsl");
const NOISE_WGSL: &str = include_str!("../shaders/noise.wgsl");
const FS_WGSL: &str = include_str!("../shaders/terrain.wgsl");

pub fn build(
    gpu: &GpuContext,
    layer_uniform_buf: &wgpu::Buffer,
    base_heightmap_view: &wgpu::TextureView,
) -> WorldLayer {
    let device = &gpu.device;

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("erosion bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("erosion"),
        source: wgpu::ShaderSource::Wgsl(
            format!("{COMMON_WGSL}\n{LAYER_WGSL}\n{NOISE_WGSL}\n{FS_WGSL}").into(),
        ),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("erosion pl"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    let pipeline = make_pipeline(
        device,
        "erosion pipeline",
        &pipeline_layout,
        &module,
        LAYER_FORMAT,
    );

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("erosion bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: layer_uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(base_heightmap_view),
            },
        ],
    });

    WorldLayer::new(gpu, "erosion pass", pipeline, bind_group)
}
