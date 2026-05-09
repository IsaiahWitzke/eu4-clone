//! World-anchored painted heightmap layer. No external texture inputs — its
//! shader just consumes `LayerUniforms` to know which world AABB to paint.

use crate::gpu::{GpuContext, make_pipeline};
use crate::world_layer::WorldLayer;

const COMMON_WGSL: &str = include_str!("../shaders/common.wgsl");
const LAYER_WGSL: &str = include_str!("../shaders/layer.wgsl");
const FS_WGSL: &str = include_str!("../shaders/base_heightmap.wgsl");

pub fn new(gpu: &GpuContext) -> WorldLayer {
    let device = &gpu.device;

    // Bind-group layout: just LayerUniforms at binding 0.
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("base_heightmap bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("base_heightmap"),
        source: wgpu::ShaderSource::Wgsl(
            format!("{COMMON_WGSL}\n{LAYER_WGSL}\n{FS_WGSL}").into(),
        ),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("base_heightmap pl"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    let pipeline = make_pipeline(
        device,
        "base_heightmap pipeline",
        &pipeline_layout,
        &module,
        wgpu::TextureFormat::Rgba32Float,
    );

    let layer_uniform_buf = WorldLayer::make_layer_uniform_buf(gpu, "base_heightmap layer ub");

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("base_heightmap bg"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: layer_uniform_buf.as_entire_binding(),
        }],
    });

    WorldLayer::new(
        gpu,
        "base_heightmap pass",
        pipeline,
        bind_group,
        layer_uniform_buf,
    )
}
