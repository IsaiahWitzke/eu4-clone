//! World-anchored detail-noise layer. Same shape as `base_heightmap`, but the
//! shader also pulls in `noise.wgsl` for the analytic gradient noise.

use crate::gpu::{GpuContext, make_pipeline};
use crate::world_layer::WorldLayer;

const COMMON_WGSL: &str = include_str!("../shaders/common.wgsl");
const LAYER_WGSL: &str = include_str!("../shaders/layer.wgsl");
const NOISE_WGSL: &str = include_str!("../shaders/noise.wgsl");
const FS_WGSL: &str = include_str!("../shaders/detail_noise.wgsl");

pub fn new(gpu: &GpuContext) -> WorldLayer {
    let device = &gpu.device;

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("detail_noise bgl"),
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
        label: Some("detail_noise"),
        source: wgpu::ShaderSource::Wgsl(
            format!("{COMMON_WGSL}\n{LAYER_WGSL}\n{NOISE_WGSL}\n{FS_WGSL}").into(),
        ),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("detail_noise pl"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    let pipeline = make_pipeline(
        device,
        "detail_noise pipeline",
        &pipeline_layout,
        &module,
        wgpu::TextureFormat::Rgba32Float,
    );

    let layer_uniform_buf = WorldLayer::make_layer_uniform_buf(gpu, "detail_noise layer ub");

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("detail_noise bg"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: layer_uniform_buf.as_entire_binding(),
        }],
    });

    WorldLayer::new(
        gpu,
        "detail_noise pass",
        pipeline,
        bind_group,
        layer_uniform_buf,
    )
}
