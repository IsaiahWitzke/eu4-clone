//! World-anchored heightmap layer. Samples the real Switzerland heightmap
//! texture (`world_heightmap`) at each fragment's world XZ position and
//! writes it into the cached layer texture.
//!
//! `bgl()` and `make_bind_group()` are exposed so the renderer can rebuild
//! the bind group whenever the world heightmap texture is swapped out (e.g.
//! when the async PNG fetch completes).

use crate::gpu::{GpuContext, make_pipeline};
use crate::world_layer::WorldLayer;

const COMMON_WGSL: &str = include_str!("../shaders/common.wgsl");
const LAYER_WGSL: &str = include_str!("../shaders/layer.wgsl");
const WORLD_WGSL: &str = include_str!("../shaders/world.wgsl");
const FS_WGSL: &str = include_str!("../shaders/base_heightmap.wgsl");

/// Bind-group layout: LayerUniforms at 0, world heightmap texture at 1,
/// a filtering sampler at 2.
pub fn bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("base_heightmap bgl"),
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
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

pub fn make_bind_group(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    layer_uniform_buf: &wgpu::Buffer,
    world_heightmap_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("base_heightmap bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: layer_uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(world_heightmap_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

pub fn build(
    gpu: &GpuContext,
    bgl: &wgpu::BindGroupLayout,
    layer_uniform_buf: &wgpu::Buffer,
    world_heightmap_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> WorldLayer {
    let device = &gpu.device;

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("base_heightmap"),
        source: wgpu::ShaderSource::Wgsl(
            format!("{COMMON_WGSL}\n{LAYER_WGSL}\n{WORLD_WGSL}\n{FS_WGSL}").into(),
        ),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("base_heightmap pl"),
        bind_group_layouts: &[Some(bgl)],
        immediate_size: 0,
    });

    let pipeline = make_pipeline(
        device,
        "base_heightmap pipeline",
        &pipeline_layout,
        &module,
        wgpu::TextureFormat::Rgba32Float,
    );

    let bind_group =
        make_bind_group(device, bgl, layer_uniform_buf, world_heightmap_view, sampler);

    WorldLayer::new(gpu, "base_heightmap pass", pipeline, bind_group)
}
