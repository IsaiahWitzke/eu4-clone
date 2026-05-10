//! Image pass — renders the final raymarched terrain to the swapchain.
//!
//! Bindings:
//!   0 — camera uniforms (Uniforms)
//!   1 — base_heightmap layer (filterable Rgba16Float)
//!   2 — detail_noise layer (filterable Rgba16Float)
//!   3 — terrain (erosion) layer (filterable Rgba16Float)
//!   4 — LayerUniforms (shared covered AABB)
//!   5 — water_mask (R8Unorm; soft-edged via filtering)
//!   6 — linear sampler used for the three filterable layer textures
//!   7 — biome_mask (R8Unorm; sampled with textureLoad, NOT filtered — biome
//!         IDs are categorical, interpolating produces fictional intermediate
//!         biome IDs at borders)
//!   8 — province_mask (Rg8Unorm carrying a 16-bit big-endian unsigned ID;
//!         categorical, sampled with textureLoad)
//!   9 — border_sdf (R8Unorm; distance-to-nearest-border, FILTERABLE —
//!         bilinearly sampled and smoothstep'd in the shader for smooth
//!         tunable borders)

use crate::gpu::{GpuContext, make_pipeline};

const COMMON_WGSL: &str = include_str!("../shaders/common.wgsl");
const CAMERA_WGSL: &str = include_str!("../shaders/camera.wgsl");
const LAYER_WGSL: &str = include_str!("../shaders/layer.wgsl");
const WORLD_WGSL: &str = include_str!("../shaders/world.wgsl");
const NOISE_WGSL: &str = include_str!("../shaders/noise.wgsl");
const FS_WGSL: &str = include_str!("../shaders/image.wgsl");

pub struct ImagePass {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
}

pub fn bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
    let texture_entry = |binding: u32, filterable: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("image bgl"),
        entries: &[
            uniform_entry(0),
            texture_entry(1, true),  // base_heightmap   (filterable, sampled)
            texture_entry(2, true),  // detail_noise     (filterable, sampled)
            texture_entry(3, true),  // terrain          (filterable, sampled)
            uniform_entry(4),
            texture_entry(5, true),  // water_mask       (filterable for soft shores)
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // Biome IDs are categorical: never interpolate. `filterable: false`
            // is the type-system enforcement of "don't bind this with a
            // Filtering sampler"; we only ever sample it via textureLoad.
            texture_entry(7, false), // biome_mask       (NEAREST only)
            texture_entry(8, false), // province_mask    (NEAREST only)
            texture_entry(9, true),  // border_sdf       (filterable; smooth borders)
        ],
    })
}

#[allow(clippy::too_many_arguments)]
pub fn make_bind_group(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    camera_uniform_buf: &wgpu::Buffer,
    base_heightmap_view: &wgpu::TextureView,
    detail_noise_view: &wgpu::TextureView,
    erosion_view: &wgpu::TextureView,
    layer_uniform_buf: &wgpu::Buffer,
    water_mask_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    biome_mask_view: &wgpu::TextureView,
    province_mask_view: &wgpu::TextureView,
    border_sdf_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("image bg"),
        layout: bgl,
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
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(water_mask_view),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(biome_mask_view),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(province_mask_view),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: wgpu::BindingResource::TextureView(border_sdf_view),
            },
        ],
    })
}

impl ImagePass {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        gpu: &GpuContext,
        bgl: &wgpu::BindGroupLayout,
        camera_uniform_buf: &wgpu::Buffer,
        base_heightmap_view: &wgpu::TextureView,
        detail_noise_view: &wgpu::TextureView,
        erosion_view: &wgpu::TextureView,
        layer_uniform_buf: &wgpu::Buffer,
        water_mask_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        biome_mask_view: &wgpu::TextureView,
        province_mask_view: &wgpu::TextureView,
        border_sdf_view: &wgpu::TextureView,
    ) -> Self {
        let device = &gpu.device;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("image"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{COMMON_WGSL}\n{CAMERA_WGSL}\n{LAYER_WGSL}\n{WORLD_WGSL}\n\
                     {NOISE_WGSL}\n{FS_WGSL}"
                )
                .into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("image pl"),
            bind_group_layouts: &[Some(bgl)],
            immediate_size: 0,
        });

        let pipeline = make_pipeline(
            device,
            "image pipeline",
            &pipeline_layout,
            &module,
            gpu.swapchain_format,
        );

        let bind_group = make_bind_group(
            device,
            bgl,
            camera_uniform_buf,
            base_heightmap_view,
            detail_noise_view,
            erosion_view,
            layer_uniform_buf,
            water_mask_view,
            sampler,
            biome_mask_view,
            province_mask_view,
            border_sdf_view,
        );

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
