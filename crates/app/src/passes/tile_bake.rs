//! Tile-bake pass — populates the four LoD atlases with shaded terrain.
//!
//! Architecture:
//!   * One pipeline shared across all LoDs (same shader, same bindings).
//!   * One bind group per LoD; the only thing that differs is the
//!     `BakeUniforms.tiles_per_side` value the vertex shader uses to
//!     place each instance inside the atlas.
//!   * One render pass per LoD (targets that LoD's atlas), issuing a
//!     single instanced draw with `LOD_TILES[lod]²` instances. Six
//!     vertices per instance form one quad.
//!
//! Total bake cost: 4 render passes, `1 + 16 + 256 + 1024 = 1297`
//! instances, ~17 million fragments at the finest LoD. One-time stall;
//! the per-frame world-mesh draw afterwards is just an atlas sample.

use bytemuck::{Pod, Zeroable};

use crate::gpu::GpuContext;
use crate::tiles::{LOD_COUNT, LOD_TILES, atlas_dim};

const SHADER_WGSL: &str = include_str!("../shaders/tile_bake.wgsl");

/// Pixel format for every atlas. sRGB means the bake shader's linear
/// colour output gets gamma-encoded on write, and `textureSample` in
/// `world_mesh` gamma-decodes back to linear before the swapchain's
/// sRGB encoding kicks in — round-trips cleanly with the sRGB
/// swapchain that `GpuContext::new` picks.
pub const ATLAS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// Mirrors `BakeUniforms` in `tile_bake.wgsl`. 16 bytes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BakeUniforms {
    tiles_per_side: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct TileBakePass {
    pipeline: wgpu::RenderPipeline,
    /// Bind-group layout retained so the renderer can rebuild bind
    /// groups when input textures swap (PNG arrivals).
    bgl: wgpu::BindGroupLayout,
    /// One bind group per LoD. Differs only in which `BakeUniforms`
    /// buffer it references (each LoD has its own tiles_per_side).
    bind_groups: [wgpu::BindGroup; LOD_COUNT],
    /// One uniform buffer per LoD; written once at construction.
    uniform_bufs: [wgpu::Buffer; LOD_COUNT],
    /// One atlas texture per LoD, plus its view.
    pub atlases: [wgpu::Texture; LOD_COUNT],
    pub atlas_views: [wgpu::TextureView; LOD_COUNT],
}

#[allow(clippy::too_many_arguments)]
pub fn build(
    gpu: &GpuContext,
    world_heightmap_view: &wgpu::TextureView,
    water_mask_view: &wgpu::TextureView,
    biome_mask_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> TileBakePass {
    let device = &gpu.device;

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tile_bake bgl"),
        entries: &[
            // 0: BakeUniforms
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
            // 1: world_heightmap
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
            // 2: water_mask
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // 3: biome_mask
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // 4: sampler
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("tile_bake"),
        source: wgpu::ShaderSource::Wgsl(SHADER_WGSL.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("tile_bake pl"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("tile_bake pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: ATLAS_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    // Build per-LoD: atlas texture + uniform buffer + bind group.
    let atlases: [wgpu::Texture; LOD_COUNT] = std::array::from_fn(|lod| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(match lod {
                0 => "atlas LoD 0",
                1 => "atlas LoD 1",
                2 => "atlas LoD 2",
                _ => "atlas LoD 3",
            }),
            size: wgpu::Extent3d {
                width: atlas_dim(lod),
                height: atlas_dim(lod),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: ATLAS_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    });
    let atlas_views: [wgpu::TextureView; LOD_COUNT] =
        std::array::from_fn(|lod| atlases[lod].create_view(&Default::default()));

    let uniform_bufs: [wgpu::Buffer; LOD_COUNT] = std::array::from_fn(|lod| {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile_bake ub"),
            size: std::mem::size_of::<BakeUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let u = BakeUniforms {
            tiles_per_side: LOD_TILES[lod],
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        gpu.queue.write_buffer(&buf, 0, bytemuck::bytes_of(&u));
        buf
    });

    let bind_groups: [wgpu::BindGroup; LOD_COUNT] = std::array::from_fn(|lod| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tile_bake bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_bufs[lod].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(world_heightmap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(water_mask_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(biome_mask_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    });

    TileBakePass {
        pipeline,
        bgl,
        bind_groups,
        uniform_bufs,
        atlases,
        atlas_views,
    }
}

impl TileBakePass {
    /// Rebuild the per-LoD bind groups against fresh texture views.
    /// Call this whenever any of the source textures (heightmap, water
    /// mask, biome mask) has been replaced — the bind groups hold
    /// references to the old views and need to be repointed.
    pub fn rebuild_bind_groups(
        &mut self,
        device: &wgpu::Device,
        world_heightmap_view: &wgpu::TextureView,
        water_mask_view: &wgpu::TextureView,
        biome_mask_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) {
        self.bind_groups = std::array::from_fn(|lod| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tile_bake bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_bufs[lod].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(world_heightmap_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(water_mask_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(biome_mask_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            })
        });
    }

    /// Bake all four LoDs into their atlases. One render pass per LoD,
    /// one instanced draw per pass. Called only when assets change.
    pub fn render_all(&self, encoder: &mut wgpu::CommandEncoder) {
        for lod in 0..LOD_COUNT {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(match lod {
                    0 => "tile_bake LoD 0",
                    1 => "tile_bake LoD 1",
                    2 => "tile_bake LoD 2",
                    _ => "tile_bake LoD 3",
                }),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.atlas_views[lod],
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_groups[lod], &[]);
            let instances = LOD_TILES[lod] * LOD_TILES[lod];
            rpass.draw(0..6, 0..instances);
        }
    }
}
