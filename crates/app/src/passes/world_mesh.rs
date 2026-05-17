//! World-mesh pass — the per-frame world disc draw.
//!
//! Renders a `MESH_GRID²` heightmap-displaced grid in world XZ,
//! depth-tested against a swapchain-sized depth buffer (lets the 3D
//! tilt work without overdraw glitches), and fragment-samples the
//! appropriate LoD atlas plus the realm-field for political tinting.
//!
//! Per-fragment cost: one `fwidth`, one atlas `textureSampleLevel`,
//! one `textureLoad` on the realm field. No raymarch, no
//! per-fragment lighting math — that was all paid for at bake time.

use crate::gpu::GpuContext;

const SHADER_WGSL: &str = include_str!("../shaders/world_mesh.wgsl");

/// Mesh grid resolution. Mirrors `MESH_GRID` in `world_mesh.wgsl`.
/// 256 → 65 025 cells, 390 150 vertex invocations per frame.
pub const MESH_GRID: u32 = 256;
pub const MESH_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub fn mesh_vertex_count() -> u32 {
    6 * (MESH_GRID - 1) * (MESH_GRID - 1)
}

pub struct WorldMeshPass {
    /// Water plane pass: flat quad at y=0 covering the world disc.
    /// Renders first so the land pass can alpha-blend on top.
    water_pipeline: wgpu::RenderPipeline,
    /// Land mesh pass: heightmap-displaced MESH_GRID² grid. Renders
    /// second with SrcAlpha/OneMinusSrcAlpha blending so the shoreline
    /// alpha falloff produces screen-pixel-wide coast AA over the
    /// water plane.
    land_pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
}

#[allow(clippy::too_many_arguments)]
pub fn build(
    gpu: &GpuContext,
    camera_uniform_buf: &wgpu::Buffer,
    world_heightmap_view: &wgpu::TextureView,
    atlas_views: [&wgpu::TextureView; 4],
    sampler: &wgpu::Sampler,
    realm_field_view: &wgpu::TextureView,
    water_sdf_view: &wgpu::TextureView,
    bathymetry_view: &wgpu::TextureView,
    target_format: wgpu::TextureFormat,
) -> WorldMeshPass {
    let device = &gpu.device;

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("world_mesh bgl"),
        entries: &[
            // 0: camera uniforms
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
            // 1: world_heightmap (sampled by vs for displacement)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // 2..5: LoD atlases
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
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // 6: shared linear sampler
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // 7: realm field (textureLoad only — no filtering required)
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // 8: water SDF — sampled per fragment for screen-pixel-accurate
            //    coastline AA + shelf gradient. R8Unorm with the
            //    encoding documented in `script/gen-water-sdf`.
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // 9: bathymetry — real ocean depth in metres, R8Unorm over
            //    [0, MAX_DEPTH_M]. Drives the long offshore colour fade
            //    where the SDF saturates. See `script/gen-bathymetry`.
            wgpu::BindGroupLayoutEntry {
                binding: 9,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("world_mesh"),
        source: wgpu::ShaderSource::Wgsl(SHADER_WGSL.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("world_mesh pl"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    // Water plane pipeline. No blending (water is the base layer);
    // depth_write=true so the land pass's depth_test=Less rejects
    // fully-water fragments cleanly.
    let water_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("world_mesh water pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_water"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_water"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: MESH_DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    // Land mesh pipeline. Alpha blending so the fragment shader's
    // shoreline alpha output produces screen-pixel-wide coast AA against
    // whatever the water pass wrote underneath.
    let land_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("world_mesh land pipeline"),
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
                format: target_format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    // Alpha channel just gets replaced — nothing
                    // downstream actually reads framebuffer alpha,
                    // so the simple OVER blend is fine.
                    alpha: wgpu::BlendComponent::OVER,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            // No face culling — the grid can be viewed from below at
            // extreme tilts.
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: MESH_DEPTH_FORMAT,
            // Land mesh does NOT write depth. The water plane already
            // owns the depth at y=0 across the world.
            depth_write_enabled: Some(false),
            // Always pass the depth test. The only gate on whether a
            // land fragment draws is the SDF discard in fs_main.
            //
            // Why not Less / LessEqual: coastal-cliff mesh triangles
            // have one vertex at water level (y=0, heightmap returns
            // 0 for sea) and another inland (y>0). Within such a
            // triangle, the rasterizer interpolates Y across all its
            // fragments. With Less, fragments whose interpolated Y
            // happens to equal the water-plane depth fail the test;
            // adjacent fragments with Y slightly above pass. That
            // produces a torn striped pattern at world zoom where
            // the triangles are only a few pixels across. Always
            // depth-test sidesteps the whole interaction.
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    let bind_group = make_bind_group(
        device,
        &bgl,
        camera_uniform_buf,
        world_heightmap_view,
        atlas_views,
        sampler,
        realm_field_view,
        water_sdf_view,
        bathymetry_view,
    );

    WorldMeshPass {
        water_pipeline,
        land_pipeline,
        bind_group_layout: bgl,
        bind_group,
    }
}

/// Build a fresh bind group. Called from `build()` and from
/// `Renderer::rebuild_world_mesh_bind_group` after any input view swaps
/// (heightmap PNG arriving, atlases rebaking, SDF arriving).
#[allow(clippy::too_many_arguments)]
pub fn make_bind_group(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    camera_uniform_buf: &wgpu::Buffer,
    world_heightmap_view: &wgpu::TextureView,
    atlas_views: [&wgpu::TextureView; 4],
    sampler: &wgpu::Sampler,
    realm_field_view: &wgpu::TextureView,
    water_sdf_view: &wgpu::TextureView,
    bathymetry_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("world_mesh bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(world_heightmap_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(atlas_views[0]),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(atlas_views[1]),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(atlas_views[2]),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(atlas_views[3]),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(realm_field_view),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(water_sdf_view),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: wgpu::BindingResource::TextureView(bathymetry_view),
            },
        ],
    })
}

impl WorldMeshPass {
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
    ) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("world_mesh"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Sky-colour clear behind the mesh; the disc covers
                    // ±2750 km in XZ so anything past the horizon falls
                    // back to this colour.
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.05,
                        g: 0.08,
                        b: 0.12,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_bind_group(0, &self.bind_group, &[]);

        // Water first: flat 6-vert quad at y=0.
        rpass.set_pipeline(&self.water_pipeline);
        rpass.draw(0..6, 0..1);

        // Land second: heightmap-displaced grid, alpha-blended over water.
        rpass.set_pipeline(&self.land_pipeline);
        rpass.draw(0..mesh_vertex_count(), 0..1);
    }
}
