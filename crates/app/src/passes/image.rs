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
//!  10 — settlements (uniform; influence-field point sources, retained for
//!         settlement markers / future per-frame use; the realm field
//!         itself is now read from binding 11)
//!  11 — realm_field (Rgba16Float; pre-baked world-space realm influence
//!         field. R = realm_id, G = alpha, B = contested-1, A = city_idx.
//!         Sampled with textureLoad: realm_id and city_idx are categorical
//!         and the smooth GB channels are at 2048² over a 5500 km world,
//!         which is plenty of resolution to skip filtering.)
//!  13 — world_heightmap (Rg8Unorm 4096²; full-resolution world heightmap
//!         with the 16-bit elevation split across R+G. Sampled by
//!         `vs_mesh` to displace mesh vertices at the source data's
//!         native ~1.34 km/px resolution instead of going through the
//!         1024² cached `terrain` layer.)
//!
//! Bindings 8, 9, and 12 used to live here but were retired:
//!   * 8/9 (province_mask, border_sdf) when settlement influence fields
//!     became the unit of control;
//!   * 12 (realm_labels) when the Canvas2D-painted name overlay was
//!     replaced by the in-engine SDF glyph atlas pass
//!     (see `passes::realm_labels`).
//! The binding numbers stay skipped rather than getting renumbered.

use crate::gpu::{GpuContext, make_pipeline};

const COMMON_WGSL: &str = include_str!("../shaders/common.wgsl");
const CAMERA_WGSL: &str = include_str!("../shaders/camera.wgsl");
const LAYER_WGSL: &str = include_str!("../shaders/layer.wgsl");
const WORLD_WGSL: &str = include_str!("../shaders/world.wgsl");
const NOISE_WGSL: &str = include_str!("../shaders/noise.wgsl");
const FS_WGSL: &str = include_str!("../shaders/image.wgsl");

/// Cells per side of the heightmap-displacement mesh (Option 3 path).
/// Must match `MESH_GRID` in `shaders/image.wgsl`. Vertex count per draw =
/// `6 * (MESH_GRID - 1)^2`.
///
/// 1024 picked to match the cached layer's texel resolution at default
/// zoom (~3 km per cell, same as one `terrain` texel), so the mesh
/// captures the full detail of the cached heightmap. ~6.3M vertex-
/// shader invocations per frame; cheap on M-class GPUs (<1 ms
/// measured). Drop to 512 if the vertex stage ever becomes a bottleneck;
/// bump to 2048 if you want sub-texel detail (will require also sampling
/// the world heightmap directly in `vs_mesh` to actually have extra
/// detail to capture).
pub const MESH_GRID: u32 = 1024;
/// Depth attachment format for the mesh pipeline. `Depth32Float` is
/// widely supported and doesn't need an extension.
pub const MESH_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Which rendering path the image pass should use for a given frame.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum RenderMode {
    /// Original fullscreen-quad raymarch (single triangle, no depth).
    #[default]
    Raymarch,
    /// Heightmap mesh rasterization (~390k verts, depth-tested).
    Mesh,
}

impl RenderMode {
    pub fn toggled(self) -> Self {
        match self {
            Self::Raymarch => Self::Mesh,
            Self::Mesh => Self::Raymarch,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::Raymarch => "raymarch",
            Self::Mesh => "mesh",
        }
    }
}

pub struct ImagePass {
    /// Original fullscreen-quad raymarch pipeline (`vs_main` + `fs_main`).
    pub pipeline: wgpu::RenderPipeline,
    /// Heightmap-mesh rasterizer (`vs_mesh` + `fs_mesh`). Depth-tested.
    pub pipeline_mesh: wgpu::RenderPipeline,
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
    let texture_entry =
        |binding: u32, filterable: bool, vis: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
    let f = wgpu::ShaderStages::FRAGMENT;
    // VERTEX_FRAGMENT: the mesh vertex shader (`vs_mesh`) samples the
    // `terrain` layer to displace its grid vertices, and uses the shared
    // linear sampler to do it. Everything else stays fragment-only.
    let vf = wgpu::ShaderStages::VERTEX_FRAGMENT;

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("image bgl"),
        entries: &[
            uniform_entry(0),
            texture_entry(1, true, f),    // base_heightmap   (filterable, sampled)
            texture_entry(2, true, f),    // detail_noise     (filterable, sampled)
            texture_entry(3, true, vf),   // terrain          (vs_mesh + fs)
            uniform_entry(4),
            // water_mask: also sampled by `vs_mesh` so submarine vertices
            // can be snapped to a flat sea level (otherwise bathymetry
            // pokes through under the water shading).
            texture_entry(5, true, vf),   // water_mask       (vs_mesh + fs)
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: vf,            // shared linear sampler (vs_mesh + fs)
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // Biome IDs are categorical: never interpolate. `filterable: false`
            // is the type-system enforcement of "don't bind this with a
            // Filtering sampler"; we only ever sample it via textureLoad.
            texture_entry(7, false, f), // biome_mask       (NEAREST only)
            // Bindings 8, 9, 12 intentionally absent — see module-level
            // doc for the history. Numbers stay skipped rather than
            // getting renumbered.
            uniform_entry(10),          // settlements     (influence-field sources)
            texture_entry(11, false, f), // realm_field    (pre-baked field; textureLoad)
            // world_heightmap: Rg8Unorm with 16-bit elevation split into
            // high/low bytes. Sampled by `vs_mesh` to get full-resolution
            // displacement (~1.34 km/px vs the layer cache's ~3 km/px
            // downsampled version) — the difference between visibly-
            // faceted ridges and smooth ones at default zoom.
            texture_entry(13, true, vf), // world_heightmap (vs_mesh)
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
    settlements_uniform_buf: &wgpu::Buffer,
    realm_field_view: &wgpu::TextureView,
    world_heightmap_view: &wgpu::TextureView,
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
                binding: 10,
                resource: settlements_uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: wgpu::BindingResource::TextureView(realm_field_view),
            },
            wgpu::BindGroupEntry {
                binding: 13,
                resource: wgpu::BindingResource::TextureView(world_heightmap_view),
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
        settlements_uniform_buf: &wgpu::Buffer,
        realm_field_view: &wgpu::TextureView,
        world_heightmap_view: &wgpu::TextureView,
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

        // Mesh pipeline: vs_mesh+fs_mesh, depth-tested, no face culling
        // (the grid can be viewed from below at extreme tilts).
        let pipeline_mesh =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("image mesh pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_mesh"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_mesh"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.swapchain_format,
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
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            });

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
            settlements_uniform_buf,
            realm_field_view,
            world_heightmap_view,
        );

        Self {
            pipeline,
            pipeline_mesh,
            bind_group,
        }
    }

    /// Total vertex count for one `vs_mesh` draw call.
    pub fn mesh_vertex_count() -> u32 {
        6 * (MESH_GRID - 1) * (MESH_GRID - 1)
    }

    /// Render via the selected path. The `depth_view` is only consulted
    /// when `mode == Mesh`; callers may pass anything (or a 1x1 dummy)
    /// for the raymarch path.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        mode: RenderMode,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let color_attachment = wgpu::RenderPassColorAttachment {
            view: target_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        };
        // Mesh needs a depth attachment; raymarch ignores depth entirely.
        let depth_attachment = match mode {
            RenderMode::Mesh => Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            RenderMode::Raymarch => None,
        };
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("image pass"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: depth_attachment,
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_bind_group(0, &self.bind_group, &[]);
        match mode {
            RenderMode::Raymarch => {
                rpass.set_pipeline(&self.pipeline);
                rpass.draw(0..3, 0..1);
            }
            RenderMode::Mesh => {
                rpass.set_pipeline(&self.pipeline_mesh);
                rpass.draw(0..Self::mesh_vertex_count(), 0..1);
            }
        }
    }
}
