//! `Renderer` owns the GPU context, the camera, the runtime asset
//! textures, the settlement list, and all render passes.
//!
//! Per-frame flow (post-rewrite):
//!   1. push camera uniforms
//!   2. if `realm_field_dirty`, re-bake the realm-influence field
//!   3. if `atlases_dirty`, re-bake all four LoD tile atlases (one
//!      stall after each asset PNG arrives; afterwards stable)
//!   4. `world_mesh` draw — heightmap-displaced grid sampling the
//!      LoD atlases + realm-field tint
//!   5. realm-name labels (SDF glyph overlay)
//!   6. frame-pacing spinner overlay
//!   7. submit / present

use bytemuck;
use web_sys::{HtmlCanvasElement, Performance};

use crate::camera::{CAMERA_FOV_Y_RAD, Camera, CameraUniforms, HOVER_PICK_Y};
use crate::gpu::{GpuContext, compute_canvas_size};
use crate::labels::{self, GlyphAtlas, LayoutSettings};
use crate::passes::{
    realm_field as realm_field_pass, realm_field::RealmFieldPass,
    realm_labels::{self as realm_labels_pass, RealmLabelsPass},
    spinner::{self as spinner_pass, SpinnerPass},
    tile_bake::{self as tile_bake_pass, TileBakePass},
    world_mesh::{self as world_mesh_pass, MESH_DEPTH_FORMAT, WorldMeshPass},
};
use crate::settlements::{
    LoadedSettlements, MAX_SETTLEMENTS, RealmInfo, Settlement, SettlementUniforms, WaterMask,
    default_swiss_settlements, dominant_at_world_xz,
    make_uniform_buf as make_settlement_uniform_buf, nearest_within_km, realm_infos,
};
use crate::tiles::total_tile_count;

/// Width/height of the asset PNGs. Mirrors `WORLD_TEX_SIZE` in
/// `script/_world.py`.
const WORLD_TEX_SIZE: u32 = 4096;

pub struct Renderer {
    pub gpu: GpuContext,
    pub camera: Camera,
    camera_uniform_buf: wgpu::Buffer,

    /// Shared linear sampler used by both bake and per-frame passes.
    sampler: wgpu::Sampler,

    /// World-data textures + views. Start as 1×1 placeholders, swapped
    /// when the async PNG fetches complete. Both the tile-bake pass
    /// (reads heightmap + water + biome) and the world-mesh pass
    /// (reads heightmap for vertex displacement) reference these.
    world_heightmap_tex: wgpu::Texture,
    world_heightmap_view: wgpu::TextureView,
    water_mask_tex: wgpu::Texture,
    water_mask_view: wgpu::TextureView,
    /// CPU copy of the water-mask bytes so the realm rasteriser can
    /// drop offshore cells.
    water_mask_cpu: Option<(Vec<u8>, u32, u32)>,
    biome_mask_tex: wgpu::Texture,
    biome_mask_view: wgpu::TextureView,

    settlements: Vec<Settlement>,
    settlements_uniform_buf: wgpu::Buffer,
    realm_names: std::collections::HashMap<u32, String>,
    realm_infos: Vec<RealmInfo>,

    realm_field: RealmFieldPass,
    realm_field_dirty: bool,

    glyph_atlas: Option<GlyphAtlas>,
    _placeholder_atlas: Option<(wgpu::Texture, wgpu::TextureView, wgpu::Sampler)>,
    realm_labels: RealmLabelsPass,

    performance: Performance,

    spinner: SpinnerPass,
    start_time_ms: f64,

    /// Tile-bake pass + the four LoD atlases it writes into.
    tile_bake: TileBakePass,
    /// True until the next `frame()` re-bakes all atlases. Set by any
    /// of `set_world_heightmap` / `set_water_mask` / `set_biome_mask`,
    /// plus the constructor (initial bake from placeholders).
    atlases_dirty: bool,

    /// Per-frame world draw.
    world_mesh: WorldMeshPass,

    /// Depth attachment for `world_mesh`. Recreated on swapchain resize.
    depth_tex: wgpu::Texture,
    depth_view: wgpu::TextureView,
    depth_size: (u32, u32),
}

impl Renderer {
    pub async fn new(canvas: HtmlCanvasElement) -> Self {
        let gpu = GpuContext::new(canvas).await;

        let performance = web_sys::window()
            .expect("no window")
            .performance()
            .expect("no performance");

        let camera = Camera::new();
        let camera_uniform_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera uniforms"),
            size: std::mem::size_of::<CameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // Placeholder asset textures so bind groups can be built before
        // the PNGs land. The first bake will use them, producing a
        // not-very-interesting atlas; the first PNG arrival re-bakes.
        let (world_heightmap_tex, world_heightmap_view) =
            placeholder_rg8unorm(&gpu, "world_heightmap (placeholder)");
        let (water_mask_tex, water_mask_view) =
            placeholder_r8unorm(&gpu, "water_mask (placeholder)");
        let (biome_mask_tex, biome_mask_view) =
            placeholder_r8unorm(&gpu, "biome_mask (placeholder)");

        // Settlements + realm-field bake target — same as before the rewrite.
        let LoadedSettlements {
            settlements,
            realm_names,
        } = default_swiss_settlements();
        let realm_infos_init = realm_infos(&settlements, &realm_names, None);
        let settlements_uniform_buf =
            make_settlement_uniform_buf(&gpu, "settlements ub");
        let settlement_data = SettlementUniforms::from_slice(&settlements);
        gpu.queue.write_buffer(
            &settlements_uniform_buf,
            0,
            bytemuck::bytes_of(&settlement_data),
        );

        let realm_field = realm_field_pass::build(&gpu, &settlements_uniform_buf);

        // Tile-bake pass — owns the four atlases. Bind groups reference
        // the placeholder views right now; rebuilt when real PNGs land.
        let tile_bake = tile_bake_pass::build(
            &gpu,
            &world_heightmap_view,
            &water_mask_view,
            &biome_mask_view,
            &sampler,
        );

        // World-mesh pass — references the atlases (stable lifetimes)
        // and the heightmap view (rebuilt on PNG arrival).
        let atlas_views_ref: [&wgpu::TextureView; 4] = [
            &tile_bake.atlas_views[0],
            &tile_bake.atlas_views[1],
            &tile_bake.atlas_views[2],
            &tile_bake.atlas_views[3],
        ];
        let world_mesh = world_mesh_pass::build(
            &gpu,
            &camera_uniform_buf,
            &world_heightmap_view,
            atlas_views_ref,
            &sampler,
            &realm_field.view,
            gpu.swapchain_format,
        );

        let (depth_tex, depth_view) = make_depth(&gpu, gpu.width, gpu.height);
        let depth_size = (gpu.width, gpu.height);

        let placeholder_atlas = realm_labels_pass::placeholder_atlas(&gpu);
        let realm_labels = RealmLabelsPass::build(
            &gpu,
            &camera_uniform_buf,
            &placeholder_atlas.1,
            &placeholder_atlas.2,
            gpu.swapchain_format,
        );

        let spinner = spinner_pass::build(&gpu, gpu.swapchain_format);
        let start_time_ms = performance.now();

        web_sys::console::log_1(
            &format!(
                "renderer ready: tile pyramid will bake {} tiles across 4 LoDs once assets load",
                total_tile_count()
            )
            .into(),
        );

        Self {
            gpu,
            camera,
            camera_uniform_buf,
            sampler,
            world_heightmap_tex,
            world_heightmap_view,
            water_mask_tex,
            water_mask_view,
            water_mask_cpu: None,
            biome_mask_tex,
            biome_mask_view,
            settlements,
            settlements_uniform_buf,
            realm_names,
            realm_infos: realm_infos_init,
            realm_field,
            realm_field_dirty: true,
            glyph_atlas: None,
            _placeholder_atlas: Some(placeholder_atlas),
            realm_labels,
            performance,
            spinner,
            start_time_ms,
            tile_bake,
            atlases_dirty: true,
            world_mesh,
            depth_tex,
            depth_view,
            depth_size,
        }
    }

    pub fn canvas(&self) -> &HtmlCanvasElement {
        &self.gpu.canvas
    }

    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }

    #[allow(dead_code)]
    pub fn settlements(&self) -> &[Settlement] {
        &self.settlements
    }

    #[allow(dead_code)]
    pub fn realm_infos(&self) -> &[RealmInfo] {
        &self.realm_infos
    }

    pub fn pick_settlement_at(
        &self,
        mx: f32,
        my: f32,
        css_w: f32,
        css_h: f32,
    ) -> Option<&Settlement> {
        let xz = self.camera.pick_world_xz(mx, my, css_w, css_h, HOVER_PICK_Y)?;
        let view_h_km = 2.0 * self.camera.distance * (CAMERA_FOV_Y_RAD * 0.5).tan();
        let radius_km = (view_h_km / 25.0).max(2.0);
        nearest_within_km(&self.settlements, xz, radius_km).map(|i| &self.settlements[i])
    }

    pub fn handle_resize(&mut self) {
        let (w, h) = compute_canvas_size(&self.gpu.canvas);
        if (w, h) == (self.gpu.width, self.gpu.height) {
            return;
        }
        self.gpu.resize(w, h);
        if self.depth_size != (w, h) {
            let (tex, view) = make_depth(&self.gpu, w, h);
            self.depth_tex = tex;
            self.depth_view = view;
            self.depth_size = (w, h);
        }
    }

    pub fn frame(&mut self) {
        let uniforms = self.camera.to_uniforms(self.gpu.width, self.gpu.height);
        self.gpu
            .queue
            .write_buffer(&self.camera_uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = self.gpu.encoder("frame");

        // Realm-field bake — fires whenever settlements changed.
        if self.realm_field_dirty {
            self.realm_field.render(&mut encoder);
            self.realm_field_dirty = false;
        }

        // Tile atlas bake — fires once after each asset PNG lands.
        if self.atlases_dirty {
            self.tile_bake.render_all(&mut encoder);
            self.atlases_dirty = false;
        }

        // Acquire swapchain frame.
        let (frame, frame_view) = self.gpu.acquire_frame();

        // World-mesh draw — heightmap-displaced grid + atlas sample + tint.
        self.world_mesh
            .render(&mut encoder, &frame_view, &self.depth_view);

        // Realm-name overlay.
        self.realm_labels.render(&mut encoder, &frame_view);

        // Spinner.
        let time_s = ((self.performance.now() - self.start_time_ms) / 1000.0) as f32;
        self.spinner
            .write_uniforms(&self.gpu, time_s, self.gpu.width, self.gpu.height);
        self.spinner.render(&mut encoder, &frame_view);

        self.gpu.submit(encoder);

        frame.present();
    }

    /// Upload a fresh world heightmap. Rebuilds the tile_bake and
    /// world_mesh bind groups (they referenced the placeholder view)
    /// and marks the atlases dirty so the next frame re-bakes.
    pub fn set_world_heightmap(&mut self, width: u32, height: u32, bytes: &[u8]) {
        if (width, height) != (WORLD_TEX_SIZE, WORLD_TEX_SIZE) {
            web_sys::console::warn_1(
                &format!(
                    "world heightmap size mismatch: got {width}x{height}, expected \
                     {WORLD_TEX_SIZE}\u{00d7}{WORLD_TEX_SIZE}"
                )
                .into(),
            );
        }
        let tex = upload_world_texture(
            &self.gpu,
            "world_heightmap",
            wgpu::TextureFormat::Rg8Unorm,
            width,
            height,
            bytes,
            2,
        );
        let view = tex.create_view(&Default::default());
        self.world_heightmap_tex = tex;
        self.world_heightmap_view = view;

        self.rebuild_bake_bind_groups();
        self.rebuild_world_mesh_bind_group();
        self.atlases_dirty = true;
    }

    /// Upload a fresh water mask + CPU-side copy for the realm rasteriser.
    pub fn set_water_mask(&mut self, width: u32, height: u32, bytes: &[u8]) {
        if (width, height) != (WORLD_TEX_SIZE, WORLD_TEX_SIZE) {
            web_sys::console::warn_1(
                &format!(
                    "water mask size mismatch: got {width}x{height}, expected \
                     {WORLD_TEX_SIZE}\u{00d7}{WORLD_TEX_SIZE}"
                )
                .into(),
            );
        }
        let tex = upload_world_texture(
            &self.gpu,
            "water_mask",
            wgpu::TextureFormat::R8Unorm,
            width,
            height,
            bytes,
            1,
        );
        let view = tex.create_view(&Default::default());
        self.water_mask_tex = tex;
        self.water_mask_view = view;

        self.rebuild_bake_bind_groups();
        self.atlases_dirty = true;

        self.water_mask_cpu = Some((bytes.to_vec(), width, height));
        self.recompute_realm_infos();
        self.rebuild_realm_label_vertices();
    }

    pub fn update_hover(&mut self, mx: f32, my: f32, css_w: f32, css_h: f32) -> bool {
        let (prev_realm, prev_city) = (self.camera.hovered_pid, self.camera.hovered_city);
        let (realm_id, city_id) = self
            .camera
            .pick_world_xz(mx, my, css_w, css_h, HOVER_PICK_Y)
            .map(|xz| {
                let hit = dominant_at_world_xz(&self.settlements, xz);
                if hit.strength < 0.5 {
                    (0_u32, 0_u32)
                } else {
                    (hit.realm_id + 1, hit.city_idx + 1)
                }
            })
            .unwrap_or((0, 0));
        let changed = realm_id != prev_realm || city_id != prev_city;
        if changed {
            self.camera.hovered_pid = realm_id;
            self.camera.hovered_city = city_id;
        }
        changed
    }

    /// Upload a fresh biome mask.
    pub fn set_biome_mask(&mut self, width: u32, height: u32, bytes: &[u8]) {
        if (width, height) != (WORLD_TEX_SIZE, WORLD_TEX_SIZE) {
            web_sys::console::warn_1(
                &format!(
                    "biome mask size mismatch: got {width}x{height}, expected \
                     {WORLD_TEX_SIZE}\u{00d7}{WORLD_TEX_SIZE}"
                )
                .into(),
            );
        }
        let tex = upload_world_texture(
            &self.gpu,
            "biome_mask",
            wgpu::TextureFormat::R8Unorm,
            width,
            height,
            bytes,
            1,
        );
        let view = tex.create_view(&Default::default());
        self.biome_mask_tex = tex;
        self.biome_mask_view = view;

        self.rebuild_bake_bind_groups();
        self.atlases_dirty = true;
    }

    fn rebuild_bake_bind_groups(&mut self) {
        self.tile_bake.rebuild_bind_groups(
            &self.gpu.device,
            &self.world_heightmap_view,
            &self.water_mask_view,
            &self.biome_mask_view,
            &self.sampler,
        );
    }

    fn rebuild_world_mesh_bind_group(&mut self) {
        let atlas_views_ref: [&wgpu::TextureView; 4] = [
            &self.tile_bake.atlas_views[0],
            &self.tile_bake.atlas_views[1],
            &self.tile_bake.atlas_views[2],
            &self.tile_bake.atlas_views[3],
        ];
        self.world_mesh.bind_group = world_mesh_pass::make_bind_group(
            &self.gpu.device,
            &self.world_mesh.bind_group_layout,
            &self.camera_uniform_buf,
            &self.world_heightmap_view,
            atlas_views_ref,
            &self.sampler,
            &self.realm_field.view,
        );
    }

    #[allow(dead_code)]
    pub fn upload_settlements(&mut self) {
        let data = SettlementUniforms::from_slice(&self.settlements);
        self.gpu
            .queue
            .write_buffer(&self.settlements_uniform_buf, 0, bytemuck::bytes_of(&data));
        self.realm_field_dirty = true;
    }

    pub fn set_settlements(&mut self, loaded: LoadedSettlements) {
        let LoadedSettlements {
            mut settlements,
            realm_names,
        } = loaded;
        if settlements.len() > MAX_SETTLEMENTS {
            web_sys::console::warn_1(
                &format!(
                    "set_settlements: got {} settlements, truncating to {}",
                    settlements.len(), MAX_SETTLEMENTS,
                )
                .into(),
            );
            settlements.truncate(MAX_SETTLEMENTS);
        }
        self.settlements = settlements;
        self.realm_names = realm_names;
        self.recompute_realm_infos();
        self.upload_settlements();
        self.rebuild_realm_label_vertices();
    }

    fn recompute_realm_infos(&mut self) {
        let wm = self.water_mask_cpu.as_ref().map(|(bytes, w, h)| WaterMask {
            bytes: bytes.as_slice(),
            width: *w,
            height: *h,
        });
        self.realm_infos = realm_infos(&self.settlements, &self.realm_names, wm.as_ref());
    }

    pub fn set_glyph_atlas(
        &mut self,
        json_text: &str,
        png_width: u32,
        png_height: u32,
        rgba_bytes: &[u8],
    ) -> Result<(), String> {
        let atlas = GlyphAtlas::build(
            &self.gpu, json_text, png_width, png_height, rgba_bytes,
        )?;
        self.realm_labels.rebuild_bind_group(
            &self.gpu,
            &self.camera_uniform_buf,
            &atlas.view,
            &atlas.sampler,
        );
        self.glyph_atlas = Some(atlas);
        self._placeholder_atlas = None;
        self.rebuild_realm_label_vertices();
        Ok(())
    }

    fn rebuild_realm_label_vertices(&mut self) {
        let Some(atlas) = self.glyph_atlas.as_ref() else {
            return;
        };
        let verts = labels::build_label_vertices(
            atlas,
            &self.realm_infos,
            &LayoutSettings::default(),
        );
        self.realm_labels.set_vertices(&self.gpu, &verts);
    }
}

// ---- Helpers ---------------------------------------------------------------

fn placeholder_rg8unorm(gpu: &GpuContext, label: &str) -> (wgpu::Texture, wgpu::TextureView) {
    placeholder_texture(gpu, label, wgpu::TextureFormat::Rg8Unorm, &[0u8, 0u8])
}

fn placeholder_r8unorm(gpu: &GpuContext, label: &str) -> (wgpu::Texture, wgpu::TextureView) {
    placeholder_texture(gpu, label, wgpu::TextureFormat::R8Unorm, &[0u8])
}

fn placeholder_texture(
    gpu: &GpuContext,
    label: &str,
    format: wgpu::TextureFormat,
    bytes: &[u8],
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bytes.len() as u32),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let view = tex.create_view(&Default::default());
    (tex, view)
}

fn upload_world_texture(
    gpu: &GpuContext,
    label: &str,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    bytes: &[u8],
    bytes_per_pixel: u32,
) -> wgpu::Texture {
    let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * bytes_per_pixel),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    tex
}

/// Build a depth attachment sized to the current swapchain for the
/// world-mesh pass.
fn make_depth(gpu: &GpuContext, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("world_mesh depth"),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: MESH_DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    (tex, view)
}
