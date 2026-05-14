//! `Renderer` ties together the GPU context, the camera, and the four passes
//! (three world-anchored layers + the screen-space image pass). The cache for
//! the world layers (which world AABB the layer textures cover) lives here as
//! a single shared piece of state, so all three layers stay in sync.
//!
//! Two PNG textures (the Switzerland heightmap and water mask) are loaded
//! asynchronously by `lib.rs`. Until they arrive, 1×1 placeholder textures
//! keep the bind groups valid. `set_world_heightmap` / `set_water_mask` are
//! called when the bytes land — they upload, swap views, rebuild the
//! bind groups that referenced the old views, and invalidate the world-layer
//! cache so the next frame regenerates from real data.

use bytemuck;
use web_sys::{HtmlCanvasElement, Performance};

use crate::camera::{Aabb2, CAMERA_FOV_Y_RAD, Camera, CameraUniforms, HOVER_PICK_Y};
use crate::gpu::{GpuContext, compute_canvas_size};
use crate::gpu_timing::{GpuTimer, Section};
use crate::labels::{self, GlyphAtlas, LayoutSettings};
use crate::passes::{
    base_heightmap, detail_noise, erosion, image as image_pass,
    image::{ImagePass, MESH_DEPTH_FORMAT, RenderMode},
    realm_field as realm_field_pass, realm_field::RealmFieldPass,
    realm_labels::{self as realm_labels_pass, RealmLabelsPass},
};
use crate::perf::{self, Span};
use crate::settlements::{
    LoadedSettlements, MAX_SETTLEMENTS, RealmInfo, Settlement, SettlementUniforms, WaterMask,
    default_swiss_settlements, dominant_at_world_xz,
    make_uniform_buf as make_settlement_uniform_buf, nearest_within_km, realm_infos,
};
use crate::world_layer::{LAYER_SIZE, LayerUniforms, WorldLayer, make_layer_uniform_buf};

/// Padding factor for the world-layer cache: re-render only when the camera
/// has moved past `view_aabb × PAD`. Bumped from 2.0 → 4.0 so the cached
/// region covers 4× the visible AABB instead of 2×. You can pan ~1.5 view
/// AABBs before triggering a regen (vs ~0.5 before), so per-frame layer
/// regen cost — the dominant source of pan-stutter at certain zoom levels
/// — fires ~4× less often. Cost per texel doubles (each 1024² layer now
/// covers 4× more world), but `vs_mesh` samples `world_heightmap`
/// directly so the mesh geometry isn't affected; only the layer-derived
/// shading paths (normals, water blend, etc.) take the small fidelity
/// hit. Worth it for the pan smoothness.
const PAD: f32 = 4.0;

/// Width/height of the asset textures (heightmap.png + water_mask.png).
/// Mirrors `WORLD_TEX_SIZE` in `script/_world.py`.
const WORLD_TEX_SIZE: u32 = 4096;

// `WORLD_BOUNDS_HALF` lives in `shaders/world.wgsl`. CPU code no longer
// needs it now that hover picks via the settlement field directly.

pub struct Renderer {
    pub gpu: GpuContext,
    pub camera: Camera,
    camera_uniform_buf: wgpu::Buffer,

    /// Shared LayerUniforms buffer used by every world layer's bind group +
    /// the image pass's bind group. The renderer rewrites it whenever the
    /// cache region changes.
    layer_uniform_buf: wgpu::Buffer,
    /// World AABB the cached layer textures currently cover. None = invalid.
    layer_covered: Option<Aabb2>,

    /// Shared linear sampler (filtering Float). Used to bilerp the world
    /// heightmap.
    sampler: wgpu::Sampler,

    /// World-data textures + their views. Start as 1×1 placeholders, swapped
    /// when the async PNG fetch completes.
    world_heightmap_tex: wgpu::Texture,
    world_heightmap_view: wgpu::TextureView,
    water_mask_tex: wgpu::Texture,
    water_mask_view: wgpu::TextureView,
    /// CPU copy of the water-mask bytes, kept around so the realm
    /// rasteriser can filter out offshore cells (otherwise coastal
    /// cities project their influence into the sea and the label
    /// drifts off the map). `None` until `set_water_mask` runs;
    /// `realm_infos` falls back to a land-everywhere assumption
    /// while it's empty.
    water_mask_cpu: Option<(Vec<u8>, u32, u32)>,
    /// Biome IDs (R8Unorm; pixel value = biome number 0..14, where 0 is
    /// "no biome / not classified"). Sampled with textureLoad in the shader.
    biome_mask_tex: wgpu::Texture,
    biome_mask_view: wgpu::TextureView,

    /// Settlement list — the influence-field point sources. The CPU keeps
    /// the full struct (with names, etc.) for hover lookup + future editing;
    /// `settlements_uniform_buf` mirrors a packed GPU view of it.
    settlements: Vec<Settlement>,
    settlements_uniform_buf: wgpu::Buffer,
    /// realm_id → display name. Populated by the cities.json loader (or
    /// the hardcoded Swiss seed) and consumed by [`RealmInfo`] for the
    /// label UI. Sparse: missing entries fall back to `"Realm {id}"`.
    realm_names: std::collections::HashMap<u32, String>,
    /// Per-realm summary (centroid + name + total strength), recomputed
    /// whenever `set_settlements` runs. The HTML overlay reads this each
    /// frame to position country labels.
    realm_infos: Vec<RealmInfo>,

    /// Pre-baked realm-influence field. Re-baked whenever the settlement
    /// list changes; the image pass reads it via `textureLoad` instead of
    /// looping over settlements per fragment.
    realm_field: RealmFieldPass,

    /// True when the realm-field bake is dirty and needs a re-render at
    /// the start of the next frame. Set by `set_settlements` /
    /// `upload_settlements`; cleared after the bake runs.
    realm_field_dirty: bool,

    /// SDF glyph atlas. `None` until `glyph_atlas.png` + JSON have been
    /// fetched; the realm-labels pass renders nothing while it's empty.
    glyph_atlas: Option<GlyphAtlas>,
    /// Placeholder glyph atlas resources kept alive while the
    /// `glyph_atlas` field is `None` so the realm-labels pass's bind
    /// group stays valid. Dropped once the real atlas lands.
    _placeholder_atlas: Option<(wgpu::Texture, wgpu::TextureView, wgpu::Sampler)>,
    /// Realm-name overlay pass. World-space SDF glyph quads, drawn
    /// after the image pass with alpha blending. Vertex buffer is
    /// rebuilt whenever `set_settlements` or `set_glyph_atlas` runs.
    realm_labels: RealmLabelsPass,

    /// Bind-group layouts kept around for swap-in.
    base_heightmap_bgl: wgpu::BindGroupLayout,
    image_bgl: wgpu::BindGroupLayout,

    /// Cached `window.performance` handle. Used by [`crate::perf::Span`]
    /// to emit User Timing marks/measures around per-frame work; cached
    /// because `window().performance()` traverses two `Option` layers
    /// and JsValue casts on every call.
    performance: Performance,

    /// Per-pass GPU timer. `None` when the backend doesn't support
    /// `TIMESTAMP_QUERY` (e.g. WebGL2 fallback).
    gpu_timer: Option<GpuTimer>,

    /// Which image-pass path to use this frame. Toggled at runtime with
    /// the `T` key (see `lib.rs`). `RenderMode::Raymarch` matches the
    /// historical behaviour; `RenderMode::Mesh` rasterizes a tessellated
    /// heightmap-displaced grid for an A/B comparison.
    render_mode: RenderMode,

    /// Depth attachment for the mesh path. Recreated whenever the canvas
    /// resizes; unused (but still bound) on the raymarch path.
    depth_tex: wgpu::Texture,
    depth_view: wgpu::TextureView,
    /// Canvas size the current `depth_tex` was built for; lets
    /// `handle_resize` skip rebuilding on no-op resizes.
    depth_size: (u32, u32),

    pub base_heightmap: WorldLayer,
    pub detail_noise: WorldLayer,
    pub erosion: WorldLayer,
    pub image: ImagePass,
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

        let layer_uniform_buf = make_layer_uniform_buf(&gpu, "shared layer ub");

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

        // 1×1 placeholders so the bind groups are valid before the PNGs land.
        // The heightmap uses Rg8Unorm — the PNG ships 16-bit grayscale as
        // [hi, lo] BE bytes per pixel, which we upload verbatim into the two
        // 8-bit channels and reassemble in the shader.
        let (world_heightmap_tex, world_heightmap_view) =
            placeholder_rg8unorm(&gpu, "world_heightmap (placeholder)");
        let (water_mask_tex, water_mask_view) =
            placeholder_r8unorm(&gpu, "water_mask (placeholder)");
        // Biome 0 → "no biome / fall through to default land color", which is
        // the safe behaviour while the real biome mask is in flight.
        let (biome_mask_tex, biome_mask_view) =
            placeholder_r8unorm(&gpu, "biome_mask (placeholder)");

        let base_heightmap_bgl = base_heightmap::bgl(&gpu.device);
        let image_bgl = image_pass::bgl(&gpu.device);

        // Build the settlement list + its uniform buffer up-front so the
        // image pass's bind group has real data on the very first frame.
        let LoadedSettlements {
            settlements,
            realm_names,
        } = default_swiss_settlements();
        // No water mask available yet (it's loaded async); rasterise
        // the field with land-everywhere assumption. Will be rebuilt
        // once the PNG lands.
        let realm_infos_init = realm_infos(&settlements, &realm_names, None);
        let settlements_uniform_buf =
            make_settlement_uniform_buf(&gpu, "settlements ub");
        let settlement_data = SettlementUniforms::from_slice(&settlements);
        gpu.queue.write_buffer(
            &settlements_uniform_buf,
            0,
            bytemuck::bytes_of(&settlement_data),
        );

        let base_heightmap = base_heightmap::build(
            &gpu,
            &base_heightmap_bgl,
            &layer_uniform_buf,
            &world_heightmap_view,
            &sampler,
        );
        let detail_noise = detail_noise::build(&gpu, &layer_uniform_buf);
        let erosion = erosion::build(&gpu, &layer_uniform_buf, &base_heightmap.view);

        // Optional per-pass GPU timer. Returns None on WebGL2 (and
        // logs an explanatory line); otherwise we'll record begin/end
        // timestamps around the world layers, the realm-field bake,
        // and the image pass each frame.
        let gpu_timer = GpuTimer::try_new(&gpu);

        // Depth attachment for the mesh path. Sized to the current
        // swapchain; we'll recreate in `handle_resize`.
        let (depth_tex, depth_view) = make_depth(&gpu, gpu.width, gpu.height);
        let depth_size = (gpu.width, gpu.height);

        // Settlement-influence bake target. Built before the image pass so
        // its texture view can feed the image pass's bind group. The
        // initial bake happens on the first `frame()` call (we set
        // `realm_field_dirty = true` below).
        let realm_field = realm_field_pass::build(&gpu, &settlements_uniform_buf);

        let image = ImagePass::build(
            &gpu,
            &image_bgl,
            &camera_uniform_buf,
            &base_heightmap.view,
            &detail_noise.view,
            &erosion.view,
            &layer_uniform_buf,
            &water_mask_view,
            &sampler,
            &biome_mask_view,
            &settlements_uniform_buf,
            &realm_field.view,
            &world_heightmap_view,
        );

        // Realm-labels pass. Built with a 1×1 placeholder atlas so the
        // bind group is valid before `glyph_atlas.png` lands. The pass
        // is a no-op (`vertex_count == 0`) until both the atlas + the
        // settlement layout are ready, so the placeholder content never
        // gets sampled.
        let placeholder_atlas = realm_labels_pass::placeholder_atlas(&gpu);
        let realm_labels = RealmLabelsPass::build(
            &gpu,
            &camera_uniform_buf,
            &placeholder_atlas.1,
            &placeholder_atlas.2,
            gpu.swapchain_format,
        );

        Self {
            gpu,
            camera,
            camera_uniform_buf,
            layer_uniform_buf,
            layer_covered: None,
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
            // Initial settlements were just uploaded; bake before the
            // first image pass so binding(11) has real data.
            realm_field_dirty: true,
            glyph_atlas: None,
            _placeholder_atlas: Some(placeholder_atlas),
            realm_labels,
            base_heightmap_bgl,
            image_bgl,
            performance,
            gpu_timer,
            render_mode: RenderMode::default(),
            depth_tex,
            depth_view,
            depth_size,
            base_heightmap,
            detail_noise,
            erosion,
            image,
        }
    }

    pub fn canvas(&self) -> &HtmlCanvasElement {
        &self.gpu.canvas
    }

    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }

    /// Read-only access to the settlement list. The UI layer uses this to
    /// look up the city the user just clicked on (name, realm, strength).
    pub fn settlements(&self) -> &[Settlement] {
        &self.settlements
    }

    /// Per-realm summaries (name + centroid + total strength). The HTML
    /// overlay's `RealmLabels` reads this each frame to draw country
    /// labels. Re-computed whenever `set_settlements` is called.
    pub fn realm_infos(&self) -> &[RealmInfo] {
        &self.realm_infos
    }

    /// Resolve a screen-space click (CSS pixels) to the nearest settlement.
    /// The click radius scales with zoom so deeply zoomed-in views give a
    /// tight 2 km radius and pulled-back views give a generous one.
    /// Returns `None` if the click misses (no city within the radius, or
    /// the click ray didn't intersect the ground plane at all).
    pub fn pick_settlement_at(
        &self,
        mx: f32,
        my: f32,
        css_w: f32,
        css_h: f32,
    ) -> Option<&Settlement> {
        let xz = self.camera.pick_world_xz(mx, my, css_w, css_h, HOVER_PICK_Y)?;
        // Visible vertical extent in world km (one full screen height,
        // matching `Camera::view_aabb`). Click radius is 1/25th of that
        // with a 2 km floor so close zooms still let you click cities.
        let view_h_km = 2.0 * self.camera.distance * (CAMERA_FOV_Y_RAD * 0.5).tan();
        let radius_km = (view_h_km / 25.0).max(2.0);
        nearest_within_km(&self.settlements, xz, radius_km).map(|i| &self.settlements[i])
    }

    /// Reconfigure the swapchain to the canvas's current backing size.
    /// Also recreates the depth texture used by the mesh path so it
    /// matches the new dimensions.
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
        // World layers are independent of swapchain size; nothing else to do.
    }

    /// Toggle between the raymarch and mesh image-pass paths. Returns
    /// the new mode so callers can log it.
    pub fn toggle_render_mode(&mut self) -> RenderMode {
        self.render_mode = self.render_mode.toggled();
        self.render_mode
    }

    /// Run one frame: push camera uniforms, regenerate world layers if the
    /// camera left the cached region, run the image pass to swapchain.
    ///
    /// Each phase is wrapped in a [`Span`] so the work shows up as
    /// labelled bars in DevTools' Performance timeline. Spans nest:
    /// `frame` is the outer bar, with `frame.world_layers`,
    /// `frame.realm_field_bake`, `frame.image_encode`,
    /// `frame.acquire`, `frame.submit`, `frame.present` as inner
    /// children.
    pub fn frame(&mut self) {
        let _frame_span = Span::new(&self.performance, "frame");

        // 1. Camera uniforms — cheap, always update.
        {
            let _s = Span::new(&self.performance, "frame.camera_uniforms");
            let uniforms = self.camera.to_uniforms(self.gpu.width, self.gpu.height);
            self.gpu
                .queue
                .write_buffer(&self.camera_uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        }

        // 2. World-layer cache: re-render iff the camera's view AABB (×PAD)
        // is no longer contained in `layer_covered`.
        let aspect = self.gpu.width as f32 / self.gpu.height.max(1) as f32;
        let view = self.camera.view_aabb(aspect);
        let need = view.expanded(PAD);
        let cache_stale = !matches!(self.layer_covered, Some(c) if c.contains(need));

        let mut encoder = self.gpu.encoder("frame");

        // Decide whether to record GPU timestamps this frame. False if
        // the timer doesn't exist (WebGL2) or the previous frame's
        // readback hasn't completed yet.
        let timing_active = self
            .gpu_timer
            .as_mut()
            .map(|t| t.begin_frame())
            .unwrap_or(false);

        if cache_stale {
            let _s = Span::new(&self.performance, "frame.world_layers");
            self.layer_covered = Some(need);
            let layer_u = LayerUniforms::from_aabb(need);
            self.gpu.queue.write_buffer(
                &self.layer_uniform_buf,
                0,
                bytemuck::bytes_of(&layer_u),
            );
            // Order matters: erosion reads base_heightmap; both must use the
            // same covered AABB this frame. We bracket the *whole* layers
            // span by writing only the begin timestamp on the first
            // sub-pass and only the end timestamp on the last.
            let begin = if timing_active {
                self.gpu_timer.as_mut().map(|t| t.writes_begin(Section::Layers))
            } else {
                None
            };
            self.base_heightmap.render(&mut encoder, begin);
            self.detail_noise.render(&mut encoder, None);
            let end = if timing_active {
                self.gpu_timer.as_mut().map(|t| t.writes_end(Section::Layers))
            } else {
                None
            };
            self.erosion.render(&mut encoder, end);
        }

        // 2b. Realm-field bake — only when the settlement list changed.
        // The image pass reads from `realm_field.view` via textureLoad; the
        // bake itself is a single fullscreen draw so it's cheap.
        if self.realm_field_dirty {
            let _s = Span::new(&self.performance, "frame.realm_field_bake");
            let bake_writes = if timing_active {
                self.gpu_timer.as_mut().map(|t| t.writes_full(Section::Bake))
            } else {
                None
            };
            self.realm_field.render(&mut encoder, bake_writes);
            self.realm_field_dirty = false;
        }

        // 3. Image pass to swapchain.
        let (frame, frame_view) = {
            let _s = Span::new(&self.performance, "frame.acquire");
            self.gpu.acquire_frame()
        };
        {
            let _s = Span::new(&self.performance, "frame.image_encode");
            let image_writes = if timing_active {
                self.gpu_timer.as_mut().map(|t| t.writes_full(Section::Image))
            } else {
                None
            };
            self.image.render(
                &mut encoder,
                &frame_view,
                &self.depth_view,
                self.render_mode,
                image_writes,
            );
        }

        // 3b. Realm-name overlay. Cheap (a few hundred glyph quads at
        // most) and a no-op until the SDF atlas + a non-empty
        // realm-info list have both landed.
        {
            let _s = Span::new(&self.performance, "frame.realm_labels");
            self.realm_labels.render(&mut encoder, &frame_view);
        }

        // Resolve the timestamps + queue the readback copy *before*
        // submission so they ride along in the same encoder.
        if timing_active {
            if let Some(t) = self.gpu_timer.as_ref() {
                t.resolve(&mut encoder);
            }
        }

        {
            let _s = Span::new(&self.performance, "frame.submit");
            self.gpu.submit(encoder);
        }

        // Schedule the async map_async *after* submit; the GPU has to
        // chew through the resolve+copy before the map can complete.
        if timing_active {
            if let Some(t) = self.gpu_timer.as_mut() {
                t.after_submit();
            }
        }

        {
            let _s = Span::new(&self.performance, "frame.present");
            frame.present();
        }

        // Outer span finishes here (drop order = reverse of construction).
        drop(_frame_span);
        // Empty the User Timing buffer so a long session doesn't
        // accumulate marks/measures forever. The DevTools recorder
        // already snapshotted them via PerformanceObserver, so clearing
        // here is safe even mid-recording.
        perf::clear_buffer(&self.performance);
    }

    /// Upload a fresh 8192² R16Unorm world heightmap from the PNG bytes
    /// returned by `assets::fetch_png`. Replaces the placeholder, rebuilds
    /// the base_heightmap and image bind groups, and invalidates the cache
    /// so the world layers regenerate on the next frame.
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
            2, // bytes per pixel: 16-bit grayscale PNG split across R + G
        );
        let view = tex.create_view(&Default::default());
        self.world_heightmap_tex = tex;
        self.world_heightmap_view = view;

        // Rebuild the bind groups that referenced the old view.
        self.base_heightmap.bind_group = base_heightmap::make_bind_group(
            &self.gpu.device,
            &self.base_heightmap_bgl,
            &self.layer_uniform_buf,
            &self.world_heightmap_view,
            &self.sampler,
        );
        self.rebuild_image_bind_group();

        // The cached world layers were built against the placeholder; toss them.
        self.layer_covered = None;
    }

    /// Upload a fresh 8192² R8Unorm water mask. Symmetrical with
    /// `set_world_heightmap`, but also stores a CPU copy of the bytes
    /// so `realm_infos` can drop offshore cells from the rasteriser,
    /// and triggers a fresh realm-info + label rebuild so the labels
    /// reposition onto land once water data lands.
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
            1, // bytes per pixel
        );
        let view = tex.create_view(&Default::default());
        self.water_mask_tex = tex;
        self.water_mask_view = view;
        self.rebuild_image_bind_group();

        // Stash the bytes CPU-side, then re-run the realm rasteriser
        // now that we can filter out sea cells. Label vertices follow.
        self.water_mask_cpu = Some((bytes.to_vec(), width, height));
        self.recompute_realm_infos();
        self.rebuild_realm_label_vertices();
    }

    /// Update the hovered-realm + hovered-city state from a screen-space
    /// mouse position (CSS pixels). Returns `true` iff *either* the realm
    /// or the city under the cursor changed.
    ///
    /// Mirrors `sample_realm_field` on the CPU — picking the world XZ
    /// under the cursor via the camera ray, then evaluating the
    /// argmax-strength settlement at that point. Both indices are stored
    /// as `value + 1` so 0 is the "nothing hovered" sentinel.
    pub fn update_hover(&mut self, mx: f32, my: f32, css_w: f32, css_h: f32) -> bool {
        // Wrap the whole pick + 683-settlement scan in a Span so its
        // cumulative cost shows up in DevTools' Performance → Timings
        // track. On a 1000 Hz trackpad this fires up to ~16× per
        // painted frame, so it's worth seeing the total time per
        // second — not just the per-call cost.
        let _s = Span::new(&self.performance, "hover.update");
        let (prev_realm, prev_city) = (self.camera.hovered_pid, self.camera.hovered_city);
        let (realm_id, city_id) = self
            .camera
            .pick_world_xz(mx, my, css_w, css_h, HOVER_PICK_Y)
            .map(|xz| {
                let hit = dominant_at_world_xz(&self.settlements, xz);
                // Below this threshold the field is so weak no realm /
                // city has any meaningful presence — treat as wilderness.
                // Threshold mirrors the shader's `field.alpha > 0.05`
                // gate (alpha = 1 - exp(-strength * 0.1) crosses 0.05 at
                // strength ≈ 0.51).
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

    /// Upload a fresh 8192² R8Unorm biome mask. Pixel values are biome IDs
    /// (0..14, with 0 = "no biome"). Mirrors `set_water_mask`.
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
            1, // bytes per pixel
        );
        let view = tex.create_view(&Default::default());
        self.biome_mask_tex = tex;
        self.biome_mask_view = view;
        self.rebuild_image_bind_group();
    }

    fn rebuild_image_bind_group(&mut self) {
        self.image.bind_group = image_pass::make_bind_group(
            &self.gpu.device,
            &self.image_bgl,
            &self.camera_uniform_buf,
            &self.base_heightmap.view,
            &self.detail_noise.view,
            &self.erosion.view,
            &self.layer_uniform_buf,
            &self.water_mask_view,
            &self.sampler,
            &self.biome_mask_view,
            &self.settlements_uniform_buf,
            &self.realm_field.view,
            &self.world_heightmap_view,
        );
    }

    /// Re-upload the settlement uniform buffer from the current
    /// `self.settlements` list and mark the realm-field bake dirty so the
    /// next frame re-runs it. Call after mutating settlements at runtime.
    #[allow(dead_code)]
    pub fn upload_settlements(&mut self) {
        let data = SettlementUniforms::from_slice(&self.settlements);
        self.gpu
            .queue
            .write_buffer(&self.settlements_uniform_buf, 0, bytemuck::bytes_of(&data));
        self.realm_field_dirty = true;
    }

    /// Replace the settlement list (and its realm-name map) and re-upload.
    /// Used by the asynchronous `cities.json` fetch in `lib.rs` once the
    /// JSON has been parsed. Truncates to `MAX_SETTLEMENTS` since that's
    /// what the GPU buffer holds, and rebuilds `realm_infos` so the label
    /// pass picks up the new realms on the next frame.
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

    /// Re-run the realm rasteriser → PCA → baseline pipeline from the
    /// current `self.settlements` + the cached water mask (if loaded).
    /// Both `set_settlements` and `set_water_mask` call this so the
    /// labels stay correct regardless of asset-load ordering.
    fn recompute_realm_infos(&mut self) {
        let wm = self.water_mask_cpu.as_ref().map(|(bytes, w, h)| WaterMask {
            bytes: bytes.as_slice(),
            width: *w,
            height: *h,
        });
        self.realm_infos = realm_infos(&self.settlements, &self.realm_names, wm.as_ref());
    }

    /// Install a freshly loaded SDF glyph atlas. Replaces the
    /// placeholder texture in the realm-labels bind group and
    /// re-runs the layout so labels appear on the next frame.
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
        // Drop the placeholder once the real atlas owns the bind group
        // — wgpu keeps the placeholder texture alive via the bind
        // group's internal refcount, but our explicit handle is no
        // longer needed.
        self._placeholder_atlas = None;
        self.rebuild_realm_label_vertices();
        Ok(())
    }

    /// Re-run the label layout from `self.realm_infos` and upload the
    /// resulting vertex buffer. No-op when the atlas isn't loaded yet.
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

/// Build a 1×1 Rg8Unorm placeholder, value (0, 0). Used for the world
/// heightmap until the real PNG lands.
fn placeholder_rg8unorm(gpu: &GpuContext, label: &str) -> (wgpu::Texture, wgpu::TextureView) {
    placeholder_texture(gpu, label, wgpu::TextureFormat::Rg8Unorm, &[0u8, 0u8])
}

/// Build a 1×1 R8Unorm placeholder, value 0.
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

/// Allocate a `width × height` texture in `format` and upload `bytes` into it.
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

// LAYER_SIZE is unused at the moment, but kept in scope for future LoD
// selection.
#[allow(dead_code)]
const _LAYER_SIZE_REFERENCE: u32 = LAYER_SIZE;

/// Build a `Depth32Float` texture sized to the current swapchain for the
/// mesh-path image pass. Lives in this module (not `gpu.rs`) because it's
/// only needed by the image pass, and lifecycle is tied to swapchain
/// resizes which the Renderer drives.
fn make_depth(gpu: &GpuContext, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("image depth"),
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
