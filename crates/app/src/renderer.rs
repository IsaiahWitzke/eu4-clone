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
use web_sys::HtmlCanvasElement;

use crate::camera::{Aabb2, Camera, CameraUniforms, HOVER_PICK_Y};
use crate::gpu::{GpuContext, compute_canvas_size};
use crate::passes::{base_heightmap, detail_noise, erosion, image::ImagePass, image as image_pass};
use crate::world_layer::{LAYER_SIZE, LayerUniforms, WorldLayer, make_layer_uniform_buf};

/// Padding factor for the world-layer cache: re-render only when the camera
/// has moved past `view_aabb × PAD`.
const PAD: f32 = 2.0;

/// Width/height of the asset textures (heightmap.png + water_mask.png).
const WORLD_TEX_SIZE: u32 = 8192;

/// Half-extent of the world rectangle (km). Mirrors `WORLD_BOUNDS_HALF` in
/// `shaders/world.wgsl`. Used by the CPU mouse-pick code to map world XZ
/// back into province-mask UV coords.
const WORLD_BOUNDS_HALF_KM: f32 = 313.037;

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
    /// Biome IDs (R8Unorm; pixel value = biome number 0..14, where 0 is
    /// "no biome / not classified"). Sampled with textureLoad in the shader.
    biome_mask_tex: wgpu::Texture,
    biome_mask_view: wgpu::TextureView,
    /// Province IDs (Rg8Unorm carrying a 16-bit big-endian unsigned integer
    /// per pixel; 0 = "no province"). Sampled with textureLoad in the shader
    /// and reassembled to a `u32` ID for political coloring.
    province_mask_tex: wgpu::Texture,
    province_mask_view: wgpu::TextureView,
    /// CPU-side copy of the province-mask bytes, kept around so we can
    /// resolve a screen-space mouse position to a province ID without
    /// round-tripping to the GPU. None until the PNG finishes loading.
    province_mask_bytes: Option<Vec<u8>>,
    /// Border signed-distance-field (R8Unorm; pixel value = clamped
    /// distance-to-nearest-border encoded as 0..255). Pre-baked offline by
    /// `script/gen-border-sdf` from the smoothed NUTS-3 boundaries.
    /// **Filterable** — the shader bilinearly samples it and runs the result
    /// through smoothstep for AA'd, thickness-controllable borders.
    border_sdf_tex: wgpu::Texture,
    border_sdf_view: wgpu::TextureView,

    /// Bind-group layouts kept around for swap-in.
    base_heightmap_bgl: wgpu::BindGroupLayout,
    image_bgl: wgpu::BindGroupLayout,

    pub base_heightmap: WorldLayer,
    pub detail_noise: WorldLayer,
    pub erosion: WorldLayer,
    pub image: ImagePass,
}

impl Renderer {
    pub async fn new(canvas: HtmlCanvasElement) -> Self {
        let gpu = GpuContext::new(canvas).await;

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
        // Province 0 → "no province". Same Rg8Unorm split-bytes encoding as
        // the heightmap (the PNG is 16-bit grayscale, big-endian).
        let (province_mask_tex, province_mask_view) =
            placeholder_rg8unorm(&gpu, "province_mask (placeholder)");
        // Border SDF: 1×1 R8Unorm value 255 → "infinite distance to any
        // border", which makes the border smoothstep return 0 (no darkening)
        // until the real PNG lands.
        let (border_sdf_tex, border_sdf_view) =
            placeholder_texture(&gpu, "border_sdf (placeholder)", wgpu::TextureFormat::R8Unorm, &[255u8]);

        let base_heightmap_bgl = base_heightmap::bgl(&gpu.device);
        let image_bgl = image_pass::bgl(&gpu.device);

        let base_heightmap = base_heightmap::build(
            &gpu,
            &base_heightmap_bgl,
            &layer_uniform_buf,
            &world_heightmap_view,
            &sampler,
        );
        let detail_noise = detail_noise::build(&gpu, &layer_uniform_buf);
        let erosion = erosion::build(&gpu, &layer_uniform_buf, &base_heightmap.view);
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
            &province_mask_view,
            &border_sdf_view,
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
            biome_mask_tex,
            biome_mask_view,
            province_mask_tex,
            province_mask_view,
            province_mask_bytes: None,
            border_sdf_tex,
            border_sdf_view,
            base_heightmap_bgl,
            image_bgl,
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

    /// Reconfigure the swapchain to the canvas's current backing size.
    pub fn handle_resize(&mut self) {
        let (w, h) = compute_canvas_size(&self.gpu.canvas);
        if (w, h) == (self.gpu.width, self.gpu.height) {
            return;
        }
        self.gpu.resize(w, h);
        // World layers are independent of swapchain size; nothing else to do.
    }

    /// Run one frame: push camera uniforms, regenerate world layers if the
    /// camera left the cached region, run the image pass to swapchain.
    pub fn frame(&mut self) {
        // 1. Camera uniforms — cheap, always update.
        let uniforms = self.camera.to_uniforms(self.gpu.width, self.gpu.height);
        self.gpu
            .queue
            .write_buffer(&self.camera_uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        // 2. World-layer cache: re-render iff the camera's view AABB (×PAD)
        // is no longer contained in `layer_covered`.
        let aspect = self.gpu.width as f32 / self.gpu.height.max(1) as f32;
        let view = self.camera.view_aabb(aspect);
        let need = view.expanded(PAD);
        let cache_stale = !matches!(self.layer_covered, Some(c) if c.contains(need));

        let mut encoder = self.gpu.encoder("frame");

        if cache_stale {
            self.layer_covered = Some(need);
            let layer_u = LayerUniforms::from_aabb(need);
            self.gpu.queue.write_buffer(
                &self.layer_uniform_buf,
                0,
                bytemuck::bytes_of(&layer_u),
            );
            // Order matters: erosion reads base_heightmap; both must use the
            // same covered AABB this frame.
            self.base_heightmap.render(&mut encoder);
            self.detail_noise.render(&mut encoder);
            self.erosion.render(&mut encoder);
        }

        // 3. Image pass to swapchain.
        let (frame, frame_view) = self.gpu.acquire_frame();
        self.image.render(&mut encoder, &frame_view);
        self.gpu.submit(encoder);
        frame.present();
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
    /// `set_world_heightmap`, but only the image pass references it.
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
    }

    /// Upload a fresh 8192² Rg8Unorm province mask. The PNG is 16-bit
    /// grayscale (big-endian); we upload the [hi, lo] bytes verbatim into
    /// R + G channels and let the shader reassemble them. ID 0 = "no
    /// province". Also keeps a CPU copy of the bytes for mouse-pick.
    pub fn set_province_mask(&mut self, width: u32, height: u32, bytes: &[u8]) {
        if (width, height) != (WORLD_TEX_SIZE, WORLD_TEX_SIZE) {
            web_sys::console::warn_1(
                &format!(
                    "province mask size mismatch: got {width}x{height}, expected \
                     {WORLD_TEX_SIZE}\u{00d7}{WORLD_TEX_SIZE}"
                )
                .into(),
            );
        }
        let tex = upload_world_texture(
            &self.gpu,
            "province_mask",
            wgpu::TextureFormat::Rg8Unorm,
            width,
            height,
            bytes,
            2, // bytes per pixel: 16-bit grayscale split across R + G
        );
        let view = tex.create_view(&Default::default());
        self.province_mask_tex = tex;
        self.province_mask_view = view;
        // Clone for CPU-side hover picking. ~128 MB at full resolution but
        // happens once per session.
        self.province_mask_bytes = Some(bytes.to_vec());
        self.rebuild_image_bind_group();
    }

    /// Update the hovered-province state from a screen-space mouse position
    /// (CSS pixels). Returns `true` iff the hovered province changed (so the
    /// caller can avoid redundant frames).
    pub fn update_hover(&mut self, mx: f32, my: f32, css_w: f32, css_h: f32) -> bool {
        let prev = self.camera.hovered_pid;
        let pid = self
            .camera
            .pick_world_xz(mx, my, css_w, css_h, HOVER_PICK_Y)
            .map(|xz| self.province_at_world_xz(xz))
            .unwrap_or(0);
        if pid != prev {
            self.camera.hovered_pid = pid;
            true
        } else {
            false
        }
    }

    /// Look up the province ID at a world XZ position, using the CPU copy
    /// of the province mask. Returns 0 if the position is out of bounds or
    /// the mask isn't loaded yet.
    fn province_at_world_xz(&self, xz: [f32; 2]) -> u32 {
        let bytes = match &self.province_mask_bytes {
            Some(b) => b,
            None => return 0,
        };
        let half = WORLD_BOUNDS_HALF_KM;
        let u = (xz[0] + half) / (2.0 * half);
        // Y flip mirrors `world_to_world_uv` in shaders/world.wgsl.
        let v = 1.0 - (xz[1] + half) / (2.0 * half);
        if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
            return 0;
        }
        let w = WORLD_TEX_SIZE;
        let px = ((u * w as f32) as i32).clamp(0, (w as i32) - 1) as u32;
        let py = ((v * w as f32) as i32).clamp(0, (w as i32) - 1) as u32;
        let idx = ((py * w + px) as usize) * 2;
        let hi = bytes[idx] as u32;
        let lo = bytes[idx + 1] as u32;
        hi * 256 + lo
    }

    /// Upload a fresh 8192² R8Unorm border SDF. Pixel value 0 = on a border;
    /// 255 = ≥ MAX_DIST_PX away. The shader bilinearly samples and runs it
    /// through smoothstep to draw smooth, AA'd, thickness-tunable borders.
    pub fn set_border_sdf(&mut self, width: u32, height: u32, bytes: &[u8]) {
        if (width, height) != (WORLD_TEX_SIZE, WORLD_TEX_SIZE) {
            web_sys::console::warn_1(
                &format!(
                    "border SDF size mismatch: got {width}x{height}, expected \
                     {WORLD_TEX_SIZE}\u{00d7}{WORLD_TEX_SIZE}"
                )
                .into(),
            );
        }
        let tex = upload_world_texture(
            &self.gpu,
            "border_sdf",
            wgpu::TextureFormat::R8Unorm,
            width,
            height,
            bytes,
            1, // bytes per pixel
        );
        let view = tex.create_view(&Default::default());
        self.border_sdf_tex = tex;
        self.border_sdf_view = view;
        self.rebuild_image_bind_group();
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
            &self.province_mask_view,
            &self.border_sdf_view,
        );
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
