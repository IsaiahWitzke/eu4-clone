use std::cell::RefCell;
use std::rc::Rc;

use bytemuck::{Pod, Zeroable};
use wasm_bindgen::JsCast;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;
use web_sys::{HtmlCanvasElement, console};

// ----------------------------------------------------------------------------
// Shaders. Common declarations (uniforms + fullscreen vs) are concatenated
// onto each fragment shader so each pass is compiled as a single module that
// has both `vs_main` and `fs_main`.
// ----------------------------------------------------------------------------
const COMMON_WGSL: &str = include_str!("shaders/common.wgsl");
const NOISE_WGSL: &str = include_str!("shaders/noise.wgsl");
const BASE_HEIGHTMAP_FS: &str = include_str!("shaders/base_heightmap.wgsl");
const TERRAIN_FS: &str = include_str!("shaders/terrain.wgsl");
const DETAIL_NOISE_FS: &str = include_str!("shaders/detail_noise.wgsl");
const IMAGE_FS: &str = include_str!("shaders/image.wgsl");

// ---- Camera defaults & derivation ----
//
// Camera lives at `look_at + eye_offset`, where eye_offset is derived from
// (distance, tilt). tilt = 0 → straight overhead; tilt approaches π/2 →
// horizontal view. The shader gets `eye_offset` as a uniform and uses it both
// to position the eye and to derive the camera basis (forward / right / up).

/// Vertical FOV used by the perspective shader. Must match `CAMERA_FOV_Y` in
/// image.wgsl — we duplicate it on the Rust side so we can size the heightmap
/// region (world_half_size) to the camera's actual coverage.
const CAMERA_FOV_Y_RAD: f32 = std::f32::consts::PI / 6.0; // 30°

/// Multiplier on the computed view extent before storing as world_half_size,
/// so the heightmap is slightly larger than what the camera sees (avoiding
/// edge artifacts when panning a little past the strict view).
const VIEW_MARGIN: f32 = 1.15;

/// Default camera distance from look_at and tilt angle from vertical (radians).
const DEFAULT_CAMERA_DISTANCE: f32 = 2.6;   // ≈ sqrt(2.5² + 0.7²) of the old setup
const DEFAULT_CAMERA_TILT: f32 = 0.27;      // ≈ 16° from vertical

/// Hard limits applied when the user adjusts the camera with keys.
const MIN_CAMERA_DISTANCE: f32 = 0.8;
const MAX_CAMERA_DISTANCE: f32 = 6.0;
const MIN_CAMERA_TILT: f32 = 0.0;                          // pure top-down
const MAX_CAMERA_TILT: f32 = std::f32::consts::FRAC_PI_2 - 0.05; // ~85°

/// Uniform block shared by every pass. Mirrors the WGSL `Uniforms` struct.
/// std140 layout: vec3 has 16-byte alignment, vec2 has 8-byte. Total size 48.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    i_resolution: [f32; 3],
    i_time: f32,
    world_center: [f32; 2],
    world_half_size: [f32; 2],
    eye_offset: [f32; 3],
    _pad: f32,
}

impl Uniforms {
    /// Build a fresh uniform block at i_time = 0 with the given camera state.
    /// `world_half_size` and `eye_offset` are both derived from
    /// (camera_distance, camera_tilt) so they stay consistent.
    fn new(
        width: u32,
        height: u32,
        world_center: [f32; 2],
        camera_distance: f32,
        camera_tilt: f32,
    ) -> Self {
        let aspect = width as f32 / height.max(1) as f32;
        // Vertical world half-extent the camera sees at this distance, scaled
        // up by VIEW_MARGIN so the heightmap covers it with some slack.
        let half_h = camera_distance * (CAMERA_FOV_Y_RAD * 0.5).tan() * VIEW_MARGIN;
        let half_w = half_h * aspect;
        // Eye offset: above look_at, pulled south by `tilt`. tilt=0 is
        // straight overhead, larger tilt is more oblique.
        let eye_offset = [
            0.0,
            camera_distance * camera_tilt.cos(),
            -camera_distance * camera_tilt.sin(),
        ];
        Self {
            i_resolution: [width as f32, height as f32, 1.0],
            i_time: 0.0,
            world_center,
            world_half_size: [half_w, half_h],
            eye_offset,
            _pad: 0.0,
        }
    }
}

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    wasm_bindgen_futures::spawn_local(run());
}

async fn run() {
    let canvas = canvas_by_id("game");
    let state = Rc::new(RefCell::new(State::new(canvas).await));
    state.borrow().render();

    let window = web_sys::window().expect("no window");

    // Resize -> reconfigure surface + offscreen textures + re-render.
    let resize_state = state.clone();
    let on_resize = Closure::<dyn FnMut()>::new(move || {
        let mut s = resize_state.borrow_mut();
        let (w, h) = compute_canvas_size(&s.canvas);
        if (w, h) != (s.width, s.height) {
            s.resize(w, h);
            drop(s);
            resize_state.borrow().render();
        }
    });
    window.set_onresize(Some(on_resize.as_ref().unchecked_ref()));
    on_resize.forget();

    // Keyboard input. Browser key-repeat (~30 Hz on Mac after ~500 ms delay)
    // gives reasonable continuous control when a key is held. No animation
    // loop needed — we re-render only on input.
    //
    //   Arrow keys     pan world_center
    //   Q / E          tilt camera (more top-down / more oblique)
    //   - / = (+)      zoom out / in
    let key_state = state.clone();
    let on_keydown = Closure::<dyn FnMut(_)>::new(move |e: web_sys::KeyboardEvent| {
        // Step sizes per keydown event.
        let pan_step: f32 = 0.01;     // world units / press
        let tilt_step: f32 = 0.025;   // radians / press (~1.4°)
        let zoom_factor: f32 = 1.05;  // 5% per press, multiplicative

        let mut s = key_state.borrow_mut();
        let mut handled = true;
        match e.key().as_str() {
            "ArrowUp"    => s.world_center[1] += pan_step,
            "ArrowDown"  => s.world_center[1] -= pan_step,
            "ArrowLeft"  => s.world_center[0] -= pan_step,
            "ArrowRight" => s.world_center[0] += pan_step,
            "q" | "Q"    => s.camera_tilt =
                (s.camera_tilt - tilt_step).clamp(MIN_CAMERA_TILT, MAX_CAMERA_TILT),
            "e" | "E"    => s.camera_tilt =
                (s.camera_tilt + tilt_step).clamp(MIN_CAMERA_TILT, MAX_CAMERA_TILT),
            "=" | "+"    => s.camera_distance =
                (s.camera_distance / zoom_factor).clamp(MIN_CAMERA_DISTANCE, MAX_CAMERA_DISTANCE),
            "-" | "_"    => s.camera_distance =
                (s.camera_distance * zoom_factor).clamp(MIN_CAMERA_DISTANCE, MAX_CAMERA_DISTANCE),
            _ => handled = false,
        }
        if !handled {
            return;
        }
        e.prevent_default();
        s.write_uniforms();
        drop(s);
        key_state.borrow().render();
    });
    window
        .add_event_listener_with_callback("keydown", on_keydown.as_ref().unchecked_ref())
        .expect("failed to attach keydown listener");
    on_keydown.forget();

    // Click-and-drag panning: grab the world and move it under the cursor.
    // mousedown anywhere starts the drag (we treat the canvas as fullscreen);
    // move and up listen on the window so the drag survives leaving the canvas.
    let down_state = state.clone();
    let on_mousedown = Closure::<dyn FnMut(_)>::new(move |e: web_sys::MouseEvent| {
        let mut s = down_state.borrow_mut();
        s.dragging = true;
        s.last_mouse = (e.client_x() as f32, e.client_y() as f32);
        e.prevent_default();
    });
    window
        .add_event_listener_with_callback("mousedown", on_mousedown.as_ref().unchecked_ref())
        .expect("failed to attach mousedown listener");
    on_mousedown.forget();

    let move_state = state.clone();
    let on_mousemove = Closure::<dyn FnMut(_)>::new(move |e: web_sys::MouseEvent| {
        let mut s = move_state.borrow_mut();
        if !s.dragging {
            return;
        }
        let (mx, my) = (e.client_x() as f32, e.client_y() as f32);
        let (lx, ly) = s.last_mouse;
        let dx_px = mx - lx;
        let dy_px = my - ly;
        s.last_mouse = (mx, my);

        // World units per CSS pixel: visible half-extent on each axis divided by
        // half the canvas's CSS dimensions. Uses raw FOV (no VIEW_MARGIN) so the
        // grabbed world point moves 1:1 with the cursor at the lookat plane.
        let css_w = s.canvas.client_width().max(1) as f32;
        let css_h = s.canvas.client_height().max(1) as f32;
        let half_y = s.camera_distance * (CAMERA_FOV_Y_RAD * 0.5).tan();
        let half_x = half_y * (css_w / css_h);
        let world_per_px_x = 2.0 * half_x / css_w;
        let world_per_px_y = 2.0 * half_y / css_h;

        // Drag right → camera moves west (so the world appears to follow the
        // mouse). Drag down → camera moves north (screen Y is top-down, world
        // Z is north-up, so the signs work out as +).
        s.world_center[0] -= dx_px * world_per_px_x;
        s.world_center[1] += dy_px * world_per_px_y;
        s.write_uniforms();
        drop(s);
        move_state.borrow().render();
    });
    window
        .add_event_listener_with_callback("mousemove", on_mousemove.as_ref().unchecked_ref())
        .expect("failed to attach mousemove listener");
    on_mousemove.forget();

    let up_state = state.clone();
    let on_mouseup = Closure::<dyn FnMut(_)>::new(move |_e: web_sys::MouseEvent| {
        up_state.borrow_mut().dragging = false;
    });
    window
        .add_event_listener_with_callback("mouseup", on_mouseup.as_ref().unchecked_ref())
        .expect("failed to attach mouseup listener");
    on_mouseup.forget();

    console::log_1(&"ready (resize + arrow-key pan + mouse drag)".into());
}

// ----------------------------------------------------------------------------
// Render state.
//
// Split into:
//  - "persistent" fields, created once in `new()` and untouched on resize
//  - "size-dependent" fields, recreated in `resize()` whenever the canvas size
//    changes (textures + bind groups that reference them).
// ----------------------------------------------------------------------------
struct State {
    canvas: HtmlCanvasElement,

    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_caps: wgpu::SurfaceCapabilities,
    swapchain_format: wgpu::TextureFormat,

    width: u32,
    height: u32,

    /// Camera pan target in world XZ. Updated by arrow keys; written into the
    /// `world_center` field of the uniform block.
    world_center: [f32; 2],

    /// Camera distance from look_at, in world units. Affects zoom + the
    /// heightmap's world coverage (so the box stays just slightly larger than
    /// the visible region).
    camera_distance: f32,

    /// Camera tilt angle in radians from vertical. 0 → straight overhead;
    /// approaches π/2 → horizontal view.
    camera_tilt: f32,

    /// Whether the user is currently click-dragging the map.
    dragging: bool,
    /// Last observed mouse position in CSS pixels, while dragging.
    last_mouse: (f32, f32),

    uniform_buf: wgpu::Buffer,
    // Layout for `image` pass (uniforms + 3 textures).
    image_layout: wgpu::BindGroupLayout,
    // Layout for `terrain` pass (uniforms + base_heightmap texture). Kept around so
    // we can rebuild the bind group on resize when base_heightmap's view changes.
    terrain_layout: wgpu::BindGroupLayout,

    base_heightmap_pipeline: wgpu::RenderPipeline,
    terrain_pipeline: wgpu::RenderPipeline,
    detail_noise_pipeline: wgpu::RenderPipeline,
    image_pipeline: wgpu::RenderPipeline,
    base_heightmap_bg: wgpu::BindGroup,
    detail_noise_bg: wgpu::BindGroup,

    _base_heightmap_tex: wgpu::Texture,
    _terrain_tex: wgpu::Texture,
    _detail_noise_tex: wgpu::Texture,
    base_heightmap_view: wgpu::TextureView,
    terrain_view: wgpu::TextureView,
    detail_noise_view: wgpu::TextureView,
    // BG referencing base_heightmap as input. Recreated on resize.
    terrain_bg: wgpu::BindGroup,
    image_bg: wgpu::BindGroup,
}

impl State {
    async fn new(canvas: HtmlCanvasElement) -> Self {
        // Size the canvas backing buffer to (CSS size * devicePixelRatio).
        let (width, height) = compute_canvas_size(&canvas);
        canvas.set_width(width);
        canvas.set_height(height);

        // ---- Instance / surface / adapter / device --------------------------------
        let instance = wgpu::util::new_instance_with_webgpu_detection(
            wgpu::InstanceDescriptor::new_without_display_handle(),
        )
        .await;

        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .expect("failed to create surface from canvas");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .expect("no compatible GPU adapter found");

        let info = adapter.get_info();
        console::log_1(
            &format!(
                "wgpu adapter: backend={:?} name={:?} driver={:?}",
                info.backend, info.name, info.driver
            )
            .into(),
        );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("eu4-app device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("failed to request device");

        // ---- Surface config -------------------------------------------------------
        let surface_caps = surface.get_capabilities(&adapter);
        let swapchain_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        surface.configure(
            &device,
            &surface_config(swapchain_format, &surface_caps, width, height),
        );

        // ---- Uniform buffer -------------------------------------------------------
        let uniforms = Uniforms::new(
            width,
            height,
            [0.0, 0.0],
            DEFAULT_CAMERA_DISTANCE,
            DEFAULT_CAMERA_TILT,
        );
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        // ---- Offscreen Rgba32Float textures for Buffers A, B, and C -----------
        let (base_heightmap_tex, base_heightmap_view) = make_offscreen(&device, "base_heightmap", width, height);
        let (terrain_tex, terrain_view) = make_offscreen(&device, "terrain", width, height);
        let (detail_noise_tex, detail_noise_view) = make_offscreen(&device, "detail_noise", width, height);

        // ---- Bind group layouts ----------------------------------------------
        let uniform_only_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("uniforms only"),
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

        let texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };

        let uniform_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        // Buffer B reads Buffer A: uniforms + 1 texture (binding 1 = base_heightmap).
        let terrain_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain layout"),
            entries: &[uniform_entry, texture_entry(1)],
        });

        // Image samples Buffer A (1), Buffer C (2), Buffer B (3) per image.wgsl.
        let image_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("image layout"),
            entries: &[
                uniform_entry,
                texture_entry(1),
                texture_entry(2),
                texture_entry(3),
            ],
        });

        // ---- Shader modules + pipelines --------------------------------------
        let base_heightmap_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("base_heightmap"),
            source: wgpu::ShaderSource::Wgsl(format!("{COMMON_WGSL}\n{BASE_HEIGHTMAP_FS}").into()),
        });
        let terrain_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain"),
            // phacelle_noise calls hash(), which lives in noise.wgsl.
            source: wgpu::ShaderSource::Wgsl(
                format!("{COMMON_WGSL}\n{NOISE_WGSL}\n{TERRAIN_FS}").into(),
            ),
        });
        let detail_noise_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("detail_noise"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{COMMON_WGSL}\n{NOISE_WGSL}\n{DETAIL_NOISE_FS}").into(),
            ),
        });
        let image_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("image"),
            source: wgpu::ShaderSource::Wgsl(format!("{COMMON_WGSL}\n{IMAGE_FS}").into()),
        });

        // Pipeline layouts: A & C only need uniforms; B needs uniforms + base_heightmap;
        // Image needs uniforms + 3 textures.
        let uniform_only_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("uniform-only pipeline layout"),
                bind_group_layouts: &[Some(&uniform_only_layout)],
                immediate_size: 0,
            });
        let terrain_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("terrain pipeline layout"),
                bind_group_layouts: &[Some(&terrain_layout)],
                immediate_size: 0,
            });
        let image_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("image layout"),
                bind_group_layouts: &[Some(&image_layout)],
                immediate_size: 0,
            });

        let base_heightmap_pipeline = make_pipeline(
            &device,
            "base_heightmap pipeline",
            &uniform_only_pipeline_layout,
            &base_heightmap_module,
            wgpu::TextureFormat::Rgba32Float,
        );
        let terrain_pipeline = make_pipeline(
            &device,
            "terrain pipeline",
            &terrain_pipeline_layout,
            &terrain_module,
            wgpu::TextureFormat::Rgba32Float,
        );
        let detail_noise_pipeline = make_pipeline(
            &device,
            "detail_noise pipeline",
            &uniform_only_pipeline_layout,
            &detail_noise_module,
            wgpu::TextureFormat::Rgba32Float,
        );
        let image_pipeline = make_pipeline(
            &device,
            "image pipeline",
            &image_pipeline_layout,
            &image_module,
            swapchain_format,
        );

        // ---- Bind groups -----------------------------------------------------
        let base_heightmap_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("base_heightmap bg"),
            layout: &uniform_only_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });
        let detail_noise_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("detail_noise bg"),
            layout: &uniform_only_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });
        let terrain_bg = make_terrain_bg(&device, &terrain_layout, &uniform_buf, &base_heightmap_view);
        let image_bg = make_image_bg(
            &device,
            &image_layout,
            &uniform_buf,
            &base_heightmap_view,
            &detail_noise_view,
            &terrain_view,
        );

        Self {
            canvas,
            surface,
            device,
            queue,
            surface_caps,
            swapchain_format,
            width,
            height,
            world_center: [0.0, 0.0],
            camera_distance: DEFAULT_CAMERA_DISTANCE,
            camera_tilt: DEFAULT_CAMERA_TILT,
            dragging: false,
            last_mouse: (0.0, 0.0),
            uniform_buf,
            image_layout,
            terrain_layout,
            base_heightmap_pipeline,
            terrain_pipeline,
            detail_noise_pipeline,
            image_pipeline,
            base_heightmap_bg,
            detail_noise_bg,
            _base_heightmap_tex: base_heightmap_tex,
            _terrain_tex: terrain_tex,
            _detail_noise_tex: detail_noise_tex,
            base_heightmap_view,
            terrain_view,
            detail_noise_view,
            terrain_bg,
            image_bg,
        }
    }

    /// Build the current uniform block from State and push it to the GPU.
    /// Called whenever any uniform field changes.
    fn write_uniforms(&self) {
        let u = Uniforms::new(
            self.width,
            self.height,
            self.world_center,
            self.camera_distance,
            self.camera_tilt,
        );
        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        self.canvas.set_width(width);
        self.canvas.set_height(height);
        self.width = width;
        self.height = height;

        // Re-configure the swapchain.
        self.surface.configure(
            &self.device,
            &surface_config(self.swapchain_format, &self.surface_caps, width, height),
        );

        // Update iResolution / aspect-derived world half-size + current pan.
        self.write_uniforms();

        // Reallocate offscreen textures and any bind groups that reference
        // them (image_bg and terrain_bg sample base_heightmap / detail_noise / terrain).
        let (a_tex, a_view) = make_offscreen(&self.device, "base_heightmap", width, height);
        let (b_tex, b_view) = make_offscreen(&self.device, "terrain", width, height);
        let (c_tex, c_view) = make_offscreen(&self.device, "detail_noise", width, height);
        self._base_heightmap_tex = a_tex;
        self._terrain_tex = b_tex;
        self._detail_noise_tex = c_tex;
        self.base_heightmap_view = a_view;
        self.terrain_view = b_view;
        self.detail_noise_view = c_view;
        self.terrain_bg = make_terrain_bg(
            &self.device,
            &self.terrain_layout,
            &self.uniform_buf,
            &self.base_heightmap_view,
        );
        self.image_bg = make_image_bg(
            &self.device,
            &self.image_layout,
            &self.uniform_buf,
            &self.base_heightmap_view,
            &self.detail_noise_view,
            &self.terrain_view,
        );
    }

    fn render(&self) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            other => panic!("failed to acquire surface texture: {other:?}"),
        };
        let frame_view = frame.texture.create_view(&Default::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame encoder"),
            });

        let mut offscreen_pass = |label: &'static str,
                                  view: &wgpu::TextureView,
                                  pipeline: &wgpu::RenderPipeline,
                                  bind_group: &wgpu::BindGroup| {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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
            rpass.set_pipeline(pipeline);
            rpass.set_bind_group(0, bind_group, &[]);
            rpass.draw(0..3, 0..1);
        };

        offscreen_pass(
            "base_heightmap pass",
            &self.base_heightmap_view,
            &self.base_heightmap_pipeline,
            &self.base_heightmap_bg,
        );
        offscreen_pass(
            "detail_noise pass",
            &self.detail_noise_view,
            &self.detail_noise_pipeline,
            &self.detail_noise_bg,
        );
        // Buffer B reads Buffer A's output — must run after base_heightmap.
        offscreen_pass(
            "terrain pass",
            &self.terrain_view,
            &self.terrain_pipeline,
            &self.terrain_bg,
        );

        // Image pass — samples Buffer A/B/C and writes to the swapchain.
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("image pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame_view,
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
            rpass.set_pipeline(&self.image_pipeline);
            rpass.set_bind_group(0, &self.image_bg, &[]);
            rpass.draw(0..3, 0..1);
        }

        self.queue.submit([encoder.finish()]);
        frame.present();
    }
}

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

/// Compute the canvas backing-buffer size as (CSS pixels * devicePixelRatio).
fn compute_canvas_size(canvas: &HtmlCanvasElement) -> (u32, u32) {
    let window = web_sys::window().expect("no window");
    let dpr = window.device_pixel_ratio();
    let css_w = canvas.client_width() as f64;
    let css_h = canvas.client_height() as f64;
    let w = ((css_w * dpr).max(1.0)) as u32;
    let h = ((css_h * dpr).max(1.0)) as u32;
    (w, h)
}

fn surface_config(
    format: wgpu::TextureFormat,
    caps: &wgpu::SurfaceCapabilities,
    width: u32,
    height: u32,
) -> wgpu::SurfaceConfiguration {
    wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width,
        height,
        present_mode: caps.present_modes[0],
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    }
}

fn make_offscreen(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    (tex, view)
}

fn make_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    target_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn make_image_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buf: &wgpu::Buffer,
    base_heightmap_view: &wgpu::TextureView,
    detail_noise_view: &wgpu::TextureView,
    terrain_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("image bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
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
                resource: wgpu::BindingResource::TextureView(terrain_view),
            },
        ],
    })
}

fn make_terrain_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buf: &wgpu::Buffer,
    base_heightmap_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("terrain bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(base_heightmap_view),
            },
        ],
    })
}

/// Look up a `<canvas>` element by id, panicking with a helpful message if it's missing.
fn canvas_by_id(id: &str) -> HtmlCanvasElement {
    web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id(id))
        .and_then(|el| el.dyn_into::<HtmlCanvasElement>().ok())
        .unwrap_or_else(|| panic!("no <canvas id=\"{id}\"> on the page"))
}
