//! Low-level wgpu wrapper. Owns the surface/device/queue + a few factory
//! helpers. Knows nothing about heightmaps, cameras, or any other app concept.

use wasm_bindgen::JsCast;
use web_sys::{HtmlCanvasElement, console};

/// Owns wgpu surface/device/queue and tracks the current swapchain size.
pub struct GpuContext {
    pub canvas: HtmlCanvasElement,
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface_caps: wgpu::SurfaceCapabilities,
    pub swapchain_format: wgpu::TextureFormat,
    pub width: u32,
    pub height: u32,
}

impl GpuContext {
    /// Build a `GpuContext` for the given on-page canvas. Sizes the canvas
    /// backing buffer to `(CSS px) * devicePixelRatio` and configures a
    /// swapchain at that size.
    pub async fn new(canvas: HtmlCanvasElement) -> Self {
        let (width, height) = compute_canvas_size(&canvas);
        canvas.set_width(width);
        canvas.set_height(height);

        // `webgpu_detection` falls back to WebGL2 if WebGPU is unavailable.
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

        Self {
            canvas,
            surface,
            device,
            queue,
            surface_caps,
            swapchain_format,
            width,
            height,
        }
    }

    /// Reconfigure the swapchain to a new size. World-anchored layers don't
    /// depend on swapchain size, so they're untouched.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.canvas.set_width(width);
        self.canvas.set_height(height);
        self.width = width;
        self.height = height;
        self.surface.configure(
            &self.device,
            &surface_config(self.swapchain_format, &self.surface_caps, width, height),
        );
    }

    /// Acquire the next swapchain texture + a default view of it.
    /// Caller must call `frame.texture.present()` after submission.
    pub fn acquire_frame(&self) -> (wgpu::SurfaceTexture, wgpu::TextureView) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            other => panic!("failed to acquire surface texture: {other:?}"),
        };
        let view = frame.texture.create_view(&Default::default());
        (frame, view)
    }

    pub fn encoder(&self, label: &str) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
    }

    pub fn submit(&self, encoder: wgpu::CommandEncoder) {
        self.queue.submit([encoder.finish()]);
    }
}

/// Pixel format used for every world-anchored offscreen layer. Rgba16Float is
/// filterable in WebGPU baseline (no `float32-filterable` feature needed), so
/// the image pass can `textureSampleLevel` the cached layer textures with a
/// linear sampler. f16 in [0, 1] has ~3 decimal digits of precision — fine
/// for elevation in fractions of HEIGHT_SCALE_M.
pub const LAYER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Build a fixed-size `LAYER_FORMAT` texture suitable as both a render target
/// and a shader input. Used for offscreen passes (world layers).
pub fn make_offscreen(
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
        format: LAYER_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    (tex, view)
}

/// Build a render pipeline with our usual fullscreen-triangle setup
/// (no vertex buffers, no depth/stencil, no MSAA).
pub fn make_pipeline(
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

/// Compute the canvas backing-buffer size as (CSS pixels * devicePixelRatio).
pub fn compute_canvas_size(canvas: &HtmlCanvasElement) -> (u32, u32) {
    let window = web_sys::window().expect("no window");
    let dpr = window.device_pixel_ratio();
    let css_w = canvas.client_width() as f64;
    let css_h = canvas.client_height() as f64;
    let w = ((css_w * dpr).max(1.0)) as u32;
    let h = ((css_h * dpr).max(1.0)) as u32;
    (w, h)
}

/// Look up a `<canvas>` element by id, panicking with a helpful message if it's missing.
pub fn canvas_by_id(id: &str) -> HtmlCanvasElement {
    web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id(id))
        .and_then(|el| el.dyn_into::<HtmlCanvasElement>().ok())
        .unwrap_or_else(|| panic!("no <canvas id=\"{id}\"> on the page"))
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
