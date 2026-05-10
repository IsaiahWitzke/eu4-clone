//! Crate entry point. The wasm binary boots here, builds a `Renderer`,
//! wires DOM events to it, and kicks off async fetches for the runtime
//! assets (heightmap + water mask). All wgpu / shader work lives in the
//! sibling modules.

mod assets;
mod camera;
mod gpu;
mod passes;
mod renderer;
mod world_layer;

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use wasm_bindgen::JsCast;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;
use web_sys::console;

use crate::camera::HOVER_PICK_Y;
use crate::gpu::canvas_by_id;
use crate::renderer::Renderer;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    wasm_bindgen_futures::spawn_local(run());
}

// ---- Frame profiler -------------------------------------------------------

/// How many frame samples to accumulate before logging stats to the console.
/// At ~60 FPS that's roughly one log line per second.
const PROFILER_LOG_EVERY: usize = 60;

/// Rolling frame-time profiler. Records:
///   * `cpu_ms` — wallclock time spent inside `frame()` (encode + submit;
///     wgpu doesn't block on GPU completion in `present()`, so this is *not*
///     the GPU cost).
///   * `pacing_ms` — inter-rAF delta (timestamp passed by the browser).
///     This is the *real* paint cadence; if it sits at 16.7 ms you're at
///     60 FPS, if it sits at 33 ms you're at 30 FPS, etc.
///   * `last_size` — the canvas backing-store resolution (in physical
///     pixels). Pixel count is the dominant factor for our raymarched
///     fragment shader, so this lets you correlate window size with cost.
///
/// Open the browser DevTools console to see lines like:
///   `frame@2880×1800 n=60: cpu avg=0.07ms (0.00–0.20) | paint 32.5ms (~31 FPS)`
struct FrameProfiler {
    cpu_samples: Vec<f64>,
    pacing_samples: Vec<f64>,
    last_size: (u32, u32),
}

impl FrameProfiler {
    fn new() -> Self {
        Self {
            cpu_samples: Vec::with_capacity(PROFILER_LOG_EVERY),
            pacing_samples: Vec::with_capacity(PROFILER_LOG_EVERY),
            last_size: (0, 0),
        }
    }

    fn record(&mut self, cpu_ms: f64, pacing_ms: f64, size: (u32, u32)) {
        self.cpu_samples.push(cpu_ms);
        // First rAF callback has no previous timestamp — skip the 0 sample
        // so it doesn't pollute min/avg.
        if pacing_ms > 0.0 {
            self.pacing_samples.push(pacing_ms);
        }
        self.last_size = size;
        if self.cpu_samples.len() >= PROFILER_LOG_EVERY {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if self.cpu_samples.is_empty() {
            return;
        }
        let stats = |samples: &[f64]| -> (f64, f64, f64) {
            let n = samples.len().max(1) as f64;
            let sum: f64 = samples.iter().sum();
            let mean = sum / n;
            let min = samples
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            let max = samples
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            (min, mean, max)
        };
        let (cmin, cmean, cmax) = stats(&self.cpu_samples);
        let (pmin, pmean, pmax) = stats(&self.pacing_samples);
        let cn = self.cpu_samples.len();
        let pfps = if pmean > 0.0 { 1000.0 / pmean } else { 0.0 };
        let (w, h) = self.last_size;
        console::log_1(
            &format!(
                "frame@{w}\u{00d7}{h} n={cn}: cpu avg={cmean:.2}ms ({cmin:.2}\u{2013}{cmax:.2}) | \
                 paint avg={pmean:.2}ms ({pmin:.2}\u{2013}{pmax:.2}) (~{pfps:.0} FPS)"
            )
            .into(),
        );
        self.cpu_samples.clear();
        self.pacing_samples.clear();
    }
}

fn now_ms() -> f64 {
    web_sys::window()
        .expect("no window")
        .performance()
        .expect("no performance")
        .now()
}

/// Schedule a frame via `requestAnimationFrame`. If a redraw is already
/// pending for the next animation frame, this is a no-op — so a flood of
/// `mousemove` events during a drag all coalesce into a single render at
/// most once per display refresh.
fn schedule_frame(
    renderer: &Rc<RefCell<Renderer>>,
    pending: &Rc<Cell<bool>>,
    profiler: &Rc<RefCell<FrameProfiler>>,
    last_raf: &Rc<Cell<Option<f64>>>,
) {
    if pending.get() {
        return;
    }
    pending.set(true);

    let r = renderer.clone();
    let p = pending.clone();
    let prof = profiler.clone();
    let last = last_raf.clone();
    // rAF callbacks receive a `DOMHighResTimeStamp` (ms since navigationStart);
    // the delta between consecutive timestamps is the real paint cadence.
    let cb = Closure::once_into_js(move |raf_t: f64| {
        p.set(false);
        let cpu_t0 = now_ms();
        let size = {
            let mut rb = r.borrow_mut();
            rb.frame();
            (rb.gpu.width, rb.gpu.height)
        };
        let cpu_dt = now_ms() - cpu_t0;
        let pacing_dt = last
            .replace(Some(raf_t))
            .map(|prev| raf_t - prev)
            .unwrap_or(0.0);
        prof.borrow_mut().record(cpu_dt, pacing_dt, size);
    });
    web_sys::window()
        .expect("no window")
        .request_animation_frame(cb.unchecked_ref())
        .expect("requestAnimationFrame failed");
}

async fn run() {
    let canvas = canvas_by_id("game");
    let renderer = Rc::new(RefCell::new(Renderer::new(canvas).await));
    renderer.borrow_mut().frame();

    // Shared frame-scheduling state. `pending` is set when a frame has been
    // requested for the next rAF tick; cleared inside the rAF callback. This
    // guarantees we render at most once per display refresh, regardless of
    // how many `mousemove` / key events fire in between.
    let pending = Rc::new(Cell::new(false));
    let profiler = Rc::new(RefCell::new(FrameProfiler::new()));
    // Last rAF timestamp — used to compute paint cadence (the real frame
    // pacing the browser is achieving, possibly capped by GPU work).
    let last_raf: Rc<Cell<Option<f64>>> = Rc::new(Cell::new(None));

    let window = web_sys::window().expect("no window");

    // ---- Resize ---------------------------------------------------------
    {
        let r = renderer.clone();
        let p = pending.clone();
        let prof = profiler.clone();
        let lr = last_raf.clone();
        let on_resize = Closure::<dyn FnMut()>::new(move || {
            r.borrow_mut().handle_resize();
            schedule_frame(&r, &p, &prof, &lr);
        });
        window.set_onresize(Some(on_resize.as_ref().unchecked_ref()));
        on_resize.forget();
    }

    // ---- Keyboard: arrow-key pan, Q/E tilt, +/- zoom ---------------------
    {
        let r = renderer.clone();
        let p = pending.clone();
        let prof = profiler.clone();
        let lr = last_raf.clone();
        let on_keydown = Closure::<dyn FnMut(_)>::new(move |e: web_sys::KeyboardEvent| {
            let pan_step: f32 = 10.0;     // 10 km / press — sized for the Swiss bbox
            let tilt_step: f32 = 0.025;
            let zoom_factor: f32 = 1.15;  // ~15% per press; 1.05 was glacial at km scale

            let mut handled = true;
            {
                let mut rb = r.borrow_mut();
                match e.key().as_str() {
                    "ArrowUp" => rb.camera_mut().pan_world(0.0, pan_step),
                    "ArrowDown" => rb.camera_mut().pan_world(0.0, -pan_step),
                    "ArrowLeft" => rb.camera_mut().pan_world(-pan_step, 0.0),
                    "ArrowRight" => rb.camera_mut().pan_world(pan_step, 0.0),
                    "q" | "Q" => rb.camera_mut().tilt_by(-tilt_step),
                    "e" | "E" => rb.camera_mut().tilt_by(tilt_step),
                    "=" | "+" => rb.camera_mut().zoom(1.0 / zoom_factor),
                    "-" | "_" => rb.camera_mut().zoom(zoom_factor),
                    "m" | "M" => rb.camera_mut().cycle_map_mode(),
                    _ => handled = false,
                }
            }
            if !handled {
                return;
            }
            e.prevent_default();
            schedule_frame(&r, &p, &prof, &lr);
        });
        window
            .add_event_listener_with_callback("keydown", on_keydown.as_ref().unchecked_ref())
            .expect("failed to attach keydown listener");
        on_keydown.forget();
    }

    // ---- Click-and-drag panning -----------------------------------------
    let dragging: Rc<RefCell<Option<(f32, f32)>>> = Rc::new(RefCell::new(None));

    {
        let drag = dragging.clone();
        let on_mousedown = Closure::<dyn FnMut(_)>::new(move |e: web_sys::MouseEvent| {
            *drag.borrow_mut() = Some((e.client_x() as f32, e.client_y() as f32));
            e.prevent_default();
        });
        window
            .add_event_listener_with_callback("mousedown", on_mousedown.as_ref().unchecked_ref())
            .expect("failed to attach mousedown listener");
        on_mousedown.forget();
    }

    {
        let r = renderer.clone();
        let p = pending.clone();
        let prof = profiler.clone();
        let lr = last_raf.clone();
        let drag = dragging.clone();
        let on_mousemove = Closure::<dyn FnMut(_)>::new(move |e: web_sys::MouseEvent| {
            let mx = e.client_x() as f32;
            let my = e.client_y() as f32;

            let needs_redraw = {
                let mut rb = r.borrow_mut();
                let css_w = rb.canvas().client_width().max(1) as f32;
                let css_h = rb.canvas().client_height().max(1) as f32;

                // Drag pan (only active while a button is held).
                let mut drag_ref = drag.borrow_mut();
                let drag_pan = if let Some((lx, ly)) = *drag_ref {
                    *drag_ref = Some((mx, my));
                    rb.camera_mut().pan_pixels(mx - lx, my - ly, css_w, css_h);
                    true
                } else {
                    false
                };
                drop(drag_ref);

                // Always update hover so the highlight follows the cursor
                // even when the user isn't dragging.
                let hover_changed = rb.update_hover(mx, my, css_w, css_h);
                drag_pan || hover_changed
            };

            if needs_redraw {
                schedule_frame(&r, &p, &prof, &lr);
            }
        });
        window
            .add_event_listener_with_callback("mousemove", on_mousemove.as_ref().unchecked_ref())
            .expect("failed to attach mousemove listener");
        on_mousemove.forget();
    }

    {
        let drag = dragging.clone();
        let on_mouseup = Closure::<dyn FnMut(_)>::new(move |_e: web_sys::MouseEvent| {
            *drag.borrow_mut() = None;
        });
        window
            .add_event_listener_with_callback("mouseup", on_mouseup.as_ref().unchecked_ref())
            .expect("failed to attach mouseup listener");
        on_mouseup.forget();
    }

    // ---- Mouse-wheel zoom (toward cursor) -------------------------------
    {
        let r = renderer.clone();
        let p = pending.clone();
        let prof = profiler.clone();
        let lr = last_raf.clone();
        let on_wheel = Closure::<dyn FnMut(_)>::new(move |e: web_sys::WheelEvent| {
            // Suppress native page scrolling; we want every wheel tick to
            // zoom the map instead.
            e.prevent_default();

            // Exponential zoom: a typical "line" wheel tick is ~100 dy units,
            // so 100 × 0.0015 ≈ +15% zoom factor per notch. Trackpads emit
            // many smaller ticks so the cumulative speed is similar.
            let dy = e.delta_y() as f32;
            let factor = (dy * 0.0015).exp();
            let mx = e.client_x() as f32;
            let my = e.client_y() as f32;

            let mut rb = r.borrow_mut();
            let css_w = rb.canvas().client_width().max(1) as f32;
            let css_h = rb.canvas().client_height().max(1) as f32;

            // Zoom toward the cursor: pick the world XZ under the cursor
            // before and after the zoom, then shift world_center by their
            // difference so the same world point stays under the cursor.
            let before = rb
                .camera_mut()
                .pick_world_xz(mx, my, css_w, css_h, HOVER_PICK_Y);
            rb.camera_mut().zoom(factor);
            let after = rb
                .camera_mut()
                .pick_world_xz(mx, my, css_w, css_h, HOVER_PICK_Y);
            if let (Some(b), Some(a)) = (before, after) {
                let cam = rb.camera_mut();
                cam.world_center[0] += b[0] - a[0];
                cam.world_center[1] += b[1] - a[1];
            }

            // Hover position changes (camera moved); refresh the highlight.
            rb.update_hover(mx, my, css_w, css_h);
            drop(rb);

            schedule_frame(&r, &p, &prof, &lr);
        });
        // `passive: false` is required for prevent_default() to work on a
        // wheel listener — the browser default for wheel/touchmove on the
        // root has been passive since 2017 for scroll-jank reasons.
        let opts = web_sys::AddEventListenerOptions::new();
        opts.set_passive(false);
        window
            .add_event_listener_with_callback_and_add_event_listener_options(
                "wheel",
                on_wheel.as_ref().unchecked_ref(),
                &opts,
            )
            .expect("failed to attach wheel listener");
        on_wheel.forget();
    }

    console::log_1(&"ready (resize + arrow-key pan + mouse drag)".into());

    // ---- Async asset loading --------------------------------------------
    spawn_load(
        renderer.clone(),
        "./heightmap.png",
        |r, decoded| r.set_world_heightmap(decoded.width, decoded.height, &decoded.bytes),
    );
    spawn_load(
        renderer.clone(),
        "./water_mask.png",
        |r, decoded| r.set_water_mask(decoded.width, decoded.height, &decoded.bytes),
    );
    spawn_load(
        renderer.clone(),
        "./biome_mask.png",
        |r, decoded| r.set_biome_mask(decoded.width, decoded.height, &decoded.bytes),
    );
    spawn_load(
        renderer.clone(),
        "./province_mask.png",
        |r, decoded| r.set_province_mask(decoded.width, decoded.height, &decoded.bytes),
    );
    spawn_load(
        renderer.clone(),
        "./border_sdf.png",
        |r, decoded| r.set_border_sdf(decoded.width, decoded.height, &decoded.bytes),
    );
}

/// Spawn a `fetch + decode + apply` task. `apply` mutates the renderer with
/// the decoded bytes; we then trigger one frame so the user sees the new data.
fn spawn_load(
    renderer: Rc<RefCell<Renderer>>,
    url: &'static str,
    apply: fn(&mut Renderer, &assets::DecodedPng),
) {
    wasm_bindgen_futures::spawn_local(async move {
        match assets::fetch_png(url).await {
            Ok(decoded) => {
                console::log_1(
                    &format!(
                        "loaded {}: {}x{} ({}-bit, {} bytes)",
                        url,
                        decoded.width,
                        decoded.height,
                        decoded.bit_depth,
                        decoded.bytes.len()
                    )
                    .into(),
                );
                let mut rb = renderer.borrow_mut();
                apply(&mut rb, &decoded);
                rb.frame();
            }
            Err(err) => {
                console::error_1(&format!("failed to load {url}: {err:?}").into());
            }
        }
    });
}
