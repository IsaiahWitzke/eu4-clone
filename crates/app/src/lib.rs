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

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::JsCast;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;
use web_sys::console;

use crate::gpu::canvas_by_id;
use crate::renderer::Renderer;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    wasm_bindgen_futures::spawn_local(run());
}

async fn run() {
    let canvas = canvas_by_id("game");
    let renderer = Rc::new(RefCell::new(Renderer::new(canvas).await));
    renderer.borrow_mut().frame();

    let window = web_sys::window().expect("no window");

    // ---- Resize ---------------------------------------------------------
    {
        let r = renderer.clone();
        let on_resize = Closure::<dyn FnMut()>::new(move || {
            let mut rb = r.borrow_mut();
            rb.handle_resize();
            rb.frame();
        });
        window.set_onresize(Some(on_resize.as_ref().unchecked_ref()));
        on_resize.forget();
    }

    // ---- Keyboard: arrow-key pan, Q/E tilt, +/- zoom ---------------------
    {
        let r = renderer.clone();
        let on_keydown = Closure::<dyn FnMut(_)>::new(move |e: web_sys::KeyboardEvent| {
            let pan_step: f32 = 1.0; // 1 km / press
            let tilt_step: f32 = 0.025;
            let zoom_factor: f32 = 1.05;

            let mut rb = r.borrow_mut();
            let mut handled = true;
            match e.key().as_str() {
                "ArrowUp" => rb.camera_mut().pan_world(0.0, pan_step),
                "ArrowDown" => rb.camera_mut().pan_world(0.0, -pan_step),
                "ArrowLeft" => rb.camera_mut().pan_world(-pan_step, 0.0),
                "ArrowRight" => rb.camera_mut().pan_world(pan_step, 0.0),
                "q" | "Q" => rb.camera_mut().tilt_by(-tilt_step),
                "e" | "E" => rb.camera_mut().tilt_by(tilt_step),
                "=" | "+" => rb.camera_mut().zoom(1.0 / zoom_factor),
                "-" | "_" => rb.camera_mut().zoom(zoom_factor),
                _ => handled = false,
            }
            if !handled {
                return;
            }
            e.prevent_default();
            rb.frame();
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
        let drag = dragging.clone();
        let on_mousemove = Closure::<dyn FnMut(_)>::new(move |e: web_sys::MouseEvent| {
            let mut drag_ref = drag.borrow_mut();
            let Some((lx, ly)) = *drag_ref else {
                return;
            };
            let mx = e.client_x() as f32;
            let my = e.client_y() as f32;
            *drag_ref = Some((mx, my));
            drop(drag_ref);

            let mut rb = r.borrow_mut();
            let css_w = rb.canvas().client_width().max(1) as f32;
            let css_h = rb.canvas().client_height().max(1) as f32;
            rb.camera_mut().pan_pixels(mx - lx, my - ly, css_w, css_h);
            rb.frame();
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
