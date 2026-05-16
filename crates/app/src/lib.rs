//! Crate entry point. The wasm binary boots here, builds a `Renderer`,
//! wires DOM events to it, and kicks off async fetches for the runtime
//! assets (heightmap + water mask). All wgpu / shader work lives in the
//! sibling modules.

mod assets;
mod camera;
mod gpu;
mod labels;
mod passes;
mod renderer;
mod settlements;
mod tiles;
mod ui;

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use wasm_bindgen::JsCast;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;
use web_sys::console;

use crate::camera::HOVER_PICK_Y;
use crate::gpu::canvas_by_id;
use crate::renderer::Renderer;
use crate::ui::CityPanel;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    wasm_bindgen_futures::spawn_local(run());
}

/// Schedule a frame via `requestAnimationFrame`. If a redraw is already
/// pending for the next animation frame, this is a no-op — so a flood of
/// `mousemove` events during a drag all coalesce into a single render at
/// most once per display refresh.
///
/// The mousemove / wheel handlers stash the cursor position in
/// `latest_mouse` without doing any work; we pull it (via `take`) once
/// per painted frame and run the drag-pan + hover update here. This
/// caps per-event cost to a single `Cell::set` regardless of how often
/// mousemove fires (1000 Hz gaming mice, 120 Hz trackpads, etc.) — the
/// real work runs at rAF cadence (≤ display refresh).
///
/// At the end of each rAF callback we call back into `schedule_frame`
/// so frames keep flowing even when there's no user input. This keeps
/// the spinner overlay spinning so its smoothness reflects whether
/// `frame()` is paced by the display refresh or by our own work.
fn schedule_frame(
    renderer: &Rc<RefCell<Renderer>>,
    pending: &Rc<Cell<bool>>,
    latest_mouse: &Rc<Cell<Option<(f32, f32)>>>,
    dragging: &Rc<RefCell<Option<(f32, f32)>>>,
) {
    if pending.get() {
        return;
    }
    pending.set(true);

    let r = renderer.clone();
    let p = pending.clone();
    let lm = latest_mouse.clone();
    let dr = dragging.clone();
    let cb = Closure::once_into_js(move |_raf_t: f64| {
        p.set(false);
        {
            let mut rb = r.borrow_mut();
            // Consume the latest cursor position. `take` (vs `get`) means
            // non-mouse schedule_frame paths (keydown, resize) don't
            // re-run hover with stale coords — the scan only fires when
            // a real mousemove has happened since the last paint.
            if let Some((mx, my)) = lm.take() {
                let css_w = rb.canvas().client_width().max(1) as f32;
                let css_h = rb.canvas().client_height().max(1) as f32;
                {
                    // Drag-pan: only active while a button is held.
                    // `dragging` stores the cursor position as-of the
                    // *last consumed mousemove*, so the pan delta is
                    // exactly (cursor_now - cursor_at_last_frame).
                    let mut drag_ref = dr.borrow_mut();
                    if let Some((lx, ly)) = *drag_ref {
                        rb.camera_mut().pan_pixels(mx - lx, my - ly, css_w, css_h);
                        *drag_ref = Some((mx, my));
                    }
                }
                rb.update_hover(mx, my, css_w, css_h);
            }
            rb.frame();
        }

        // Tail-chain the next frame. The `pending` flag guard in
        // `schedule_frame` means stacking calls from event handlers
        // (mousemove, wheel, keydown, …) still coalesces to one rAF
        // per refresh — this call just keeps the loop going when no
        // events fire, so the spinner overlay keeps animating.
        schedule_frame(&r, &p, &lm, &dr);
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

    // ---- UI overlay -----------------------------------------------------
    // The HTML-side markup lives in `web/index.html`; we just hold cached
    // element handles in the Rust side. Wrap in `Rc<RefCell<...>>` so
    // multiple event closures can share it.
    let document = web_sys::window()
        .expect("no window")
        .document()
        .expect("no document");
    let city_panel = Rc::new(CityPanel::new(&document));

    // Wire the panel's close button to hide the panel.
    {
        let panel = city_panel.clone();
        if let Some(btn) = document.get_element_by_id("cp-close") {
            let on_close = Closure::<dyn FnMut(_)>::new(move |_e: web_sys::MouseEvent| {
                panel.hide();
            });
            btn.add_event_listener_with_callback("click", on_close.as_ref().unchecked_ref())
                .expect("failed to attach close listener");
            on_close.forget();
        }
    }

    // Shared frame-scheduling state. `pending` is set when a frame has been
    // requested for the next rAF tick; cleared inside the rAF callback. This
    // guarantees we render at most once per display refresh, regardless of
    // how many `mousemove` / key events fire in between.
    let pending = Rc::new(Cell::new(false));
    // Most recent cursor position (in CSS pixels) from a mousemove /
    // wheel event. The handlers store into this Cell without doing any
    // work; the rAF callback `take`s it and runs the drag-pan + hover
    // update once per painted frame. Decouples per-event cost from
    // per-frame cost — a 1000 Hz mouse stops translating into 1000
    // hover scans per second.
    let latest_mouse: Rc<Cell<Option<(f32, f32)>>> = Rc::new(Cell::new(None));
    // `dragging` is Some(pos) iff a mouse button is held. The position
    // is updated by the rAF callback to whichever cursor pos it just
    // consumed, so the next frame's pan delta is `(cursor_now -
    // cursor_at_last_paint)`. On mouseup, we compare the stored value
    // against the mouseup position to tell a click from a drag.
    let dragging: Rc<RefCell<Option<(f32, f32)>>> = Rc::new(RefCell::new(None));

    let window = web_sys::window().expect("no window");

    // ---- Resize ---------------------------------------------------------
    {
        let r = renderer.clone();
        let p = pending.clone();
        let lm = latest_mouse.clone();
        let drag = dragging.clone();
        let on_resize = Closure::<dyn FnMut()>::new(move || {
            r.borrow_mut().handle_resize();
            schedule_frame(&r, &p, &lm, &drag);
        });
        window.set_onresize(Some(on_resize.as_ref().unchecked_ref()));
        on_resize.forget();
    }

    // ---- Keyboard: arrow-key pan, Q/E tilt, +/- zoom ---------------------
    {
        let r = renderer.clone();
        let p = pending.clone();
        let lm = latest_mouse.clone();
        let drag = dragging.clone();
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
                    // The old T-key toggle between raymarch and mesh paths is gone
                    // — there's only one render path now (tile pyramid, landing in
                    // Phase 2/3 of the rewrite).
                    _ => handled = false,
                }
            }
            if !handled {
                return;
            }
            e.prevent_default();
            schedule_frame(&r, &p, &lm, &drag);
        });
        window
            .add_event_listener_with_callback("keydown", on_keydown.as_ref().unchecked_ref())
            .expect("failed to attach keydown listener");
        on_keydown.forget();
    }

    // ---- Click-and-drag panning -----------------------------------------
    /// Pixel slop within which a mousedown→mouseup pair counts as a
    /// "click" instead of a "drag". 4 px feels right for desktop mice;
    /// trackpads occasionally jiggle a couple of pixels even on a sharp
    /// click, but more than that and the user clearly meant to pan.
    const CLICK_SLOP_PX: f32 = 4.0;

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
        let lm = latest_mouse.clone();
        let drag = dragging.clone();
        let on_mousemove = Closure::<dyn FnMut(_)>::new(move |e: web_sys::MouseEvent| {
            // Per-event work is now exactly this: stash the cursor
            // position. The drag-pan + hover scan that used to run
            // here have moved into the rAF callback (`schedule_frame`),
            // so a 1000 Hz mouse no longer pays for 1000 hover scans
            // per second — the work runs once per painted frame.
            let mx = e.client_x() as f32;
            let my = e.client_y() as f32;
            lm.set(Some((mx, my)));
            schedule_frame(&r, &p, &lm, &drag);
        });
        window
            .add_event_listener_with_callback("mousemove", on_mousemove.as_ref().unchecked_ref())
            .expect("failed to attach mousemove listener");
        on_mousemove.forget();
    }

    {
        let drag = dragging.clone();
        let r = renderer.clone();
        let panel = city_panel.clone();
        let on_mouseup = Closure::<dyn FnMut(_)>::new(move |e: web_sys::MouseEvent| {
            // Convert mousedown→mouseup into a click iff the cursor
            // didn't move far. Otherwise it was a drag-pan and we just
            // clear the drag state.
            let down = drag.borrow_mut().take();
            if let Some((dx, dy)) = down {
                let mx = e.client_x() as f32;
                let my = e.client_y() as f32;
                let moved = ((mx - dx).powi(2) + (my - dy).powi(2)).sqrt();
                if moved <= CLICK_SLOP_PX {
                    let rb = r.borrow();
                    let css_w = rb.canvas().client_width().max(1) as f32;
                    let css_h = rb.canvas().client_height().max(1) as f32;
                    match rb.pick_settlement_at(mx, my, css_w, css_h) {
                        Some(city) => panel.show(city),
                        None => panel.hide(),
                    }
                }
            }
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
        let lm = latest_mouse.clone();
        let drag = dragging.clone();
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

            {
                let mut rb = r.borrow_mut();
                let css_w = rb.canvas().client_width().max(1) as f32;
                let css_h = rb.canvas().client_height().max(1) as f32;

                // Zoom toward the cursor: pick the world XZ under the cursor
                // before and after the zoom, then shift world_center by their
                // difference so the same world point stays under the cursor.
                // This has to run synchronously — the math depends on the
                // *current* camera, not the post-rAF state.
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
            }

            // Camera moved; the cursor now picks a different world point,
            // so leave hover-refresh to the rAF callback (it'll consume
            // this cursor pos and re-run `update_hover`).
            lm.set(Some((mx, my)));
            schedule_frame(&r, &p, &lm, &drag);
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

    // Kick off the continuous rAF loop. From here on the rAF callback
    // tail-chains itself, so `frame()` runs once per display refresh
    // even when no input events fire. This drives the spinner overlay's
    // smoothness as a live frame-pacing indicator.
    schedule_frame(&renderer, &pending, &latest_mouse, &dragging);

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

    // cities.json — parsed into a LoadedSettlements (settlements +
    // realm-name map) and pushed into the renderer once the bytes
    // land. Until then the renderer keeps using its built-in Swiss
    // defaults.
    {
        let r = renderer.clone();
        wasm_bindgen_futures::spawn_local(async move {
            match assets::fetch_text("./cities.json").await {
                Ok(text) => match settlements::from_cities_json(&text) {
                    Ok(loaded) => {
                        console::log_1(
                            &format!(
                                "cities.json: loaded {} settlements ({} realms)",
                                loaded.settlements.len(),
                                loaded.realm_names.len(),
                            )
                            .into(),
                        );
                        let mut rb = r.borrow_mut();
                        rb.set_settlements(loaded);
                        rb.frame();
                    }
                    Err(e) => {
                        console::error_1(&format!("cities.json parse failed: {e}").into());
                    }
                },
                Err(err) => {
                    console::error_1(
                        &format!("failed to load cities.json: {err:?}").into(),
                    );
                }
            }
        });
    }

    // SDF glyph atlas — fetch the JSON metrics + the RGBA8 PNG
    // (`script/gen-glyph-atlas` produced both), then install them on
    // the renderer. The realm-labels render pass is a no-op until
    // both arrive, so failures here just leave the map unlabelled.
    {
        let r = renderer.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let json = match assets::fetch_text("./glyph_atlas.json").await {
                Ok(t) => t,
                Err(err) => {
                    console::error_1(
                        &format!("failed to load glyph_atlas.json: {err:?}").into(),
                    );
                    return;
                }
            };
            let png = match assets::fetch_rgba_png("./glyph_atlas.png").await {
                Ok(p) => p,
                Err(err) => {
                    console::error_1(
                        &format!("failed to load glyph_atlas.png: {err:?}").into(),
                    );
                    return;
                }
            };
            console::log_1(
                &format!(
                    "loaded glyph_atlas.png: {}x{} ({} bytes)",
                    png.width, png.height, png.bytes.len(),
                )
                .into(),
            );
            let mut rb = r.borrow_mut();
            match rb.set_glyph_atlas(&json, png.width, png.height, &png.bytes) {
                Ok(()) => rb.frame(),
                Err(e) => {
                    console::error_1(&format!("glyph_atlas: {e}").into());
                }
            }
        });
    }
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
