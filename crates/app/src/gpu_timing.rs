//! Per-pass GPU timing via `TIMESTAMP_QUERY`.
//!
//! The CPU-side `FrameProfiler` only sees how long it took to *encode +
//! submit* commands, which on wgpu is essentially free (drivers queue the
//! work and return). To see what the GPU is actually spending on each
//! pass, we ask the driver to record timestamps before and after each
//! pass; after the frame is submitted, we copy the timestamp buffer
//! back to the CPU and decode the deltas into milliseconds.
//!
//! # Slot layout
//!
//! The query set holds 6 timestamps, paired into 3 sections:
//!
//!   `[ layers_begin, layers_end, bake_begin, bake_end, image_begin, image_end ]`
//!
//! Each section corresponds to one `Section` variant; the renderer
//! attaches the appropriate `RenderPassTimestampWrites` to its render
//! passes. The 3 world-layer sub-passes (base_heightmap, detail_noise,
//! erosion) share the `Layers` section: the begin timestamp lives on
//! the first sub-pass, the end timestamp on the last.
//!
//! # Backpressure
//!
//! There is exactly *one* readback buffer in flight. If the previous
//! frame's `map_async` hasn't completed by the time the next frame
//! tries to record, we skip recording for that frame. With the readback
//! ready in <1 frame on a modern GPU, this is rare.
//!
//! # Backend support
//!
//! `TIMESTAMP_QUERY` is supported by the WebGPU backend in modern
//! Chrome (it must be enabled at the device level via the WebGPU API,
//! which wgpu does for us when we `required_features` it).
//! It is **not** supported by wgpu's WebGL2 backend, so on browsers
//! that fall back to WebGL the timer is disabled and `try_new` returns
//! `None`. The user-facing log line in that case explains why.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use web_sys::console;

use crate::gpu::GpuContext;

/// Total timestamp count: begin + end for each of 3 sections.
pub const TIMESTAMP_COUNT: u32 = 6;

/// Each section is a logically-grouped span of GPU work that we want
/// reported as a single line in the per-second log.
#[derive(Copy, Clone, Debug)]
pub enum Section {
    /// Combined time for the three world-layer sub-passes (only valid
    /// when the layer cache invalidated this frame; otherwise zero).
    Layers = 0,
    /// Realm-influence-field bake. Fires once per `set_settlements`
    /// (= once on startup, once when cities.json loads, etc.).
    Bake = 1,
    /// The big one: the swapchain raymarch pass.
    Image = 2,
}

impl Section {
    fn begin_index(self) -> u32 {
        2 * self as u32
    }
    fn end_index(self) -> u32 {
        2 * self as u32 + 1
    }
}

pub struct GpuTimer {
    /// Nanoseconds per timestamp tick (from `Queue::get_timestamp_period`).
    pub timestamp_period_ns: f32,
    pub query_set: wgpu::QuerySet,
    /// Transient buffer that `resolve_query_set` writes into. Must have
    /// `QUERY_RESOLVE | COPY_SRC` usage. Not mappable on its own.
    resolve_buf: wgpu::Buffer,
    /// CPU-readable buffer that we `copy_buffer_to_buffer` the resolve
    /// data into. `MAP_READ | COPY_DST`. Single-buffered to keep the
    /// state machine trivial; if we ever need overlapping frames, turn
    /// this into a small ring.
    readback_buf: wgpu::Buffer,
    /// True between submitting a readback and the `map_async` callback
    /// firing. Prevents us from issuing a second readback while one is
    /// still in flight.
    pending: Rc<Cell<bool>>,
    /// Rolling stats updated by the readback callback.
    stats: Rc<RefCell<GpuStats>>,
    /// Per-section "did we record this frame?" flags. Reset by
    /// `begin_frame`, set by `writes_for`, consumed by `after_submit`.
    section_recorded: [bool; 3],
}

impl GpuTimer {
    /// Try to build a `GpuTimer`. Returns `None` and logs a hint if the
    /// adapter doesn't expose `TIMESTAMP_QUERY` (e.g. running over the
    /// WebGL2 backend, or Chrome without unsafe-WebGPU enabled).
    pub fn try_new(gpu: &GpuContext) -> Option<Self> {
        if !gpu.device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            console::log_1(
                &"gpu_timing: TIMESTAMP_QUERY unsupported on this backend; \
                 per-pass GPU timings disabled. (WebGL2 fallback or Chrome \
                 without WebGPU? Try chrome://flags/#enable-unsafe-webgpu \
                 + restart.)"
                    .into(),
            );
            return None;
        }

        let timestamp_period_ns = gpu.queue.get_timestamp_period();

        let query_set = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gpu_timing query set"),
            ty: wgpu::QueryType::Timestamp,
            count: TIMESTAMP_COUNT,
        });

        let buffer_size = (TIMESTAMP_COUNT as u64) * 8;
        let resolve_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_timing resolve"),
            size: buffer_size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_timing readback"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Some(Self {
            timestamp_period_ns,
            query_set,
            resolve_buf,
            readback_buf,
            pending: Rc::new(Cell::new(false)),
            stats: Rc::new(RefCell::new(GpuStats::default())),
            section_recorded: [false; 3],
        })
    }

    /// Reset the per-frame state. Returns `true` iff the frame should
    /// record timestamps (= no readback in flight). When this returns
    /// `false`, callers must pass `None` for `timestamp_writes` to all
    /// passes this frame.
    pub fn begin_frame(&mut self) -> bool {
        self.section_recorded = [false; 3];
        !self.pending.get()
    }

    /// `RenderPassTimestampWrites` for the *whole* span of a section
    /// (begin + end on a single render pass). Use for `Bake` and
    /// `Image`.
    pub fn writes_full(&mut self, section: Section) -> wgpu::RenderPassTimestampWrites<'_> {
        self.section_recorded[section as usize] = true;
        wgpu::RenderPassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(section.begin_index()),
            end_of_pass_write_index: Some(section.end_index()),
        }
    }

    /// `RenderPassTimestampWrites` that records *only* the begin
    /// timestamp for a section. Pair with `writes_end` on a later
    /// render pass to bracket multiple passes (used for `Layers`).
    pub fn writes_begin(&mut self, section: Section) -> wgpu::RenderPassTimestampWrites<'_> {
        self.section_recorded[section as usize] = true;
        wgpu::RenderPassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(section.begin_index()),
            end_of_pass_write_index: None,
        }
    }

    /// `RenderPassTimestampWrites` that records *only* the end
    /// timestamp for a section.
    pub fn writes_end(&mut self, section: Section) -> wgpu::RenderPassTimestampWrites<'_> {
        // `section_recorded` already set by `writes_begin`; setting
        // it again is harmless.
        wgpu::RenderPassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: None,
            end_of_pass_write_index: Some(section.end_index()),
        }
    }

    /// Append the `resolve_query_set` + `copy_buffer_to_buffer` commands
    /// to `encoder`. Caller submits afterward, then calls
    /// `after_submit`.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        if !self.section_recorded.iter().any(|b| *b) {
            return;
        }
        encoder.resolve_query_set(
            &self.query_set,
            0..TIMESTAMP_COUNT,
            &self.resolve_buf,
            0,
        );
        encoder.copy_buffer_to_buffer(
            &self.resolve_buf,
            0,
            &self.readback_buf,
            0,
            (TIMESTAMP_COUNT as u64) * 8,
        );
    }

    /// Schedule the async readback. Must be called *after* the encoder
    /// containing the resolve commands has been submitted, so the GPU
    /// has work to drive the map_async to completion.
    pub fn after_submit(&mut self) {
        if !self.section_recorded.iter().any(|b| *b) {
            return;
        }
        self.pending.set(true);

        // Capture into the closure. Buffer is Arc-internal, so
        // cloning is cheap and gives the closure independent ownership
        // for `unmap`. Stats and pending are Rc.
        let buf = self.readback_buf.clone();
        let stats = self.stats.clone();
        let pending = self.pending.clone();
        let period_ns = self.timestamp_period_ns;
        let mask = self.section_recorded;

        self.readback_buf
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_err() {
                    pending.set(false);
                    return;
                }
                {
                    let data = buf.slice(..).get_mapped_range();
                    // Read each u64 via from_le_bytes to avoid
                    // alignment assumptions on the mapped range.
                    let mut tss = [0u64; TIMESTAMP_COUNT as usize];
                    for i in 0..TIMESTAMP_COUNT as usize {
                        let mut b = [0u8; 8];
                        b.copy_from_slice(&data[i * 8..(i + 1) * 8]);
                        tss[i] = u64::from_le_bytes(b);
                    }

                    let mut s = stats.borrow_mut();
                    if mask[0] {
                        s.layers.record(diff_ms(tss[0], tss[1], period_ns));
                    }
                    if mask[1] {
                        s.bake.record(diff_ms(tss[2], tss[3], period_ns));
                    }
                    if mask[2] {
                        s.image.record(diff_ms(tss[4], tss[5], period_ns));
                    }
                    s.maybe_log();
                }
                buf.unmap();
                pending.set(false);
            });
    }
}

/// Convert a (begin, end) timestamp pair into milliseconds. `end` may
/// equal `begin` (zero-cost section) — guard against subtractive
/// underflow with `saturating_sub`.
fn diff_ms(begin: u64, end: u64, period_ns: f32) -> f64 {
    let ticks = end.saturating_sub(begin) as f64;
    ticks * (period_ns as f64) / 1_000_000.0
}

#[derive(Default)]
struct PassAvg {
    n: u32,
    sum_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

impl PassAvg {
    fn record(&mut self, ms: f64) {
        if self.n == 0 {
            self.min_ms = ms;
            self.max_ms = ms;
        } else {
            self.min_ms = self.min_ms.min(ms);
            self.max_ms = self.max_ms.max(ms);
        }
        self.n += 1;
        self.sum_ms += ms;
    }

    fn fmt_inline(&self, label: &str) -> String {
        if self.n == 0 {
            format!("{label}=–")
        } else {
            let avg = self.sum_ms / self.n as f64;
            format!(
                "{label}={avg:.2}ms ({:.2}\u{2013}{:.2}, n={})",
                self.min_ms, self.max_ms, self.n
            )
        }
    }

    fn reset(&mut self) {
        *self = PassAvg::default();
    }
}

#[derive(Default)]
struct GpuStats {
    layers: PassAvg,
    bake: PassAvg,
    image: PassAvg,
    /// Total samples since the last log line. We log every 60 *image*
    /// samples (~once per second at 60 fps). Layers/bake just hitchhike
    /// and reset together.
    image_samples_since_log: u32,
}

impl GpuStats {
    fn maybe_log(&mut self) {
        self.image_samples_since_log = self.image_samples_since_log.saturating_add(1);
        if self.image_samples_since_log < 60 {
            return;
        }
        console::log_1(
            &format!(
                "gpu: {} | {} | {}",
                self.image.fmt_inline("image"),
                self.layers.fmt_inline("layers"),
                self.bake.fmt_inline("bake"),
            )
            .into(),
        );
        self.image.reset();
        self.layers.reset();
        self.bake.reset();
        self.image_samples_since_log = 0;
    }
}
