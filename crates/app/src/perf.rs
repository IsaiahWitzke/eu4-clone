//! Lightweight wrapper around the Web User Timing API
//! (`performance.mark` / `performance.measure`).
//!
//! Each [`Span`] you construct emits a `mark` on creation and, on drop,
//! a `measure` between that mark and the moment of drop. Those measures
//! show up as labelled bars in Chrome DevTools' Performance timeline
//! (and as live counters in the Performance Monitor sidebar), so you
//! can see exactly which slice of `Renderer::frame` is dominating any
//! given frame without sprinkling `console.log` calls or eyeballing
//! flame chart shapes.
//!
//! Cost is negligible: each `mark`/`measure` is a sub-microsecond JS
//! call that pushes a fixed-size struct into the browser's User Timing
//! buffer. Callers should still invoke [`clear_buffer`] once per frame
//! so the buffer doesn't accumulate over a long session — the DevTools
//! recorder uses a `PerformanceObserver` under the hood, so clearing
//! after `measure` is fine: anything already recorded stays in the
//! captured trace.

use web_sys::Performance;

/// RAII timing span. `Span::new("foo")` emits `mark("foo.start")`, and
/// `drop` emits `measure("foo", "foo.start")` — together they render
/// as a single bar named `foo` in DevTools.
pub struct Span<'a> {
    perf: &'a Performance,
    name: &'static str,
    start_mark: String,
}

impl<'a> Span<'a> {
    pub fn new(perf: &'a Performance, name: &'static str) -> Self {
        // Allocates a small `String` per span. Each frame creates a
        // handful, so the allocator churn is fine; if it ever shows up
        // in profiling, switch to a `concat!`-based macro variant.
        let start_mark = format!("{}.start", name);
        let _ = perf.mark(&start_mark);
        Self {
            perf,
            name,
            start_mark,
        }
    }
}

impl Drop for Span<'_> {
    fn drop(&mut self) {
        let _ = self
            .perf
            .measure_with_start_mark(self.name, &self.start_mark);
    }
}

/// Drop the User Timing buffer. Call once per frame (after all spans
/// for the frame have ended) so a long session doesn't slowly leak
/// memory into the buffer. Already-recorded entries inside an active
/// DevTools recording survive — `clear*` only empties the on-page
/// retention buffer, not the recorder's snapshot.
pub fn clear_buffer(perf: &Performance) {
    perf.clear_marks();
    perf.clear_measures();
}
