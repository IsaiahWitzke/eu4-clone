//! Camera state + the GPU uniform block fed to the image pass.
//!
//! The camera owns a world-XZ pan target plus distance/tilt; it can derive its
//! world-AABB (visible rectangle) and a `CameraUniforms` block. Mouse pan
//! conversions live here too so input handlers don't have to do the math.

use bytemuck::{Pod, Zeroable};

// ---- Camera defaults & limits ---------------------------------------------

/// Vertical FOV used by the perspective shader. Must match `CAMERA_FOV_Y` in
/// `image.wgsl`.
pub const CAMERA_FOV_Y_RAD: f32 = std::f32::consts::PI / 6.0; // 30°

pub const DEFAULT_CAMERA_DISTANCE: f32 = 2.6;
pub const DEFAULT_CAMERA_TILT: f32 = 0.27;

pub const MIN_CAMERA_DISTANCE: f32 = 0.8;
pub const MAX_CAMERA_DISTANCE: f32 = 6.0;
pub const MIN_CAMERA_TILT: f32 = 0.0;
pub const MAX_CAMERA_TILT: f32 = std::f32::consts::FRAC_PI_2 - 0.05;

// ---- Aabb2 ---------------------------------------------------------------

/// Axis-aligned 2D box in world XZ.
#[derive(Copy, Clone, Debug)]
pub struct Aabb2 {
    pub min: [f32; 2],
    pub max: [f32; 2],
}

impl Aabb2 {
    pub fn from_center_half(center: [f32; 2], half: [f32; 2]) -> Self {
        Self {
            min: [center[0] - half[0], center[1] - half[1]],
            max: [center[0] + half[0], center[1] + half[1]],
        }
    }

    pub fn center(self) -> [f32; 2] {
        [
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
        ]
    }

    pub fn half_size(self) -> [f32; 2] {
        [
            0.5 * (self.max[0] - self.min[0]),
            0.5 * (self.max[1] - self.min[1]),
        ]
    }

    /// True if `inner` is fully contained within `self` (inclusive).
    pub fn contains(self, inner: Aabb2) -> bool {
        self.min[0] <= inner.min[0]
            && self.min[1] <= inner.min[1]
            && self.max[0] >= inner.max[0]
            && self.max[1] >= inner.max[1]
    }

    /// Returns a new Aabb2 expanded by a multiplicative pad factor around its center.
    /// `pad = 1.0` is identity; `pad = 2.0` doubles each axis.
    pub fn expanded(self, pad: f32) -> Self {
        let c = self.center();
        let h = self.half_size();
        Self::from_center_half(c, [h[0] * pad, h[1] * pad])
    }
}

// ---- Camera --------------------------------------------------------------

#[derive(Copy, Clone, Debug)]
pub struct Camera {
    pub world_center: [f32; 2],
    pub distance: f32,
    pub tilt: f32,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            world_center: [0.0, 0.0],
            distance: DEFAULT_CAMERA_DISTANCE,
            tilt: DEFAULT_CAMERA_TILT,
        }
    }

    /// Visible world rectangle for the given canvas aspect ratio (width/height).
    /// Derived from `distance` and the vertical FOV — no extra padding here;
    /// padding is the world-layer cache's responsibility.
    pub fn view_aabb(&self, aspect: f32) -> Aabb2 {
        let half_h = self.distance * (CAMERA_FOV_Y_RAD * 0.5).tan();
        let half_w = half_h * aspect;
        Aabb2::from_center_half(self.world_center, [half_w, half_h])
    }

    /// Apply a CSS-pixel mouse drag, panning so the world point under the
    /// cursor follows it. Sign convention: drag right → camera moves west;
    /// drag down → camera moves north (screen Y is top-down, world Z is +north).
    pub fn pan_pixels(&mut self, dx_px: f32, dy_px: f32, css_w: f32, css_h: f32) {
        let half_h = self.distance * (CAMERA_FOV_Y_RAD * 0.5).tan();
        let half_w = half_h * (css_w / css_h.max(1.0));
        let per_px_x = 2.0 * half_w / css_w.max(1.0);
        let per_px_y = 2.0 * half_h / css_h.max(1.0);
        self.world_center[0] -= dx_px * per_px_x;
        self.world_center[1] += dy_px * per_px_y;
    }

    /// Pan by a fixed amount in world units (used by arrow keys).
    pub fn pan_world(&mut self, dx: f32, dy: f32) {
        self.world_center[0] += dx;
        self.world_center[1] += dy;
    }

    /// Adjust tilt by `delta` radians; clamped to (0, π/2).
    pub fn tilt_by(&mut self, delta: f32) {
        self.tilt = (self.tilt + delta).clamp(MIN_CAMERA_TILT, MAX_CAMERA_TILT);
    }

    /// Multiply distance by `factor`, clamped. `factor < 1` zooms in.
    pub fn zoom(&mut self, factor: f32) {
        self.distance =
            (self.distance * factor).clamp(MIN_CAMERA_DISTANCE, MAX_CAMERA_DISTANCE);
    }

    /// Eye offset relative to the look-at point. Encodes both distance and tilt.
    fn eye_offset(&self) -> [f32; 3] {
        [
            0.0,
            self.distance * self.tilt.cos(),
            -self.distance * self.tilt.sin(),
        ]
    }

    /// Build the GPU uniform block for the image pass.
    pub fn to_uniforms(&self, width: u32, height: u32) -> CameraUniforms {
        CameraUniforms {
            i_resolution: [width as f32, height as f32, 1.0],
            i_time: 0.0,
            world_center: self.world_center,
            _pad0: [0.0; 2],
            eye_offset: self.eye_offset(),
            _pad1: 0.0,
        }
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new()
    }
}

// ---- GPU uniform block ---------------------------------------------------

/// Mirrors the WGSL `Uniforms` struct in `shaders/camera.wgsl`. Layout:
/// vec3 (16-byte aligned), then the trailing f32, then vec2, vec2 pad,
/// vec3, f32 pad → 48 bytes total.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CameraUniforms {
    pub i_resolution: [f32; 3],
    pub i_time: f32,
    pub world_center: [f32; 2],
    pub _pad0: [f32; 2],
    pub eye_offset: [f32; 3],
    pub _pad1: f32,
}
