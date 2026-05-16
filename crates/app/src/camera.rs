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

/// Y altitude of the look-at point, in world units (post-VERTICAL_EXAGGERATION).
/// Must match `CAMERA_TARGET_Y` in `image.wgsl`. Used by both the GPU camera
/// and the CPU mouse-pick reconstruction.
pub const CAMERA_TARGET_Y: f32 = 4.0;

/// Approximate world Y of "average ground" for mouse-pick ray intersection.
/// Picked roughly at the mean elevation in the heightmap (~1500 m × VE/5000 m
/// = 3.0). Picking error scales with abs(actual_h - HOVER_PICK_Y) * tan(tilt);
/// at default 15.5° tilt, max worst-case error is ~10 km on 4 km peaks.
pub const HOVER_PICK_Y: f32 = 3.0;

// World units = km, so distances are in km. The world is now ~5500 km
// wide (Western/Central Europe in mercator units), so the default zoom
// shows a wide regional view; max distance is enough to fit the entire
// world on screen, min keeps single-village zooms reachable.
pub const DEFAULT_CAMERA_DISTANCE: f32 = 3000.0;
pub const DEFAULT_CAMERA_TILT: f32 = 0.27;

pub const MIN_CAMERA_DISTANCE: f32 = 5.0;
pub const MAX_CAMERA_DISTANCE: f32 = 8000.0;
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

// ---- Map modes -----------------------------------------------------------

/// Which information layer the renderer is showing right now.
///
/// Discriminants are mirrored as `MAP_MODE_*` constants in `image.wgsl`; do
/// not reorder without updating both sides.
#[repr(u32)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum MapMode {
    #[default]
    Terrain = 0,
    Political = 1,
    /// Debug overlay: shade each fragment by which LoD atlas the
    /// world_mesh fragment shader sampled. Press M to cycle into this
    /// mode — lets you see at a glance whether LoD selection is
    /// actually switching as you zoom.
    DebugLod = 2,
}

impl MapMode {
    pub fn next(self) -> Self {
        match self {
            Self::Terrain => Self::Political,
            Self::Political => Self::DebugLod,
            Self::DebugLod => Self::Terrain,
        }
    }
}

// ---- Camera --------------------------------------------------------------

#[derive(Copy, Clone, Debug)]
pub struct Camera {
    pub world_center: [f32; 2],
    pub distance: f32,
    pub tilt: f32,
    pub map_mode: MapMode,
    /// Currently-hovered realm ID encoded as `realm_id + 1` (so 0 means
    /// "nothing hovered"). Drives the realm-wide hover wash. The name
    /// `_pid` is historical — originally a province ID; now it carries the
    /// influence-field's dominant realm.
    pub hovered_pid: u32,
    /// Currently-hovered settlement encoded as `settlement_index + 1`
    /// (0 = nothing). Drives the *hinterland* hover wash — a stronger
    /// highlight on just the cells dominated by the same city the cursor
    /// is over, layered on top of the realm-wide wash.
    pub hovered_city: u32,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            world_center: [0.0, 0.0],
            distance: DEFAULT_CAMERA_DISTANCE,
            tilt: DEFAULT_CAMERA_TILT,
            map_mode: MapMode::default(),
            hovered_pid: 0,
            hovered_city: 0,
        }
    }

    /// Cycle to the next map mode.
    pub fn cycle_map_mode(&mut self) {
        self.map_mode = self.map_mode.next();
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

    /// Reconstruct the ray for a CSS-pixel screen position and intersect it
    /// with the horizontal plane at `y_target`. Returns the world XZ at the
    /// intersection, or `None` if the ray is parallel to / above the plane.
    /// This mirrors the shader's `get_ray()` exactly.
    pub fn pick_world_xz(
        &self,
        mx: f32,
        my: f32,
        css_w: f32,
        css_h: f32,
        y_target: f32,
    ) -> Option<[f32; 2]> {
        let look_at = [
            self.world_center[0],
            CAMERA_TARGET_Y,
            self.world_center[1],
        ];
        let off = self.eye_offset();
        let eye = [
            look_at[0] + off[0],
            look_at[1] + off[1],
            look_at[2] + off[2],
        ];

        // forward = normalize(look_at - eye)
        let forward = normalize3([
            look_at[0] - eye[0],
            look_at[1] - eye[1],
            look_at[2] - eye[2],
        ]);
        let world_up = [0.0_f32, 1.0, 0.0];
        let right = normalize3(cross3(world_up, forward));
        let up = cross3(forward, right);

        let aspect = css_w / css_h.max(1.0);
        let tan_half_y = (CAMERA_FOV_Y_RAD * 0.5).tan();
        let tan_half_x = tan_half_y * aspect;
        let ndc_x = (mx / css_w.max(1.0)) * 2.0 - 1.0;
        let ndc_y = 1.0 - (my / css_h.max(1.0)) * 2.0;

        let rd = normalize3([
            forward[0] + right[0] * ndc_x * tan_half_x + up[0] * ndc_y * tan_half_y,
            forward[1] + right[1] * ndc_x * tan_half_x + up[1] * ndc_y * tan_half_y,
            forward[2] + right[2] * ndc_x * tan_half_x + up[2] * ndc_y * tan_half_y,
        ]);

        if rd[1].abs() < 1e-6 {
            return None;
        }
        let t = (y_target - eye[1]) / rd[1];
        if !t.is_finite() || t <= 0.0 {
            return None;
        }
        Some([eye[0] + rd[0] * t, eye[2] + rd[2] * t])
    }

    /// Project a world point at (xz, y) to a screen-space CSS-pixel
    /// position. Inverse of [`pick_world_xz`]: takes a 3D world point
    /// and returns where on the canvas it lands. Returns `None` if the
    /// point is behind the camera (or numerically unstable).
    ///
    /// Used by the HTML overlay to position country labels at each
    /// realm's centroid every frame. Same camera basis as
    /// `pick_world_xz` so the two stay in sync; if you tweak one,
    /// tweak the other.
    pub fn project_world_xz(
        &self,
        p_xz: [f32; 2],
        y: f32,
        css_w: f32,
        css_h: f32,
    ) -> Option<[f32; 2]> {
        let look_at = [
            self.world_center[0],
            CAMERA_TARGET_Y,
            self.world_center[1],
        ];
        let off = self.eye_offset();
        let eye = [
            look_at[0] + off[0],
            look_at[1] + off[1],
            look_at[2] + off[2],
        ];

        let forward = normalize3([
            look_at[0] - eye[0],
            look_at[1] - eye[1],
            look_at[2] - eye[2],
        ]);
        let world_up = [0.0_f32, 1.0, 0.0];
        let right = normalize3(cross3(world_up, forward));
        let up = cross3(forward, right);

        // Vector from camera to the world point, decomposed along
        // (right, up, forward). `depth` is how far in front of the
        // camera the point is; (right, up) components / depth give
        // tan-space NDC.
        let v = [p_xz[0] - eye[0], y - eye[1], p_xz[1] - eye[2]];
        let depth = dot3(v, forward);
        if depth <= 1e-3 {
            return None;
        }

        let aspect = css_w / css_h.max(1.0);
        let tan_half_y = (CAMERA_FOV_Y_RAD * 0.5).tan();
        let tan_half_x = tan_half_y * aspect;

        let ndc_x = (dot3(v, right) / depth) / tan_half_x;
        let ndc_y = (dot3(v, up) / depth) / tan_half_y;

        if !ndc_x.is_finite() || !ndc_y.is_finite() {
            return None;
        }

        // NDC → screen px. NDC y is +up, screen y is +down (matches
        // the inverse `ndc_y = 1.0 - my/css_h * 2.0` in pick_world_xz).
        let mx = (ndc_x + 1.0) * 0.5 * css_w;
        let my = (1.0 - ndc_y) * 0.5 * css_h;
        Some([mx, my])
    }

    /// Build the GPU uniform block for the image pass.
    pub fn to_uniforms(&self, width: u32, height: u32) -> CameraUniforms {
        CameraUniforms {
            i_resolution: [width as f32, height as f32, 1.0],
            i_time: 0.0,
            world_center: self.world_center,
            hovered_pid: self.hovered_pid,
            hovered_city: self.hovered_city,
            eye_offset: self.eye_offset(),
            map_mode: self.map_mode as u32,
        }
    }
}

// ---- vec3 helpers (private) ----------------------------------------------
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-20);
    [v[0] / len, v[1] / len, v[2] / len]
}
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

impl Default for Camera {
    fn default() -> Self {
        Self::new()
    }
}

// ---- GPU uniform block ---------------------------------------------------

/// Mirrors the WGSL `Uniforms` struct in `shaders/camera.wgsl`. Layout:
/// vec3 (16-byte aligned), trailing f32, vec2, then `hovered_pid` and
/// `hovered_city` filling the next 8 bytes, then vec3 + `map_mode: u32`
/// in the vec3's trailing pad. 48 bytes total.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CameraUniforms {
    pub i_resolution: [f32; 3],
    pub i_time: f32,
    pub world_center: [f32; 2],
    /// Currently-hovered realm encoded as `realm_id + 1` (0 = none).
    pub hovered_pid: u32,
    /// Currently-hovered settlement encoded as `settlement_index + 1`
    /// (0 = none). Used by the shader to draw a hinterland highlight on
    /// top of the realm-wide hover wash.
    pub hovered_city: u32,
    pub eye_offset: [f32; 3],
    /// Mirrors `MapMode` discriminants. Stored as `u32` so the shader can
    /// branch directly on it without a float→int conversion.
    pub map_mode: u32,
}
