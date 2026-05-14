//! Settlements + the influence-field uniform buffer the image pass reads.
//!
//! Each settlement is a point in world XZ that emits a radial influence field
//! `strength * exp(-distance_km / e_fold_km)` for its realm. The shader
//! per-fragment loops over the settlements, takes the realm whose total
//! field is largest at that pixel, and renders that. Borders fall out as
//! the iso-line where two realms tie (no SDF needed).
//!
//! Coordinates: world XZ is in km, with the origin at the centre of the
//! 5500 km Web-Mercator world bbox (see `script/_world.py`). We accept
//! lat/lon for editing convenience and project on the CPU; the result is
//! shipped to the GPU as `world_xz` so the shader doesn't need any
//! projection math.
//!
//! The hover path on the CPU side mirrors the same field eval (see
//! `dominant_at_world_xz`) so hover-highlighting picks the same realm and
//! the same dominant city the shader would.

use bytemuck::{Pod, Zeroable};

use crate::camera::Aabb2;
use crate::gpu::GpuContext;

/// Hard cap on the number of settlements packed into the uniform buffer.
/// Bumping this requires bumping the `array<GpuSettlement, N>` (and
/// `MAX_SETTLEMENTS` const) in `image.wgsl` to match.
///
/// At 1024 entries the uniform buffer is `16 + 1024*16 = 16,400 bytes`,
/// well under the 64 KB minimum guaranteed by WebGPU. Per-fragment cost
/// stays tractable on modern GPUs (the inner loop is just `exp` + a few
/// mul/adds per settlement).
pub const MAX_SETTLEMENTS: usize = 1024;

/// E-folding distance for a settlement's *long-range*, population-driven
/// influence, in km. At distance `E_FOLD_KM` the field has fallen to
/// `1/e ≈ 0.37` of its peak; at `3 × E_FOLD_KM` it's down to ~5%.
///
/// **Mirrored** in `image.wgsl` and `realm_field.wgsl`. Keep in sync.
pub const E_FOLD_KM: f32 = 30.0;

/// Peak strength of the *local-core* component, applied to every
/// settlement regardless of population. Without this, a small village
/// next to a megacity has no hinterland — the megacity's long-range
/// field swamps the village even at d = 0.
///
/// We pick `LOCAL_BONUS` large enough that every settlement, however
/// tiny, dominates within `LOCAL_E_FOLD_KM` of itself even when a much
/// bigger city is nearby. Falls off with its own (short) e-fold so it
/// only gives each city a guaranteed local hinterland of a few km.
///
/// **Mirrored** in `realm_field.wgsl`. Keep in sync.
pub const LOCAL_BONUS: f32 = 2000.0;

/// E-folding distance for the local-core bump, in km. Roughly the
/// radius of the guaranteed hinterland every settlement gets to itself.
///
/// **Mirrored** in `realm_field.wgsl`. Keep in sync.
pub const LOCAL_E_FOLD_KM: f32 = 5.0;

// ---- Mercator projection -----------------------------------------------
//
// World is EPSG:3857. World origin (0, 0) sits at the centre of the bbox
// from `script/_world.py`: `(WORLD_BBOX_CENTER_X, WORLD_BBOX_CENTER_Y) =
// ((-1500000 + 4000000) / 2, (4250000 + 9750000) / 2) = (1250000, 7000000)`
// metres.

const MERCATOR_R_M: f64 = 6_378_137.0;
const WORLD_CENTER_MERC_X_M: f64 = 1_250_000.0;
const WORLD_CENTER_MERC_Y_M: f64 = 7_000_000.0;

/// Project geographic (lat, lon) in degrees to world XZ in km. World
/// convention: +X east, +Z north.
pub fn lat_lon_to_world_xz(lat_deg: f64, lon_deg: f64) -> [f32; 2] {
    let merc_x = lon_deg.to_radians() * MERCATOR_R_M;
    let lat_rad = lat_deg.to_radians();
    // Standard Mercator y projection: ln(tan(π/4 + φ/2)).
    let merc_y = MERCATOR_R_M
        * 0.5
        * ((1.0 + lat_rad.sin()) / (1.0 - lat_rad.sin())).ln();
    [
        ((merc_x - WORLD_CENTER_MERC_X_M) / 1000.0) as f32,
        ((merc_y - WORLD_CENTER_MERC_Y_M) / 1000.0) as f32,
    ]
}

// ---- CPU-side data ------------------------------------------------------

/// One settlement: a population gravity well that projects realm influence.
#[derive(Clone, Debug)]
pub struct Settlement {
    /// Position in world XZ (km).
    pub world_xz: [f32; 2],
    /// Population-ish (unitless). Higher = bigger reach. ~city population
    /// in thousands works as a starting heuristic.
    pub strength: f32,
    /// Realm this settlement belongs to. Indexes the same 16-entry palette
    /// the shader uses for province colouring (`realm_palette` in WGSL).
    pub realm_id: u32,
    /// Display name (handy for debugging; not sent to the GPU). Owned
    /// `String` rather than `&'static str` so loaders that decode names
    /// from JSON / disk don't have to leak.
    pub name: String,
}

// ---- GPU layout ---------------------------------------------------------

/// One settlement on the GPU. 16 bytes, vec4-aligned. Mirrors WGSL:
/// ```wgsl
/// struct GpuSettlement {
///   world_xz: vec2<f32>,
///   strength: f32,
///   realm_id: u32,
/// }
/// ```
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct GpuSettlement {
    pub world_xz: [f32; 2],
    pub strength: f32,
    pub realm_id: u32,
}

/// Top-level uniform block. Layout:
/// `count` (4 B) + `_pad` (12 B) + array of `MAX_SETTLEMENTS` × 16 B.
///
/// std140 requires the array start to be 16-aligned, hence the `_pad`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SettlementUniforms {
    pub count: u32,
    pub _pad: [u32; 3],
    pub items: [GpuSettlement; MAX_SETTLEMENTS],
}

impl SettlementUniforms {
    pub fn from_slice(settlements: &[Settlement]) -> Self {
        let mut items = [GpuSettlement::zeroed(); MAX_SETTLEMENTS];
        let n = settlements.len().min(MAX_SETTLEMENTS);
        for (i, s) in settlements.iter().take(n).enumerate() {
            items[i] = GpuSettlement {
                world_xz: s.world_xz,
                strength: s.strength,
                realm_id: s.realm_id,
            };
        }
        Self {
            count: n as u32,
            _pad: [0; 3],
            items,
        }
    }
}

// ---- Buffer ----------------------------------------------------------------

/// Allocate the uniform buffer that backs the settlements array. Big enough
/// for `MAX_SETTLEMENTS`; not resized at runtime.
pub fn make_uniform_buf(gpu: &GpuContext, label: &str) -> wgpu::Buffer {
    gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: std::mem::size_of::<SettlementUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

// ---- Default Swiss city set -----------------------------------------------
//
// Twelve real Swiss cities, projected from lat/lon, grouped into five
// fictional realms. Strengths are roughly population in thousands so the
// relative balance feels right (Zürich >> tiny mountain towns). The realm
// groupings are deliberately a bit weird so the borders don't just trace
// modern cantons.
//
// Realm IDs map to `realm_palette(idx)` in image.wgsl:
//   0 crimson, 1 burnt orange, 2 gold, 3 olive, 4 forest green,
//   5 teal, 6 sky blue, 7 navy, 8 royal purple, …
//
// Realm assignments here:
//   0 (crimson)        — German-Swiss core: Zürich, Luzern, St. Gallen
//   1 (burnt orange)   — Rhine north: Basel
//   2 (gold)           — French-speaking west: Geneva, Lausanne, Neuchâtel
//   3 (olive)          — Bernese: Bern, Sion
//   4 (forest green)   — Italian Alps: Lugano, Bellinzona, Chur
/// Settlement list + the realm-id→display-name map that goes with it.
/// Returned by both the hardcoded Swiss seed and the cities.json loader so
/// downstream consumers (the renderer + the realm-label UI) get a single
/// bundle to plumb through.
pub struct LoadedSettlements {
    pub settlements: Vec<Settlement>,
    /// realm_id → display name. Sparse: realms without a known name fall
    /// back to `format!("Realm {id}")` at the call site.
    pub realm_names: std::collections::HashMap<u32, String>,
}

pub fn default_swiss_settlements() -> LoadedSettlements {
    fn city(name: &'static str, lat: f64, lon: f64, strength: f32, realm_id: u32) -> Settlement {
        Settlement {
            world_xz: lat_lon_to_world_xz(lat, lon),
            strength,
            realm_id,
            name: name.to_string(),
        }
    }
    let settlements = vec![
        // Realm 0 — German-Swiss core.
        city("Zürich",     47.3769, 8.5417, 400.0, 0),
        city("Luzern",     47.0502, 8.3093,  82.0, 0),
        city("St. Gallen", 47.4239, 9.3748,  76.0, 0),
        // Realm 1 — Rhine north.
        city("Basel",      47.5596, 7.5886, 175.0, 1),
        // Realm 2 — French-speaking west.
        city("Genève",     46.2044, 6.1432, 200.0, 2),
        city("Lausanne",   46.5197, 6.6323, 140.0, 2),
        city("Neuchâtel",  46.9899, 6.9292,  35.0, 2),
        // Realm 3 — Bernese.
        city("Bern",       46.9481, 7.4474, 134.0, 3),
        city("Sion",       46.2331, 7.3596,  35.0, 3),
        // Realm 4 — Italian Alps.
        city("Lugano",     46.0050, 8.9522,  63.0, 4),
        city("Bellinzona", 46.1947, 9.0238,  18.0, 4),
        city("Chur",       46.8508, 9.5320,  35.0, 4),
    ];
    let realm_names = std::collections::HashMap::from([
        (0_u32, "Argaria".to_string()),
        (1_u32, "Rhinelands".to_string()),
        (2_u32, "Helvetia".to_string()),
        (3_u32, "Bernese".to_string()),
        (4_u32, "Lombardia".to_string()),
    ]);
    LoadedSettlements { settlements, realm_names }
}

// ---- cities.json loader ----------------------------------------------------

/// Build a `Vec<Settlement>` from the JSON produced by
/// `script/gen-cities`. Each entry already has `world_x_km` /
/// `world_z_km` pre-computed (mercator metres minus bbox centre, divided
/// by 1000), so we don't need to reproject; just copy.
///
/// Realm assignment: each unique `country` string gets a fresh integer
/// realm id (in encounter order, so the assignment is deterministic per
/// JSON payload). Strength = `modern_population / 1000`, which keeps
/// the same units the hand-tuned Swiss defaults use (Zürich's ~400k pop
/// landed at strength=400). Top-N truncation falls out of caller-side
/// sorting; we apply `MAX_SETTLEMENTS` here as a final safety cap.
pub fn from_cities_json(json_text: &str) -> Result<LoadedSettlements, String> {
    use serde::Deserialize;
    use std::collections::HashMap;

    #[derive(Deserialize)]
    struct CityJson {
        name: String,
        country: Option<String>,
        world_x_km: f32,
        world_z_km: f32,
        modern_population: Option<u64>,
    }
    #[derive(Deserialize)]
    struct Document {
        cities: Vec<CityJson>,
    }

    let doc: Document = serde_json::from_str(json_text).map_err(|e| e.to_string())?;

    let mut country_to_realm: HashMap<String, u32> = HashMap::new();
    let mut realm_names: HashMap<u32, String> = HashMap::new();
    let mut next_realm: u32 = 0;

    let mut settlements = Vec::with_capacity(doc.cities.len());
    for city in doc.cities {
        let country_key = city.country.clone().unwrap_or_else(|| "_".into());
        let realm_id = match country_to_realm.get(&country_key) {
            Some(&id) => id,
            None => {
                let id = next_realm;
                next_realm = next_realm.wrapping_add(1);
                country_to_realm.insert(country_key.clone(), id);
                // Stash the country string so the label UI can show it.
                realm_names.insert(id, country_key);
                id
            }
        };
        // Population/1000 → strength. Cities with no population fall back
        // to a small constant so they still show up but don't dominate.
        let strength = city
            .modern_population
            .map(|p| (p as f32 / 1000.0).max(1.0))
            .unwrap_or(10.0);
        settlements.push(Settlement {
            world_xz: [city.world_x_km, city.world_z_km],
            strength,
            realm_id,
            name: city.name,
        });
        if settlements.len() >= MAX_SETTLEMENTS {
            break;
        }
    }
    Ok(LoadedSettlements { settlements, realm_names })
}

// ---- CPU-side field eval (used by hover) ----------------------------------

/// Result of evaluating the influence field on the CPU side. Carries the
/// argmax realm + the index of the *specific city* whose contribution
/// won — callers can use the city index to highlight just that city's
/// hinterland (the area it's the dominant settlement for), independent of
/// realm-wide highlighting.
pub struct DominantHit {
    pub realm_id: u32,
    pub city_idx: u32,
    pub strength: f32,
}

/// Realm count for the per-realm reinforcement buckets. Mirrors
/// `MAX_REALMS` in `shaders/realm_field.wgsl`. Cities are bucketed via
/// `realm_id % MAX_REALMS`, so this is the maximum number of *distinct*
/// realms the engine can tell apart — anything beyond this collides on
/// the same bucket and shares hover state, borders, etc.
///
/// 64 fits any country-scale dataset (Europe has ~50 countries; the
/// whole world has ~200). The per-fragment cost in `image.wgsl` /
/// `realm_field.wgsl` is three 64-entry stack arrays — cheap.
///
/// The visual *palette* in `image.wgsl::realm_palette` still wraps at
/// 16 (`REALM_PALETTE_SIZE`); that's a separate cycle. Different realms
/// may share a colour, but they remain logically distinct for hover
/// and bookkeeping.
pub const MAX_REALMS: usize = 64;

/// World-XZ half-extent in km. Mirrors `WORLD_BOUNDS_HALF` in
/// `shaders/world.wgsl`. Used by the CPU-side water-mask sampler in
/// `realm_infos` so the rasterised territory matches the on-screen
/// `world_to_world_uv` mapping.
const WORLD_BOUNDS_HALF_KM: f32 = 2750.0;

/// CPU view of a pre-loaded water mask, used to drop offshore cells
/// from the realm rasteriser so labels don't drift into the sea
/// when a country's cities cluster near the coast.
///
/// Format mirrors the GPU upload: a `width × height` R8Unorm pixel
/// buffer, row 0 = geographic north, byte value 0 = land, 255 = water.
/// `sample` reproduces `world_to_world_uv` in `shaders/world.wgsl`.
pub struct WaterMask<'a> {
    pub bytes: &'a [u8],
    pub width: u32,
    pub height: u32,
}

impl<'a> WaterMask<'a> {
    /// Sample the mask at a world XZ point. Returns 0..1 (high = water).
    /// Points outside the world bounds return 0 — "treat as land" — so
    /// the rasteriser pad past the city bbox doesn't accidentally clip
    /// inland cells.
    pub fn sample(&self, world_xz: [f32; 2]) -> f32 {
        let u = (world_xz[0] + WORLD_BOUNDS_HALF_KM) / (2.0 * WORLD_BOUNDS_HALF_KM);
        let v = 1.0 - (world_xz[1] + WORLD_BOUNDS_HALF_KM) / (2.0 * WORLD_BOUNDS_HALF_KM);
        let px = (u * self.width as f32) as i32;
        let py = (v * self.height as f32) as i32;
        if px < 0
            || py < 0
            || px >= self.width as i32
            || py >= self.height as i32
        {
            return 0.0;
        }
        let idx = (py as usize) * (self.width as usize) + (px as usize);
        self.bytes[idx] as f32 / 255.0
    }
}

/// Evaluate the same-realm-reinforced influence field at a world XZ and
/// return the dominant realm + its dominant single city. Mirrors the
/// WGSL bake pass (`shaders/realm_field.wgsl`) so hover stays in lockstep
/// with what the GPU paints:
///   1. Sum each settlement's contribution into its realm bucket.
///   2. Argmax across realm buckets to pick the dominant *cluster*.
///   3. Within that cluster, return the index of the strongest single
///      contributor (so per-city hinterland highlights still work).
///
/// Returns `(realm = 0, city = 0, strength = 0.0)` when no settlement is
/// in range.
pub fn dominant_at_world_xz(settlements: &[Settlement], xz: [f32; 2]) -> DominantHit {
    let mut realm_sums = [0.0_f32; MAX_REALMS];
    let mut realm_best_strength = [0.0_f32; MAX_REALMS];
    let mut realm_best_city = [0_u32; MAX_REALMS];

    for (i, s) in settlements.iter().enumerate() {
        let dx = xz[0] - s.world_xz[0];
        let dy = xz[1] - s.world_xz[1];
        let d_km = (dx * dx + dy * dy).sqrt();
        // Two components, summed: long-range (population-scaled, slowly
        // falling) + local-core (fixed amplitude, sharply falling). The
        // local bump guarantees every settlement owns its immediate
        // surroundings even next to a much bigger neighbour.
        let v_long  = s.strength * (-d_km / E_FOLD_KM).exp();
        let v_local = LOCAL_BONUS * (-d_km / LOCAL_E_FOLD_KM).exp();
        let v = v_long + v_local;
        let r = (s.realm_id as usize) % MAX_REALMS;
        realm_sums[r] += v;
        if v > realm_best_strength[r] {
            realm_best_strength[r] = v;
            realm_best_city[r] = i as u32;
        }
    }

    let mut best_strength = 0.0_f32;
    let mut best_realm = 0_u32;
    for (r, &sum) in realm_sums.iter().enumerate() {
        if sum > best_strength {
            best_strength = sum;
            best_realm = r as u32;
        }
    }

    DominantHit {
        realm_id: best_realm,
        city_idx: realm_best_city[best_realm as usize],
        strength: best_strength,
    }
}

/// Pick the nearest settlement to a world XZ within `max_km`. Returns the
/// index in the slice, or `None` if nothing's close enough. Used by the
/// click-to-select UI path — we want to find the literal closest city,
/// not the dominant one (which can be a far-away megacity even when
/// you're standing in a small town that has its own local-core).
pub fn nearest_within_km(
    settlements: &[Settlement],
    xz: [f32; 2],
    max_km: f32,
) -> Option<usize> {
    let mut best_d2 = max_km * max_km;
    let mut best_idx: Option<usize> = None;
    for (i, s) in settlements.iter().enumerate() {
        let dx = xz[0] - s.world_xz[0];
        let dy = xz[1] - s.world_xz[1];
        let d2 = dx * dx + dy * dy;
        if d2 < best_d2 {
            best_d2 = d2;
            best_idx = Some(i);
        }
    }
    best_idx
}

// ---- Realm-level summary (used by the SDF label pass) ----------------------

/// One realm's display info: name + spatial extent + a baseline curve.
/// Consumed by `crate::labels` which lays the realm name out along the
/// baseline using the SDF glyph atlas.
pub struct RealmInfo {
    pub realm_id: u32,
    pub name: String,
    /// Geometric centroid of the realm's *largest connected
    /// influence-field component* in world km (uniform mean of cell
    /// positions). Falls back to the city-position mean when no
    /// component qualifies.
    pub centroid: [f32; 2],
    /// World-XZ AABB enclosing every cell of the largest connected
    /// component (or, fallback, every member city). Used to size
    /// the font (label width ≈ a fraction of the bbox-long
    /// dimension), and as a strength-independent measure of "how
    /// big is this realm visually".
    pub bbox: Aabb2,
    /// Sum of member cities' `strength`. Useful as a filter ("don't
    /// label realms whose total population is < N").
    pub total_strength: f32,
    /// Major-axis baseline endpoints in world XZ. The label is laid
    /// along this segment with a small fraction of the segment length
    /// reserved as side-padding. Computed by uniform PCA over the
    /// cells of the realm's largest connected influence-field
    /// component, so it tracks the *shape of the territory* rather
    /// than the (often skewed) distribution of cities. For example,
    /// a single megacity in the south can't drag France's label
    /// southward when most of the country is north of it.
    ///
    /// Realms with no qualifying component (single tiny village,
    /// degenerate field) collapse to a zero-length segment
    /// `(centroid, centroid)`; the layout side then falls back to a
    /// horizontal baseline sized to the rendered text width.
    pub baseline_start: [f32; 2],
    pub baseline_end: [f32; 2],
}

/// Sentinel value used in the realm-component raster to mean "no
/// realm dominates here / wilderness". Anything < `MAX_REALMS` is a
/// real realm id; this is well outside the legal range and packs
/// into a `u32` for cache-friendly indexing.
const NO_REALM: u32 = u32::MAX;

/// World-km size of one cell in the realm-component raster. Smaller
/// = sharper realm boundaries + more cells (and more work); larger =
/// faster bake + coarser PCA direction. ~20 km matches the
/// `LOCAL_E_FOLD_KM` × 4 reach, which is plenty to resolve realm
/// shapes at country zoom.
const REALM_RASTER_STEP_KM: f32 = 20.0;

/// Strength threshold below which a cell is treated as wilderness.
/// Mirrors the `field.alpha > 0.05` gate the hover path uses, so
/// the CPU-rasterised territory matches what the player sees on the
/// map.
const REALM_RASTER_MIN_STRENGTH: f32 = 0.5;

/// Compute one `RealmInfo` per distinct realm in `settlements`.
///
/// The centroid + baseline come from a CPU rasterisation of the
/// influence field:
///   1. Lay a coarse grid (`REALM_RASTER_STEP_KM`) over the world
///      bbox padded by ~one e-fold so we don't crop realms whose
///      cities cluster near the edges.
///   2. For each cell, run the same argmax-realm field eval the
///      shader does (`dominant_at_world_xz`), tagging the cell
///      with the winning realm (or `NO_REALM` if below the
///      wilderness threshold).
///   3. Per realm: flood-fill the cell grid to find connected
///      components, pick the largest, and run uniform PCA on its
///      cell positions. The major-axis eigenvector becomes the
///      baseline direction; the projected half-extent sets the
///      baseline length; the cell mean becomes the centroid.
///
/// Compared to weighting by city positions, this:
///   * Centres labels on the realm's *actual territory*, not its
///     city distribution — so a big-coastal-city / small-interior
///     country (think Australia) doesn't get its label dragged into
///     the ocean.
///   * Orients labels along the territory's natural elongation, even
///     when there's only one city (which the city-weighted PCA
///     can't do at all).
///   * Drops disconnected splinters (offshore islands, exclaves) so
///     the label lives on the mainland.
pub fn realm_infos(
    settlements: &[Settlement],
    realm_names: &std::collections::HashMap<u32, String>,
    water_mask: Option<&WaterMask<'_>>,
) -> Vec<RealmInfo> {
    if settlements.is_empty() {
        return Vec::new();
    }

    // ---- 1. Rasterisation grid -------------------------------------
    //
    // Cover the bbox of all settlements + a generous pad. The pad
    // captures cells beyond any city where the realm still wins
    // (e.g. between two same-realm cities, 25 km north of the
    // northernmost city, etc.). 3× the long-range e-fold is enough
    // for the field to drop below the wilderness threshold.
    let pad = E_FOLD_KM * 3.0;
    let mut xmin = f32::INFINITY;
    let mut xmax = f32::NEG_INFINITY;
    let mut ymin = f32::INFINITY;
    let mut ymax = f32::NEG_INFINITY;
    for s in settlements {
        xmin = xmin.min(s.world_xz[0]);
        xmax = xmax.max(s.world_xz[0]);
        ymin = ymin.min(s.world_xz[1]);
        ymax = ymax.max(s.world_xz[1]);
    }
    xmin -= pad;
    xmax += pad;
    ymin -= pad;
    ymax += pad;
    let step = REALM_RASTER_STEP_KM;
    let nx = (((xmax - xmin) / step).ceil() as usize).max(2);
    let ny = (((ymax - ymin) / step).ceil() as usize).max(2);

    // Rasterised realm winner per cell. `NO_REALM` = wilderness.
    // Two filters:
    //   * Influence-field strength must clear `REALM_RASTER_MIN_STRENGTH`
    //     (else the cell is wilderness, no realm dominates).
    //   * Cell must be on land, where a water mask is available. Without
    //     this, coastal cities project their influence into the sea and
    //     the largest-component centroid drifts offshore (e.g. the
    //     Netherlands label sliding into the North Sea before this
    //     filter went in).
    let mut grid = vec![NO_REALM; nx * ny];
    for j in 0..ny {
        let cy = ymin + (j as f32 + 0.5) * step;
        for i in 0..nx {
            let cx = xmin + (i as f32 + 0.5) * step;
            if let Some(wm) = water_mask {
                if wm.sample([cx, cy]) > 0.5 {
                    continue;
                }
            }
            let hit = dominant_at_world_xz(settlements, [cx, cy]);
            if hit.strength >= REALM_RASTER_MIN_STRENGTH {
                grid[j * nx + i] = hit.realm_id;
            }
        }
    }

    // ---- 2. Per-realm: flood-fill + keep the largest component ----
    //
    // We walk every grid cell once; the first time we hit an
    // unvisited realm cell we BFS its entire component, then keep
    // it iff it's the largest seen for that realm. Visited cells
    // are marked so the outer loop doesn't re-scan them.
    let mut visited = vec![false; nx * ny];
    use std::collections::HashMap;
    let mut largest_per_realm: HashMap<u32, Vec<usize>> = HashMap::new();
    let mut queue: Vec<usize> = Vec::new();

    for start_idx in 0..nx * ny {
        if visited[start_idx] {
            continue;
        }
        let realm = grid[start_idx];
        if realm == NO_REALM {
            visited[start_idx] = true;
            continue;
        }
        // BFS — collect every same-realm cell connected to `start_idx`
        // via 4-connectivity.
        queue.clear();
        queue.push(start_idx);
        visited[start_idx] = true;
        let mut component: Vec<usize> = Vec::new();
        while let Some(idx) = queue.pop() {
            component.push(idx);
            let i = idx % nx;
            let j = idx / nx;
            // 4-neighbours.
            let nbrs = [
                (i.wrapping_sub(1), j, i > 0),
                (i + 1, j, i + 1 < nx),
                (i, j.wrapping_sub(1), j > 0),
                (i, j + 1, j + 1 < ny),
            ];
            for (ni, nj, ok) in nbrs {
                if !ok {
                    continue;
                }
                let nidx = nj * nx + ni;
                if visited[nidx] || grid[nidx] != realm {
                    continue;
                }
                visited[nidx] = true;
                queue.push(nidx);
            }
        }
        // Keep the largest component per realm.
        match largest_per_realm.get(&realm) {
            Some(prev) if prev.len() >= component.len() => {}
            _ => {
                largest_per_realm.insert(realm, component);
            }
        }
    }

    // ---- 3. Per-realm: PCA + bbox over the chosen component -------
    use std::collections::BTreeMap;
    // Sort by realm_id for deterministic ordering downstream.
    let mut out: BTreeMap<u32, RealmInfo> = BTreeMap::new();

    // Per-realm city totals (used for the strength filter + the
    // city-fallback path).
    struct CityAccum {
        sum_strength: f32,
        sum_xy: [f32; 2],
        bbox_min: [f32; 2],
        bbox_max: [f32; 2],
    }
    let mut by_city: BTreeMap<u32, CityAccum> = BTreeMap::new();
    for s in settlements {
        let e = by_city.entry(s.realm_id).or_insert(CityAccum {
            sum_strength: 0.0,
            sum_xy: [0.0, 0.0],
            bbox_min: [f32::INFINITY, f32::INFINITY],
            bbox_max: [f32::NEG_INFINITY, f32::NEG_INFINITY],
        });
        e.sum_strength += s.strength;
        e.sum_xy[0] += s.world_xz[0] * s.strength;
        e.sum_xy[1] += s.world_xz[1] * s.strength;
        e.bbox_min[0] = e.bbox_min[0].min(s.world_xz[0]);
        e.bbox_min[1] = e.bbox_min[1].min(s.world_xz[1]);
        e.bbox_max[0] = e.bbox_max[0].max(s.world_xz[0]);
        e.bbox_max[1] = e.bbox_max[1].max(s.world_xz[1]);
    }

    for (realm_id, city) in &by_city {
        if city.sum_strength <= 0.0 {
            continue;
        }
        let name = realm_names
            .get(realm_id)
            .cloned()
            .unwrap_or_else(|| format!("Realm {realm_id}"));

        let info = match largest_per_realm.get(realm_id) {
            Some(cells) if cells.len() >= 2 => {
                pca_from_cells(
                    cells, nx, xmin, ymin, step, *realm_id, name, city.sum_strength,
                )
            }
            _ => {
                // No connected component (or just one cell) — fall
                // back to the city centroid + a degenerate baseline.
                // The layout side then expands the baseline to the
                // rendered text width.
                let centroid = [
                    city.sum_xy[0] / city.sum_strength,
                    city.sum_xy[1] / city.sum_strength,
                ];
                RealmInfo {
                    realm_id: *realm_id,
                    name,
                    centroid,
                    bbox: Aabb2 {
                        min: city.bbox_min,
                        max: city.bbox_max,
                    },
                    total_strength: city.sum_strength,
                    baseline_start: centroid,
                    baseline_end: centroid,
                }
            }
        };
        out.insert(*realm_id, info);
    }

    out.into_values().collect()
}

/// Helper: compute centroid + bbox + PCA major axis from a list of
/// raster cells (indices into the `nx × ny` grid starting at
/// `(xmin, ymin)` with step `step`). Used by `realm_infos` for the
/// dominant-component branch.
#[allow(clippy::too_many_arguments)]
fn pca_from_cells(
    cells: &[usize],
    nx: usize,
    xmin: f32,
    ymin: f32,
    step: f32,
    realm_id: u32,
    name: String,
    total_strength: f32,
) -> RealmInfo {
    let n = cells.len() as f32;
    // First pass: centroid + bbox.
    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;
    let mut bbox_min = [f32::INFINITY, f32::INFINITY];
    let mut bbox_max = [f32::NEG_INFINITY, f32::NEG_INFINITY];
    for &idx in cells {
        let i = idx % nx;
        let j = idx / nx;
        let x = xmin + (i as f32 + 0.5) * step;
        let y = ymin + (j as f32 + 0.5) * step;
        sum_x += x;
        sum_y += y;
        bbox_min[0] = bbox_min[0].min(x);
        bbox_min[1] = bbox_min[1].min(y);
        bbox_max[0] = bbox_max[0].max(x);
        bbox_max[1] = bbox_max[1].max(y);
    }
    let centroid = [sum_x / n, sum_y / n];

    // Second pass: 2×2 covariance (uniform weight — 1 per cell).
    let mut cxx = 0.0f32;
    let mut cxy = 0.0f32;
    let mut cyy = 0.0f32;
    for &idx in cells {
        let i = idx % nx;
        let j = idx / nx;
        let x = xmin + (i as f32 + 0.5) * step;
        let y = ymin + (j as f32 + 0.5) * step;
        let dx = x - centroid[0];
        let dy = y - centroid[1];
        cxx += dx * dx;
        cxy += dx * dy;
        cyy += dy * dy;
    }
    cxx /= n;
    cxy /= n;
    cyy /= n;

    // Largest eigenvalue / eigenvector of the symmetric 2×2 covariance.
    let tr = cxx + cyy;
    let det = cxx * cyy - cxy * cxy;
    let disc = ((tr * 0.5) * (tr * 0.5) - det).max(0.0);
    let lambda = tr * 0.5 + disc.sqrt();
    let raw_dir = if cxy.abs() > 1e-6 {
        let dx = cxy;
        let dy = lambda - cxx;
        let len = (dx * dx + dy * dy).sqrt().max(1e-9);
        [dx / len, dy / len]
    } else if cxx >= cyy {
        [1.0, 0.0]
    } else {
        [0.0, 1.0]
    };

    // Keep the raw PCA major axis so labels tilt with the actual
    // shape of the country (UK at NW–SE, Italy at NW–SE, etc.), and
    // only normalise the *sign* of the eigenvector.
    //
    // PCA eigenvectors are line directions, not rays — ±`raw_dir`
    // are equally valid. The atlas convention is "text reads
    // top-to-bottom" for any non-horizontal baseline (head-tilt-right
    // when you turn the page); that means we want
    // `dir.y <= 0` (text goes south-ish). Tiebreaker for purely
    // horizontal baselines: prefer `dir.x > 0` so text reads
    // left-to-right.
    let mut dir = raw_dir;
    if dir[1] > 0.0 {
        dir = [-dir[0], -dir[1]];
    } else if dir[1].abs() < 1e-6 && dir[0] < 0.0 {
        dir = [-dir[0], -dir[1]];
    }

    // Half-extent along the major axis: project every cell onto
    // `dir` and take the max |offset|.
    let mut half = 0.0f32;
    for &idx in cells {
        let i = idx % nx;
        let j = idx / nx;
        let x = xmin + (i as f32 + 0.5) * step;
        let y = ymin + (j as f32 + 0.5) * step;
        let d = (x - centroid[0]) * dir[0] + (y - centroid[1]) * dir[1];
        half = half.max(d.abs());
    }
    let baseline_start = [
        centroid[0] - dir[0] * half,
        centroid[1] - dir[1] * half,
    ];
    let baseline_end = [
        centroid[0] + dir[0] * half,
        centroid[1] + dir[1] * half,
    ];

    RealmInfo {
        realm_id,
        name,
        centroid,
        bbox: Aabb2 {
            min: bbox_min,
            max: bbox_max,
        },
        total_strength,
        baseline_start,
        baseline_end,
    }
}

/// Hex colour string for a realm id. Mirrors the 16-entry palette in
/// `image.wgsl::realm_palette`. Used by the HTML-side UI panel for the
/// per-city colour swatch — the GPU side is the source of truth, so
/// keep this table in sync if you tweak the WGSL palette.
pub fn realm_color_hex(realm_id: u32) -> &'static str {
    const PALETTE: [&str; 16] = [
        "#c73333", //  0 crimson
        "#d97326", //  1 burnt orange
        "#e6c633", //  2 gold
        "#8ca633", //  3 olive
        "#338c4d", //  4 forest green
        "#1a8c8c", //  5 teal
        "#4d8ccc", //  6 sky blue
        "#334da6", //  7 navy
        "#7340a6", //  8 royal purple
        "#cc408c", //  9 magenta
        "#994d4d", // 10 brick
        "#bfa64d", // 11 mustard
        "#739973", // 12 sage
        "#8c80bf", // 13 lavender
        "#d98ca6", // 14 rose pink
        "#8c6640", // 15 brown
    ];
    PALETTE[(realm_id as usize) % PALETTE.len()]
}
