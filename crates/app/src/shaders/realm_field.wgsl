// ============================================================================
// Realm-field bake pass. Runs *once* per settlement-list change to evaluate
// the per-pixel argmax-realm at every cell of a 2048² RGBA8Unorm texture
// indexed by world_xz. The image pass then `textureLoad`s this once per
// fragment instead of looping over the full settlements array.
//
// Layout matches the same Settlements struct used by `image.wgsl`. Keep
// `MAX_SETTLEMENTS` / `SETTLEMENT_E_FOLD_KM` / `WORLD_BOUNDS_HALF_KM` in
// lock-step with `image.wgsl` + `world.wgsl`.
//
// Output channels (Rgba16Float — 11 bits of mantissa, so integer ids up
// through 2048 round-trip exactly):
//   R: realm_id          (f32(id), categorical — unpacked via u32(round(x)))
//   G: alpha             (0 in wilderness, → 1 well inside any realm)
//   B: contested - 1     (0 right on a border iso-line, → 1 deep inside)
//   A: city_idx          (f32(idx) of the *specific* dominant settlement;
//                         used by hover hinterland highlighting)
// ============================================================================

const MAX_SETTLEMENTS: u32 = 1024u;
struct GpuSettlement {
    world_xz: vec2<f32>,
    strength: f32,
    realm_id: u32,
}
struct Settlements {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    items: array<GpuSettlement, 1024>,
}
@group(0) @binding(0) var<uniform> settlements: Settlements;

const SETTLEMENT_E_FOLD_KM: f32 = 30.0;

// Short-range local-core component. Every settlement gets a fixed-amplitude
// bump that falls off in a few km, so even tiny villages own their
// immediate surroundings against bigger neighbours. Mirrors
// `LOCAL_BONUS` / `LOCAL_E_FOLD_KM` in `settlements.rs`.
const SETTLEMENT_LOCAL_BONUS: f32 = 2000.0;
const SETTLEMENT_LOCAL_E_FOLD_KM: f32 = 5.0;

// Must match `WORLD_BOUNDS_HALF` in `shaders/world.wgsl`.
const WORLD_BOUNDS_HALF_KM: vec2<f32> = vec2<f32>(2750.0, 2750.0);

// Realm count for the per-realm reinforcement buckets. Cities are
// bucketed via `realm_id % MAX_REALMS`. Sized to fit any realistic
// country-scale dataset (≥50 — with room to spare); collisions in
// this bucket cause hover state + borders to fuse across realms, so
// it deliberately doesn't share the smaller `REALM_PALETTE_SIZE`
// modulo used purely for *colour* in `image.wgsl::realm_palette`.
//
// Mirrored on the CPU side as `MAX_REALMS` in `settlements.rs`.
const MAX_REALMS: u32 = 64u;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Fullscreen triangle (no vertex buffers).
@vertex
fn vs_main(@builtin(vertex_index) vidx: u32) -> VsOut {
    let x = f32((vidx << 1u) & 2u);
    let y = f32(vidx & 2u);
    let pos_ndc = vec2<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0);
    let uv = vec2<f32>(x, y);
    return VsOut(vec4<f32>(pos_ndc, 0.0, 1.0), uv);
}

// Bake the field with same-realm reinforcement. Two-stage algorithm:
//
//   1. Walk every settlement once. Bucket each contribution
//      (`strength * exp(-d/E_FOLD)`) into `realm_sums[realm % MAX_REALMS]`,
//      and remember the *single strongest* contributor per realm in
//      `realm_best_*` so we can still hand `city_idx` back to the hover
//      path (it points at the dominant city of the dominant realm).
//   2. Argmax across the per-realm sums. The "competitor" is now the
//      second-best *realm cluster*, not just the second-best individual
//      city — so the contested / iso-line lives between cluster fields
//      rather than between point fields, producing organic curved borders
//      around city clusters instead of straight perpendicular bisectors.
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // UV → world XZ. `world_to_world_uv` in world.wgsl maps
    // world_x ∈ [-HALF, +HALF] → uv.x ∈ [0, 1] linearly, and
    // world_z ∈ [+HALF, -HALF] → uv.y ∈ [0, 1] (Y-flipped, because PNG
    // row 0 sits at geographic north). Inverting:
    let world_x = (in.uv.x - 0.5) * 2.0 * WORLD_BOUNDS_HALF_KM.x;
    let world_z = (0.5 - in.uv.y) * 2.0 * WORLD_BOUNDS_HALF_KM.y;
    let xz = vec2<f32>(world_x, world_z);

    // Per-realm accumulators. WGSL var-arrays default-initialise to zero
    // so we don't need an explicit clearing loop. Array sizes must equal
    // `MAX_REALMS`.
    var realm_sums:          array<f32, 64>;
    var realm_best_strength: array<f32, 64>;
    var realm_best_city:     array<u32, 64>;

    let n = min(settlements.count, MAX_SETTLEMENTS);
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let s = settlements.items[i];
        let d_km = distance(xz, s.world_xz);
        // Long-range population pull + short-range local-core bump.
        // The local term gives every settlement a guaranteed hinterland
        // of roughly LOCAL_E_FOLD_KM around itself; both terms feed the
        // same realm sum and per-realm best-city tracker, so within a
        // realm the dominant *city* (used for hinterland highlighting)
        // is whichever settlement owns this point's local core.
        let v_long  = s.strength * exp(-d_km / SETTLEMENT_E_FOLD_KM);
        let v_local = SETTLEMENT_LOCAL_BONUS
            * exp(-d_km / SETTLEMENT_LOCAL_E_FOLD_KM);
        let v = v_long + v_local;
        let r = s.realm_id % MAX_REALMS;
        realm_sums[r] = realm_sums[r] + v;
        if (v > realm_best_strength[r]) {
            realm_best_strength[r] = v;
            realm_best_city[r] = i;
        }
    }

    // Argmax + second-best across realm clusters.
    var best_strength: f32 = 0.0;
    var best_realm: u32 = 0u;
    var second_strength: f32 = 0.0;
    for (var r: u32 = 0u; r < MAX_REALMS; r = r + 1u) {
        let v = realm_sums[r];
        if (v > best_strength) {
            second_strength = best_strength;
            best_strength = v;
            best_realm = r;
        } else if (v > second_strength) {
            second_strength = v;
        }
    }
    let best_city = realm_best_city[best_realm];

    // Saturating fade: the 0.05 multiplier is half the old single-city
    // value of 0.1, since `best_strength` is now a SUM (typically ~2×
    // a single city's contribution in dense areas). Keeps the alpha
    // calibrated to the same "saturates inside the realm core" feel.
    let alpha = 1.0 - exp(-best_strength * 0.05);
    // contested = best / second (∞ in interior, 1.0 on the iso-line);
    // we pack (contested - 1) clamped to [0, 1] into the B channel and
    // unpack on the read side.
    let contested_minus_one = clamp(
        best_strength / max(second_strength, 1e-6) - 1.0,
        0.0, 1.0,
    );

    // Rgba16Float — store realm_id and city_idx as raw float values; both
    // round-trip exactly through f16 since they're < 2048. The image pass
    // recovers them with `u32(round(s.x))` and `u32(round(s.w))`.
    return vec4<f32>(
        f32(best_realm),
        alpha,
        contested_minus_one,
        f32(best_city),
    );
}
