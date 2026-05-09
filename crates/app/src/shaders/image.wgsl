// ============================================================================
// Image pass: raymarched terrain rendering. Renders to the swapchain.
//
// Reads:
//   u              (binding 0) — camera uniforms (from camera.wgsl)
//   base_heightmap (binding 1) — world-anchored painted heightmap
//   detail_noise   (binding 2) — world-anchored detail noise
//   terrain        (binding 3) — world-anchored eroded heightmap (the terrain)
//   layer          (binding 4) — world AABB covered by the three layers above
//   water_mask     (binding 5) — binary mask, 1.0 where water, 0.0 elsewhere
// ============================================================================
@group(0) @binding(1) var base_heightmap: texture_2d<f32>;
@group(0) @binding(2) var detail_noise: texture_2d<f32>;
@group(0) @binding(3) var terrain: texture_2d<f32>;
@group(0) @binding(4) var<uniform> layer: LayerUniforms;
@group(0) @binding(5) var water_mask: texture_2d<f32>;

// ----------------------------------------------------------------------------
// Constants
// ----------------------------------------------------------------------------
const PI: f32 = 3.14159265358979;
const DEG_TO_RAD: f32 = PI / 180.0;

// Camera: EU4-style top-down. Eye position relative to look_at comes from the
// `eye_offset` uniform (driven by camera_distance + camera_tilt on the Rust
// side). North = +Z, so north stays "up" on the screen.
const CAMERA_FOV_Y:    f32 = 30.0;  // Vertical FOV in degrees — must match CAMERA_FOV_Y_RAD in lib.rs.
const CAMERA_TARGET_Y: f32 = 0.4;   // Lookat height — sits roughly at terrain level.

// Material IDs.
const M_GROUND: i32 = 0;
const M_STRATA: i32 = 1;
const M_WATER:  i32 = 2;

// Color palette.
const CLIFF_COLOR       = vec3<f32>(0.22, 0.20, 0.20);
const DIRT_COLOR        = vec3<f32>(0.60, 0.50, 0.40);
const GRASS_COLOR1      = vec3<f32>(0.15, 0.30, 0.10);
const GRASS_COLOR2      = vec3<f32>(0.40, 0.50, 0.20);
const SAND_COLOR        = vec3<f32>(0.80, 0.70, 0.60);
const WATER_COLOR       = vec3<f32>(0.00, 0.05, 0.10);
const WATER_SHORE_COLOR = vec3<f32>(0.00, 0.25, 0.25);
const SUN_COLOR         = vec3<f32>(2.0, 1.96, 1.90);
const AMBIENT_COLOR     = vec3<f32>(0.03, 0.05, 0.07);

// Heights / quality. Heights are fractions of HEIGHT_SCALE_M (= 5000 m), so
// y = 0 is sea level, y = 1 is the highest representable peak. Lake Maggiore
// (Switzerland's lowest body of water) sits at ~190 m → 0.038.
const WATER_HEIGHT:    f32 = 0.038;
const GRASS_HEIGHT:    f32 = 0.045;
const RAYMARCH_QUALITY: f32 = 2.0;

// Atmosphere coefficients (Rayleigh + Mie).
const C_RAYLEIGH = vec3<f32>(5.802e-6, 13.558e-6, 33.100e-6);
const C_MIE      = vec3<f32>(3.996e-6,  3.996e-6,  3.996e-6);

// Half-extents of the playable box in world space. XZ comes from the cached
// world layer's covered AABB — the box matches the region for which the
// heightmap is valid. Y stays fixed: it's the maximum terrain height.
fn box_size() -> vec3<f32> {
    let half = (layer.covered_max - layer.covered_min) * 0.5;
    return vec3<f32>(half.x, 1.0, half.y);
}

// Center of the playable box in world space (Y = 0, XZ at the layer center).
fn box_center() -> vec3<f32> {
    let mid = (layer.covered_min + layer.covered_max) * 0.5;
    return vec3<f32>(mid.x, 0.0, mid.y);
}

// ----------------------------------------------------------------------------
// Math helpers
// ----------------------------------------------------------------------------
fn clamp01(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }
fn sq(x: f32) -> f32 { return x * x; }
fn pow5(x: f32) -> f32 { let x2 = x * x; return x2 * x2 * x; }

// ----------------------------------------------------------------------------
// Box intersection (Inigo Quilez)
// ----------------------------------------------------------------------------
struct BoxHit {
    t_near: f32,
    t_far:  f32,
    normal: vec3<f32>,
}

fn box_intersection(ro: vec3<f32>, rd: vec3<f32>, box_size: vec3<f32>) -> BoxHit {
    let m = 1.0 / rd;
    let n = m * ro;
    let k = abs(m) * box_size;
    let t1 = -n - k;
    let t2 = -n + k;
    let t_near = max(max(t1.x, t1.y), t1.z);
    let t_far  = min(min(t2.x, t2.y), t2.z);
    if (t_near > t_far || t_far < 0.0) {
        return BoxHit(-1.0, -1.0, vec3<f32>(0.0));
    }
    let normal = -sign(rd) * step(t1.yzx, t1.xyz) * step(t1.zxy, t1.xyz);
    return BoxHit(t_near, t_far, normal);
}

// ----------------------------------------------------------------------------
// Camera: top-down look-at, panned via `u.world_center`.
//
// The framebuffer is Y-down but our world is Y-up. The single Y-flip needed
// to bridge the two lives in the `ndc_y` line below.
// ----------------------------------------------------------------------------
struct Ray { ro: vec3<f32>, rd: vec3<f32> }

fn get_ray(frag_pos: vec2<f32>) -> Ray {
    // Lookat point: pan via world_center.x/y (mapped to world XZ).
    // (Don't name this `target` — reserved keyword in WGSL.)
    let look_at = vec3<f32>(u.world_center.x, CAMERA_TARGET_Y, u.world_center.y);
    // Eye: look_at + uniform-driven offset (encodes both distance and tilt).
    let eye = look_at + u.eye_offset;

    // Orthonormal camera basis (right-handed, Y-up).
    let forward = normalize(look_at - eye);
    let world_up = vec3<f32>(0.0, 1.0, 0.0);
    let right = normalize(cross(world_up, forward));   // east
    let up = cross(forward, right);                    // perp to both

    // Pixel → ray direction. NDC in [-1, 1] both axes; aspect-correct on x.
    let aspect = u.i_resolution.x / u.i_resolution.y;
    let tan_half_fov_y = tan(CAMERA_FOV_Y * 0.5 * DEG_TO_RAD);
    let tan_half_fov_x = tan_half_fov_y * aspect;
    let ndc_x = (frag_pos.x / u.i_resolution.x) * 2.0 - 1.0;
    // Y-flip: framebuffer-Y is top-down, but we want top-of-screen pixels to
    // produce rays tilted in the +up direction.
    let ndc_y = 1.0 - (frag_pos.y / u.i_resolution.y) * 2.0;

    let rd = normalize(
        forward
        + right * (ndc_x * tan_half_fov_x)
        + up    * (ndc_y * tan_half_fov_y)
    );
    return Ray(eye, rd);
}

// ----------------------------------------------------------------------------
// Sky / atmosphere
// ----------------------------------------------------------------------------
fn sky_color(rd: vec3<f32>, sun: vec3<f32>) -> vec3<f32> {
    let costh = dot(rd, sun);
    return AMBIENT_COLOR * PI * (1.0 - abs(costh) * 0.8);
}

fn phase_rayleigh(costh: f32) -> f32 {
    return 3.0 * (1.0 + costh * costh) / (16.0 * PI);
}

fn phase_mie(costh: f32, g_in: f32) -> f32 {
    let g = min(g_in, 0.9381);
    let k = 1.55 * g - 0.55 * g * g * g;
    let kcosth = k * costh;
    return (1.0 - k * k) / ((4.0 * PI) * (1.0 - kcosth) * (1.0 - kcosth));
}

// ----------------------------------------------------------------------------
// Filament-style BRDF
// ----------------------------------------------------------------------------
fn d_ggx(linear_roughness: f32, n_o_h: f32) -> f32 {
    let one_minus_noh_squared = 1.0 - n_o_h * n_o_h;
    let a = n_o_h * linear_roughness;
    let k = linear_roughness / (one_minus_noh_squared + a * a);
    return k * k * (1.0 / PI);
}

fn v_smith(linear_roughness: f32, n_o_v: f32, n_o_l: f32) -> f32 {
    let a2 = linear_roughness * linear_roughness;
    let ggxv = n_o_l * sqrt((n_o_v - a2 * n_o_v) * n_o_v + a2);
    let ggxl = n_o_v * sqrt((n_o_l - a2 * n_o_l) * n_o_l + a2);
    return 0.5 / (ggxv + ggxl);
}

fn f_schlick(f0: vec3<f32>, v_o_h: f32) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow5(1.0 - v_o_h);
}

fn f_schlick_scalar(f0: f32, f90: f32, v_o_h: f32) -> f32 {
    return f0 + (f90 - f0) * pow5(1.0 - v_o_h);
}

fn fd_burley(linear_roughness: f32, n_o_v: f32, n_o_l: f32, l_o_h: f32) -> f32 {
    let f90 = 0.5 + 2.0 * linear_roughness * l_o_h * l_o_h;
    let light_scatter = f_schlick_scalar(1.0, f90, n_o_l);
    let view_scatter  = f_schlick_scalar(1.0, f90, n_o_v);
    return light_scatter * view_scatter * (1.0 / PI);
}

fn fd_lambert() -> f32 { return 1.0 / PI; }

fn shade(
    diffuse: vec3<f32>, f0: vec3<f32>, smoothness: f32,
    n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, lc: vec3<f32>,
) -> vec3<f32> {
    let h = normalize(v + l);
    let n_o_v = abs(dot(n, v)) + 1e-5;
    let n_o_l = clamp01(dot(n, l));
    let n_o_h = clamp01(dot(n, h));
    let l_o_h = clamp01(dot(l, h));

    let roughness = 1.0 - smoothness;
    let lr = roughness * roughness;
    let d  = d_ggx(lr, n_o_h);
    let vt = v_smith(lr, n_o_v, n_o_l);
    let fresnel = f_schlick(f0, l_o_h);
    let fr = (d * vt) * fresnel;
    let fd = diffuse * fd_burley(lr, n_o_v, n_o_l, l_o_h);
    return (fd + fr) * lc * n_o_l;
}

// ACES filmic tonemapping (Narkowicz 2015).
fn tonemap_aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return (x * (a * x + b)) / (x * (c * x + d) + e);
}

// ----------------------------------------------------------------------------
// Heightmap sampling
// ----------------------------------------------------------------------------
fn uv_to_coord(uv: vec2<f32>) -> vec2<i32> {
    let dim_f = vec2<f32>(LAYER_SIZE);
    let dim_i = vec2<i32>(i32(LAYER_SIZE));
    return clamp(vec2<i32>(uv * dim_f), vec2<i32>(0), dim_i - vec2<i32>(1));
}

// Map a world XZ to the [0,1] UV of the world layers (which all share the
// same covered AABB). Clamped one texel inside the edge so neighbour
// fetches in `map_full` stay within the valid region.
fn world_xz_to_uv(p: vec3<f32>) -> vec2<f32> {
    let pixel = 1.0 / vec2<f32>(LAYER_SIZE);
    let uv = world_to_layer_uv(p.xz, layer);
    return clamp(uv, pixel, vec2<f32>(1.0) - pixel);
}

fn map_height(uv: vec2<f32>) -> f32 {
    return textureLoad(terrain, uv_to_coord(uv), 0).x;
}

// height in .x, normal in .yzw (Y-up world space).
fn map_full(uv: vec2<f32>) -> vec4<f32> {
    let height = map_height(uv);
    let pixel = 1.0 / vec2<f32>(LAYER_SIZE);
    let uv1 = uv + vec2<f32>(pixel.x, 0.0);
    let uv2 = uv + vec2<f32>(0.0, pixel.y);
    let h1 = map_height(uv1);
    let h2 = map_height(uv2);
    let v1 = vec3<f32>(uv1 - uv, h1 - height);
    let v2 = vec3<f32>(uv2 - uv, h2 - height);
    // Cross product gives normal in (u, v, h) local coords; .xzy reorders to
    // (u, h, v) i.e. Y-up world.
    let raw = normalize(cross(v1, v2));
    return vec4<f32>(height, raw.x, raw.z, raw.y);
}

// ----------------------------------------------------------------------------
// Raymarching
// ----------------------------------------------------------------------------
struct MarchResult {
    t: f32,             // distance along ray to hit, or -1 if miss
    normal: vec3<f32>,  // only valid for STRATA / WATER (GROUND filled by caller)
    material: i32,
    s_t: f32,           // soft-shadow factor
}

fn march(ro: vec3<f32>, rd: vec3<f32>) -> MarchResult {
    var s_t: f32 = 9999.0;
    let bs = box_size();
    let bc = box_center();
    // The box is axis-aligned but not centered at world origin (it tracks the
    // camera via the world layer's covered AABB). Translate the ray origin
    // into the box's local frame for the intersection; the returned
    // (t_near, t_far, normal) are in ray-space and need no further fix-up.
    let local_ro = ro - bc;
    let box = box_intersection(local_ro, rd, bs);

    let t_start = max(0.0, box.t_near) + 1e-2;
    let t_end   = box.t_far - 1e-2;

    var material: i32 = M_GROUND;
    var normal = vec3<f32>(0.0);
    var step_size:  f32 = 0.0;
    var step_scale: f32 = 1.0 / RAYMARCH_QUALITY;
    let samples: i32 = i32(48.0 * RAYMARCH_QUALITY);
    var t: f32 = t_start;
    var hit_strata: bool = false;

    for (var i: i32 = 0; i < samples; i = i + 1) {
        let pos = ro + rd * t;
        let h = map_height(world_xz_to_uv(pos));
        let altitude = pos.y - h;
        s_t = max(0.0, min(s_t, altitude / t));

        if (altitude < 0.0 && i < 1) {
            // Hit a side wall of the bounding box on first sample. Below sea
            // level (y < 0) we escape to sky; otherwise it's an exposed cliff
            // (strata material).
            if (pos.y < 0.0) {
                return MarchResult(-1.0, vec3<f32>(0.0), M_GROUND, 9999.0);
            }
            normal = box.normal;
            material = M_STRATA;
            hit_strata = true;
            break;
        }

        if (altitude < 0.0) {
            // Refine: undo last step and halve the step scale.
            step_scale *= 0.5;
            t -= step_size * step_scale;
        } else {
            step_size = abs(altitude) + min(1e-2, abs(altitude) * 0.01);
            t += step_size * step_scale;
        }
    }

    // Possibly cap on water surface (same XZ extent as the playable box, but
    // Y half-extent = WATER_HEIGHT so its top face sits at +WATER_HEIGHT).
    let water_box = box_intersection(local_ro, rd, vec3<f32>(bs.x, WATER_HEIGHT, bs.z));
    if ((water_box.t_far > 0.0 && (water_box.t_near < t || t < 0.0)) && !hit_strata) {
        t = max(0.0, water_box.t_near);
        normal = water_box.normal;
        material = M_WATER;
    }

    if (box.t_far < 0.0) {
        return MarchResult(-1.0, vec3<f32>(0.0), M_GROUND, 9999.0);
    }
    if (t > t_end) {
        return MarchResult(-1.0, normal, material, s_t);
    }
    return MarchResult(t, normal, material, s_t);
}

fn get_reflection(p: vec3<f32>, r: vec3<f32>, sun: vec3<f32>, smoothness: f32) -> vec3<f32> {
    let refl = sky_color(r, sun) * 4.0;
    let m = march(p, r);
    return refl * (1.0 - exp(-m.s_t * 10.0 * sq(smoothness)));
}

// ----------------------------------------------------------------------------
// Main fragment shader
// ----------------------------------------------------------------------------
@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let frag_pos = frag.xy;

    // Camera setup
    let ray = get_ray(frag_pos);
    let ro = ray.ro;
    let rd = ray.rd;

    // Fixed sun direction (Y-up world).
    let sun = normalize(vec3<f32>(-1.0, 0.4, 0.05));

    // Trace the ray.
    let m = march(ro, rd);

    let fog_color = vec3<f32>(1.0) - exp(-sky_color(rd, sun) * 2.0);

    var color = vec3<f32>(0.0);

    if (m.t < 0.0) {
        // Sky. Brighter at top of screen — Y-down screen so flip the gradient term.
        let sky_t = 1.0 - frag_pos.y / u.i_resolution.y;
        color = fog_color * (1.0 + pow(sky_t, 3.0) * 3.0) * 0.5;
    } else {
        let pos = ro + rd * m.t;
        var normal = m.normal;
        var material = m.material;

        if (material == M_GROUND) {
            normal = map_full(world_xz_to_uv(pos)).yzw;
        }

        // Water-mask override: if the real Switzerland water mask says this
        // XZ is water, classify the hit as water regardless of what the
        // box-intersection-based march decided. Skip strata cliffs since
        // those are exposed rock walls (not on the surface).
        if (material != M_STRATA) {
            let mask_size_i = i32(WORLD_HEIGHTMAP_SIZE);
            let mask_uv = world_to_world_uv(pos.xz);
            let mask_coord = clamp(
                vec2<i32>(mask_uv * vec2<f32>(WORLD_HEIGHTMAP_SIZE)),
                vec2<i32>(0),
                vec2<i32>(mask_size_i - 1),
            );
            let mask = textureLoad(water_mask, mask_coord, 0).x;
            if (mask > 0.5) {
                material = M_WATER;
                normal = vec3<f32>(0.0, 1.0, 0.0);
            }
        }

        // Detail breakup texture.
        let breakup_tex = textureLoad(
            detail_noise,
            uv_to_coord(world_xz_to_uv(pos)),
            0,
        );
        let breakup = breakup_tex.x;
        if (material == M_WATER) {
            normal = normalize(normal + vec3<f32>(breakup_tex.z, 0.0, breakup_tex.y) * 0.1);
        }

        var diffuse_color = vec3<f32>(0.5);
        var f0 = vec3<f32>(0.04);
        var smoothness: f32 = 0.0;
        // Hard-coded since we deferred packed-data extraction.
        let occlusion: f32 = 1.0;
        let trees_v: f32 = -1.0;

        let r_dir = reflect(rd, normal);

        if (material == M_GROUND) {
            // Cliff base.
            diffuse_color = CLIFF_COLOR * smoothstep(0.4, 0.52, pos.y);
            // Dirt over cliff (smoothstep edges flipped to keep WGSL well-defined).
            diffuse_color = mix(
                diffuse_color, DIRT_COLOR,
                1.0 - smoothstep(0.0, 0.6, occlusion + breakup * 1.5),
            );
            // Snow.
            diffuse_color = mix(
                diffuse_color, vec3<f32>(1.0),
                smoothstep(0.53, 0.6, pos.y + breakup * 0.1),
            );
            // Sand (beach) — smoothstep edges flipped.
            diffuse_color = mix(
                diffuse_color, SAND_COLOR,
                1.0 - smoothstep(WATER_HEIGHT, WATER_HEIGHT + 0.005, pos.y + breakup * 0.01),
            );
            // Grass.
            let grass_mix = mix(
                GRASS_COLOR1, GRASS_COLOR2,
                smoothstep(0.4, 0.6, pos.y + breakup * 0.3),
            );
            let grass_height_mask =
                1.0 - smoothstep(GRASS_HEIGHT + 0.02, GRASS_HEIGHT + 0.05,
                    pos.y + 0.01 + (occlusion - 0.8) * 0.05 - breakup * 0.02);
            let grass_slope_mask = smoothstep(
                0.8, 1.0,
                1.0 - (1.0 - normal.y) * (1.0 - trees_v) + breakup * 0.1,
            );
            diffuse_color = mix(
                diffuse_color, grass_mix,
                grass_height_mask * grass_slope_mask,
            );
            diffuse_color *= 1.0 + breakup * 0.5;
        } else if (material == M_STRATA) {
            let diff = pos.y - map_height(world_xz_to_uv(pos));
            let strata = smoothstep(
                vec3<f32>(0.0), vec3<f32>(1.0),
                cos(diff * vec3<f32>(130.0, 190.0, 250.0)),
            );
            diffuse_color = vec3<f32>(0.3);
            diffuse_color = mix(diffuse_color, vec3<f32>(0.50), strata.x);
            diffuse_color = mix(diffuse_color, vec3<f32>(0.55), strata.y);
            diffuse_color = mix(diffuse_color, vec3<f32>(0.60), strata.z);
            diffuse_color *= exp(diff * 10.0) * vec3<f32>(1.0, 0.9, 0.7);
        } else { // M_WATER
            let diff = pos.y - map_height(world_xz_to_uv(pos));
            let shore = select(0.0, exp(-diff * 60.0), normal.y > 1e-2);
            // smoothstep(0.005, 0.0, x) is reversed → flip.
            let foam = select(
                0.0,
                1.0 - smoothstep(0.0, 0.005, diff + breakup * 0.005),
                normal.y > 1e-2,
            );
            diffuse_color = mix(WATER_COLOR, WATER_SHORE_COLOR, shore);
            diffuse_color = mix(diffuse_color, vec3<f32>(1.0), foam);
            smoothness = 0.95;
        }

        // Shadow ray (skip for the inside-of-strata case).
        var shadow: f32 = 1.0;
        if (material != M_STRATA) {
            let sh = march(pos + vec3<f32>(0.0, 1.0, 0.0) * 1e-4, sun);
            shadow = 1.0 - exp(-sh.s_t * 20.0);
        }

        // Lighting decomposition.
        color = diffuse_color * sky_color(normal, sun) * fd_lambert();
        color *= occlusion;
        color += shade(diffuse_color, f0, smoothness, normal, -rd, sun, SUN_COLOR * shadow);
        // Fake bounce.
        color += diffuse_color * SUN_COLOR
            * (dot(normal, sun * vec3<f32>(1.0, -1.0, 1.0)) * 0.5 + 0.5)
            * fd_lambert() / PI;
        // Reflection.
        color += get_reflection(pos, r_dir, sun, smoothness)
            * f_schlick(f0, dot(-rd, normal));
    }

    // ---- Atmospheric scattering integral --------------------------------
    let bc_atm = box_center();
    let box = box_intersection(ro - bc_atm, rd, box_size());
    let costh = dot(rd, sun);
    let phase_r = phase_rayleigh(costh);
    let phase_m = phase_mie(costh, 0.6);

    var od = vec2<f32>(0.0);
    var tsm = vec3<f32>(1.0);
    var sct = vec3<f32>(0.0);
    let ray_length = select(box.t_far, m.t, m.t > 0.0) - box.t_near;
    let stepsize = ray_length / 16.0;
    for (var i: f32 = 0.0; i < 16.0; i = i + 1.0) {
        let p = ro + rd * (box.t_near + (i + 0.5) * stepsize);
        // Stylised low-altitude haze: full density at sea level (y = 0),
        // fading to nothing by 1000 m (y = 0.2). Below sea level we have no
        // atmosphere (the ray is underground inside the box).
        var d = 1.0 - clamp01(max(0.0, p.y) / 0.2);
        if (p.y < 0.0) { d = 0.0; }
        let density_r = d * 1e5;
        let density_m = d * 1e5;
        od += stepsize * vec2<f32>(density_r, density_m);
        tsm = exp(-(od.x * C_RAYLEIGH + od.y * C_MIE));
        sct += tsm * C_RAYLEIGH * phase_r * density_r * stepsize;
        sct += tsm * C_MIE      * phase_m * density_m * stepsize;
    }
    color = color * tsm + sct * 10.0;

    // Tonemap + gamma.
    color = tonemap_aces(color);
    color = pow(color, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(color, 1.0);
}
