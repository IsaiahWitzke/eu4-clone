// ============================================================================
// Shared 2D noise primitives, used by Buffer B (erosion), Buffer C (detail
// texture), and the Image pass (tree placement).
// Translated from the Common tab of the original Shadertoy.
// ============================================================================

// Cheap 2D hash → vec2 in [-1, 1]^2 (a pseudo-random gradient direction).
fn hash(input_x: vec2<f32>) -> vec2<f32> {
    let k = vec2<f32>(0.3183099, 0.3678794);
    let x = input_x * k + k.yx;
    return -1.0 + 2.0 * fract(16.0 * k * fract(x.x * x.y * (x.x + x.y)));
}

// Analytic gradient noise. Returns (value, derivative.x, derivative.y).
// Value is roughly in [-1, 1]; derivatives are the partial derivatives of
// `value` with respect to `p`. Source: Inigo Quilez,
// https://www.shadertoy.com/view/XdXBRH
fn noised(p: vec2<f32>) -> vec3<f32> {
    let i = floor(p);
    let f = fract(p);

    // Quintic smoothstep and its derivative — gives C2-continuous noise.
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);

    let ga = hash(i + vec2<f32>(0.0, 0.0));
    let gb = hash(i + vec2<f32>(1.0, 0.0));
    let gc = hash(i + vec2<f32>(0.0, 1.0));
    let gd = hash(i + vec2<f32>(1.0, 1.0));

    let va = dot(ga, f - vec2<f32>(0.0, 0.0));
    let vb = dot(gb, f - vec2<f32>(1.0, 0.0));
    let vc = dot(gc, f - vec2<f32>(0.0, 1.0));
    let vd = dot(gd, f - vec2<f32>(1.0, 1.0));

    let value = va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * (va - vb - vc + vd);
    let deriv = ga
        + u.x * (gb - ga)
        + u.y * (gc - ga)
        + u.x * u.y * (ga - gb - gc + gd)
        + du * (u.yx * (va - vb - vc + vd) + vec2<f32>(vb, vc) - va);
    return vec3<f32>(value, deriv);
}
