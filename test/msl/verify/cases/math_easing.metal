// Ported from test/wesl/math-easing.test.ts (same inputs and expected values).
// The invCubic and invQuartic round trips use epsilon 0.01, so they're in
// math_easing_loose.metal. The parabola test also checks parabola(0.25, 2) <
// parabola(0.25, 1), which is implied by the exact values below (0.5625 < 0.75);
// its toBeCloseTo(x, 2) checks are looser than the exact checks, so they're covered too.
#include <metal_stdlib>
using namespace metal;
#include "lygia/math/cubic.msl"
#include "lygia/math/quartic.msl"
#include "lygia/math/quintic.msl"
#include "lygia/math/gain.msl"
#include "lygia/math/parabola.msl"
#include "lygia/math/gaussian.msl"

// @test cubic
// @expect cubic[0] 0.0 0.5 1.0 0.0
kernel void cubic(device float4* results [[buffer(0)]]) {
    results[0] = float4(cubic(0.0), cubic(0.5), cubic(1.0), 0.0);
}

// @test quartic
// @expect quartic[0] 0.0 0.4375 1.0 0.0
kernel void quartic(device float4* results [[buffer(0)]]) {
    results[0] = float4(quartic(0.0), quartic(0.5), quartic(1.0), 0.0);
}

// @test quintic
// @expect quintic[0] 0.0 0.5 1.0 0.0
kernel void quintic(device float4* results [[buffer(0)]]) {
    results[0] = float4(quintic(0.0), quintic(0.5), quintic(1.0), 0.0);
}

// @test gain
// @expect gain[0] 0.5 0.125 0.875 0.0
kernel void gain(device float4* results [[buffer(0)]]) {
    results[0] = float4(gain(0.5, 2.0), gain(0.25, 2.0), gain(0.75, 2.0), 0.0);
}

// @test parabola 2
// @expect parabola[0] 0.0 0.75 1.0 0.75
// @expect parabola[1] 0.5625 1.0 0.5625 0.0
kernel void parabola(device float4* results [[buffer(0)]]) {
    results[0] = float4(parabola(0.0, 1.0), parabola(0.25, 1.0), parabola(0.5, 1.0), parabola(0.75, 1.0));
    results[1] = float4(parabola(0.25, 2.0), parabola(0.5, 2.0), parabola(0.75, 2.0), 0.0);
}

// @test gaussian
// @expect gaussian[0] 1.0 0.6065
kernel void gaussian(device float4* results [[buffer(0)]]) {
    results[0] = float4(gaussian(0.0, 1.0), gaussian(1.0, 1.0), 0.0, 0.0);
}
