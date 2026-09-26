// Ported from test/wesl/shaders/animation_easing.test.wesl (same inputs and expected values).
// The WESL test uses expectNear/expectNearVec4 (relative 1e-3, absolute 1e-6); here the
// default absolute 0.0001 is used (stricter than 1e-3 relative for |values| > 0.1, looser
// only for the elastic values near 0.02, where WGSL allows ~2e-5).
// Each kernel: results[0] = (xIn(0.5), xOut(0.5)), results[1] = xInOut(0, 0.25, 0.75, 1).
// INV_SQRT2 = 0.70710678; sineIn(0.5) = 1 - INV_SQRT2 = 0.29289322, sineInOut(0.25/0.75) = 0.14644661 / 0.85355339.
#include <metal_stdlib>
using namespace metal;
#include "lygia/animation/easing.msl"

// @test back 2
// @expect back[0] -0.375 1.375
// @expect back[1] 0.0 -0.1875 1.1875 1.0
kernel void back(device float4* results [[buffer(0)]]) {
    results[0] = float4(backIn(0.5), backOut(0.5), 0.0, 0.0);
    results[1] = float4(backInOut(0.0), backInOut(0.25), backInOut(0.75), backInOut(1.0));
}

// @test bounce 2
// @expect bounce[0] 0.28125 0.71875
// @expect bounce[1] 0.0 0.140625 0.859375 1.0
kernel void bounce(device float4* results [[buffer(0)]]) {
    results[0] = float4(bounceIn(0.5), bounceOut(0.5), 0.0, 0.0);
    results[1] = float4(bounceInOut(0.0), bounceInOut(0.25), bounceInOut(0.75), bounceInOut(1.0));
}

// @test circular 2
// @expect circular[0] 0.134 0.866
// @expect circular[1] 0.0 0.067 0.933 1.0
kernel void circular(device float4* results [[buffer(0)]]) {
    results[0] = float4(circularIn(0.5), circularOut(0.5), 0.0, 0.0);
    results[1] = float4(circularInOut(0.0), circularInOut(0.25), circularInOut(0.75), circularInOut(1.0));
}

// @test cubic 2
// @expect cubic[0] 0.125 0.875
// @expect cubic[1] 0.0 0.0625 0.9375 1.0
kernel void cubic(device float4* results [[buffer(0)]]) {
    results[0] = float4(cubicIn(0.5), cubicOut(0.5), 0.0, 0.0);
    results[1] = float4(cubicInOut(0.0), cubicInOut(0.25), cubicInOut(0.75), cubicInOut(1.0));
}

// @test elastic 2
// @expect elastic[0] -0.0221 1.0221
// @expect elastic[1] 0.0 -0.01105 1.01105 1.0
kernel void elastic(device float4* results [[buffer(0)]]) {
    results[0] = float4(elasticIn(0.5), elasticOut(0.5), 0.0, 0.0);
    results[1] = float4(elasticInOut(0.0), elasticInOut(0.25), elasticInOut(0.75), elasticInOut(1.0));
}

// @test exponential 2
// @expect exponential[0] 0.03125 0.96875
// @expect exponential[1] 0.0 0.015625 0.984375 1.0
kernel void exponential(device float4* results [[buffer(0)]]) {
    results[0] = float4(exponentialIn(0.5), exponentialOut(0.5), 0.0, 0.0);
    results[1] = float4(exponentialInOut(0.0), exponentialInOut(0.25), exponentialInOut(0.75), exponentialInOut(1.0));
}

// @test quadratic 2
// @expect quadratic[0] 0.25 0.75
// @expect quadratic[1] 0.0 0.125 0.875 1.0
kernel void quadratic(device float4* results [[buffer(0)]]) {
    results[0] = float4(quadraticIn(0.5), quadraticOut(0.5), 0.0, 0.0);
    results[1] = float4(quadraticInOut(0.0), quadraticInOut(0.25), quadraticInOut(0.75), quadraticInOut(1.0));
}

// @test quartic 2
// @expect quartic[0] 0.0625 0.9375
// @expect quartic[1] 0.0 0.03125 0.96875 1.0
kernel void quartic(device float4* results [[buffer(0)]]) {
    results[0] = float4(quarticIn(0.5), quarticOut(0.5), 0.0, 0.0);
    results[1] = float4(quarticInOut(0.0), quarticInOut(0.25), quarticInOut(0.75), quarticInOut(1.0));
}

// @test linear
// @expect linear[0] 0.5 0.5 0.5
kernel void linear(device float4* results [[buffer(0)]]) {
    results[0] = float4(linearIn(0.5), linearOut(0.5), linearInOut(0.5), 0.0);
}

// @test quintic
// @expect quintic[0] 0.03125 1.03125 1.5
kernel void quintic(device float4* results [[buffer(0)]]) {
    results[0] = float4(quinticIn(0.5), quinticOut(0.5), quinticInOut(0.5), 0.0);
}

// @test sine 2
// @expect sine[0] 0.29289322 0.70710678
// @expect sine[1] 0.0 0.14644661 0.85355339 1.0
kernel void sine(device float4* results [[buffer(0)]]) {
    results[0] = float4(sineIn(0.5), sineOut(0.5), 0.0, 0.0);
    results[1] = float4(sineInOut(0.0), sineInOut(0.25), sineInOut(0.75), sineInOut(1.0));
}
