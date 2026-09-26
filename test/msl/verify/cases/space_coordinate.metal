// Ported from test/wesl/space-coordinate.test.ts (same inputs and expected values).
// WGSL cart2polar2(vec2f) is MSL cart2polar(float2). SQRT2 (lygia::math::consts, WESL only;
// not in math/const.msl) is written as a literal.
#include <metal_stdlib>
using namespace metal;
#include "lygia/math/const.msl"
#include "lygia/space/cart2polar.msl"
#include "lygia/space/polar2cart.msl"
#include "lygia/space/equirect2xyz.msl"
#include "lygia/space/xyz2equirect.msl"
#include "lygia/space/fisheye2xyz.msl"
#include "lygia/space/nearest.msl"

// (QTR_PI, SQRT2)
// @test cart2polar2
// @expect cart2polar2[0] 0.78539816 1.41421356
kernel void cart2polar2(device float4* results [[buffer(0)]]) {
    results[0] = float4(cart2polar(float2(1.0, 1.0)), 0.0, 0.0);
}

// @test polar2cart
// @expect polar2cart[0] 1.0 1.0
kernel void polar2cart(device float4* results [[buffer(0)]]) {
    results[0] = float4(polar2cart(float2(QTR_PI, 1.4142135623730951)), 0.0, 0.0);
}

// @test equirect2xyz
// @expect equirect2xyz[0] -1.0 0.0 0.0 0.0
kernel void equirect2xyz(device float4* results [[buffer(0)]]) {
    results[0] = float4(equirect2xyz(float2(0.5, 0.5)), 0.0);
}

// @test xyz2equirect
// @expect xyz2equirect[0] 0.5 0.5
kernel void xyz2equirect(device float4* results [[buffer(0)]]) {
    results[0] = float4(xyz2equirect(float3(1.0, 0.0, 0.0)), 0.0, 0.0);
}

// "fisheye2xyz": the direction is normalized (results[0].x = length)
// @test fisheye2xyz
// @expect fisheye2xyz[0] 1.0
kernel void fisheye2xyz(device float4* results [[buffer(0)]]) {
    float3 r = fisheye2xyz(float2(0.75, 0.5));
    results[0] = float4(length(r), r);
}

// "fisheye2xyz - division by zero at center"
// Note: MSL fisheye2xyz returns (0, 1, 0) at the center like the WESL fix; GLSL divides by 0.
// With a literal uv the Metal compiler constant-folds the 0/0 away and returns (0, 1, 0), so the
// input goes through `zero` (results[0].x, 0 at runtime) to get what a shader would compute.
// @test fisheye2xyzCenter
// @expect fisheye2xyzCenter[0] 0.0 1.0 0.0
kernel void fisheye2xyzCenter(device float4* results [[buffer(0)]]) {
    float zero = results[0].x;
    results[0] = float4(fisheye2xyz(float2(0.5 + zero, 0.5)), 0.0);
}

// WGSL result[0..3] = nearest(...).xy, nearest(...).xy
// @test nearest
// @expect nearest[0] 0.7534 0.2569 0.75026 0.25046
kernel void nearest(device float4* results [[buffer(0)]]) {
    float2 r1 = nearest(float2(0.7533, 0.2567), float2(1920.0, 1080.0));
    float2 r2 = nearest(float2(0.7500, 0.2500), float2(1920.0, 1080.0));
    results[0] = float4(r1, r2);
}
