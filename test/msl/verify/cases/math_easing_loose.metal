// Ported from test/wesl/math-easing.test.ts: the checks that use epsilon 0.01
// (the rest are in math_easing.metal with the default 0.0001).
// @eps 0.01
#include <metal_stdlib>
using namespace metal;
#include "lygia/math/cubic.msl"
#include "lygia/math/quartic.msl"
#include "lygia/math/invCubic.msl"
#include "lygia/math/invQuartic.msl"

// @test invCubic
// @expect invCubic[0] 0.3 0.3
kernel void invCubic(device float4* results [[buffer(0)]]) {
    float x = 0.3;
    results[0] = float4(x, invCubic(cubic(x)), 0.0, 0.0);
}

// @test invQuartic
// @expect invQuartic[0] 0.7 0.7
kernel void invQuartic(device float4* results [[buffer(0)]]) {
    float x = 0.7;
    results[0] = float4(x, invQuartic(quartic(x)), 0.0, 0.0);
}
