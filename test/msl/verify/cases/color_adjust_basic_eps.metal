// Ported from test/wesl/color-adjust-basic.test.ts (same inputs and expected values).
// Only the hueShift test, which the WGSL test checks with tolerance 0.001 (the rest of the
// file is in color_adjust_basic.metal with the default 0.0001).
// WGSL hueShift(vec3f, f32) == MSL hueShift(float3, float); both default to the angle mode
// (HUESHIFT_AMOUNT not defined). WGSL lygia::math::consts::TAU == MSL TAU (math/const.msl).
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/hueShift.msl"

// @eps 0.001

// @test hueShift_test
// @expect hueShift_test[0] 0.0 1.0 0.0
kernel void hueShift_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(hueShift(float3(1.0, 0.0, 0.0), TAU * 0.3333), 0.0);
}
