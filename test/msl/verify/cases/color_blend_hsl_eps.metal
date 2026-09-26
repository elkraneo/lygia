// Ported from test/wesl/color-blend-hsl.test.ts (same inputs and expected values).
// Only blendSaturation and blendLuminosity, which the WGSL test checks with tolerance 0.05
// (the rest are in color_blend_hsl.metal).
// WGSL blendSaturation/blendLuminosity(vec3f, vec3f) == MSL overloads with float3; both
// the WESL and MSL versions go through rgb2hsv/hsv2rgb.
// Relational expects are in-kernel flags (1.0 = true):
//   toBeCloseTo(a, b, 1) is |a - b| < 0.05; toBeGreaterThan / toBeLessThan as > / <.
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/blend/saturation.msl"
#include "lygia/color/blend/luminosity.msl"

// @eps 0.05

// [0] = result; [1] = flags (x: |r - g| < 0.05, y: |g - b| < 0.05)
// @test blendSaturation_test 2
// @expect blendSaturation_test[0] 1.0 1.0 1.0
// @expect blendSaturation_test[1] 1 1
kernel void blendSaturation_test(device float4* results [[buffer(0)]]) {
    float3 r = blendSaturation(float3(1.0, 0.0, 0.0), float3(0.5, 0.5, 0.5));
    results[0] = float4(r, 0.0);
    results[1] = float4(abs(r.x - r.y) < 0.05 ? 1.0 : 0.0, abs(r.y - r.z) < 0.05 ? 1.0 : 0.0, 0.0, 0.0);
}

// [0] = result; [1] = flags (x: r > g, y: r > b, z: r < 0.5)
// @test blendLuminosity_test 2
// @expect blendLuminosity_test[0] 0.1 0.0 0.0
// @expect blendLuminosity_test[1] 1 1 1
kernel void blendLuminosity_test(device float4* results [[buffer(0)]]) {
    float3 r = blendLuminosity(float3(1.0, 0.0, 0.0), float3(0.1, 0.1, 0.1));
    results[0] = float4(r, 0.0);
    results[1] = float4(r.x > r.y ? 1.0 : 0.0, r.x > r.z ? 1.0 : 0.0, r.x < 0.5 ? 1.0 : 0.0, 0.0);
}
