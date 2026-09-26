// Ported from test/wesl/color-space-xyz.test.ts (same inputs and expected values).
// The tests that pass epsilon 0.001 to expectCloseTo (default-tolerance tests are in
// color_space_xyz.metal). WGSL xxx4(vec4f) == MSL xxx(float4) overload.
// vec3f results are written as float4(result, 0) and only x y z are checked.
// @eps 0.001
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/space/xyY2rgb.msl"
#include "lygia/color/space/xyY2srgb.msl"
#include "lygia/color/space/xyz2srgb.msl"

// @test xyY2rgb_test
// @expect xyY2rgb_test[0] 1.0 0.0 0.0
kernel void xyY2rgb_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(xyY2rgb(float3(0.64, 0.33, 21.26)), 0.0);
}

// @test xyY2srgb_test
// @expect xyY2srgb_test[0] 1.0 0.0 0.0
kernel void xyY2srgb_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(xyY2srgb(float3(0.64, 0.33, 21.26)), 0.0);
}

// @test xyz2srgb_test
// @expect xyz2srgb_test[0] 1.0 0.0 0.0
kernel void xyz2srgb_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(xyz2srgb(float3(41.24, 21.26, 1.93)), 0.0);
}

// WGSL xyY2rgb4
// @test xyY2rgb4_test
// @expect xyY2rgb4_test[0] 1.0 0.0 0.0 0.5
kernel void xyY2rgb4_test(device float4* results [[buffer(0)]]) {
    results[0] = xyY2rgb(float4(0.64, 0.33, 21.26, 0.5));
}

// WGSL xyY2srgb4
// @test xyY2srgb4_test
// @expect xyY2srgb4_test[0] 1.0 0.0 0.0 0.85
kernel void xyY2srgb4_test(device float4* results [[buffer(0)]]) {
    results[0] = xyY2srgb(float4(0.64, 0.33, 21.26, 0.85));
}

// WGSL xyz2srgb4
// @test xyz2srgb4_test
// @expect xyz2srgb4_test[0] 1.0 0.0 0.0 0.2
kernel void xyz2srgb4_test(device float4* results [[buffer(0)]]) {
    results[0] = xyz2srgb(float4(41.24, 21.26, 1.93, 0.2));
}
