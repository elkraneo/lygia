// Ported from test/wesl/color-space-xyz.test.ts (same inputs and expected values).
// The tests that pass epsilon 0.01 to expectCloseTo (default-tolerance tests are in
// color_space_xyz.metal). WGSL xxx4(vec4f) == MSL xxx(float4) overload.
// vec3f results are written as float4(result, 0) and only x y z are checked.
// @eps 0.01
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/space/xyY2xyz.msl"
#include "lygia/color/space/rgb2xyY.msl"
#include "lygia/color/space/srgb2xyz.msl"

// @test xyY2xyz_test
// @expect xyY2xyz_test[0] 0.9505 1.0 1.089
kernel void xyY2xyz_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(xyY2xyz(float3(0.3127, 0.3290, 1.0)), 0.0);
}

// WGSL rgb2xyY4
// Note: MSL rgb2xyz returns XYZ in 0-100 like the WESL fix (#271); GLSL still returns 0-1.
// @test rgb2xyY4_test
// @expect rgb2xyY4_test[0] 0.64 0.33 21.26 0.4
kernel void rgb2xyY4_test(device float4* results [[buffer(0)]]) {
    results[0] = rgb2xyY(float4(1.0, 0.0, 0.0, 0.4));
}

// WGSL srgb2xyz4
// @test srgb2xyz4_test
// @expect srgb2xyz4_test[0] 41.24 21.26 1.93 0.65
kernel void srgb2xyz4_test(device float4* results [[buffer(0)]]) {
    results[0] = srgb2xyz(float4(1.0, 0.0, 0.0, 0.65));
}

// WGSL xyY2xyz4
// @test xyY2xyz4_test
// @expect xyY2xyz4_test[0] 0.9505 1.0 1.089 0.4
kernel void xyY2xyz4_test(device float4* results [[buffer(0)]]) {
    results[0] = xyY2xyz(float4(0.3127, 0.3290, 1.0, 0.4));
}
