// Ported from test/wesl/color-space-xyz.test.ts (same inputs and expected values).
// WGSL names encode the argument type; in MSL they're overloads:
//   WGSL xxx(vec3f) / xxx4(vec4f)  ==  MSL xxx(float3) / xxx(float4)
// vec3f results are written as float4(result, 0) and only x y z are checked.
// This file holds the tests with the default tolerance (0.0001). Others:
//   color_space_xyz_eps.metal         tests using expectCloseTo(..., 0.001)
//   color_space_xyz_eps_coarse.metal  tests using expectCloseTo(..., 0.01)
//   color_space_xyz_d50.metal         rgb2xyz with conditions { CIE_D50: true }
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/space/rgb2xyz.msl"
#include "lygia/color/space/srgb2xyz.msl"
#include "lygia/color/space/rgb2xyY.msl"
#include "lygia/color/space/xyz2xyY.msl"
#include "lygia/color/space/xyz2rgb.msl"

// Note: MSL rgb2xyz returns XYZ in 0-100 like the WESL fix (#271); GLSL still returns 0-1.
// @test rgb2xyz_test
// @expect rgb2xyz_test[0] 67.0487 70.6832 57.4054
kernel void rgb2xyz_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(rgb2xyz(float3(0.8, 0.7, 0.5)), 0.0);
}

// @test srgb2xyz_test
// @expect srgb2xyz_test[0] 41.2456 21.2673 1.9334
kernel void srgb2xyz_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(srgb2xyz(float3(1.0, 0.0, 0.0)), 0.0);
}

// @test rgb2xyY_test
// @expect rgb2xyY_test[0] 0.64 0.33 21.2673
kernel void rgb2xyY_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(rgb2xyY(float3(1.0, 0.0, 0.0)), 0.0);
}

// @test xyz2xyY_test
// @expect xyz2xyY_test[0] 0.3127 0.329 1.0
kernel void xyz2xyY_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(xyz2xyY(float3(0.9505, 1.0, 1.089)), 0.0);
}

// @test xyz2rgb_test
// @expect xyz2rgb_test[0] 1.0 0.0 0.0
kernel void xyz2rgb_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(xyz2rgb(float3(41.24, 21.26, 1.93)), 0.0);
}

// WGSL rgb2xyz4
// @test rgb2xyz4_test
// @expect rgb2xyz4_test[0] 67.0487 70.6832 57.4054 0.2
kernel void rgb2xyz4_test(device float4* results [[buffer(0)]]) {
    results[0] = rgb2xyz(float4(0.8, 0.7, 0.5, 0.2));
}

// WGSL xyz2rgb4
// @test xyz2rgb4_test
// @expect xyz2rgb4_test[0] 1.0 0.0 0.0 0.6
kernel void xyz2rgb4_test(device float4* results [[buffer(0)]]) {
    results[0] = xyz2rgb(float4(41.24, 21.26, 1.93, 0.6));
}

// WGSL xyz2xyY4
// @test xyz2xyY4_test
// @expect xyz2xyY4_test[0] 0.3127 0.329 1.0 0.75
kernel void xyz2xyY4_test(device float4* results [[buffer(0)]]) {
    results[0] = xyz2xyY(float4(0.9505, 1.0, 1.089, 0.75));
}
