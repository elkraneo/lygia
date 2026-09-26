// Ported from test/wesl/color-space-specialty.test.ts (same inputs and expected values).
// WGSL names encode the argument type; in MSL they're overloads:
//   WGSL xxx(vec3f) / xxx4(vec4f)  ==  MSL xxx(float3) / xxx(float4)
// vec3f results are written as float4(result, 0) and only x y z are checked;
// f32 results go to .x.
// rgb2ryb / ryb2rgb: both WESL and MSL default to the homogeneous (cubic) version
// (no RYB_FAST), so no #defines are needed.
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/space/rgb2heat.msl"
#include "lygia/color/space/cmyk2rgb.msl"
#include "lygia/color/space/rgb2cmyk.msl"
#include "lygia/color/space/k2rgb.msl"
#include "lygia/color/space/rgb2lms.msl"
#include "lygia/color/space/lms2rgb.msl"
#include "lygia/color/space/rgb2ryb.msl"
#include "lygia/color/space/ryb2rgb.msl"

// @test rgb2heat_test
// @expect rgb2heat_test[0] 0.854
kernel void rgb2heat_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(rgb2heat(float3(0.8, 0.7, 0.5)), 0.0, 0.0, 0.0);
}

// @test cmyk2rgb_test
// @expect cmyk2rgb_test[0] 0.5 0.5 0.5
kernel void cmyk2rgb_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(cmyk2rgb(float4(0.0, 0.0, 0.0, 0.5)), 0.0);
}

// @test rgb2cmyk_test
// @expect rgb2cmyk_test[0] 0.0 1.0 1.0 0.0
kernel void rgb2cmyk_test(device float4* results [[buffer(0)]]) {
    results[0] = rgb2cmyk(float3(1.0, 0.0, 0.0));
}

// WGSL "k2rgb - color temperature gradient": relational expects, computed in-kernel as flags.
//   results[0].x = warm.r > cool.r   (warm has more red than cool)
//   results[0].y = cool.b > warm.b   (cool has more blue than warm)
// @test k2rgb_test
// @expect k2rgb_test[0] 1 1
kernel void k2rgb_test(device float4* results [[buffer(0)]]) {
    float3 warm = k2rgb(3000.0);
    float3 cool = k2rgb(8000.0);
    results[0] = float4(warm.r > cool.r ? 1.0 : 0.0, cool.b > warm.b ? 1.0 : 0.0, 0.0, 0.0);
}

// @test rgb2lms_test
// @expect rgb2lms_test[0] 17.8824 3.45565 0.02996
kernel void rgb2lms_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(rgb2lms(float3(1.0, 0.0, 0.0)), 0.0);
}

// @test lms2rgb_test
// @expect lms2rgb_test[0] 0.00985 -0.00363 0.06842
kernel void lms2rgb_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(lms2rgb(float3(0.3, 0.2, 0.1)), 0.0);
}

// WGSL lms2rgb4
// @test lms2rgb4_test
// @expect lms2rgb4_test[0] 0.00985 -0.00363 0.06842 0.55
kernel void lms2rgb4_test(device float4* results [[buffer(0)]]) {
    results[0] = lms2rgb(float4(0.3, 0.2, 0.1, 0.55));
}

// WGSL rgb2lms4
// Note: MSL rgb2lms(float4) converts like the WESL fix; GLSL returns the input.
// @test rgb2lms4_test
// @expect rgb2lms4_test[0] 17.8824 3.45565 0.02996 0.3
kernel void rgb2lms4_test(device float4* results [[buffer(0)]]) {
    results[0] = rgb2lms(float4(1.0, 0.0, 0.0, 0.3));
}

// WGSL "rgb2ryb - default mode"
// @test rgb2ryb_test
// @expect rgb2ryb_test[0] 1.0 0.0 0.0
kernel void rgb2ryb_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(rgb2ryb(float3(1.0, 0.0, 0.0)), 0.0);
}

// WGSL rgb2ryb4
// @test rgb2ryb4_test
// @expect rgb2ryb4_test[0] 0.0 1.0 0.483 0.5
kernel void rgb2ryb4_test(device float4* results [[buffer(0)]]) {
    results[0] = rgb2ryb(float4(0.0, 1.0, 0.0, 0.5));
}

// WGSL "ryb2rgb - default mode"
// @test ryb2rgb_test
// @expect ryb2rgb_test[0] 1.0 0.0 0.0
kernel void ryb2rgb_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(ryb2rgb(float3(1.0, 0.0, 0.0)), 0.0);
}

// WGSL ryb2rgb4
// @test ryb2rgb4_test
// @expect ryb2rgb4_test[0] 1.0 1.0 0.0 0.75
kernel void ryb2rgb4_test(device float4* results [[buffer(0)]]) {
    results[0] = ryb2rgb(float4(0.0, 1.0, 0.0, 0.75));
}

// WGSL rgb2heat4 returns vec4f(heat, heat, heat, a); MSL (like GLSL) rgb2heat(float4) returns
// only the float heat, so the vec4 is built here from it (API difference, not a value mismatch).
// @test rgb2heat4_test
// @expect rgb2heat4_test[0] 0.854 0.854 0.854 0.6
kernel void rgb2heat4_test(device float4* results [[buffer(0)]]) {
    float4 rgb = float4(0.8, 0.7, 0.5, 0.6);
    float h = rgb2heat(rgb);
    results[0] = float4(h, h, h, rgb.a);
}
