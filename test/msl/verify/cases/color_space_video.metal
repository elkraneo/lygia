// Ported from test/wesl/color-space-video.test.ts (same inputs and expected values).
// WGSL names encode the argument type; in MSL they're overloads:
//   WGSL xxx(vec3f) / xxx4(vec4f)  ==  MSL xxx(float3) / xxx(float4)
// vec3f results are written as float4(result, 0) and only x y z are checked.
// Default (HDTV) configuration here; the YPBPR_SDTV / YUV_SDTV variants of the
// rgb2YPbPr, rgb2yuv and yuv2rgb tests are in color_space_video_sdtv.metal.
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/space/rgb2YPbPr.msl"
#include "lygia/color/space/rgb2yuv.msl"
#include "lygia/color/space/yuv2rgb.msl"
#include "lygia/color/space/YCbCr2rgb.msl"
#include "lygia/color/space/YPbPr2rgb.msl"
#include "lygia/color/space/rgb2YCbCr.msl"
#include "lygia/color/space/yiq2rgb.msl"
#include "lygia/color/space/rgb2yiq.msl"

// @test rgb2YPbPr_test
// @expect rgb2YPbPr_test[0] 0.6643 -0.0885 -0.0408
kernel void rgb2YPbPr_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(rgb2YPbPr(float3(0.6, 0.7, 0.5)), 0.0);
}

// @test rgb2yuv_test
// @expect rgb2yuv_test[0] 0.6643 -0.0822 -0.0502
kernel void rgb2yuv_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(rgb2yuv(float3(0.6, 0.7, 0.5)), 0.0);
}

// @test yuv2rgb_test
// @expect yuv2rgb_test[0] 1.2402 0.2593 2.0896
kernel void yuv2rgb_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(yuv2rgb(float3(0.6, 0.7, 0.5)), 0.0);
}

// @test YCbCr2rgb_test
// @expect YCbCr2rgb_test[0] 0.5 0.5 0.5
kernel void YCbCr2rgb_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(YCbCr2rgb(float3(0.5, 0.5, 0.5)), 0.0);
}

// @test YPbPr2rgb_test
// @expect YPbPr2rgb_test[0] 0.5 0.5 0.5
kernel void YPbPr2rgb_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(YPbPr2rgb(float3(0.5, 0.0, 0.0)), 0.0);
}

// @test rgb2YCbCr_test
// @expect rgb2YCbCr_test[0] 0.5 0.5 0.5
kernel void rgb2YCbCr_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(rgb2YCbCr(float3(0.5, 0.5, 0.5)), 0.0);
}

// @test yiq2rgb_test
// @expect yiq2rgb_test[0] 0.5 0.4735 0.3117
kernel void yiq2rgb_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(yiq2rgb(float3(0.5, 0.0, 0.0)), 0.0);
}

// @test rgb2yiq_test
// @expect rgb2yiq_test[0] 0.3 0.59 0.11
kernel void rgb2yiq_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(rgb2yiq(float3(1.0, 0.0, 0.0)), 0.0);
}

// WGSL YCbCr2rgb4
// @test YCbCr2rgb4_test
// @expect YCbCr2rgb4_test[0] 0.5 0.5 0.5 0.7
kernel void YCbCr2rgb4_test(device float4* results [[buffer(0)]]) {
    results[0] = YCbCr2rgb(float4(0.5, 0.5, 0.5, 0.7));
}

// WGSL YPbPr2rgb4
// @test YPbPr2rgb4_test
// @expect YPbPr2rgb4_test[0] 0.5 0.5 0.5 0.8
kernel void YPbPr2rgb4_test(device float4* results [[buffer(0)]]) {
    results[0] = YPbPr2rgb(float4(0.5, 0.0, 0.0, 0.8));
}

// WGSL rgb2YCbCr4
// @test rgb2YCbCr4_test
// @expect rgb2YCbCr4_test[0] 0.5 0.5 0.5 0.25
kernel void rgb2YCbCr4_test(device float4* results [[buffer(0)]]) {
    results[0] = rgb2YCbCr(float4(0.5, 0.5, 0.5, 0.25));
}

// WGSL rgb2YPbPr4
// @test rgb2YPbPr4_test
// @expect rgb2YPbPr4_test[0] 0.6643 -0.0885 -0.0408 0.15
kernel void rgb2YPbPr4_test(device float4* results [[buffer(0)]]) {
    results[0] = rgb2YPbPr(float4(0.6, 0.7, 0.5, 0.15));
}

// WGSL rgb2yiq4
// @test rgb2yiq4_test
// @expect rgb2yiq4_test[0] 0.3 0.59 0.11 0.8
kernel void rgb2yiq4_test(device float4* results [[buffer(0)]]) {
    results[0] = rgb2yiq(float4(1.0, 0.0, 0.0, 0.8));
}

// WGSL rgb2yuv4
// @test rgb2yuv4_test
// @expect rgb2yuv4_test[0] 0.6643 -0.0822 -0.0502 0.3
kernel void rgb2yuv4_test(device float4* results [[buffer(0)]]) {
    results[0] = rgb2yuv(float4(0.6, 0.7, 0.5, 0.3));
}

// WGSL yiq2rgb4
// @test yiq2rgb4_test
// @expect yiq2rgb4_test[0] 0.5 0.4735 0.3117 0.55
kernel void yiq2rgb4_test(device float4* results [[buffer(0)]]) {
    results[0] = yiq2rgb(float4(0.5, 0.0, 0.0, 0.55));
}

// WGSL yuv2rgb4
// @test yuv2rgb4_test
// @expect yuv2rgb4_test[0] 1.2402 0.2593 2.0896 0.95
kernel void yuv2rgb4_test(device float4* results [[buffer(0)]]) {
    results[0] = yuv2rgb(float4(0.6, 0.7, 0.5, 0.95));
}
