// Ported from test/wesl/color-space-video.test.ts (same inputs and expected values).
// SDTV variants of the rgb2YPbPr / rgb2yuv / yuv2rgb tests, which the WGSL test runs with
// conditions { YPBPR_SDTV: true } and { YUV_SDTV: true }. They need different #defines
// than color_space_video.metal. The two flags select independent matrices (YPBPR_SDTV
// only affects RGB2PBPR/YPBPR2RGB, YUV_SDTV only RGB2YUV/YUV2RGB), so both fit here.
#include <metal_stdlib>
using namespace metal;
#define YPBPR_SDTV
#define YUV_SDTV
#include "lygia/color/space/rgb2YPbPr.msl"
#include "lygia/color/space/rgb2yuv.msl"
#include "lygia/color/space/yuv2rgb.msl"

// @test rgb2YPbPr_sdtv
// @expect rgb2YPbPr_sdtv[0] 0.6473 -0.0831 -0.0338
kernel void rgb2YPbPr_sdtv(device float4* results [[buffer(0)]]) {
    results[0] = float4(rgb2YPbPr(float3(0.6, 0.7, 0.5)), 0.0);
}

// @test rgb2yuv_sdtv
// @expect rgb2yuv_sdtv[0] 0.6473 -0.0725 -0.0415
kernel void rgb2yuv_sdtv(device float4* results [[buffer(0)]]) {
    results[0] = float4(rgb2yuv(float3(0.6, 0.7, 0.5)), 0.0);
}

// @test yuv2rgb_sdtv
// @expect yuv2rgb_sdtv[0] 1.1699 0.0334 2.0225
kernel void yuv2rgb_sdtv(device float4* results [[buffer(0)]]) {
    results[0] = float4(yuv2rgb(float3(0.6, 0.7, 0.5)), 0.0);
}
