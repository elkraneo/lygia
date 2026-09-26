// Ported from test/wesl/shaders/color_space_roundtrip.test.wesl (same inputs and expected values).
// The WESL test uses expectNearVec4 (relative 1e-3, absolute 1e-6); here the default
// absolute 0.0001 is used, which is stricter for these values (all in [0.2, 0.9]).
// WGSL xxx4(vec4f) == MSL xxx(float4). YUV_SDTV is off in both (HDTV matrices).
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/space/rgb2hsl.msl"
#include "lygia/color/space/hsl2rgb.msl"
#include "lygia/color/space/rgb2hsv.msl"
#include "lygia/color/space/hsv2rgb.msl"
#include "lygia/color/space/rgb2oklab.msl"
#include "lygia/color/space/oklab2rgb.msl"
#include "lygia/color/space/rgb2yiq.msl"
#include "lygia/color/space/yiq2rgb.msl"
#include "lygia/color/space/rgb2yuv.msl"
#include "lygia/color/space/yuv2rgb.msl"

// @test rgb_hsl_roundtrip
// @expect rgb_hsl_roundtrip[0] 0.7 0.3 0.5 0.8
kernel void rgb_hsl_roundtrip(device float4* results [[buffer(0)]]) { results[0] = hsl2rgb(rgb2hsl(float4(0.7, 0.3, 0.5, 0.8))); }

// @test rgb_hsv_roundtrip
// @expect rgb_hsv_roundtrip[0] 0.8 0.2 0.6 0.5
kernel void rgb_hsv_roundtrip(device float4* results [[buffer(0)]]) { results[0] = hsv2rgb(rgb2hsv(float4(0.8, 0.2, 0.6, 0.5))); }

// @test rgb_oklab_roundtrip
// @expect rgb_oklab_roundtrip[0] 0.6 0.4 0.2 0.9
kernel void rgb_oklab_roundtrip(device float4* results [[buffer(0)]]) { results[0] = oklab2rgb(rgb2oklab(float4(0.6, 0.4, 0.2, 0.9))); }

// @test rgb_yiq_roundtrip
// @expect rgb_yiq_roundtrip[0] 0.7 0.40003 0.19989 0.6
kernel void rgb_yiq_roundtrip(device float4* results [[buffer(0)]]) { results[0] = yiq2rgb(rgb2yiq(float4(0.7, 0.4, 0.2, 0.6))); }

// @test rgb_yuv_roundtrip
// @expect rgb_yuv_roundtrip[0] 0.5 0.60064 0.29362 0.8
kernel void rgb_yuv_roundtrip(device float4* results [[buffer(0)]]) { results[0] = yuv2rgb(rgb2yuv(float4(0.5, 0.6, 0.3, 0.8))); }
