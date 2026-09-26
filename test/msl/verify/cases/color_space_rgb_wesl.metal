// Ported from test/wesl/shaders/color_space_rgb.test.wesl (same inputs and expected values).
// The WESL test uses expectNear/expectNearVec3/Vec4 (relative 1e-3, absolute 1e-6); here the
// default absolute 0.0001 is used, which is stricter for these values (all <= 1).
// Name mapping (WGSL can't overload):
//   WGSL xxx4(vec4f)                  == MSL xxx(float4) (hsl2rgb, rgb2hsl, hsv2rgb, rgb2hsv, hcy2rgb, rgb2hcy, rgb2hcv, rgb2srgb, srgb2rgb)
//   WGSL rgb2srgb_mono / srgb2rgb_mono == MSL rgb2srgb(float) / srgb2rgb(float)
//   WGSL rgb2hue4, rgb2luma4, srgb2luma4 return vec4f(v, v, v, a); MSL rgb2hue/rgb2luma/srgb2luma(float4)
//   return the float v, so the kernels build float4(v, v, v, a) from it.
// Config: HSV2RYB_FAST is not set in the WESL test, and RYB_FAST is off in both, so hsv2ryb uses
// the default ryb2rgb(hsv2rgb(v)) - saturate(1 - v.z) path in both.
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/space/hsl2rgb.msl"
#include "lygia/color/space/rgb2hsl.msl"
#include "lygia/color/space/hsv2rgb.msl"
#include "lygia/color/space/rgb2hsv.msl"
#include "lygia/color/space/hcy2rgb.msl"
#include "lygia/color/space/rgb2hcy.msl"
#include "lygia/color/space/hsv2ryb.msl"
#include "lygia/color/space/rgb2luma.msl"
#include "lygia/color/space/srgb2luma.msl"
#include "lygia/color/space/rgb2hcv.msl"
#include "lygia/color/space/rgb2hue.msl"
#include "lygia/color/space/hue2rgb.msl"
#include "lygia/color/space/rgb2srgb.msl"
#include "lygia/color/space/srgb2rgb.msl"

// @test hsl2rgbCyan
// @expect hsl2rgbCyan[0] 0.1 0.9 0.9
kernel void hsl2rgbCyan(device float4* results [[buffer(0)]]) { results[0] = float4(hsl2rgb(float3(0.5, 0.8, 0.5)), 0.0); }

// @test rgb2hslCyan
// @expect rgb2hslCyan[0] 0.5 0.8 0.5
kernel void rgb2hslCyan(device float4* results [[buffer(0)]]) { results[0] = float4(rgb2hsl(float3(0.1, 0.9, 0.9)), 0.0); }

// @test hsv2rgbBlue
// @expect hsv2rgbBlue[0] 0.0002 0.0 1.0
kernel void hsv2rgbBlue(device float4* results [[buffer(0)]]) { results[0] = float4(hsv2rgb(float3(0.6667, 1.0, 1.0)), 0.0); }

// @test rgb2hsvBlue
// @expect rgb2hsvBlue[0] 0.6667 1.0 1.0
kernel void rgb2hsvBlue(device float4* results [[buffer(0)]]) { results[0] = float4(rgb2hsv(float3(0.0, 0.0, 1.0)), 0.0); }

// @test hcy2rgbRed
// @expect hcy2rgbRed[0] 0.75 0.3934 0.3934
kernel void hcy2rgbRed(device float4* results [[buffer(0)]]) { results[0] = float4(hcy2rgb(float3(0.0, 0.5, 0.5)), 0.0); }

// @test rgb2hcyRed
// @expect rgb2hcyRed[0] 0.0 1.0 0.2989
kernel void rgb2hcyRed(device float4* results [[buffer(0)]]) { results[0] = float4(rgb2hcy(float3(1.0, 0.0, 0.0)), 0.0); }

// @test hsv2rybRed
// @expect hsv2rybRed[0] 1.0 0.0 0.0
kernel void hsv2rybRed(device float4* results [[buffer(0)]]) { results[0] = float4(hsv2ryb(float3(0.0, 1.0, 1.0)), 0.0); }

// @test rgb2lumaOrange
// @expect rgb2lumaOrange[0] 0.5702
kernel void rgb2lumaOrange(device float4* results [[buffer(0)]]) { results[0] = float4(rgb2luma(float3(1.0, 0.5, 0.0)), 0.0, 0.0, 0.0); }

// @test srgb2lumaOrange
// @expect srgb2lumaOrange[0] 0.5925
kernel void srgb2lumaOrange(device float4* results [[buffer(0)]]) { results[0] = float4(srgb2luma(float3(1.0, 0.5, 0.0)), 0.0, 0.0, 0.0); }

// @test rgb2hcvRed
// @expect rgb2hcvRed[0] 0.0 1.0 1.0
kernel void rgb2hcvRed(device float4* results [[buffer(0)]]) { results[0] = float4(rgb2hcv(float3(1.0, 0.0, 0.0)), 0.0); }

// @test rgb2hueGreen
// @expect rgb2hueGreen[0] 0.3333
kernel void rgb2hueGreen(device float4* results [[buffer(0)]]) { results[0] = float4(rgb2hue(float3(0.0, 1.0, 0.0)), 0.0, 0.0, 0.0); }

// @test hue2rgbGreen
// @expect hue2rgbGreen[0] 0.0002 1.0 0.0
kernel void hue2rgbGreen(device float4* results [[buffer(0)]]) { results[0] = float4(hue2rgb(0.3333), 0.0); }

// ---- vec4 overloads with alpha preservation ----

// @test hcy2rgb4Alpha
// @expect hcy2rgb4Alpha[0] 0.75 0.3934 0.3934 0.6
kernel void hcy2rgb4Alpha(device float4* results [[buffer(0)]]) { results[0] = hcy2rgb(float4(0.0, 0.5, 0.5, 0.6)); }

// @test hsl2rgb4Alpha
// @expect hsl2rgb4Alpha[0] 0.1 0.9 0.9 0.9
kernel void hsl2rgb4Alpha(device float4* results [[buffer(0)]]) { results[0] = hsl2rgb(float4(0.5, 0.8, 0.5, 0.9)); }

// @test hsv2rgb4Alpha
// @expect hsv2rgb4Alpha[0] 0.0002 0.0 1.0 0.5
kernel void hsv2rgb4Alpha(device float4* results [[buffer(0)]]) { results[0] = hsv2rgb(float4(0.6667, 1.0, 1.0, 0.5)); }

// @test rgb2hcy4Alpha
// @expect rgb2hcy4Alpha[0] 0.0 1.0 0.2989 0.1
kernel void rgb2hcy4Alpha(device float4* results [[buffer(0)]]) { results[0] = rgb2hcy(float4(1.0, 0.0, 0.0, 0.1)); }

// @test rgb2hsl4Alpha
// @expect rgb2hsl4Alpha[0] 0.5 0.8 0.5 0.5
kernel void rgb2hsl4Alpha(device float4* results [[buffer(0)]]) { results[0] = rgb2hsl(float4(0.1, 0.9, 0.9, 0.5)); }

// @test rgb2hsv4Alpha
// @expect rgb2hsv4Alpha[0] 0.6667 1.0 1.0 0.9
kernel void rgb2hsv4Alpha(device float4* results [[buffer(0)]]) { results[0] = rgb2hsv(float4(0.0, 0.0, 1.0, 0.9)); }

// @test rgb2hcv4Alpha
// @expect rgb2hcv4Alpha[0] 0.0833 1.0 1.0 0.6
kernel void rgb2hcv4Alpha(device float4* results [[buffer(0)]]) { results[0] = rgb2hcv(float4(1.0, 0.5, 0.0, 0.6)); }

// WGSL rgb2hue4 returns vec4f(h, h, h, a); MSL rgb2hue(float4) returns h.
// @test rgb2hue4Alpha
// @expect rgb2hue4Alpha[0] 0.3333 0.3333 0.3333 0.75
kernel void rgb2hue4Alpha(device float4* results [[buffer(0)]]) {
    float4 c = float4(0.0, 1.0, 0.0, 0.75);
    float h = rgb2hue(c);
    results[0] = float4(h, h, h, c.a);
}

// WGSL rgb2luma4 returns vec4f(l, l, l, a); MSL rgb2luma(float4) returns l.
// @test rgb2luma4Alpha
// @expect rgb2luma4Alpha[0] 0.5702 0.5702 0.5702 0.85
kernel void rgb2luma4Alpha(device float4* results [[buffer(0)]]) {
    float4 c = float4(1.0, 0.5, 0.0, 0.85);
    float l = rgb2luma(c);
    results[0] = float4(l, l, l, c.a);
}

// WGSL srgb2luma4 returns vec4f(l, l, l, a); MSL srgb2luma(float4) returns l.
// @test srgb2luma4Alpha
// @expect srgb2luma4Alpha[0] 0.5925 0.5925 0.5925 0.95
kernel void srgb2luma4Alpha(device float4* results [[buffer(0)]]) {
    float4 c = float4(1.0, 0.5, 0.0, 0.95);
    float l = srgb2luma(c);
    results[0] = float4(l, l, l, c.a);
}

// ---- RGB <-> sRGB ----

// WGSL rgb2srgb_mono(0.002) (linear branch), rgb2srgb_mono(0.5) (pow branch)
// @test rgb2srgbMono
// @expect rgb2srgbMono[0] 0.02584 0.73536 0.0 0.0
kernel void rgb2srgbMono(device float4* results [[buffer(0)]]) { results[0] = float4(rgb2srgb(0.002), rgb2srgb(0.5), 0.0, 0.0); }

// WGSL srgb2rgb_mono(0.03) (linear branch), srgb2rgb_mono(0.735) (pow branch)
// @test srgb2rgbMono
// @expect srgb2rgbMono[0] 0.00232 0.49946 0.0 0.0
kernel void srgb2rgbMono(device float4* results [[buffer(0)]]) { results[0] = float4(srgb2rgb(0.03), srgb2rgb(0.735), 0.0, 0.0); }

// @test rgb2srgbVec3
// @expect rgb2srgbVec3[0] 0.7354 0.5838 0.3492
kernel void rgb2srgbVec3(device float4* results [[buffer(0)]]) { results[0] = float4(rgb2srgb(float3(0.5, 0.3, 0.1)), 0.0); }

// @test srgb2rgbVec3
// @expect srgb2rgbVec3[0] 0.4995 0.3002 0.0999
kernel void srgb2rgbVec3(device float4* results [[buffer(0)]]) { results[0] = float4(srgb2rgb(float3(0.735, 0.584, 0.349)), 0.0); }

// @test rgb2srgb4Alpha
// @expect rgb2srgb4Alpha[0] 0.7354 0.5838 0.3492 0.4
kernel void rgb2srgb4Alpha(device float4* results [[buffer(0)]]) { results[0] = rgb2srgb(float4(0.5, 0.3, 0.1, 0.4)); }

// @test srgb2rgb4Alpha
// @expect srgb2rgb4Alpha[0] 0.49946 0.30019 0.09989 0.3
kernel void srgb2rgb4Alpha(device float4* results [[buffer(0)]]) { results[0] = srgb2rgb(float4(0.735, 0.584, 0.349, 0.3)); }
