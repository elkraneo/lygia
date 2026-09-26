// Ported from test/wesl/color-dither.test.ts (same inputs and expected values).
// Name mapping (WGSL names encode the type; MSL uses overloads):
//   ditherBayerPrecision(v, xy, p)      -> ditherBayer(float3(v), xy, p).r (MSL has no scalar+precision overload;
//                                          WGSL's is defined as exactly this)
//   ditherBayer3(c, xy) / ditherBayer3Precision(c, xy, p) / ditherBayer4(c, xy) -> ditherBayer(c, xy[, p])
//   ditherVlachos4(c, xy)               -> ditherVlachos(float4, xy)
//   ditherBlueNoise1/3/3Precision/4     -> ditherBlueNoise(float|float3|float4, xy[, p])
// Default precision is 256 in both WESL and MSL (DITHER_*_PRECISION); BLUENOISE_TEXTURE is not
// defined, so MSL uses the procedural blue noise like WESL.
// The Vlachos checks with tolerances 0.004 / 0.07 are in color_dither_eps.metal / color_dither_eps07.metal.
//
// Skipped (need fragment rendering and image snapshots, expectDither):
//   ditherBayer3 - gradient with Bayer pattern (image snapshot)
//   ditherVlachos3 - gradient with Vlachos noise (image snapshot)
//   ditherBlueNoise3 - gradient with blue noise pattern (image snapshot)
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/dither/bayer.msl"
#include "lygia/color/dither/vlachos.msl"
#include "lygia/color/dither/blueNoise.msl"

// ditherBayer - all wrapper functions (precision 16; both Bayer patterns give the same result here)
// @test ditherBayer_wrappers 2
// @expect ditherBayer_wrappers[0] 0.5 0.0 0.0 0.0
// @expect ditherBayer_wrappers[1] 0.5 0.625 0.4375 0.0
kernel void ditherBayer_wrappers(device float4* results [[buffer(0)]]) {
    float value = 0.53;
    float3 color3 = float3(0.53, 0.62, 0.47);
    float2 xy = float2(2.0, 3.0);
    float ditheredScalar = ditherBayer(float3(value), xy, 16).r;
    float3 dithered3Custom = ditherBayer(color3, xy, 16);
    results[0] = float4(ditheredScalar, 0.0, 0.0, 0.0);
    results[1] = float4(dithered3Custom, 0.0);
}

// ditherBayer - all wrapper functions (default precision 256)
// of the kern[x + y*8] table GLSL uses by default (bayer.glsl:43-55) and WESL uses; at xy (2,3) MSL gives
// 36/64 instead of 52/64, so 0.53 rounds up to 136/256 = 0.53125 instead of 135/256 = 0.5273.
// @test ditherBayer_default 2
// @expect ditherBayer_default[0] 0.5273 0.6172 0.4688 0.0
// @expect ditherBayer_default[1] 0.5273 0.6172 0.4688 0.85
kernel void ditherBayer_default(device float4* results [[buffer(0)]]) {
    float3 color3 = float3(0.53, 0.62, 0.47);
    float4 color4 = float4(0.53, 0.62, 0.47, 0.85);
    float2 xy = float2(2.0, 3.0);
    results[0] = float4(ditherBayer(color3, xy), 0.0);
    results[1] = ditherBayer(color4, xy);
}

// ditherVlachos - all wrapper functions: only the alpha check uses the default tolerance
// @test ditherVlachos_alpha
// @expect ditherVlachos_alpha[0] 0.0 0.0 0.0 0.85
kernel void ditherVlachos_alpha(device float4* results [[buffer(0)]]) {
    float4 color4 = float4(0.53, 0.62, 0.47, 0.85);
    float2 xy = float2(2.0, 3.0);
    float4 dithered4 = ditherVlachos(color4, xy);
    results[0] = float4(0.0, 0.0, 0.0, dithered4.a);
}

// ditherBlueNoise - all wrapper functions
// @test ditherBlueNoise_wrappers 4
// @expect ditherBlueNoise_wrappers[0] 0.5312
// @expect ditherBlueNoise_wrappers[1] 0.5312 0.6211 0.4727 0.0
// @expect ditherBlueNoise_wrappers[2] 0.5625 0.625 0.5 0.0
// @expect ditherBlueNoise_wrappers[3] 0.5312 0.6211 0.4727 0.85
kernel void ditherBlueNoise_wrappers(device float4* results [[buffer(0)]]) {
    float value = 0.53;
    float3 color3 = float3(0.53, 0.62, 0.47);
    float4 color4 = float4(0.53, 0.62, 0.47, 0.85);
    float2 xy = float2(2.0, 3.0);
    float dithered1 = ditherBlueNoise(value, xy);
    float3 dithered3Default = ditherBlueNoise(color3, xy);
    float3 dithered3Custom = ditherBlueNoise(color3, xy, 16);
    float4 dithered4 = ditherBlueNoise(color4, xy);
    results[0] = float4(dithered1, 0.0, 0.0, 0.0);
    results[1] = float4(dithered3Default, 0.0);
    results[2] = float4(dithered3Custom, 0.0);
    results[3] = dithered4;
}
