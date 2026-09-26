// Ported from test/wesl/color-blend-hsl.test.ts (same inputs and expected values).
// WGSL blendHue/blendColor(vec3f, vec3f) == MSL blendHue/blendColor(float3, float3). Both the
// WESL and MSL versions go through rgb2hsv/hsv2rgb (despite the "HSL" names).
// blendSaturation and blendLuminosity (WGSL tolerance 0.05) are in color_blend_hsl_eps.metal.
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/blend/hue.msl"
#include "lygia/color/blend/color.msl"

// @test blendHue_test
// @expect blendHue_test[0] 0.2 0.6 0.8
kernel void blendHue_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendHue(float3(0.8, 0.4, 0.2), float3(0.2, 0.6, 0.8)), 0.0);
}

// @test blendColor_test
// @expect blendColor_test[0] 0.2 0.6 0.8
kernel void blendColor_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendColor(float3(0.8, 0.4, 0.2), float3(0.2, 0.6, 0.8)), 0.0);
}
