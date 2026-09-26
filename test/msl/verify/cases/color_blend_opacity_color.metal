// Ported from test/wesl/color-blend-opacity-color.test.ts (same inputs and expected values).
// WGSL blendHueOpacity(vec3f, vec3f, f32)  ==  MSL blendHue(float3, float3, float)
// The WGSL vec3f result is results[0].xyz here.
// blendSaturationOpacity and blendLuminosityOpacity use epsilon 0.1, so they are in
// color_blend_opacity_color_eps.metal.
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/blend/hue.msl"

// @test blendHueOpacity_test
// @expect blendHueOpacity_test[0] 0.5 0.5 0.5
kernel void blendHueOpacity_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendHue(float3(0.8, 0.4, 0.2), float3(0.2, 0.6, 0.8), 0.5), 0.0);
}
