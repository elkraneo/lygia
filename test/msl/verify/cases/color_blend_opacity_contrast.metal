// Ported from test/wesl/color-blend-opacity-contrast.test.ts (same inputs and expected values).
// WGSL blendX3Opacity(vec3f, vec3f, f32)  ==  MSL blendX(float3, float3, float)
// The WGSL vec3f result is results[0].xyz here.
// blendSoftLight3Opacity and blendVividLight3Opacity use epsilon 0.01, so they are in
// color_blend_opacity_contrast_eps.metal.
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/blend/overlay.msl"
#include "lygia/color/blend/hardLight.msl"
#include "lygia/color/blend/pinLight.msl"
#include "lygia/color/blend/linearLight.msl"
#include "lygia/color/blend/hardMix.msl"

// @test blendOverlay3Opacity_test
// @expect blendOverlay3Opacity_test[0] 0.32 0.6 0.84
kernel void blendOverlay3Opacity_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendOverlay(float3(0.4, 0.6, 0.8), float3(0.3, 0.5, 0.7), 0.5), 0.0);
}

// @test blendHardLight3Opacity_test
// @expect blendHardLight3Opacity_test[0] 0.32 0.6 0.84
kernel void blendHardLight3Opacity_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendHardLight(float3(0.4, 0.6, 0.8), float3(0.3, 0.5, 0.7), 0.5), 0.0);
}

// @test blendPinLight3Opacity_test
// @expect blendPinLight3Opacity_test[0] 0.25 0.75 0.5
kernel void blendPinLight3Opacity_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendPinLight(float3(0.3, 0.7, 0.5), float3(0.1, 0.9, 0.5), 0.5), 0.0);
}

// @test blendLinearLight3Opacity_test
// @expect blendLinearLight3Opacity_test[0] 0.2 0.5 0.8
kernel void blendLinearLight3Opacity_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendLinearLight(float3(0.4, 0.5, 0.6), float3(0.3, 0.5, 0.7), 0.5), 0.0);
}

// @test blendHardMix3Opacity_test
// @expect blendHardMix3Opacity_test[0] 0.2 0.8 0.9
kernel void blendHardMix3Opacity_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendHardMix(float3(0.4, 0.6, 0.8), float3(0.3, 0.5, 0.2), 0.5), 0.0);
}
