// Ported from test/wesl/color-blend-opacity-contrast.test.ts (same inputs and expected values).
// The two tests that use expectCloseTo(..., 0.01); the rest are in color_blend_opacity_contrast.metal.
// WGSL blendSoftLight3Opacity / blendVividLight3Opacity(vec3f, vec3f, f32)
//   ==  MSL blendSoftLight / blendVividLight(float3, float3, float)
// The WGSL vec3f result is results[0].xyz here.
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/blend/softLight.msl"
#include "lygia/color/blend/vividLight.msl"
// @eps 0.01

// @test blendSoftLight3Opacity_test
// @expect blendSoftLight3Opacity_test[0] 0.45 0.6 0.447
kernel void blendSoftLight3Opacity_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendSoftLight(float3(0.5, 0.6, 0.4), float3(0.3, 0.5, 0.7), 0.5), 0.0);
}

// @test blendVividLight3Opacity_test
// @expect blendVividLight3Opacity_test[0] 0.334 0.6 0.534
kernel void blendVividLight3Opacity_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendVividLight(float3(0.5, 0.6, 0.4), float3(0.3, 0.5, 0.7), 0.5), 0.0);
}
