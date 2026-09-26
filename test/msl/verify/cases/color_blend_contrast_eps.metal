// Ported from test/wesl/color-blend-contrast.test.ts (same inputs and expected values).
// Only the tests the WGSL file checks with tolerance 0.01; the rest are in color_blend_contrast.metal.
// WGSL blendX(f32, f32) / blendX3(vec3f, vec3f) == MSL blendX(float|float3, float|float3)
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/blend/vividLight.msl"
#include "lygia/color/blend/glow.msl"

// @eps 0.01

// WGSL blendVividLight3
// @test blendVividLight3_test
// @expect blendVividLight3_test[0] 0.167 0.6 0.667
kernel void blendVividLight3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendVividLight(float3(0.5, 0.6, 0.4), float3(0.3, 0.5, 0.7)), 0.0);
}

// WGSL blendGlow3
// @test blendGlow3_test
// @expect blendGlow3_test[0] 0.417 0.225 0.8
kernel void blendGlow3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendGlow(float3(0.4, 0.6, 0.2), float3(0.5, 0.3, 0.8)), 0.0);
}

// "blendVividLight - f32"
// @test blendVividLight_f32
// @expect blendVividLight_f32[0] 0.167
kernel void blendVividLight_f32(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendVividLight(0.5, 0.3), 0.0, 0.0, 0.0);
}

// "blendGlow - f32"
// @test blendGlow_f32
// @expect blendGlow_f32[0] 0.417
kernel void blendGlow_f32(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendGlow(0.4, 0.5), 0.0, 0.0, 0.0);
}
