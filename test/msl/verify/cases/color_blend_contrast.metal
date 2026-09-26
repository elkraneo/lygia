// Ported from test/wesl/color-blend-contrast.test.ts (same inputs and expected values).
// WGSL names encode the argument type; in MSL they're overloads:
//   WGSL blendX(f32, f32) / blendX3(vec3f, vec3f) == MSL blendX(float|float3, float|float3)
// The tests the WGSL file checks with tolerance 0.01 (blendVividLight3, blendGlow3,
// blendVividLight - f32, blendGlow - f32) are in color_blend_contrast_eps.metal.
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/blend/hardLight.msl"
#include "lygia/color/blend/overlay.msl"
#include "lygia/color/blend/softLight.msl"
#include "lygia/color/blend/pinLight.msl"
#include "lygia/color/blend/linearLight.msl"
#include "lygia/color/blend/hardMix.msl"

// WGSL blendHardLight3
// @test blendHardLight3_test
// @expect blendHardLight3_test[0] 0.24 0.6 0.88
kernel void blendHardLight3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendHardLight(float3(0.4, 0.6, 0.8), float3(0.3, 0.5, 0.7)), 0.0);
}

// WGSL blendOverlay3
// @test blendOverlay3_test
// @expect blendOverlay3_test[0] 0.24 0.6 0.88
kernel void blendOverlay3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendOverlay(float3(0.4, 0.6, 0.8), float3(0.3, 0.5, 0.7)), 0.0);
}

// WGSL blendSoftLight3
// @test blendSoftLight3_test
// @expect blendSoftLight3_test[0] 0.4 0.6 0.493
kernel void blendSoftLight3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendSoftLight(float3(0.5, 0.6, 0.4), float3(0.3, 0.5, 0.7)), 0.0);
}

// WGSL blendPinLight3
// @test blendPinLight3_test
// @expect blendPinLight3_test[0] 0.2 0.8 0.5
kernel void blendPinLight3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendPinLight(float3(0.3, 0.7, 0.5), float3(0.1, 0.9, 0.5)), 0.0);
}

// WGSL blendLinearLight3
// @test blendLinearLight3_test
// @expect blendLinearLight3_test[0] 0.0 0.5 1.0
kernel void blendLinearLight3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendLinearLight(float3(0.4, 0.5, 0.6), float3(0.3, 0.5, 0.7)), 0.0);
}

// WGSL blendHardMix3
// @test blendHardMix3_test
// @expect blendHardMix3_test[0] 0.0 1.0 1.0
kernel void blendHardMix3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendHardMix(float3(0.4, 0.6, 0.8), float3(0.3, 0.5, 0.2)), 0.0);
}

// "blendOverlay - f32"
// @test blendOverlay_f32
// @expect blendOverlay_f32[0] 0.24
kernel void blendOverlay_f32(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendOverlay(0.4, 0.3), 0.0, 0.0, 0.0);
}

// "blendSoftLight - f32"
// @test blendSoftLight_f32
// @expect blendSoftLight_f32[0] 0.4
kernel void blendSoftLight_f32(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendSoftLight(0.5, 0.3), 0.0, 0.0, 0.0);
}

// "blendHardLight - f32"
// @test blendHardLight_f32
// @expect blendHardLight_f32[0] 0.24
kernel void blendHardLight_f32(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendHardLight(0.4, 0.3), 0.0, 0.0, 0.0);
}

// "blendPinLight - f32"
// @test blendPinLight_f32
// @expect blendPinLight_f32[0] 0.2 0.8
kernel void blendPinLight_f32(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendPinLight(0.3, 0.1), blendPinLight(0.7, 0.9), 0.0, 0.0);
}

// "blendLinearLight - f32"
// @test blendLinearLight_f32
// @expect blendLinearLight_f32[0] 0.0
kernel void blendLinearLight_f32(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendLinearLight(0.4, 0.3), 0.0, 0.0, 0.0);
}

// "blendHardMix - f32"
// @test blendHardMix_f32
// @expect blendHardMix_f32[0] 0.0
kernel void blendHardMix_f32(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendHardMix(0.4, 0.3), 0.0, 0.0, 0.0);
}
