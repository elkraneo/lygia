// Ported from test/wesl/shaders/color_blend_dodge.test.wesl (same inputs and expected values).
// The WESL test uses expectNear/expectNearVec3 (relative 1e-3, absolute 1e-6); here the
// default absolute 0.0001 is used, which is stricter for these values.
// 4/7 = 0.5714286, 5/6 = 0.8333333.
// WGSL blendX3(vec3f, vec3f) / blendX(f32, f32) == MSL overloads blendX(float3, float3) / blendX(float, float).
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/blend/colorDodge.msl"
#include "lygia/color/blend/linearDodge.msl"

// @test blendColorDodge3_basic
// @expect blendColorDodge3_basic[0] 0.5714286 0.8333333 1.0
kernel void blendColorDodge3_basic(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendColorDodge(float3(0.4, 0.5, 0.6), float3(0.3, 0.4, 0.5)), 0.0);
}

// @test blendLinearDodge3_basic
// @expect blendLinearDodge3_basic[0] 0.7 0.7 0.7
kernel void blendLinearDodge3_basic(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendLinearDodge(float3(0.4, 0.5, 0.6), float3(0.3, 0.2, 0.1)), 0.0);
}

// f32 tests: blendColorDodge_f32, blendLinearDodge_f32
// @test blend_f32
// @expect blend_f32[0] 0.5714286 0.7
kernel void blend_f32(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendColorDodge(0.4, 0.3), blendLinearDodge(0.4, 0.3), 0.0, 0.0);
}
