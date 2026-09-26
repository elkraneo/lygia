// Ported from test/wesl/shaders/color_blend_darken.test.wesl (same inputs and expected values).
// The WESL test uses expectNear/expectNearVec3 (relative 1e-3, absolute 1e-6); here the
// default absolute 0.0001 is used (stricter for the non-zero values; for the expected 0.0
// results WESL allows 1e-6).
// WGSL blendX3(vec3f, vec3f) / blendX(f32, f32) == MSL overloads blendX(float3, float3) / blendX(float, float).
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/blend/darken.msl"
#include "lygia/color/blend/colorBurn.msl"
#include "lygia/color/blend/linearBurn.msl"

// @test blendDarken3_basic
// @expect blendDarken3_basic[0] 0.3 0.4 0.5
kernel void blendDarken3_basic(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendDarken(float3(0.6, 0.4, 0.5), float3(0.3, 0.7, 0.5)), 0.0);
}

// @test blendColorBurn3_basic
// @expect blendColorBurn3_basic[0] 0.0 0.0 0.0
kernel void blendColorBurn3_basic(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendColorBurn(float3(0.6, 0.5, 0.4), float3(0.3, 0.4, 0.5)), 0.0);
}

// @test blendLinearBurn3_basic
// @expect blendLinearBurn3_basic[0] 0.0 0.0 0.0
kernel void blendLinearBurn3_basic(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendLinearBurn(float3(0.6, 0.5, 0.7), float3(0.4, 0.3, 0.2)), 0.0);
}

// f32 tests: blendDarken_f32, blendColorBurn_f32, blendLinearBurn_f32
// @test blend_f32
// @expect blend_f32[0] 0.3 0.0 0.0
kernel void blend_f32(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendDarken(0.6, 0.3), blendColorBurn(0.6, 0.3), blendLinearBurn(0.6, 0.4), 0.0);
}
