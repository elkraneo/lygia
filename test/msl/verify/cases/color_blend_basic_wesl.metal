// Ported from test/wesl/shaders/color_blend_basic.test.wesl (same inputs and expected values).
// The WESL test uses expectNear/expectNearVec3 (relative 1e-3, absolute 1e-6); here the
// default absolute 0.0001 is used, which is stricter for these values (all in [0.2, 1]).
// WGSL blendX3(vec3f, vec3f) / blendX(f32, f32) == MSL overloads blendX(float3, float3) / blendX(float, float).
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/blend/add.msl"
#include "lygia/color/blend/multiply.msl"
#include "lygia/color/blend/screen.msl"
#include "lygia/color/blend/average.msl"
#include "lygia/color/blend/lighten.msl"

// @test blendAdd3_basic
// @expect blendAdd3_basic[0] 0.7 0.7 0.8
kernel void blendAdd3_basic(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendAdd(float3(0.5, 0.3, 0.2), float3(0.2, 0.4, 0.6)), 0.0);
}

// @test blendMultiply3_basic
// @expect blendMultiply3_basic[0] 0.4 0.3 0.2
kernel void blendMultiply3_basic(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendMultiply(float3(0.8, 0.6, 0.4), float3(0.5, 0.5, 0.5)), 0.0);
}

// @test blendScreen3_basic
// @expect blendScreen3_basic[0] 0.58 0.7 0.8
kernel void blendScreen3_basic(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendScreen(float3(0.4, 0.5, 0.6), float3(0.3, 0.4, 0.5)), 0.0);
}

// @test blendAverage3_basic
// @expect blendAverage3_basic[0] 0.5 0.6 0.5
kernel void blendAverage3_basic(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendAverage(float3(0.6, 0.4, 0.8), float3(0.4, 0.8, 0.2)), 0.0);
}

// @test blendLighten3_basic
// @expect blendLighten3_basic[0] 0.6 0.7 0.5
kernel void blendLighten3_basic(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendLighten(float3(0.6, 0.4, 0.5), float3(0.3, 0.7, 0.5)), 0.0);
}

// f32 tests: blendAdd_f32, blendMultiply_f32, blendScreen_f32, blendAverage_f32 in [0].xyzw, blendLighten_f32 in [1].x
// @test blend_f32 2
// @expect blend_f32[0] 0.8 0.4 0.7 0.5
// @expect blend_f32[1] 0.6
kernel void blend_f32(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendAdd(0.5, 0.3), blendMultiply(0.8, 0.5), blendScreen(0.4, 0.5), blendAverage(0.6, 0.4));
    results[1] = float4(blendLighten(0.6, 0.3), 0.0, 0.0, 0.0);
}
