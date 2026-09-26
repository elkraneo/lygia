// Ported from test/wesl/shaders/color_blend_opacity.test.wesl (same inputs and expected values).
// The WESL test uses expectNearVec3 (relative 1e-3, absolute 1e-6); here the default
// absolute 0.0001 is used, which is stricter for these values (all in [0.2, 0.8]).
// WGSL blendX3Opacity(vec3f, vec3f, f32) == MSL overload blendX(float3, float3, float).
// Expected values that the WESL test computes in-shader:
//   reflect: mix(base, (0.16/0.5, 0.36/0.7, 0.04/0.2), 0.5) = (0.36, 0.5571429, 0.2)
//   glow:    mix(base, (0.25/0.6, 0.09/0.4, 0.64/0.8), 0.5) = (0.4083333, 0.4125, 0.5)
//   colorDodge: (2/7 + 0.2, 5/12 + 0.25, 0.8) = (0.4857143, 0.6666667, 0.8)
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/blend/difference.msl"
#include "lygia/color/blend/exclusion.msl"
#include "lygia/color/blend/negation.msl"
#include "lygia/color/blend/phoenix.msl"
#include "lygia/color/blend/reflect.msl"
#include "lygia/color/blend/subtract.msl"
#include "lygia/color/blend/glow.msl"
#include "lygia/color/blend/colorBurn.msl"
#include "lygia/color/blend/colorDodge.msl"
#include "lygia/color/blend/linearBurn.msl"
#include "lygia/color/blend/linearDodge.msl"

// @test blendDifference3Opacity_basic
// @expect blendDifference3Opacity_basic[0] 0.55 0.35 0.4
kernel void blendDifference3Opacity_basic(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendDifference(float3(0.8, 0.3, 0.6), float3(0.5, 0.7, 0.4), 0.5), 0.0);
}

// @test blendDifference3Opacity_zero
// @expect blendDifference3Opacity_zero[0] 0.8 0.3 0.6
kernel void blendDifference3Opacity_zero(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendDifference(float3(0.8, 0.3, 0.6), float3(0.5, 0.7, 0.4), 0.0), 0.0);
}

// @test blendExclusion3Opacity_basic
// @expect blendExclusion3Opacity_basic[0] 0.57 0.47 0.74
kernel void blendExclusion3Opacity_basic(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendExclusion(float3(0.6, 0.4, 0.8), float3(0.3, 0.7, 0.2), 0.5), 0.0);
}

// @test blendNegation3Opacity_basic
// @expect blendNegation3Opacity_basic[0] 0.8 0.7 0.6
kernel void blendNegation3Opacity_basic(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendNegation(float3(0.7, 0.5, 0.3), float3(0.4, 0.6, 0.8), 0.5), 0.0);
}

// @test blendPhoenix3Opacity_basic
// @expect blendPhoenix3Opacity_basic[0] 0.7 0.7 0.4
kernel void blendPhoenix3Opacity_basic(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendPhoenix(float3(0.7, 0.5, 0.3), float3(0.4, 0.6, 0.8), 0.5), 0.0);
}

// @test blendReflect3Opacity_basic
// @expect blendReflect3Opacity_basic[0] 0.36 0.5571429 0.2
kernel void blendReflect3Opacity_basic(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendReflect(float3(0.4, 0.6, 0.2), float3(0.5, 0.3, 0.8), 0.5), 0.0);
}

// @test blendSubtract3Opacity_basic
// @expect blendSubtract3Opacity_basic[0] 0.45 0.3 0.25
kernel void blendSubtract3Opacity_basic(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendSubtract(float3(0.8, 0.6, 0.5), float3(0.3, 0.4, 0.2), 0.5), 0.0);
}

// @test blendGlow3Opacity_basic
// @expect blendGlow3Opacity_basic[0] 0.4083333 0.4125 0.5
kernel void blendGlow3Opacity_basic(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendGlow(float3(0.4, 0.6, 0.2), float3(0.5, 0.3, 0.8), 0.5), 0.0);
}

// @test blendColorBurn3Opacity_basic
// @expect blendColorBurn3Opacity_basic[0] 0.3 0.25 0.2
kernel void blendColorBurn3Opacity_basic(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendColorBurn(float3(0.6, 0.5, 0.4), float3(0.3, 0.4, 0.5), 0.5), 0.0);
}

// @test blendColorDodge3Opacity_basic
// @expect blendColorDodge3Opacity_basic[0] 0.4857143 0.6666667 0.8
kernel void blendColorDodge3Opacity_basic(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendColorDodge(float3(0.4, 0.5, 0.6), float3(0.3, 0.4, 0.5), 0.5), 0.0);
}

// @test blendLinearBurn3Opacity_basic
// @expect blendLinearBurn3Opacity_basic[0] 0.3 0.25 0.35
kernel void blendLinearBurn3Opacity_basic(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendLinearBurn(float3(0.6, 0.5, 0.7), float3(0.4, 0.3, 0.2), 0.5), 0.0);
}

// @test blendLinearDodge3Opacity_basic
// @expect blendLinearDodge3Opacity_basic[0] 0.55 0.6 0.65
kernel void blendLinearDodge3Opacity_basic(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendLinearDodge(float3(0.4, 0.5, 0.6), float3(0.3, 0.2, 0.1), 0.5), 0.0);
}
