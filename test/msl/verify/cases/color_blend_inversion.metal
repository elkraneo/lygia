// Ported from test/wesl/color-blend-inversion.test.ts (same inputs and expected values).
// WGSL names encode the argument type; in MSL they're overloads:
//   WGSL blendX(f32, f32) / blendX3(vec3f, vec3f) == MSL blendX(float|float3, float|float3)
// blendReflect3 (WGSL tolerance 0.01) is in color_blend_inversion_eps.metal.
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/blend/difference.msl"
#include "lygia/color/blend/exclusion.msl"
#include "lygia/color/blend/negation.msl"
#include "lygia/color/blend/phoenix.msl"
#include "lygia/color/blend/reflect.msl"
#include "lygia/color/blend/subtract.msl"

// WGSL blendDifference3
// @test blendDifference3_test
// @expect blendDifference3_test[0] 0.3 0.4 0.2
kernel void blendDifference3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendDifference(float3(0.8, 0.3, 0.6), float3(0.5, 0.7, 0.4)), 0.0);
}

// WGSL blendExclusion3
// @test blendExclusion3_test
// @expect blendExclusion3_test[0] 0.54 0.54 0.68
kernel void blendExclusion3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendExclusion(float3(0.6, 0.4, 0.8), float3(0.3, 0.7, 0.2)), 0.0);
}

// WGSL blendNegation3
// @test blendNegation3_test
// @expect blendNegation3_test[0] 0.9 0.9 0.9
kernel void blendNegation3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendNegation(float3(0.7, 0.5, 0.3), float3(0.4, 0.6, 0.8)), 0.0);
}

// WGSL blendPhoenix3
// @test blendPhoenix3_test
// @expect blendPhoenix3_test[0] 0.7 0.9 0.5
kernel void blendPhoenix3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendPhoenix(float3(0.7, 0.5, 0.3), float3(0.4, 0.6, 0.8)), 0.0);
}

// WGSL blendSubtract3
// @test blendSubtract3_test
// @expect blendSubtract3_test[0] 0.1 0.0 0.0
kernel void blendSubtract3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendSubtract(float3(0.8, 0.6, 0.5), float3(0.3, 0.4, 0.2)), 0.0);
}

// "blendDifference - f32"
// @test blendDifference_f32
// @expect blendDifference_f32[0] 0.3
kernel void blendDifference_f32(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendDifference(0.8, 0.5), 0.0, 0.0, 0.0);
}

// "blendExclusion - f32"
// @test blendExclusion_f32
// @expect blendExclusion_f32[0] 0.54
kernel void blendExclusion_f32(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendExclusion(0.6, 0.3), 0.0, 0.0, 0.0);
}

// "blendNegation - f32"
// @test blendNegation_f32
// @expect blendNegation_f32[0] 0.9
kernel void blendNegation_f32(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendNegation(0.7, 0.4), 0.0, 0.0, 0.0);
}

// "blendPhoenix - f32"
// @test blendPhoenix_f32
// @expect blendPhoenix_f32[0] 0.7
kernel void blendPhoenix_f32(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendPhoenix(0.7, 0.4), 0.0, 0.0, 0.0);
}

// "blendReflect - f32"
// @test blendReflect_f32
// @expect blendReflect_f32[0] 0.32
kernel void blendReflect_f32(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendReflect(0.4, 0.5), 0.0, 0.0, 0.0);
}

// "blendSubtract - f32"
// @test blendSubtract_f32
// @expect blendSubtract_f32[0] 0.1
kernel void blendSubtract_f32(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendSubtract(0.8, 0.3), 0.0, 0.0, 0.0);
}
