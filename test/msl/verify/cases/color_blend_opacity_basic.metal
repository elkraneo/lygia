// Ported from test/wesl/color-blend-opacity-basic.test.ts (same inputs and expected values).
// WGSL names encode the argument type; in MSL they're overloads:
//   WGSL blendX3Opacity(vec3f, vec3f, f32) / blendScreenWithOpacity3  ==  MSL blendX(float3, float3, float)
// Each WGSL test writes one vec3f to results[0]; here it's results[0].xyz.
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/blend/add.msl"
#include "lygia/color/blend/multiply.msl"
#include "lygia/color/blend/screen.msl"
#include "lygia/color/blend/average.msl"
#include "lygia/color/blend/lighten.msl"
#include "lygia/color/blend/darken.msl"

// blendAdd3Opacity - opacity 0.5
// @test blendAdd3Opacity_half
// @expect blendAdd3Opacity_half[0] 0.65 0.6 0.85
kernel void blendAdd3Opacity_half(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendAdd(float3(0.3, 0.5, 0.7), float3(0.8, 0.2, 0.4), 0.5), 0.0);
}

// blendAdd3Opacity - opacity 0
// @test blendAdd3Opacity_zero
// @expect blendAdd3Opacity_zero[0] 0.3 0.5 0.7
kernel void blendAdd3Opacity_zero(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendAdd(float3(0.3, 0.5, 0.7), float3(0.8, 0.2, 0.4), 0.0), 0.0);
}

// blendAdd3Opacity - opacity 1
// @test blendAdd3Opacity_one
// @expect blendAdd3Opacity_one[0] 1.0 0.7 1.0
kernel void blendAdd3Opacity_one(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendAdd(float3(0.3, 0.5, 0.7), float3(0.8, 0.2, 0.4), 1.0), 0.0);
}

// @test blendMultiply3Opacity_test
// @expect blendMultiply3Opacity_test[0] 0.6 0.45 0.3
kernel void blendMultiply3Opacity_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendMultiply(float3(0.8, 0.6, 0.4), float3(0.5, 0.5, 0.5), 0.5), 0.0);
}

// WGSL blendScreenWithOpacity3
// @test blendScreen3Opacity_test
// @expect blendScreen3Opacity_test[0] 0.49 0.6 0.7
kernel void blendScreen3Opacity_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendScreen(float3(0.4, 0.5, 0.6), float3(0.3, 0.4, 0.5), 0.5), 0.0);
}

// @test blendAverage3Opacity_test
// @expect blendAverage3Opacity_test[0] 0.55 0.5 0.65
kernel void blendAverage3Opacity_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendAverage(float3(0.6, 0.4, 0.8), float3(0.4, 0.8, 0.2), 0.5), 0.0);
}

// @test blendLighten3Opacity_test
// @expect blendLighten3Opacity_test[0] 0.6 0.55 0.5
kernel void blendLighten3Opacity_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendLighten(float3(0.6, 0.4, 0.5), float3(0.3, 0.7, 0.5), 0.5), 0.0);
}

// @test blendDarken3Opacity_test
// @expect blendDarken3Opacity_test[0] 0.45 0.4 0.5
kernel void blendDarken3Opacity_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendDarken(float3(0.6, 0.4, 0.5), float3(0.3, 0.7, 0.5), 0.5), 0.0);
}
