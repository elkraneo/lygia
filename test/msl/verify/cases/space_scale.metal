// Ported from test/wesl/space-scale.test.ts (same inputs and expected values), the tests without
// CENTER_2D/CENTER_3D; the ones with a custom center are in space_scale_center.metal.
// Name mapping (WGSL -> MSL): scale2(vec2f, vec2f) / scale2_f(vec2f, f32) / scale3(vec3f, vec3f)
// -> scale(float2, float2) / scale(float2, float) / scale(float3, float3);
// math scale2dXY(x, y) -> scale2d(float, float).
#include <metal_stdlib>
using namespace metal;
#include "lygia/space/scale.msl"
#include "lygia/math/scale2d.msl"

// @test scale2
// @expect scale2[0] 1.0 0.375
kernel void scale2(device float4* results [[buffer(0)]]) {
    results[0] = float4(scale(float2(0.75, 0.25), float2(2.0, 0.5)), 0.0, 0.0);
}

// @test scale2_f
// @expect scale2_f[0] 1.0 0.0
kernel void scale2_f(device float4* results [[buffer(0)]]) {
    results[0] = float4(scale(float2(0.75, 0.25), 2.0), 0.0, 0.0);
}

// "scale2dXY - matrix construction"
// @test scale2dXY
// @expect scale2dXY[0] 8.0 15.0
kernel void scale2dXY(device float4* results [[buffer(0)]]) {
    results[0] = float4(scale2d(2.0, 3.0) * float2(4.0, 5.0), 0.0, 0.0);
}

// @test scale3
// @expect scale3[0] 1.0 0.375 0.5 0.0
kernel void scale3(device float4* results [[buffer(0)]]) {
    results[0] = float4(scale(float3(0.75, 0.25, 0.5), float3(2.0, 0.5, 1.0)), 0.0);
}
