// Ported from test/wesl/shaders/math_minmax.test.wesl (same inputs and expected values).
// The WESL test uses expectNearVec4 (relative 1e-3, absolute 1e-6); here the default
// absolute 0.0001 is used, which is at least as strict for these values.
// WGSL mmax2/mmax3/mmin2/mmin3(vecN) are the MSL overloads mmax/mmin(floatN).
#include <metal_stdlib>
using namespace metal;
#include "lygia/math/mmax.msl"
#include "lygia/math/mmin.msl"

// @test mmax2Cases
// @expect mmax2Cases[0] 7.0 9.0 -2.0 4.0
kernel void mmax2Cases(device float4* results [[buffer(0)]]) {
    results[0] = float4(mmax(float2(3.0, 7.0)), mmax(float2(9.0, 2.0)), mmax(float2(-5.0, -2.0)), mmax(float2(-3.0, 4.0)));
}

// @test mmax3Cases
// @expect mmax3Cases[0] 7.0 9.0 8.0 -2.0
kernel void mmax3Cases(device float4* results [[buffer(0)]]) {
    results[0] = float4(mmax(float3(3.0, 7.0, 5.0)), mmax(float3(9.0, 2.0, 4.0)), mmax(float3(1.0, 3.0, 8.0)), mmax(float3(-6.0, -2.0, -4.0)));
}

// @test mmin2Cases
// @expect mmin2Cases[0] 3.0 2.0 -5.0 -3.0
kernel void mmin2Cases(device float4* results [[buffer(0)]]) {
    results[0] = float4(mmin(float2(3.0, 7.0)), mmin(float2(9.0, 2.0)), mmin(float2(-5.0, -2.0)), mmin(float2(-3.0, 4.0)));
}

// @test mmin3Cases
// @expect mmin3Cases[0] 3.0 2.0 1.0 -8.0
kernel void mmin3Cases(device float4* results [[buffer(0)]]) {
    results[0] = float4(mmin(float3(3.0, 7.0, 5.0)), mmin(float3(9.0, 2.0, 4.0)), mmin(float3(6.0, 8.0, 1.0)), mmin(float3(-2.0, 5.0, -8.0)));
}
