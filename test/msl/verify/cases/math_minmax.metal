// Ported from test/wesl/math-minmax.test.ts (same inputs and expected values).
// WGSL mmax2/mmax3/mmin2/mmin3(vecN) are the MSL overloads mmax/mmin(floatN).
#include <metal_stdlib>
using namespace metal;
#include "lygia/math/mmax.msl"
#include "lygia/math/mmin.msl"

// @test mmax2
// @expect mmax2[0] 7.0 9.0 -2.0 4.0
kernel void mmax2(device float4* results [[buffer(0)]]) {
    results[0] = float4(mmax(float2(3.0, 7.0)), mmax(float2(9.0, 2.0)), mmax(float2(-5.0, -2.0)), mmax(float2(-3.0, 4.0)));
}

// @test mmax3
// @expect mmax3[0] 7.0 9.0 8.0 -2.0
kernel void mmax3(device float4* results [[buffer(0)]]) {
    results[0] = float4(mmax(float3(3.0, 7.0, 5.0)), mmax(float3(9.0, 2.0, 4.0)), mmax(float3(1.0, 3.0, 8.0)), mmax(float3(-6.0, -2.0, -4.0)));
}

// @test mmin2
// @expect mmin2[0] 3.0 2.0 -5.0 -3.0
kernel void mmin2(device float4* results [[buffer(0)]]) {
    results[0] = float4(mmin(float2(3.0, 7.0)), mmin(float2(9.0, 2.0)), mmin(float2(-5.0, -2.0)), mmin(float2(-3.0, 4.0)));
}

// @test mmin3
// @expect mmin3[0] 3.0 2.0 1.0 -8.0
kernel void mmin3(device float4* results [[buffer(0)]]) {
    results[0] = float4(mmin(float3(3.0, 7.0, 5.0)), mmin(float3(9.0, 2.0, 4.0)), mmin(float3(6.0, 8.0, 1.0)), mmin(float3(-2.0, 5.0, -8.0)));
}
