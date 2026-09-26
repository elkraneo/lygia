// Ported from test/wesl/math-sampling.test.ts (same inputs and expected values).
// All sources match the WESL ones (nyquist: NYQUIST_FILTER_CENTER 0.5 / WIDTH 0.25 in both).
// WGSL hammersley(u32, i32) is MSL hammersley(uint, int).
// Inequality checks from the WGSL tests (nyquist 0.5 < v2 < 0.8, the hemisphereCosSample
// z ordering, |grad4| < 3) are implied by the exact values, which are checked too; the grad4
// "not equal" checks are written as any(a != b) flags.
#include <metal_stdlib>
using namespace metal;
#include "lygia/math/hammersley.msl"
#include "lygia/math/nyquist.msl"
#include "lygia/math/permute.msl"
#include "lygia/math/grad4.msl"

// @test hammersley
// @expect hammersley[0] 0.0 0.0 0.125 0.5
kernel void hammersley(device float4* results [[buffer(0)]]) {
    float2 h0 = hammersley(0u, 8), h1 = hammersley(1u, 8);
    results[0] = float4(h0.x, h0.y, h1.x, h1.y);
}

// "hammersley - bit reversal verification"
// @test hammersleyBits
// @expect hammersleyBits[0] 0.25 0.25 0.375 0.75
kernel void hammersleyBits(device float4* results [[buffer(0)]]) {
    float2 h2 = hammersley(2u, 8), h3 = hammersley(3u, 8);
    results[0] = float4(h2.x, h2.y, h3.x, h3.y);
}

// @test nyquist
// @expect nyquist[0] 0.8 0.65 0.5 0.5
kernel void nyquist(device float4* results [[buffer(0)]]) {
    results[0] = float4(nyquist(0.8, 0.1), nyquist(0.8, 0.5), nyquist(0.8, 1.0), nyquist(0.5, 0.8));
}

// @test permute
// @expect permute[0] 35.0 231.0 236.0 35.0
kernel void permute(device float4* results [[buffer(0)]]) {
    results[0] = float4(permute(1.0), permute(10.0), permute(100.0), permute(1.0));
}

// "grad4 - noise gradient helper": results[0..3] = g1a, g1b, g2, g3;
// results[4] = (any(g1a != g2), any(g1a != g3), 0, 0)
// @test grad4 5
// @expect grad4[0] -0.866 1.402 1.402 -3.438
// @same grad4[0] grad4[1]
// @expect grad4[4] 1.0 1.0
kernel void grad4(device float4* results [[buffer(0)]]) {
    float4 g1a = grad4(123.456, float4(0.789, 0.234, 0.567, 0.891));
    float4 g1b = grad4(123.456, float4(0.789, 0.234, 0.567, 0.891));
    float4 g2 = grad4(42.0, float4(0.789, 0.234, 0.567, 0.891));
    float4 g3 = grad4(123.456, float4(0.33, 0.67, 0.89, 1.23));
    results[0] = g1a;
    results[1] = g1b;
    results[2] = g2;
    results[3] = g3;
    results[4] = float4(any(g1a != g2) ? 1.0 : 0.0, any(g1a != g3) ? 1.0 : 0.0, 0.0, 0.0);
}

// "grad4 - gradient range validation"
// @test grad4Range
// @expect grad4Range[0] 0.0 0.0 -0.8 0.0
// @range grad4Range -3 3
kernel void grad4Range(device float4* results [[buffer(0)]]) {
    float4 g1 = grad4(50.0, float4(0.0, 0.0, 0.0, 0.0));
    float4 g2 = grad4(75.0, float4(0.2, 0.3, 0.4, 0.5));
    results[0] = float4(g1.y, g1.z, g2.y, g2.z);
}

// "hemisphereCosSample - unit vector property"
// @test hemisphereUnit
// @expect hemisphereUnit[0] 1.0 1.0 1.0 0.0
kernel void hemisphereUnit(device float4* results [[buffer(0)]]) {
    results[0] = float4(length(hemisphereCosSample(float2(0.0, 0.0))),
                        length(hemisphereCosSample(float2(1.0, 1.0))),
                        length(hemisphereCosSample(float2(0.5, 0.5))), 0.0);
}

// "hemisphereCosSample - positive hemisphere": (1, 0, SQRT1_2, 0.5)
// @test hemispherePositive
// @expect hemispherePositive[0] 1.0 0.0 0.70710678 0.5
kernel void hemispherePositive(device float4* results [[buffer(0)]]) {
    results[0] = float4(hemisphereCosSample(float2(0.0, 0.0)).z, hemisphereCosSample(float2(1.0, 1.0)).z,
                        hemisphereCosSample(float2(0.5, 0.5)).z, hemisphereCosSample(float2(0.25, 0.75)).z);
}

// "hemisphereCosSample - known values"
// @test hemisphereKnown
// @expect hemisphereKnown[0] 1.0 1.0 0.0 0.0
kernel void hemisphereKnown(device float4* results [[buffer(0)]]) {
    float3 v1 = hemisphereCosSample(float2(0.0, 0.0));
    float3 v2 = hemisphereCosSample(float2(0.0, 1.0));
    results[0] = float4(v1.z, v2.x, v2.y, v2.z);
}

// "hemisphereCosSample - cosine distribution": (0.94868, SQRT1_2, 1/sqrt(10), 0)
// @test hemisphereCosine
// @expect hemisphereCosine[0] 0.94868 0.70710678 0.31622777 0.0
kernel void hemisphereCosine(device float4* results [[buffer(0)]]) {
    results[0] = float4(hemisphereCosSample(float2(0.5, 0.1)).z, hemisphereCosSample(float2(0.5, 0.5)).z,
                        hemisphereCosSample(float2(0.5, 0.9)).z, 0.0);
}
