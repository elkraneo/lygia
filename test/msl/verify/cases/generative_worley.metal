// Ported from test/wesl/generative-worley.test.ts (same inputs and expected values).
//
// WGSL's random defaults to the sin-less hash with the .1031 scale (worley uses
// random22/random33 from it), while GLSL and MSL default to the sin hash, so use
// WGSL's configuration to compare the values.
//
// Name mapping: WGSL worley2(vec2f) / worley3(vec3f) -> MSL worley(float2) / worley(float3),
// WGSL worley22 / worley32 -> MSL worley2(float2) / worley2(float3), returning (F1, F2).
#include <metal_stdlib>
using namespace metal;
#define RANDOM_SINLESS
#define RANDOM_HIGHER_RANGE
#include "lygia/generative/worley.msl"

static inline float flag(bool b) { return b ? 1.0 : 0.0; }

// WGSL worley2: determinism, not toBeCloseTo(w3, 1) (|diff| >= 0.05, same as @differ), regression
// @test worley2 2
// @expect worley2[0] 0.7471 0.7471
// @differ worley2[0] worley2[1]
kernel void worley2(device float4* results [[buffer(0)]]) {
    float w1 = worley(float2(1.0, 2.0)), w2 = worley(float2(1.0, 2.0)), w3 = worley(float2(1.5, 2.5));
    results[0] = float4(w1, w2, w3, 0.0);
    results[1] = float4(w3, 0.0, 0.0, 0.0);
}

// WGSL worley22: determinism, F1 and F2 in [0, 1.5], F1 <= F2 + 0.001, regression
// @test worley22 2
// @expect worley22[0] 0.2529
// @same worley22[0] worley22[1]
// @expect worley22[1] 0.2529
kernel void worley22(device float4* results [[buffer(0)]]) {
    float2 w1 = worley2(float2(1.0, 2.0)), w2 = worley2(float2(1.0, 2.0));
    results[0] = float4(w1, w2);
    bool ok = w1.x >= 0.0 && w1.x <= 1.5 && w1.y >= 0.0 && w1.y <= 1.5 && w1.x <= w1.y + 0.001;
    results[1] = float4(w2, w1) * flag(ok);   // zeroed if a range check fails
}

// WGSL worley3: determinism, not toBeCloseTo(w3, 1), regression
// @test worley3 2
// @expect worley3[0] 0.3876 0.3876
// @differ worley3[0] worley3[1]
kernel void worley3(device float4* results [[buffer(0)]]) {
    float w1 = worley(float3(1.0, 2.0, 3.0)), w2 = worley(float3(1.0, 2.0, 3.0)), w3 = worley(float3(4.0, 5.0, 6.0));
    results[0] = float4(w1, w2, w3, 0.0);
    results[1] = float4(w3, 0.0, 0.0, 0.0);
}

// WGSL worley32: determinism, F1 and F2 in [0, 2], F1 <= F2 + 0.001, regression
// @test worley32 2
// @expect worley32[0] 0.6124
// @same worley32[0] worley32[1]
// @expect worley32[1] 0.6124
kernel void worley32(device float4* results [[buffer(0)]]) {
    float2 w1 = worley2(float3(1.0, 2.0, 3.0)), w2 = worley2(float3(1.0, 2.0, 3.0));
    results[0] = float4(w1, w2);
    bool ok = w1.x >= 0.0 && w1.x <= 2.0 && w1.y >= 0.0 && w1.y <= 2.0 && w1.x <= w1.y + 0.001;
    results[1] = float4(w2, w1) * flag(ok);   // zeroed if a range check fails
}
