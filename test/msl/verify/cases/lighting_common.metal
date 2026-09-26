// Ported from test/wesl/lighting-common.test.ts (same inputs and expected values).
// Name mapping (WGSL -> MSL): GGXPrecise(N, H, NoH, r) -> GGX(N, H, NoH, r);
// schlickVec3 / schlickF32 -> schlick(float3, float3, float) / schlick(float, float, float).
// No TARGET_MOBILE, as in the WGSL test. Inequality checks are evaluated in the kernel
// and written as 1/0 flags.
#include <metal_stdlib>
using namespace metal;
#include "lygia/lighting/common/ggx.msl"
#include "lygia/lighting/common/schlick.msl"
#include "lygia/lighting/common/smithGGXCorrelated.msl"

static inline float flag(bool b) { return b ? 1.0 : 0.0; }

// [1] flags: r0 > 0.3, r1 < r0, r2 > r0
// @test ggx 2
// @expect ggx[0] 1.2732 0.2943 31.831
// @expect ggx[1] 1 1 1
kernel void ggx(device float4* results [[buffer(0)]]) {
    float r1 = GGX(1.0, 0.5), r2 = GGX(0.8, 0.5), r3 = GGX(1.0, 0.1);
    results[0] = float4(r1, r2, r3, 0.0);
    results[1] = float4(flag(r1 > 0.3), flag(r2 < r1), flag(r3 > r1), 0.0);
}

// WGSL GGXPrecise; [1] flags: r0 > 0.3
// @test ggxPrecise 2
// @expect ggxPrecise[0] 1.2732 1.2732
// @expect ggxPrecise[1] 1
kernel void ggxPrecise(device float4* results [[buffer(0)]]) {
    float3 N = float3(0.0, 0.0, 1.0), H = float3(0.0, 0.0, 1.0);
    float NoH = dot(N, H);
    float precise = GGX(N, H, NoH, 0.5), standard = GGX(NoH, 0.5);
    results[0] = float4(precise, standard, 0.0, 0.0);
    results[1] = float4(flag(precise > 0.3), 0.0, 0.0, 0.0);
}

// [1] flags: r2 > 0.7
// @test importanceSamplingGGX 2
// @expect importanceSamplingGGX[0] 1.0 1.0 0.995
// @expect importanceSamplingGGX[1] 1
kernel void importanceSamplingGGX(device float4* results [[buffer(0)]]) {
    float3 s1 = importanceSamplingGGX(float2(0.0, 0.0), 0.5);
    float3 s2 = importanceSamplingGGX(float2(0.5, 0.5), 0.5);
    float3 s3 = importanceSamplingGGX(float2(0.5, 0.5), 0.1);
    results[0] = float4(s1.z, length(s2), s3.z, 0.0);
    results[1] = float4(flag(s3.z > 0.7), 0.0, 0.0, 0.0);
}

// WGSL schlick(vec3f, f32, f32); [1] flags: 0.04 < r2 < 1
// @test schlickVec3F90Scalar 2
// @expect schlickVec3F90Scalar[0] 0.04 1.0 0.07
// @expect schlickVec3F90Scalar[1] 1 1
kernel void schlickVec3F90Scalar(device float4* results [[buffer(0)]]) {
    float3 f0 = float3(0.04);
    float3 n = schlick(f0, 1.0, 1.0), g = schlick(f0, 1.0, 0.0), m = schlick(f0, 1.0, 0.5);
    results[0] = float4(n.x, g.x, m.x, 0.0);
    results[1] = float4(flag(m.x > 0.04), flag(m.x < 1.0), 0.0, 0.0);
}

// WGSL schlickVec3; [1] flags: r3 > 0.85
// @test schlickVec3 2
// @expect schlickVec3[0] 1.0 0.71 0.29 1.0
// @expect schlickVec3[1] 1
kernel void schlickVec3(device float4* results [[buffer(0)]]) {
    float3 f0 = float3(1.0, 0.71, 0.29), f90 = float3(1.0, 0.95, 0.9);
    float3 n = schlick(f0, f90, 1.0), g = schlick(f0, f90, 0.0);
    results[0] = float4(n, g.x);
    results[1] = float4(flag(g.x > 0.85), 0.0, 0.0, 0.0);
}

// WGSL schlickF32; [1] flags: 0.04 < r2 < 1
// @test schlickF32 2
// @expect schlickF32[0] 0.04 1.0 0.07
// @expect schlickF32[1] 1 1
kernel void schlickF32(device float4* results [[buffer(0)]]) {
    float n = schlick(0.04, 1.0, 1.0), g = schlick(0.04, 1.0, 0.0), m = schlick(0.04, 1.0, 0.5);
    results[0] = float4(n, g, m, 0.0);
    results[1] = float4(flag(m > 0.04), flag(m < 1.0), 0.0, 0.0);
}

// [1] flags: r0 > r1, r1 > 0, r2 > 0.2
// @test smithGGXCorrelated 2
// @expect smithGGXCorrelated[0] 0.4447 0.3482 0.25
// @expect smithGGXCorrelated[1] 1 1 1
kernel void smithGGXCorrelated(device float4* results [[buffer(0)]]) {
    float s = smithGGXCorrelated(0.8, 0.7, 0.1), r = smithGGXCorrelated(0.8, 0.7, 0.9), p = smithGGXCorrelated(1.0, 1.0, 0.5);
    results[0] = float4(s, r, p, 0.0);
    results[1] = float4(flag(s > r), flag(r > 0.0), flag(p > 0.2), 0.0);
}

// [1] flags: |r0 - r1| < 0.1, r2 > r3
// @test smithGGXCorrelatedFast 2
// @expect smithGGXCorrelatedFast[0] 0.4076 0.3817 0.4318 0.342
// @expect smithGGXCorrelatedFast[1] 1 1
kernel void smithGGXCorrelatedFast(device float4* results [[buffer(0)]]) {
    float standard = smithGGXCorrelated(0.8, 0.7, 0.5);
    float fast = smithGGXCorrelated_Fast(0.8, 0.7, 0.5);
    float fastSmooth = smithGGXCorrelated_Fast(0.8, 0.7, 0.1);
    float fastRough = smithGGXCorrelated_Fast(0.8, 0.7, 0.9);
    results[0] = float4(standard, fast, fastSmooth, fastRough);
    results[1] = float4(flag(abs(standard - fast) < 0.1), flag(fastSmooth > fastRough), 0.0, 0.0);
}
