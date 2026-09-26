// Ported from test/wesl/generative-wavelet.test.ts (same inputs and expected values).
//
// WGSL's random defaults to the sin-less hash with the .1031 scale (wavelet uses
// random2(vec2f) -> f32 from it), while GLSL and MSL default to the sin hash, so use
// WGSL's configuration to compare the values. WGSL's WAVELET_VORTICITY const is 0.0,
// which is the same as leaving the MSL define unset.
//
// Name mapping (WGSL -> MSL): noised2/noised3 -> noised(float2)/noised(float3);
// wavelet2(p) -> wavelet(float2); wavelet3(p) -> wavelet(float3);
// waveletScaled2(p, phase) -> wavelet(float2, float); waveletScaled3(p, k) -> wavelet(float3, float);
// wavelet(p, phase, k) -> wavelet(float2, float, float).
//
// "not toBeCloseTo(x, 1)" (|diff| >= 0.05) is @differ. The noised checks with epsilon 2
// and the inequality checks are evaluated in the kernel and written as 1/0 flags.
//
// The wavelet kernels fail on this branch: MSL math/rotate2d.msl:15 builds
// float2x2(c, -s, s, c), GLSL math/rotate2d.glsl:15 builds mat2(c, s, -s, c) (both
// constructors are column-major), so MSL rotates the other way. With a GLSL-layout
// rotate2d swapped in, every wavelet check below passes.
#include <metal_stdlib>
using namespace metal;
#define RANDOM_SINLESS
#define RANDOM_HIGHER_RANGE
#include "lygia/generative/noised.msl"
#include "lygia/generative/wavelet.msl"

static inline float flag(bool b) { return b ? 1.0 : 0.0; }

// [0] = dx analytical, dx numerical, dy analytical, dy numerical
// [1] flags: |dx_a - dx_n| < 2, |dy_a - dy_n| < 2, |dx_a| < 5, |dy_a| < 5
// @test noised2 2
// @expect noised2[0] -0.3648
// @expect noised2[1] 1 1 1 1
kernel void noised2(device float4* results [[buffer(0)]]) {
    float h = 0.001;
    float2 p = float2(1.0, 2.0);
    float3 nd = noised(p);
    float dxn = (noised(p + float2(h, 0.0)).x - noised(p - float2(h, 0.0)).x) / (2.0 * h);
    float dyn = (noised(p + float2(0.0, h)).x - noised(p - float2(0.0, h)).x) / (2.0 * h);
    results[0] = float4(nd.y, dxn, nd.z, dyn);
    results[1] = float4(flag(abs(nd.y - dxn) < 2.0), flag(abs(nd.z - dyn) < 2.0), flag(abs(nd.y) < 5.0), flag(abs(nd.z) < 5.0));
}

// same checks as noised2, no regression value
// @test noised3 2
// @expect noised3[1] 1 1 1 1
kernel void noised3(device float4* results [[buffer(0)]]) {
    float h = 0.001;
    float3 p = float3(1.0, 2.0, 3.0);
    float4 nd = noised(p);
    float dxn = (noised(p + float3(h, 0.0, 0.0)).x - noised(p - float3(h, 0.0, 0.0)).x) / (2.0 * h);
    float dyn = (noised(p + float3(0.0, h, 0.0)).x - noised(p - float3(0.0, h, 0.0)).x) / (2.0 * h);
    results[0] = float4(nd.y, dxn, nd.z, dyn);
    results[1] = float4(flag(abs(nd.y - dxn) < 2.0), flag(abs(nd.z - dyn) < 2.0), flag(abs(nd.y) < 5.0), flag(abs(nd.z) < 5.0));
}

// WGSL wavelet2
// @test wavelet2 2
// @expect wavelet2[0] -0.1946 -0.1946
// @differ wavelet2[0] wavelet2[1]
kernel void wavelet2(device float4* results [[buffer(0)]]) {
    float w1 = wavelet(float2(1.0, 2.0)), w2 = wavelet(float2(1.0, 2.0)), w3 = wavelet(float2(3.0, 4.0));
    results[0] = float4(w1, w2, w3, 0.0);
    results[1] = float4(w3, 0.0, 0.0, 0.0);
}

// WGSL wavelet3(vec3f(p, phase))
// @test wavelet3 3
// @expect wavelet3[0] -0.1946
// @same wavelet3[0] wavelet3[1]
// @differ wavelet3[0] wavelet3[2]
kernel void wavelet3(device float4* results [[buffer(0)]]) {
    float2 p = float2(1.0, 2.0);
    float w1 = wavelet(float3(p, 0.0)), w2 = wavelet(float3(p, 1.0)), w3 = wavelet(float3(p, 0.0));
    results[0] = float4(w1, w2, w3, 0.0);
    results[1] = float4(w3, w2, w1, 0.0);
    results[2] = float4(w2, 0.0, 0.0, 0.0);
}

// WGSL waveletScaled2(p, phase)
// @test waveletScaled2 2
// @expect waveletScaled2[0] -0.1114 -0.1114
// @differ waveletScaled2[0] waveletScaled2[1]
kernel void waveletScaled2(device float4* results [[buffer(0)]]) {
    float2 p = float2(1.0, 2.0);
    float w1 = wavelet(p, 0.5), w2 = wavelet(p, 0.5), w3 = wavelet(p * 2.0, 0.5);
    results[0] = float4(w1, w2, w3, 0.0);
    results[1] = float4(w3, 0.0, 0.0, 0.0);
}

// WGSL waveletScaled3(vec3f(p, phase), scale)
// @test waveletScaled3 2
// @expect waveletScaled3[0] 0.0945 0.0945
// @differ waveletScaled3[0] waveletScaled3[1]
kernel void waveletScaled3(device float4* results [[buffer(0)]]) {
    float3 p = float3(1.0, 2.0, 0.5);
    float w1 = wavelet(p, 1.0), w2 = wavelet(p, 1.0), w3 = wavelet(p, 2.0);
    results[0] = float4(w1, w2, w3, 0.0);
    results[1] = float4(w3, 0.0, 0.0, 0.0);
}

// WGSL wavelet(p, phase, scale)
// @test waveletBase 3
// @expect waveletBase[0] 0.1884 0.1884
// @differ waveletBase[0] waveletBase[1]
// @differ waveletBase[0] waveletBase[2]
kernel void waveletBase(device float4* results [[buffer(0)]]) {
    float2 p = float2(1.0, 2.0);
    float w1 = wavelet(p, 0.5, 1.5), w2 = wavelet(p, 0.5, 1.5), w3 = wavelet(p, 1.0, 1.5), w4 = wavelet(p, 0.5, 2.0);
    results[0] = float4(w1, w2, w3, w4);
    results[1] = float4(w3, 0.0, 0.0, 0.0);
    results[2] = float4(w4, 0.0, 0.0, 0.0);
}
