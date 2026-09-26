// Ported from test/wesl/generative-random.test.ts (same inputs and expected values).
// WGSL names encode the argument type; in MSL they're overloads:
//   WGSL random/random2/random3/random4(vecN) -> float   ==  MSL random(floatN)
//   WGSL randomMN(vecN) -> vecM                          ==  MSL randomM(floatN)
//
// WGSL's random defaults to the sin-less hash with the .1031 scale, while GLSL and
// MSL default to the sin hash, so use WGSL's configuration to compare the values.
#include <metal_stdlib>
using namespace metal;
#define RANDOM_SINLESS
#define RANDOM_HIGHER_RANGE
#include "lygia/generative/random.msl"

// @test random 2
// @expect random[0] 0.763 0.763
// @differ random[0] random[1]
kernel void random(device float4* results [[buffer(0)]]) {
    float r1 = random(1.0), r2 = random(1.0), r3 = random(2.0);
    results[0] = float4(r1, r2, r3, 0.0);
    results[1] = float4(r3, 0.0, 0.0, 0.0);
}

// @test random2 2
// @expect random2[0] 0.6153 0.6153
// @differ random2[0] random2[1]
kernel void random2(device float4* results [[buffer(0)]]) {
    float r1 = random(float2(1.0, 2.0)), r2 = random(float2(1.0, 2.0)), r3 = random(float2(3.0, 4.0));
    results[0] = float4(r1, r2, r3, 0.0);
    results[1] = float4(r3, 0.0, 0.0, 0.0);
}

// @test random3 2
// @expect random3[0] 0.372 0.372
// @differ random3[0] random3[1]
kernel void random3(device float4* results [[buffer(0)]]) {
    float r1 = random(float3(1.0, 2.0, 3.0)), r2 = random(float3(1.0, 2.0, 3.0)), r3 = random(float3(4.0, 5.0, 6.0));
    results[0] = float4(r1, r2, r3, 0.0);
    results[1] = float4(r3, 0.0, 0.0, 0.0);
}

// @test random4 2
// @expect random4[0] 0.5181 0.5181
// @differ random4[0] random4[1]
kernel void random4(device float4* results [[buffer(0)]]) {
    float r1 = random(float4(1.0, 2.0, 3.0, 4.0)), r2 = random(float4(1.0, 2.0, 3.0, 4.0)), r3 = random(float4(5.0, 6.0, 7.0, 8.0));
    results[0] = float4(r1, r2, r3, 0.0);
    results[1] = float4(r3, 0.0, 0.0, 0.0);
}

// WGSL random21(1.0)
// @test random21
// @expect random21[0] 0.8786
kernel void random21(device float4* results [[buffer(0)]]) {
    float2 r1 = random2(1.0), r2 = random2(1.0);
    results[0] = float4(r1.x, r1.y, r2.x, r2.y);
}

// WGSL random22(vec2f(1, 2))
// @test random22
// @expect random22[0] 0.2333
kernel void random22(device float4* results [[buffer(0)]]) {
    float2 r1 = random2(float2(1.0, 2.0)), r2 = random2(float2(1.0, 2.0));
    results[0] = float4(r1.x, r1.y, r2.x, r2.y);
}

// @test distribution 256
// @range distribution 0 1
kernel void distribution(device float4* results [[buffer(0)]]) {
    for (uint i = 0; i < 256; i++) {
        results[i] = float4(random(float(i)), random(float2(i, i * 2)), random(float3(i, 1, 2)), random(float4(i, 1, 2, 3)));
    }
}
