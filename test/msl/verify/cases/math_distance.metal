// Ported from test/wesl/math-distance.test.ts (same inputs and expected values).
// WGSL names encode the argument type; in MSL they're overloads:
//   WGSL lengthSq2/lengthSq3(vecN)             ==  MSL lengthSq(floatN)
//   WGSL distEuclidean2/distManhattan2(vec2f)  ==  MSL distEuclidean/distManhattan(float2)
#include <metal_stdlib>
using namespace metal;
#include "lygia/math/lengthSq.msl"
#include "lygia/math/dist.msl"

// @test lengthSq2
// @expect lengthSq2[0] 25.0
kernel void lengthSq2(device float4* results [[buffer(0)]]) {
    results[0] = float4(lengthSq(float2(3.0, 4.0)), 0.0, 0.0, 0.0);
}

// @test lengthSq3
// @expect lengthSq3[0] 9.0
kernel void lengthSq3(device float4* results [[buffer(0)]]) {
    results[0] = float4(lengthSq(float3(1.0, 2.0, 2.0)), 0.0, 0.0, 0.0);
}

// @test distEuclidean2
// @expect distEuclidean2[0] 5.0
kernel void distEuclidean2(device float4* results [[buffer(0)]]) {
    results[0] = float4(distEuclidean(float2(0.0, 0.0), float2(3.0, 4.0)), 0.0, 0.0, 0.0);
}

// @test distManhattan2
// @expect distManhattan2[0] 7.0
kernel void distManhattan2(device float4* results [[buffer(0)]]) {
    results[0] = float4(distManhattan(float2(0.0, 0.0), float2(3.0, 4.0)), 0.0, 0.0, 0.0);
}
