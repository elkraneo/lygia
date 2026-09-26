// Ported from test/wesl/shaders/math_distance.test.wesl (same inputs and expected values).
// The WESL test uses expectNear (relative 1e-3, absolute 1e-6); here the default
// absolute 0.0001 is used, which is at least as strict for these values.
// WGSL lengthSq2/3 and distEuclidean2/distManhattan2 are MSL overloads (see math_distance.metal).
#include <metal_stdlib>
using namespace metal;
#include "lygia/math/lengthSq.msl"
#include "lygia/math/dist.msl"

// results[0] = (lengthSq2_pythagorean, lengthSq3_unit_vector, euclidean_3_4_5_triangle, manhattan_taxicab)
// @test distance
// @expect distance[0] 25.0 9.0 5.0 7.0
kernel void distance(device float4* results [[buffer(0)]]) {
    results[0] = float4(lengthSq(float2(3.0, 4.0)),
                        lengthSq(float3(1.0, 2.0, 2.0)),
                        distEuclidean(float2(0.0, 0.0), float2(3.0, 4.0)),
                        distManhattan(float2(0.0, 0.0), float2(3.0, 4.0)));
}
