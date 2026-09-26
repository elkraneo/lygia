// Ported from test/wesl/geometry-aabb.test.ts (same inputs and expected values).
// WGSL expand/expand2/expand3 (ptr + f32 / vec3f / AABB) are MSL expand(thread AABB&, float|float3|AABB).
#include <metal_stdlib>
using namespace metal;
#include "lygia/geometry/aabb/centroid.msl"
#include "lygia/geometry/aabb/contain.msl"
#include "lygia/geometry/aabb/diagonal.msl"
#include "lygia/geometry/aabb/expand.msl"
#include "lygia/geometry/aabb/square.msl"

// @test centroid
// @expect centroid[0] 0 0 0
kernel void centroid(device float4* results [[buffer(0)]]) {
    AABB box = AABB{float3(-2.0, -4.0, -6.0), float3(2.0, 4.0, 6.0)};
    results[0] = float4(centroid(box), 0.0);
}

// @test contain
// @expect contain[0] 1 0 0
kernel void contain(device float4* results [[buffer(0)]]) {
    AABB box = AABB{float3(-1.0), float3(1.0)};
    bool inside = contain(box, float3(0.0, 0.0, 0.0));
    bool outside = contain(box, float3(2.0, 0.0, 0.0));
    results[0] = float4(inside ? 1.0 : 0.0, outside ? 1.0 : 0.0, 0.0, 0.0);
}

// @test diagonal
// @expect diagonal[0] 2 4 6
kernel void diagonal(device float4* results [[buffer(0)]]) {
    AABB box = AABB{float3(-1.0, -2.0, -3.0), float3(1.0, 2.0, 3.0)};
    results[0] = float4(diagonal(box), 0.0);
}

// WGSL expand(&box, 0.5)
// @test expandScalar
// @expect expandScalar[0] -1.5 1.5 0
kernel void expandScalar(device float4* results [[buffer(0)]]) {
    AABB box = AABB{float3(-1.0), float3(1.0)};
    expand(box, 0.5);
    results[0] = float4(box.min.x, box.max.x, 0.0, 0.0);
}

// WGSL expand2(&box, vec3f(2, -2, 0.5))
// @test expandPoint
// @expect expandPoint[0] -2 2 0
kernel void expandPoint(device float4* results [[buffer(0)]]) {
    AABB box = AABB{float3(-1.0), float3(1.0)};
    expand(box, float3(2.0, -2.0, 0.5));
    results[0] = float4(box.min.y, box.max.x, 0.0, 0.0);
}

// WGSL expand3(&box1, box2)
// @test expandBox
// @expect expandBox[0] -2 2 2
kernel void expandBox(device float4* results [[buffer(0)]]) {
    AABB box1 = AABB{float3(-1.0), float3(1.0)};
    AABB box2 = AABB{float3(0.0, -2.0, 0.0), float3(2.0, 0.0, 2.0)};
    expand(box1, box2);
    results[0] = float4(box1.min.y, box1.max.x, box1.max.z, 0.0);
}

// @test square
// @expect square[0] 4 4 4
kernel void square(device float4* results [[buffer(0)]]) {
    AABB box = AABB{float3(-1.0, -2.0, -0.5), float3(1.0, 2.0, 0.5)};
    square(box);
    results[0] = float4(box.max - box.min, 0.0);
}
