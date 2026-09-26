// Ported from test/wesl/math-quat.test.ts: the check that uses epsilon 0.1
// (the rest are in math_quat.metal with the default 0.0001).
// WGSL quatForward(f) is MSL quat(float3); WGSL quatMulVec3(q, v) is MSL rotate(QUAT, float3).
// @eps 0.1
#include <metal_stdlib>
using namespace metal;
#include "lygia/math/quat.msl"
#include "lygia/space/rotate.msl"

// "quatForward": the default forward (0, 0, 1) rotated by quatForward(+X) points to +X
// @test quatForward
// @expect quatForward[0] 1.0 0.0 0.0
kernel void quatForward(device float4* results [[buffer(0)]]) {
    float4 q = quat(normalize(float3(1.0, 0.0, 0.0)));
    float3 rotated = rotate(q, float3(0.0, 0.0, 1.0));
    results[0] = float4(rotated, sqrt(dot(q, q)));
}
