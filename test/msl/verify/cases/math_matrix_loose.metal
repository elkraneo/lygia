// Ported from test/wesl/math-matrix.test.ts: the checks that use epsilon 0.01
// (the rest are in math_matrix.metal with the default 0.0001).
// @eps 0.01
#include <metal_stdlib>
using namespace metal;
#include "lygia/math/inverse.msl"

// "inverse - mat3" (only x, y, z are checked, as in the WGSL test)
// @test inverse3
// @expect inverse3[0] 1.0 0.5 0.333
kernel void inverse3(device float4* results [[buffer(0)]]) {
    float3x3 m = float3x3(float3(1.0, 0.0, 0.0), float3(0.0, 2.0, 0.0), float3(0.0, 0.0, 3.0));
    float3x3 mInv = inverse(m);
    results[0] = float4(mInv[0][0], mInv[1][1], mInv[2][2], 0.0);
}
