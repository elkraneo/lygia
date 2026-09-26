// Ported from test/wesl/space-transform.test.ts (same inputs and expected values).
// INV_SQRT2 (lygia::math::consts, WESL only; not in math/const.msl) is written as a literal.
// The WGSL test's inequality checks (diff_nearby < 0.2, diff_distant > 0.3) are @range checks
// on separate kernels.
#include <metal_stdlib>
using namespace metal;
#include "lygia/space/decimateNormal.msl"

static float3 decimateNormalCase(int i) {
    float prec = 4.0;
    float3 n1 = normalize(float3(0.70710678118654752, 0.70710678118654752, 0.0));
    float3 n2 = normalize(float3(0.710, 0.690, 0.0));
    float3 n3 = normalize(float3(1.0, 0.0, 0.0));
    return decimateNormal(i == 1 ? n1 : (i == 2 ? n2 : n3), prec);
}

// results[0] = (d1.x, length(d1))
// @test decimateNormal
// @expect decimateNormal[0] 0.72986 1.0
kernel void decimateNormal(device float4* results [[buffer(0)]]) {
    float3 d1 = decimateNormalCase(1);
    results[0] = float4(d1.x, length(d1), 0.0, 0.0);
}

// diff_nearby = |d1 - d2| (sum of components) < 0.2
// @test decimateNormalNearby
// @range decimateNormalNearby 0 0.2
kernel void decimateNormalNearby(device float4* results [[buffer(0)]]) {
    float3 d = abs(decimateNormalCase(1) - decimateNormalCase(2));
    results[0] = float4(d.x + d.y + d.z);
}

// diff_distant = |d1 - d3| (sum of components) > 0.3
// @test decimateNormalDistant
// @range decimateNormalDistant 0.3 1000
kernel void decimateNormalDistant(device float4* results [[buffer(0)]]) {
    float3 d = abs(decimateNormalCase(1) - decimateNormalCase(3));
    results[0] = float4(d.x + d.y + d.z);
}
