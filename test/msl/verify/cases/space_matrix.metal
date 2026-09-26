// Ported from test/wesl/space-matrix.test.ts (same inputs and expected values).
// WGSL mat4x4f(...) and MSL float4x4(...) constructors are both column-major.
#include <metal_stdlib>
using namespace metal;
#include "lygia/math/const.msl"
#include "lygia/space/tbn.msl"
#include "lygia/space/perspective.msl"
#include "lygia/space/orthographic.msl"
#include "lygia/space/translate.msl"

// @test tbn
// @expect tbn[0] 1.0 1.0 1.0 0.0
kernel void tbn(device float4* results [[buffer(0)]]) {
    float3x3 m = tbn(float3(1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0), float3(0.0, 0.0, 1.0));
    results[0] = float4(m * float3(1.0, 1.0, 1.0), 0.0);
}

// @test perspective
// @expect perspective[0] 0.5625
kernel void perspective(device float4* results [[buffer(0)]]) {
    float4x4 m = perspective(HALF_PI, 16.0 / 9.0, 0.1, 100.0);
    results[0] = float4(m[0][0], 0.0, 0.0, 0.0);
}

// @test orthographic
// @expect orthographic[0] 1.0
kernel void orthographic(device float4* results [[buffer(0)]]) {
    float4x4 m = orthographic(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0);
    results[0] = float4(m[0][0], 0.0, 0.0, 0.0);
}

// @test translate
// @expect translate[0] 10.0 20.0 30.0 1.0
kernel void translate(device float4* results [[buffer(0)]]) {
    float3x3 m = float3x3(float3(1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0), float3(0.0, 0.0, 1.0));
    results[0] = translate(m, float3(10.0, 20.0, 30.0))[3];
}
