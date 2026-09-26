// Ported from test/wesl/math-matrix.test.ts (same inputs and expected values).
// WGSL names encode the argument type; in MSL they're overloads:
//   WGSL scale2d(f32) / scale2dVec(vec2f)          ==  MSL scale2d(float) / scale2d(float2)
//   WGSL translate4dXYZ(x, y, z)                   ==  MSL translate4d(float, float, float)
// WGSL mat4x4f(...) and MSL float4x4(...) constructors are both column-major.
// The "inverse - mat3" test uses epsilon 0.01, so it's in math_matrix_loose.metal.
#include <metal_stdlib>
using namespace metal;
#include "lygia/math/toMat3.msl"
#include "lygia/math/toMat4.msl"
#include "lygia/math/scale2d.msl"
#include "lygia/math/scale3d.msl"
#include "lygia/math/scale4d.msl"
#include "lygia/math/translate4d.msl"

// @test toMat3
// @expect toMat3[0] 1.0 6.0 11.0 0.0
kernel void toMat3(device float4* results [[buffer(0)]]) {
    float4x4 m4 = float4x4(float4(1.0, 2.0, 3.0, 4.0), float4(5.0, 6.0, 7.0, 8.0),
                           float4(9.0, 10.0, 11.0, 12.0), float4(13.0, 14.0, 15.0, 16.0));
    float3x3 m3 = toMat3(m4);
    results[0] = float4(m3[0][0], m3[1][1], m3[2][2], 0.0);
}

// @test toMat4
// @expect toMat4[0] 1.0 5.0 9.0 1.0
kernel void toMat4(device float4* results [[buffer(0)]]) {
    float3x3 m3 = float3x3(float3(1.0, 2.0, 3.0), float3(4.0, 5.0, 6.0), float3(7.0, 8.0, 9.0));
    float4x4 m4 = toMat4(m3);
    results[0] = float4(m4[0][0], m4[1][1], m4[2][2], m4[3][3]);
}

// "scale2d - uniform scale"
// @test scale2d
// @expect scale2d[0] 6.0 8.0 0.0 0.0
kernel void scale2d(device float4* results [[buffer(0)]]) {
    float2 r = scale2d(2.0) * float2(3.0, 4.0);
    results[0] = float4(r.x, r.y, 0.0, 0.0);
}

// "scale2dVec - non-uniform scale"
// @test scale2dVec
// @expect scale2dVec[0] 8.0 15.0 0.0 0.0
kernel void scale2dVec(device float4* results [[buffer(0)]]) {
    float2 r = scale2d(float2(2.0, 3.0)) * float2(4.0, 5.0);
    results[0] = float4(r.x, r.y, 0.0, 0.0);
}

// @test scale3d
// @expect scale3d[0] 2.0 6.0 12.0 0.0
kernel void scale3d(device float4* results [[buffer(0)]]) {
    float3 r = scale3d(float3(2.0, 3.0, 4.0)) * float3(1.0, 2.0, 3.0);
    results[0] = float4(r, 0.0);
}

// @test scale4d
// @expect scale4d[0] 2.0 6.0 12.0 1.0
kernel void scale4d(device float4* results [[buffer(0)]]) {
    results[0] = scale4d(float3(2.0, 3.0, 4.0)) * float4(1.0, 2.0, 3.0, 1.0);
}

// @test translate4d
// @expect translate4d[0] 11.0 22.0 33.0 1.0
kernel void translate4d(device float4* results [[buffer(0)]]) {
    results[0] = translate4d(float3(10.0, 20.0, 30.0)) * float4(1.0, 2.0, 3.0, 1.0);
}

// @test translate4dXYZ
// @expect translate4dXYZ[0] 6.0 12.0 18.0 1.0
kernel void translate4dXYZ(device float4* results [[buffer(0)]]) {
    results[0] = translate4d(5.0, 10.0, 15.0) * float4(1.0, 2.0, 3.0, 1.0);
}
