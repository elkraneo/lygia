// Ported from test/wesl/math-quat.test.ts (same inputs and expected values).
// Name mapping (WGSL -> MSL):
//   quat(axis, angle)          ->  quat(float3, float)
//   quatForward(f)             ->  quat(float3)
//   quatForwardUp(f, up)       ->  quat(float3, float3)
//   quatMulVec3(q, v)          ->  rotate(QUAT, float3) from space/rotate.msl (q * v * conj(q), the
//                                  same rotation for unit quaternions). MSL has no quatMulVec3;
//                                  include quat/mul.msl before space/rotate.msl to get this overload.
// INV_SQRT2 (lygia::math::consts, WESL only; not in math/const.msl or .glsl) is written as a literal.
// The quatForward test checks the rotated vector with epsilon 0.1, so that check is in
// math_quat_loose.metal; its length check (default epsilon) is here.
#include <metal_stdlib>
using namespace metal;
#include "lygia/math/const.msl"
#include "lygia/math/quat.msl"
#include "lygia/math/quat/add.msl"
#include "lygia/math/quat/neg.msl"
#include "lygia/math/quat/sub.msl"
#include "lygia/math/quat/mul.msl"
#include "lygia/math/quat/conj.msl"
#include "lygia/math/quat/norm.msl"
#include "lygia/math/quat/length.msl"
#include "lygia/math/quat/lengthSq.msl"
#include "lygia/math/quat/identity.msl"
#include "lygia/math/quat/lerp.msl"
#include "lygia/math/quat/2mat3.msl"
#include "lygia/math/quat/2mat4.msl"
#include "lygia/math/quat/div.msl"
#include "lygia/math/quat/inverse.msl"
#include "lygia/space/rotate.msl"

// @test quatAdd
// @expect quatAdd[0] 1.5 2.5 3.5 4.5
kernel void quatAdd(device float4* results [[buffer(0)]]) {
    results[0] = quatAdd(float4(1.0, 2.0, 3.0, 4.0), float4(0.5, 0.5, 0.5, 0.5));
}

// @test quatSub
// @expect quatSub[0] 0.5 1.5 2.5 3.5
kernel void quatSub(device float4* results [[buffer(0)]]) {
    results[0] = quatSub(float4(1.0, 2.0, 3.0, 4.0), float4(0.5, 0.5, 0.5, 0.5));
}

// @test quatMul
// @expect quatMul[0] 0.5 0.5 0.5 0.5
kernel void quatMul(device float4* results [[buffer(0)]]) {
    float4 q1 = normalize(float4(1.0, 0.0, 0.0, 1.0));
    float4 q2 = normalize(float4(0.0, 1.0, 0.0, 1.0));
    results[0] = quatMul(q1, q2);
}

// @test quatConj
// @expect quatConj[0] -1.0 -2.0 -3.0 4.0
kernel void quatConj(device float4* results [[buffer(0)]]) {
    results[0] = quatConj(float4(1.0, 2.0, 3.0, 4.0));
}

// @test quatNorm
// @expect quatNorm[0] 0.1826 0.3651 0.5477 0.7303
kernel void quatNorm(device float4* results [[buffer(0)]]) {
    results[0] = quatNorm(float4(1.0, 2.0, 3.0, 4.0));
}

// @test quatLength
// @expect quatLength[0] 5.47723
kernel void quatLength(device float4* results [[buffer(0)]]) {
    results[0] = float4(quatLength(float4(1.0, 2.0, 3.0, 4.0)), 0.0, 0.0, 0.0);
}

// @test quatLengthSq
// @expect quatLengthSq[0] 30.0
kernel void quatLengthSq(device float4* results [[buffer(0)]]) {
    results[0] = float4(quatLengthSq(float4(1.0, 2.0, 3.0, 4.0)), 0.0, 0.0, 0.0);
}

// @test quatIdentity
// @expect quatIdentity[0] 0.0 0.0 0.0 1.0
kernel void quatIdentity(device float4* results [[buffer(0)]]) {
    results[0] = QUAT_IDENTITY;
}

// @test quatLerp
// @expect quatLerp[0] 0.4082 0.4082 0.0 0.8165
kernel void quatLerp(device float4* results [[buffer(0)]]) {
    float4 q1 = normalize(float4(1.0, 0.0, 0.0, 1.0));
    float4 q2 = normalize(float4(0.0, 1.0, 0.0, 1.0));
    results[0] = quatLerp(q1, q2, 0.5);
}

// @test quat2mat3
// @expect quat2mat3[0] 0.0 0.0 1.0 0.0
kernel void quat2mat3(device float4* results [[buffer(0)]]) {
    float4 q = normalize(float4(0.0, 0.0, 0.70710678118654752, 0.70710678118654752));
    float3x3 m = quat2mat3(q);
    results[0] = float4(m[0][0], m[1][1], m[2][2], 0.0);
}

// @test quat2mat4
// @expect quat2mat4[0] 0.0 0.0 1.0 1.0
kernel void quat2mat4(device float4* results [[buffer(0)]]) {
    float4 q = normalize(float4(0.0, 0.0, 0.70710678118654752, 0.70710678118654752));
    float4x4 m = quat2mat4(q);
    results[0] = float4(m[0][0], m[1][1], m[2][2], m[3][3]);
}

// "quat - create from axis and angle": (0, SQRT1_2, 0, SQRT1_2)
// @test quatAxisAngle
// @expect quatAxisAngle[0] 0.0 0.70710678 0.0 0.70710678
kernel void quatAxisAngle(device float4* results [[buffer(0)]]) {
    float3 axis = normalize(float3(0.0, 1.0, 0.0));
    results[0] = quat(axis, HALF_PI);
}

// @test quatDiv
// @expect quatDiv[0] 1.0 2.0 3.0 4.0
kernel void quatDiv(device float4* results [[buffer(0)]]) {
    results[0] = quatDiv(float4(2.0, 4.0, 6.0, 8.0), 2.0);
}

// @test quatNeg
// @expect quatNeg[0] -1.0 -2.0 -3.0 -4.0
kernel void quatNeg(device float4* results [[buffer(0)]]) {
    results[0] = quatNeg(float4(1.0, 2.0, 3.0, 4.0));
}

// @test quatInverse
// @expect quatInverse[0] 0.0 0.0 0.0 1.0
kernel void quatInverse(device float4* results [[buffer(0)]]) {
    float4 q = normalize(float4(1.0, 2.0, 3.0, 4.0));
    results[0] = quatMul(q, quatInverse(q));
}

// "quatForward": WGSL result[3] (the quaternion's length); the rotated vector
// (result[0..2], epsilon 0.1) is checked in math_quat_loose.metal.
// @test quatForward
// @expect quatForward[0] 1.0
kernel void quatForward(device float4* results [[buffer(0)]]) {
    float4 q = quat(normalize(float3(1.0, 0.0, 0.0)));
    results[0] = float4(sqrt(dot(q, q)), 0.0, 0.0, 0.0);
}

// "quatForwardUp": results[0] = (rotatedForward.x, rotatedUp.y, length, 0)
// @test quatForwardUp
// @expect quatForwardUp[0] 1.0 1.0 1.0
kernel void quatForwardUp(device float4* results [[buffer(0)]]) {
    float3 forward = normalize(float3(1.0, 0.0, 0.0));
    float3 up = normalize(float3(0.0, 1.0, 0.0));
    float4 q = quat(forward, up);
    float3 rotatedForward = rotate(q, float3(0.0, 0.0, 1.0));
    float3 rotatedUp = rotate(q, float3(0.0, 1.0, 0.0));
    results[0] = float4(rotatedForward.x, rotatedUp.y, sqrt(dot(q, q)), 0.0);
}
