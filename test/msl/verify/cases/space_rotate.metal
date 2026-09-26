// Ported from test/wesl/space-rotate.test.ts (same inputs and expected values), the tests without
// CENTER_2D/CENTER_3D; the ones with a custom center are in space_rotate_center.metal.
// Name mapping (WGSL -> MSL): rotateX3/rotateY3/rotateZ3(vec3f, r) -> rotateX/rotateY/rotateZ(float3, float);
// rotate(vec2f, r) / rotate_c(vec2f, r, c) -> rotate(float2, float) / rotate(float2, float, float2);
// rotate3(vec3f, r, axis) -> rotate(float3, float, float3);
// bracketing(dir) -> BracketingResult  ->  bracketing(dir, out vAxis0, out vAxis1, out blendAlpha).
//
// space/rotate*.msl build on math/rotate2d/4d/4dX/4dY/4dZ.msl, which still have the old (clockwise,
// transposed) matrices that GLSL dropped in 9a3a036c (see math_rotate.metal). WESL only updated
// rotate2d, so the rotate/rotate_c expectations (counterclockwise) fail on Metal, while the
// rotateX3/rotateZ3/rotate3 expectations follow the old convention and pass (GLSL would differ).
#include <metal_stdlib>
using namespace metal;
#include "lygia/math/const.msl"
#include "lygia/space/rotateX.msl"
#include "lygia/space/rotateY.msl"
#include "lygia/space/rotateZ.msl"
#include "lygia/space/rotate.msl"
#include "lygia/space/bracketing.msl"

// @test rotateX3
// @xfail rotateX3[0] WGSL kept the rotation direction from before upstream commit 9a3a036c; MSL now matches GLSL
// @expect rotateX3[0] 1.0 0.0 -1.0 0.0
kernel void rotateX3(device float4* results [[buffer(0)]]) {
    results[0] = float4(rotateX(float3(1.0, 1.0, 0.0), HALF_PI), 0.0);
}

// @test rotateY3
// @expect rotateY3[0] 0.0 1.0 -1.0 0.0
kernel void rotateY3(device float4* results [[buffer(0)]]) {
    results[0] = float4(rotateY(float3(1.0, 1.0, 0.0), HALF_PI), 0.0);
}

// @test rotateZ3
// @xfail rotateZ3[0] WGSL kept the rotation direction from before upstream commit 9a3a036c; MSL now matches GLSL
// @expect rotateZ3[0] 0.0 -1.0 1.0 0.0
kernel void rotateZ3(device float4* results [[buffer(0)]]) {
    results[0] = float4(rotateZ(float3(1.0, 0.0, 1.0), HALF_PI), 0.0);
}

// @test rotate
// @expect rotate[0] 0.5 1.0
kernel void rotate(device float4* results [[buffer(0)]]) {
    results[0] = float4(rotate(float2(1.0, 0.5), HALF_PI), 0.0, 0.0);
}

// @test rotate_c
// @expect rotate_c[0] 0.0 1.0
kernel void rotate_c(device float4* results [[buffer(0)]]) {
    results[0] = float4(rotate(float2(1.0, 0.0), HALF_PI, float2(0.0)), 0.0, 0.0);
}

// results[0] = (rotate3(...), 0), results[1].x = its length
// @test rotate3 2
// @xfail rotate3[0] WGSL kept the rotation direction from before upstream commit 9a3a036c; MSL now matches GLSL
// @expect rotate3[0] 0.0 -1.0 0.0 0.0
// @expect rotate3[1] 1.0
kernel void rotate3(device float4* results [[buffer(0)]]) {
    float3 r = rotate(float3(1.0, 0.0, 0.0), HALF_PI, float3(0.0, 0.0, 1.0));
    results[0] = float4(r, 0.0);
    results[1] = float4(length(r), 0.0, 0.0, 0.0);
}

// WGSL result[0..3] = (r1.vAxis0.x, r1.blendAlpha, r2.blendAlpha, abs(r2.vAxis0.x - r2.vAxis1.x));
// its range checks (0.2 < r[2] < 0.8, r[3] > 0.01) are implied by the exact values.
// @test bracketing
// @expect bracketing[0] 1.0 0.0 0.5 0.01231
kernel void bracketing(device float4* results [[buffer(0)]]) {
    float2 a0, a1, b0, b1;
    float alpha1, alpha2;
    bracketing(float2(1.0, 0.0), a0, a1, alpha1);
    float angle_between = PI / 40.0;
    bracketing(float2(cos(angle_between), sin(angle_between)), b0, b1, alpha2);
    results[0] = float4(a0.x, alpha1, alpha2, abs(b0.x - b1.x));
}
