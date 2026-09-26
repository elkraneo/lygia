// Ported from test/wesl/space-view.test.ts (same inputs and expected values).
// Name mapping (WGSL -> MSL): lookAtViewRoll(pos, center, roll) -> lookAtView(float3, float3, float);
// lookAtViewFromDirection(pos, dir) -> lookAtView(float3, float3) (both use dir as the target, up +Y).
// Note: MSL lookAt(eye, target, up) (used by lookAtView) computes x = cross(z, up), while WESL lookAt3
// goes through lookAt(forward, up) (x = cross(up, z) unless LOOK_AT_RIGHT_HANDED); the tests only
// check the translation column, so this doesn't show here.
#include <metal_stdlib>
using namespace metal;
#include "lygia/math/const.msl"
#include "lygia/space/eulerView.msl"
#include "lygia/space/lookAt.msl"
#include "lygia/space/lookAtView.msl"

// @test eulerView
// @expect eulerView[0] 0.0 0.0 -1.0 1.0
kernel void eulerView(device float4* results [[buffer(0)]]) {
    float4x4 m = eulerView(float3(0.0, 0.0, 0.0), float3(0.0, HALF_PI, 0.0));
    results[0] = m * float4(1.0, 0.0, 0.0, 1.0);
}

// @test lookAt
// @expect lookAt[0] 0.0 0.0 -1.0 0.0
kernel void lookAt(device float4* results [[buffer(0)]]) {
    float3x3 m = lookAt(float3(0.0, 0.0, -1.0), float3(0.0, 1.0, 0.0));
    results[0] = float4(m * float3(0.0, 0.0, 1.0), 0.0);
}

// @test lookAtView
// @expect lookAtView[0] 5.0 0.0 0.0 1.0
kernel void lookAtView(device float4* results [[buffer(0)]]) {
    float4x4 m = lookAtView(float3(5.0, 0.0, 0.0), float3(0.0, 0.0, 0.0), float3(0.0, 1.0, 0.0));
    results[0] = m * float4(0.0, 0.0, 0.0, 1.0);
}

// @test lookAtViewRoll
// @expect lookAtViewRoll[0] 0.0 5.0 0.0 1.0
kernel void lookAtViewRoll(device float4* results [[buffer(0)]]) {
    float4x4 m = lookAtView(float3(0.0, 5.0, 0.0), float3(0.0, 0.0, 0.0), float(HALF_PI));
    results[0] = m * float4(0.0, 0.0, 0.0, 1.0);
}

// @test lookAtViewFromDirection
// @expect lookAtViewFromDirection[0] 3.0 0.0 0.0 1.0
kernel void lookAtViewFromDirection(device float4* results [[buffer(0)]]) {
    float4x4 m = lookAtView(float3(3.0, 0.0, 0.0), float3(1.0, 0.0, 0.0));
    results[0] = m * float4(0.0, 0.0, 0.0, 1.0);
}
