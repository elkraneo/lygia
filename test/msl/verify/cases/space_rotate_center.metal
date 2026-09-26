// Ported from test/wesl/space-rotate.test.ts (same inputs and expected values): the tests that set
// CENTER_2D = vec2f(0.3, 0.3) / CENTER_3D = vec3f(0.5, 0.5, 0.5) through WESL conditions and
// constants, which are #defines in MSL. The other tests are in space_rotate.metal.
#include <metal_stdlib>
using namespace metal;
#define CENTER_2D float2(0.3, 0.3)
#define CENTER_3D float3(0.5, 0.5, 0.5)
#include "lygia/math/const.msl"
#include "lygia/space/rotateX.msl"
#include "lygia/space/rotateY.msl"
#include "lygia/space/rotateZ.msl"
#include "lygia/space/rotate.msl"

// "rotate - with custom CENTER_2D via constants"
// @test rotateCenter2D
// @expect rotateCenter2D[0] 0.3 0.8
kernel void rotateCenter2D(device float4* results [[buffer(0)]]) {
    results[0] = float4(rotate(float2(0.8, 0.3), HALF_PI), 0.0, 0.0);
}

// "rotateX3 - with custom CENTER_3D via constants"
// @test rotateX3Center
// @xfail rotateX3Center[0] WGSL kept the rotation direction from before upstream commit 9a3a036c; MSL now matches GLSL
// @expect rotateX3Center[0] 1.0 0.5 -0.5 0.0
kernel void rotateX3Center(device float4* results [[buffer(0)]]) {
    results[0] = float4(rotateX(float3(1.0, 1.5, 0.5), HALF_PI), 0.0);
}

// "rotateY3 - with custom CENTER_3D via constants"
// @test rotateY3Center
// @expect rotateY3Center[0] 0.5 1.0 -0.5 0.0
kernel void rotateY3Center(device float4* results [[buffer(0)]]) {
    results[0] = float4(rotateY(float3(1.5, 1.0, 0.5), HALF_PI), 0.0);
}

// "rotateZ3 - with custom CENTER_3D via constants"
// @test rotateZ3Center
// @xfail rotateZ3Center[0] WGSL kept the rotation direction from before upstream commit 9a3a036c; MSL now matches GLSL
// @expect rotateZ3Center[0] 0.5 -0.5 1.0 0.0
kernel void rotateZ3Center(device float4* results [[buffer(0)]]) {
    results[0] = float4(rotateZ(float3(1.5, 0.5, 1.0), HALF_PI), 0.0);
}
