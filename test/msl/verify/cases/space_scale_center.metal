// Ported from test/wesl/space-scale.test.ts (same inputs and expected values): the tests that set
// CENTER_2D = vec2f(0.3, 0.7) / CENTER_3D = vec3f(0.2, 0.3, 0.4) through WESL conditions and
// constants, which are #defines in MSL. The other tests are in space_scale.metal.
#include <metal_stdlib>
using namespace metal;
#define CENTER_2D float2(0.3, 0.7)
#define CENTER_3D float3(0.2, 0.3, 0.4)
#include "lygia/space/scale.msl"

// "scale2 - with custom CENTER_2D via constants"
// @test scale2Center
// @expect scale2Center[0] 1.3 1.1
kernel void scale2Center(device float4* results [[buffer(0)]]) {
    results[0] = float4(scale(float2(0.8, 0.9), float2(2.0, 2.0)), 0.0, 0.0);
}

// "scale3 - with custom CENTER_3D via constants"
// @test scale3Center
// @expect scale3Center[0] 1.2 1.8 0.65 0.0
kernel void scale3Center(device float4* results [[buffer(0)]]) {
    results[0] = float4(scale(float3(0.7, 0.8, 0.9), float3(2.0, 3.0, 0.5)), 0.0);
}
