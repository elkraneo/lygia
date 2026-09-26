// Ported from test/wesl/color-blend-inversion.test.ts (same inputs and expected values).
// Only blendReflect3, which the WGSL test checks with tolerance 0.01; the rest are in color_blend_inversion.metal.
// WGSL blendReflect3(vec3f, vec3f) == MSL blendReflect(float3, float3)
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/blend/reflect.msl"

// @eps 0.01

// WGSL blendReflect3
// @test blendReflect3_test
// @expect blendReflect3_test[0] 0.32 0.514 0.2
kernel void blendReflect3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(blendReflect(float3(0.4, 0.6, 0.2), float3(0.5, 0.3, 0.8)), 0.0);
}
