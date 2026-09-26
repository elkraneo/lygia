// Ported from test/wesl/color-blend-opacity-color.test.ts (same inputs and expected values).
// The two tests that use expectCloseTo(..., 0.1); blendHueOpacity (default epsilon) is in
// color_blend_opacity_color.metal.
// WGSL blendSaturationOpacity / blendLuminosityOpacity(vec3f, vec3f, f32)
//   ==  MSL blendSaturation / blendLuminosity(float3, float3, float)
// The WGSL vec3f result is results[0].xyz here; the relational expects are computed
// in-kernel as 1/0 flags in results[1].
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/blend/saturation.msl"
#include "lygia/color/blend/luminosity.msl"
// @eps 0.1

// results[1]: x = (r > g), y = (r > b)
// @test blendSaturationOpacity_test 2
// @expect blendSaturationOpacity_test[0] 1.0 0.5 0.5
// @expect blendSaturationOpacity_test[1] 1 1
kernel void blendSaturationOpacity_test(device float4* results [[buffer(0)]]) {
    float3 r = blendSaturation(float3(1.0, 0.0, 0.0), float3(0.5, 0.5, 0.5), 0.5);
    results[0] = float4(r, 0.0);
    results[1] = float4(r.x > r.y ? 1.0 : 0.0, r.x > r.z ? 1.0 : 0.0, 0.0, 0.0);
}

// results[1]: x = (r > g), y = (r > b), z = (r < 0.7)
// @test blendLuminosityOpacity_test 2
// @expect blendLuminosityOpacity_test[0] 0.55 0.05 0.05
// @expect blendLuminosityOpacity_test[1] 1 1 1
kernel void blendLuminosityOpacity_test(device float4* results [[buffer(0)]]) {
    float3 r = blendLuminosity(float3(1.0, 0.0, 0.0), float3(0.1, 0.1, 0.1), 0.5);
    results[0] = float4(r, 0.0);
    results[1] = float4(r.x > r.y ? 1.0 : 0.0, r.x > r.z ? 1.0 : 0.0, r.x < 0.7 ? 1.0 : 0.0, 0.0);
}
