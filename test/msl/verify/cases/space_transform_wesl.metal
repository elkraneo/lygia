// Ported from test/wesl/shaders/space_transform.test.wesl (same inputs and expected values).
// The WESL test uses expectNear/expectNearVec2 (relative 1e-3, absolute 1e-6); here the default
// absolute 0.0001 is used, which is at least as strict for these values (the exact-zero
// expectations are within 1e-6 in WGSL; they are exact here too).
// WGSL center2/uncenter2/flipY2(vec2f) are the MSL overloads center/uncenter/flipY(float2).
#include <metal_stdlib>
using namespace metal;
#include "lygia/space/center.msl"
#include "lygia/space/uncenter.msl"
#include "lygia/space/flipY.msl"
#include "lygia/space/aspect.msl"
#include "lygia/space/ratio.msl"
#include "lygia/space/unratio.msl"

// results[0] = (centerScalar, uncenterScalar), results[1] = centerVec2.xy, uncenterVec2.xy
// @test centerUncenter 2
// @expect centerUncenter[0] 0.0 0.5
// @expect centerUncenter[1] 0.0 1.0 0.0 1.0
kernel void centerUncenter(device float4* results [[buffer(0)]]) {
    results[0] = float4(center(0.5), uncenter(0.0), 0.0, 0.0);
    results[1] = float4(center(float2(0.5, 1.0)), uncenter(float2(-1.0, 1.0)));
}

// @test flipYVec2
// @expect flipYVec2[0] 0.5 0.75
kernel void flipYVec2(device float4* results [[buffer(0)]]) {
    results[0] = float4(flipY(float2(0.5, 0.25)), 0.0, 0.0);
}

// aspectWidescreen: (0.5 * 1920 / 1080, 0.5)
// @test aspectWidescreen
// @expect aspectWidescreen[0] 0.88888889 0.5
kernel void aspectWidescreen(device float4* results [[buffer(0)]]) {
    results[0] = float4(aspect(float2(0.5, 0.5), float2(1920.0, 1080.0)), 0.0, 0.0);
}

// @test ratioWidescreen
// @expect ratioWidescreen[0] 0.5 0.5
kernel void ratioWidescreen(device float4* results [[buffer(0)]]) {
    results[0] = float4(ratio(float2(0.5, 0.5), float2(1920.0, 1080.0)), 0.0, 0.0);
}

// @test unratioWidescreen
// @expect unratioWidescreen[0] 0.5 0.5
kernel void unratioWidescreen(device float4* results [[buffer(0)]]) {
    results[0] = float4(unratio(float2(0.5, 0.5), float2(1920.0, 1080.0)), 0.0, 0.0);
}
