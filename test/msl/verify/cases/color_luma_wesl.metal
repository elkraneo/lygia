// Ported from test/wesl/shaders/color_luma.test.wesl (same inputs and expected values).
// The WESL test uses expectNear/expectNearVec2 (relative 1e-3, absolute 1e-6); here the
// default absolute 0.0001 is used, which is stricter for these values.
// WGSL luminance4(vec4f) == MSL luminance(float4); WGSL luma/luma3/luma4 == MSL luma(float/float3/float4).
// Both WESL and MSL luma use rgb2luma with Rec709 weights.
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/luminance.msl"
#include "lygia/color/luma.msl"

// @test luminanceOrange
// @expect luminanceOrange[0] 0.5702
kernel void luminanceOrange(device float4* results [[buffer(0)]]) {
    results[0] = float4(luminance(float3(1.0, 0.5, 0.0)), 0.0, 0.0, 0.0);
}

// @test luminance4Orange
// @expect luminance4Orange[0] 0.5702
kernel void luminance4Orange(device float4* results [[buffer(0)]]) {
    results[0] = float4(luminance(float4(1.0, 0.5, 0.0, 0.8)), 0.0, 0.0, 0.0);
}

// @test luma3Orange
// @expect luma3Orange[0] 0.5702
kernel void luma3Orange(device float4* results [[buffer(0)]]) {
    results[0] = float4(luma(float3(1.0, 0.5, 0.0)), 0.0, 0.0, 0.0);
}

// @test lumaGrayscaleConsistency
// @expect lumaGrayscaleConsistency[0] 0.75 0.75
kernel void lumaGrayscaleConsistency(device float4* results [[buffer(0)]]) {
    results[0] = float4(luma(0.75), luma(float3(0.75, 0.75, 0.75)), 0.0, 0.0);
}

// @test luma4Mixed
// @expect luma4Mixed[0] 0.39186
kernel void luma4Mixed(device float4* results [[buffer(0)]]) {
    results[0] = float4(luma(float4(0.8, 0.3, 0.1, 0.6)), 0.0, 0.0, 0.0);
}
