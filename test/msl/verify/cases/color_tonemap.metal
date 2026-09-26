// Ported from test/wesl/color-tonemap.test.ts (same inputs and expected values).
// WGSL names encode the type (tonemapX3 / tonemapX4); in MSL they're tonemapX(float3|float4) overloads.
//   tonemapUncharted23/24 (color/tonemap/uncharted2) -> tonemapUncharted2(float3|float4)
//   uncharted2Tonemap (color/tonemap/uncharted)       -> uncharted2Tonemap(float3)
// Kernel names get a _test suffix so they don't clash with the LYGIA functions.
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/tonemap/aces.msl"
#include "lygia/color/tonemap/debug.msl"
#include "lygia/color/tonemap/filmic.msl"
#include "lygia/color/tonemap/linear.msl"
#include "lygia/color/tonemap/reinhard.msl"
#include "lygia/color/tonemap/reinhardJodie.msl"
#include "lygia/color/tonemap/uncharted.msl"
#include "lygia/color/tonemap/uncharted2.msl"
#include "lygia/color/tonemap/unreal.msl"

// tonemapACES3
// @test tonemapACES3_test
// @expect tonemapACES3_test[0] 0.9149 0.8768 0.8038
kernel void tonemapACES3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(tonemapACES(float3(2.0, 1.5, 1.0)), 0.0);
}

// tonemapACES4
// @test tonemapACES4_test
// @expect tonemapACES4_test[0] 0.9149 0.8768 0.8038 0.8
kernel void tonemapACES4_test(device float4* results [[buffer(0)]]) {
    results[0] = tonemapACES(float4(2.0, 1.5, 1.0, 0.8));
}

// tonemapDebug3
// @test tonemapDebug3_test
// @expect tonemapDebug3_test[0] 0.5718 0.9076 0.0
kernel void tonemapDebug3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(tonemapDebug(float3(1.5, 1.0, 0.5)), 0.0);
}

// tonemapFilmic3
// @test tonemapFilmic3_test
// @expect tonemapFilmic3_test[0] 0.9128 0.8874 0.8412
kernel void tonemapFilmic3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(tonemapFilmic(float3(2.0, 1.5, 1.0)), 0.0);
}

// tonemapLinear3
// @test tonemapLinear3_test
// @expect tonemapLinear3_test[0] 2.0 1.5 1.0
kernel void tonemapLinear3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(tonemapLinear(float3(2.0, 1.5, 1.0)), 0.0);
}

// tonemapReinhard3
// @test tonemapReinhard3_test
// @expect tonemapReinhard3_test[0] 0.7782 0.5836 0.3891
kernel void tonemapReinhard3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(tonemapReinhard(float3(2.0, 1.5, 1.0)), 0.0);
}

// tonemapReinhardJodie3
// @test tonemapReinhardJodie3_test
// @expect tonemapReinhardJodie3_test[0] 0.7038 0.5935 0.4445
kernel void tonemapReinhardJodie3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(tonemapReinhardJodie(float3(2.0, 1.5, 1.0)), 0.0);
}

// tonemapUncharted3
// @test tonemapUncharted3_test
// @expect tonemapUncharted3_test[0] 0.7132 0.6208 0.4929
kernel void tonemapUncharted3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(tonemapUncharted(float3(2.0, 1.5, 1.0)), 0.0);
}

// tonemapUncharted23
// @test tonemapUncharted23_test
// @expect tonemapUncharted23_test[0] 0.4929 0.4086 0.3043
kernel void tonemapUncharted23_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(tonemapUncharted2(float3(2.0, 1.5, 1.0)), 0.0);
}

// tonemapUnreal3
// @test tonemapUnreal3_test
// @expect tonemapUnreal3_test[0] 0.9457 0.9236 0.8823
kernel void tonemapUnreal3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(tonemapUnreal(float3(2.0, 1.5, 1.0)), 0.0);
}

// tonemapDebug4
// @test tonemapDebug4_test
// @expect tonemapDebug4_test[0] 0.5718 0.9076 0.0 0.7
kernel void tonemapDebug4_test(device float4* results [[buffer(0)]]) {
    results[0] = tonemapDebug(float4(1.5, 1.0, 0.5, 0.7));
}

// tonemapFilmic4
// @test tonemapFilmic4_test
// @expect tonemapFilmic4_test[0] 0.9128 0.8874 0.8412 0.8
kernel void tonemapFilmic4_test(device float4* results [[buffer(0)]]) {
    results[0] = tonemapFilmic(float4(2.0, 1.5, 1.0, 0.8));
}

// tonemapLinear4
// @test tonemapLinear4_test
// @expect tonemapLinear4_test[0] 2.0 1.5 1.0 0.5
kernel void tonemapLinear4_test(device float4* results [[buffer(0)]]) {
    results[0] = tonemapLinear(float4(2.0, 1.5, 1.0, 0.5));
}

// tonemapReinhard4
// @test tonemapReinhard4_test
// @expect tonemapReinhard4_test[0] 0.7782 0.5836 0.3891 0.6
kernel void tonemapReinhard4_test(device float4* results [[buffer(0)]]) {
    results[0] = tonemapReinhard(float4(2.0, 1.5, 1.0, 0.6));
}

// tonemapReinhardJodie4
// @test tonemapReinhardJodie4_test
// @expect tonemapReinhardJodie4_test[0] 0.7038 0.5935 0.4445 0.75
kernel void tonemapReinhardJodie4_test(device float4* results [[buffer(0)]]) {
    results[0] = tonemapReinhardJodie(float4(2.0, 1.5, 1.0, 0.75));
}

// tonemapUncharted4
// @test tonemapUncharted4_test
// @expect tonemapUncharted4_test[0] 0.7132 0.6208 0.4929 0.9
kernel void tonemapUncharted4_test(device float4* results [[buffer(0)]]) {
    results[0] = tonemapUncharted(float4(2.0, 1.5, 1.0, 0.9));
}

// uncharted2Tonemap
// @test uncharted2Tonemap_test
// @expect uncharted2Tonemap_test[0] 0.3574 0.2963 0.2207
kernel void uncharted2Tonemap_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(uncharted2Tonemap(float3(2.0, 1.5, 1.0)), 0.0);
}

// tonemapUncharted24
// @test tonemapUncharted24_test
// @expect tonemapUncharted24_test[0] 0.4929 0.4086 0.3043 0.85
kernel void tonemapUncharted24_test(device float4* results [[buffer(0)]]) {
    results[0] = tonemapUncharted2(float4(2.0, 1.5, 1.0, 0.85));
}

// tonemapUnreal4
// @test tonemapUnreal4_test
// @expect tonemapUnreal4_test[0] 0.9457 0.9236 0.8823 0.65
kernel void tonemapUnreal4_test(device float4* results [[buffer(0)]]) {
    results[0] = tonemapUnreal(float4(2.0, 1.5, 1.0, 0.65));
}
