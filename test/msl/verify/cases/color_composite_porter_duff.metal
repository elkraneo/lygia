// Ported from test/wesl/color-composite-porter-duff.test.ts (same inputs and expected values).
// WGSL compositeX4(vec4f, vec4f)            ==  MSL compositeX(float4, float4)
// WGSL compositeX3(vec3f, vec3f, f32, f32)  ==  MSL compositeX(float3, float3, float, float)
// The vec3 tests write vec4f(result, 0.0), as in WGSL.
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/composite/sourceOver.msl"
#include "lygia/color/composite/sourceIn.msl"
#include "lygia/color/composite/sourceOut.msl"
#include "lygia/color/composite/sourceAtop.msl"
#include "lygia/color/composite/destinationOver.msl"
#include "lygia/color/composite/destinationIn.msl"
#include "lygia/color/composite/destinationOut.msl"
#include "lygia/color/composite/destinationAtop.msl"
#include "lygia/color/composite/compositeXor.msl"

// @test compositeSourceOver4_test
// @expect compositeSourceOver4_test[0] 0.5 0.0 0.25 0.75
kernel void compositeSourceOver4_test(device float4* results [[buffer(0)]]) {
    results[0] = compositeSourceOver(float4(1.0, 0.0, 0.0, 0.5), float4(0.0, 0.0, 1.0, 0.5));
}

// @test compositeSourceIn4_test
// @expect compositeSourceIn4_test[0] 0.5 0.0 0.0 0.4
kernel void compositeSourceIn4_test(device float4* results [[buffer(0)]]) {
    results[0] = compositeSourceIn(float4(1.0, 0.0, 0.0, 0.8), float4(0.0, 0.0, 1.0, 0.5));
}

// @test compositeSourceOut4_test
// @expect compositeSourceOut4_test[0] 0.6 0.0 0.0 0.48
kernel void compositeSourceOut4_test(device float4* results [[buffer(0)]]) {
    results[0] = compositeSourceOut(float4(1.0, 0.0, 0.0, 0.8), float4(0.0, 1.0, 0.0, 0.4));
}

// @test compositeSourceAtop4_test
// @expect compositeSourceAtop4_test[0] 0.5 0.0 0.4 0.5
kernel void compositeSourceAtop4_test(device float4* results [[buffer(0)]]) {
    results[0] = compositeSourceAtop(float4(1.0, 0.0, 0.0, 0.6), float4(0.0, 0.0, 1.0, 0.5));
}

// @test compositeDestinationOver4_test
// @expect compositeDestinationOver4_test[0] 0.4 0.0 1.0 0.8
kernel void compositeDestinationOver4_test(device float4* results [[buffer(0)]]) {
    results[0] = compositeDestinationOver(float4(1.0, 0.0, 0.0, 0.5), float4(0.0, 0.0, 1.0, 0.6));
}

// @test compositeDestinationIn4_test
// @expect compositeDestinationIn4_test[0] 0.0 0.6 0.0 0.48
kernel void compositeDestinationIn4_test(device float4* results [[buffer(0)]]) {
    results[0] = compositeDestinationIn(float4(1.0, 0.0, 0.0, 0.6), float4(0.0, 1.0, 0.0, 0.8));
}

// @test compositeDestinationOut4_test
// @expect compositeDestinationOut4_test[0] 0.0 0.7 0.0 0.49
kernel void compositeDestinationOut4_test(device float4* results [[buffer(0)]]) {
    results[0] = compositeDestinationOut(float4(1.0, 0.0, 0.0, 0.3), float4(0.0, 1.0, 0.0, 0.7));
}

// @test compositeDestinationAtop4_test
// @expect compositeDestinationAtop4_test[0] 0.5 0.0 0.7 0.7
kernel void compositeDestinationAtop4_test(device float4* results [[buffer(0)]]) {
    results[0] = compositeDestinationAtop(float4(1.0, 0.0, 0.0, 0.7), float4(0.0, 0.0, 1.0, 0.5));
}

// @test compositeXor4_test
// @expect compositeXor4_test[0] 0.6 0.0 0.4 0.52
kernel void compositeXor4_test(device float4* results [[buffer(0)]]) {
    results[0] = compositeXor(float4(1.0, 0.0, 0.0, 0.6), float4(0.0, 0.0, 1.0, 0.4));
}

// @test compositeSourceOver3_test
// @expect compositeSourceOver3_test[0] 0.5 0.0 0.25 0.0
kernel void compositeSourceOver3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(compositeSourceOver(float3(1.0, 0.0, 0.0), float3(0.0, 0.0, 1.0), 0.5, 0.5), 0.0);
}

// @test compositeSourceIn3_test
// @expect compositeSourceIn3_test[0] 0.6 0.0 0.0 0.0
kernel void compositeSourceIn3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(compositeSourceIn(float3(1.0, 0.0, 0.0), float3(0.0, 0.0, 1.0), 0.8, 0.6), 0.0);
}

// @test compositeSourceOut3_test
// @expect compositeSourceOut3_test[0] 0.7 0.0 0.0 0.0
kernel void compositeSourceOut3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(compositeSourceOut(float3(1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0), 0.8, 0.3), 0.0);
}

// @test compositeSourceAtop3_test
// @expect compositeSourceAtop3_test[0] 0.5 0.0 0.4 0.0
kernel void compositeSourceAtop3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(compositeSourceAtop(float3(1.0, 0.0, 0.0), float3(0.0, 0.0, 1.0), 0.6, 0.5), 0.0);
}

// @test compositeDestinationOver3_test
// @expect compositeDestinationOver3_test[0] 0.4 0.0 1.0 0.0
kernel void compositeDestinationOver3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(compositeDestinationOver(float3(1.0, 0.0, 0.0), float3(0.0, 0.0, 1.0), 0.5, 0.6), 0.0);
}

// @test compositeDestinationIn3_test
// @expect compositeDestinationIn3_test[0] 0.0 0.7 0.0 0.0
kernel void compositeDestinationIn3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(compositeDestinationIn(float3(1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0), 0.7, 0.8), 0.0);
}

// @test compositeDestinationOut3_test
// @expect compositeDestinationOut3_test[0] 0.0 0.6 0.0 0.0
kernel void compositeDestinationOut3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(compositeDestinationOut(float3(1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0), 0.4, 0.7), 0.0);
}

// @test compositeDestinationAtop3_test
// @expect compositeDestinationAtop3_test[0] 0.5 0.0 0.7 0.0
kernel void compositeDestinationAtop3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(compositeDestinationAtop(float3(1.0, 0.0, 0.0), float3(0.0, 0.0, 1.0), 0.7, 0.5), 0.0);
}

// @test compositeXor3_test
// @expect compositeXor3_test[0] 0.6 0.0 0.4 0.0
kernel void compositeXor3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(compositeXor(float3(1.0, 0.0, 0.0), float3(0.0, 0.0, 1.0), 0.6, 0.4), 0.0);
}
