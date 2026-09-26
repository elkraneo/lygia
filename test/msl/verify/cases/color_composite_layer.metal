// Ported from test/wesl/color-composite-layer.test.ts (same inputs and expected values).
// WGSL layerXSourceOver4(vec4f, vec4f)  ==  MSL layerXSourceOver(float4, float4)
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/layer/multiplySourceOver.msl"
#include "lygia/color/layer/screenSourceOver.msl"
#include "lygia/color/layer/addSourceOver.msl"
#include "lygia/color/layer/overlaySourceOver.msl"
#include "lygia/color/layer/darkenSourceOver.msl"
#include "lygia/color/layer/lightenSourceOver.msl"
#include "lygia/color/layer/differenceSourceOver.msl"
#include "lygia/color/layer/exclusionSourceOver.msl"
#include "lygia/color/layer/phoenixSourceOver.msl"
#include "lygia/color/layer/subtractSourceOver.msl"

// @test layerMultiplySourceOver4_test
// @expect layerMultiplySourceOver4_test[0] 0.3625 0.4025 0.3825 0.875
kernel void layerMultiplySourceOver4_test(device float4* results [[buffer(0)]]) {
    results[0] = layerMultiplySourceOver(float4(0.8, 0.6, 0.4, 0.75), float4(0.5, 0.7, 0.9, 0.5));
}

// @test layerScreenSourceOver4_test
// @expect layerScreenSourceOver4_test[0] 0.45 0.5 0.59 0.8
kernel void layerScreenSourceOver4_test(device float4* results [[buffer(0)]]) {
    results[0] = layerScreenSourceOver(float4(0.6, 0.4, 0.2, 0.5), float4(0.3, 0.5, 0.7, 0.6));
}

// @test layerAddSourceOver4_test
// @expect layerAddSourceOver4_test[0] 0.34 0.48 0.62 0.8
kernel void layerAddSourceOver4_test(device float4* results [[buffer(0)]]) {
    results[0] = layerAddSourceOver(float4(0.3, 0.4, 0.5, 0.6), float4(0.2, 0.3, 0.4, 0.5));
}

// @test layerOverlaySourceOver4_test
// @expect layerOverlaySourceOver4_test[0] 0.544 0.336 0.44 0.88
kernel void layerOverlaySourceOver4_test(device float4* results [[buffer(0)]]) {
    results[0] = layerOverlaySourceOver(float4(0.7, 0.3, 0.5, 0.8), float4(0.4, 0.6, 0.5, 0.4));
}

// @test layerDarkenSourceOver4_test
// @expect layerDarkenSourceOver4_test[0] 0.3 0.3 0.375 0.75
kernel void layerDarkenSourceOver4_test(device float4* results [[buffer(0)]]) {
    results[0] = layerDarkenSourceOver(float4(0.3, 0.7, 0.5, 0.5), float4(0.6, 0.4, 0.5, 0.5));
}

// @test layerLightenSourceOver4_test
// @expect layerLightenSourceOver4_test[0] 0.45 0.45 0.375 0.75
kernel void layerLightenSourceOver4_test(device float4* results [[buffer(0)]]) {
    results[0] = layerLightenSourceOver(float4(0.3, 0.7, 0.5, 0.5), float4(0.6, 0.4, 0.5, 0.5));
}

// @test layerDifferenceSourceOver4_test
// @expect layerDifferenceSourceOver4_test[0] 0.3 0.406 0.212 0.88
kernel void layerDifferenceSourceOver4_test(device float4* results [[buffer(0)]]) {
    results[0] = layerDifferenceSourceOver(float4(0.8, 0.3, 0.6, 0.7), float4(0.5, 0.7, 0.4, 0.6));
}

// @test layerExclusionSourceOver4_test
// @expect layerExclusionSourceOver4_test[0] 0.345 0.445 0.39 0.75
kernel void layerExclusionSourceOver4_test(device float4* results [[buffer(0)]]) {
    results[0] = layerExclusionSourceOver(float4(0.6, 0.4, 0.8, 0.5), float4(0.3, 0.7, 0.2, 0.5));
}

// @test layerPhoenixSourceOver4_test
// @expect layerPhoenixSourceOver4_test[0] 0.484 0.636 0.428 0.76
kernel void layerPhoenixSourceOver4_test(device float4* results [[buffer(0)]]) {
    results[0] = layerPhoenixSourceOver(float4(0.7, 0.5, 0.3, 0.6), float4(0.4, 0.6, 0.8, 0.4));
}

// @test layerSubtractSourceOver4_test
// @expect layerSubtractSourceOver4_test[0] 0.2 0.2 0.175 0.75
kernel void layerSubtractSourceOver4_test(device float4* results [[buffer(0)]]) {
    results[0] = layerSubtractSourceOver(float4(0.8, 0.5, 0.3, 0.5), float4(0.4, 0.6, 0.7, 0.5));
}
