// Ported from test/wesl/color-distance.test.ts (same inputs and expected values).
// WGSL colorDistance(vec3f) / colorDistance4(vec4f)  ==  MSL colorDistance(float3) / colorDistance(float4)
// (both default to colorDistanceLABCIE94 in WESL and MSL; COLORDISTANCE_FNC is left at its default).
// Other names are the same in both. f32 results[k] -> results[k].x here; the vec4f results map 1:1.
// Relational expects (toBeGreaterThan/toBeLessThan) are computed in-kernel as 1/0 flags in results[1].
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/distance.msl"

// Note: MSL rgb2xyz returns XYZ in 0-100 like the WESL fix (#271); GLSL still returns 0-1.
// @test colorDistance_test
// @expect colorDistance_test[0] 71.0491
kernel void colorDistance_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(colorDistance(float3(1.0, 0.0, 0.0), float3(0.0, 0.0, 1.0)), 0.0, 0.0, 0.0);
}

// @test colorDistance4_test
// @expect colorDistance4_test[0] 71.0491
kernel void colorDistance4_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(colorDistance(float4(1.0, 0.0, 0.0, 0.8), float4(0.0, 0.0, 1.0, 0.6)), 0.0, 0.0, 0.0);
}

// @test colorDistanceLAB_test
// @expect colorDistanceLAB_test[0] 176.314
kernel void colorDistanceLAB_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(colorDistanceLAB(float3(1.0, 0.0, 0.0), float3(0.0, 0.0, 1.0)), 0.0, 0.0, 0.0);
}

// @test colorDistanceLABCIE94_test
// @expect colorDistanceLABCIE94_test[0] 10.0626
kernel void colorDistanceLABCIE94_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(colorDistanceLABCIE94(float3(0.0, 1.0, 0.0), float3(1.0, 1.0, 0.0)), 0.0, 0.0, 0.0);
}

// results[1]: x = redToOrange < redToBlue, y = redToOrange > 0.05, z = redToOrange < 0.3, w = redToBlue > 0.3
// @test colorDistanceOKLAB_test 2
// @expect colorDistanceOKLAB_test[0] 0.2917 0.5371
// @expect colorDistanceOKLAB_test[1] 1 1 1 1
kernel void colorDistanceOKLAB_test(device float4* results [[buffer(0)]]) {
    float3 red = float3(1.0, 0.0, 0.0), orange = float3(1.0, 0.5, 0.0), blue = float3(0.0, 0.0, 1.0);
    float redToOrange = colorDistanceOKLAB(red, orange);
    float redToBlue = colorDistanceOKLAB(red, blue);
    results[0] = float4(redToOrange, redToBlue, 0.0, 0.0);
    results[1] = float4(redToOrange < redToBlue ? 1.0 : 0.0, redToOrange > 0.05 ? 1.0 : 0.0,
                        redToOrange < 0.3 ? 1.0 : 0.0, redToBlue > 0.3 ? 1.0 : 0.0);
}

// results[1]: x = chromaDist > 0.5, y = chromaDist < 1.5
// @test colorDistanceYCbCr_test 2
// @expect colorDistanceYCbCr_test[0] 0.5316 0.0
// @expect colorDistanceYCbCr_test[1] 1 1
kernel void colorDistanceYCbCr_test(device float4* results [[buffer(0)]]) {
    float chromaDist = colorDistanceYCbCr(float3(0.8, 0.2, 0.2), float3(0.2, 0.2, 0.8));
    float lumaDist = colorDistanceYCbCr(float3(0.3, 0.3, 0.3), float3(0.7, 0.7, 0.7));
    results[0] = float4(chromaDist, lumaDist, 0.0, 0.0);
    results[1] = float4(chromaDist > 0.5 ? 1.0 : 0.0, chromaDist < 1.5 ? 1.0 : 0.0, 0.0, 0.0);
}

// results[0] = (complementaryDist, similarDist, dist1, dist2); dist1 == dist2 is checked with @same on results[1..2].
// results[3]: x = complementaryDist > similarDist, y = complementaryDist > 0.5
// @test colorDistanceYPbPr_test 4
// @expect colorDistanceYPbPr_test[0] 0.9919 0.5957
// @same colorDistanceYPbPr_test[1] colorDistanceYPbPr_test[2]
// @expect colorDistanceYPbPr_test[3] 1 1
kernel void colorDistanceYPbPr_test(device float4* results [[buffer(0)]]) {
    float3 magenta = float3(1.0, 0.0, 1.0), cyan = float3(0.0, 1.0, 1.0), blue = float3(0.0, 0.0, 1.0);
    float complementaryDist = colorDistanceYPbPr(magenta, cyan);
    float similarDist = colorDistanceYPbPr(cyan, blue);
    float dist1 = colorDistanceYPbPr(magenta, cyan);
    float dist2 = colorDistanceYPbPr(cyan, magenta);
    results[0] = float4(complementaryDist, similarDist, dist1, dist2);
    results[1] = float4(dist1, 0.0, 0.0, 0.0);
    results[2] = float4(dist2, 0.0, 0.0, 0.0);
    results[3] = float4(complementaryDist > similarDist ? 1.0 : 0.0, complementaryDist > 0.5 ? 1.0 : 0.0, 0.0, 0.0);
}

// @test colorDistanceYUV_test
// @expect colorDistanceYUV_test[0] 0.5
kernel void colorDistanceYUV_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(colorDistanceYUV(float3(1.0, 1.0, 1.0), float3(0.5, 0.5, 0.5)), 0.0, 0.0, 0.0);
}
