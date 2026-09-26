// Ported from test/wesl/color-space-lab-lch.test.ts (same inputs and expected values).
// WGSL names encode the argument type; in MSL they're overloads:
//   WGSL xxx(vec3f) / xxx4(vec4f)  ==  MSL xxx(float3) / xxx(float4)
// vec3f results are written as float4(result, 0) and only x y z are checked.
// No config #defines: WESL and MSL share the defaults (D65 white, 0-100 L and XYZ scale).
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/space/lab2srgb.msl"
#include "lygia/color/space/lab2rgb.msl"
#include "lygia/color/space/srgb2lab.msl"
#include "lygia/color/space/rgb2lab.msl"
#include "lygia/color/space/lch2srgb.msl"
#include "lygia/color/space/lch2rgb.msl"
#include "lygia/color/space/srgb2lch.msl"
#include "lygia/color/space/rgb2lch.msl"
#include "lygia/color/space/lab2lch.msl"
#include "lygia/color/space/lch2lab.msl"
#include "lygia/color/space/lab2xyz.msl"
#include "lygia/color/space/xyz2lab.msl"

// @test lab2srgb_test
// @expect lab2srgb_test[0] 0.5524 0.413 0.634
kernel void lab2srgb_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(lab2srgb(float3(50.0, 25.0, -25.0)), 0.0);
}

// @test lab2rgb_test
// @expect lab2rgb_test[0] 1.0 0.0 0.0
kernel void lab2rgb_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(lab2rgb(float3(53.24, 80.09, 67.20)), 0.0);
}

// Note: MSL rgb2xyz returns XYZ in 0-100 like the WESL fix (#271); GLSL still returns 0-1.
// @test srgb2lab_test
// @expect srgb2lab_test[0] 53.2408 80.0925 67.2032
kernel void srgb2lab_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(srgb2lab(float3(1.0, 0.0, 0.0)), 0.0);
}

// @test rgb2lab_test
// @expect rgb2lab_test[0] 76.0693 0 0.00001
kernel void rgb2lab_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(rgb2lab(float3(0.5, 0.5, 0.5)), 0.0);
}

// WGSL test "lch2srgb3"
// @test lch2srgb_test
// @expect lch2srgb_test[0] 0.4277 0.4903 0.2895
kernel void lch2srgb_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(lch2srgb(float3(50.0, 30.0, 120.0)), 0.0);
}

// @test lch2rgb_test
// @expect lch2rgb_test[0] 1.0 0.0 0.0
kernel void lch2rgb_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(lch2rgb(float3(53.24, 104.55, 40.0)), 0.0);
}

// @test srgb2lch_test
// @expect srgb2lch_test[0] 53.2408 104.5518 39.999
kernel void srgb2lch_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(srgb2lch(float3(1.0, 0.0, 0.0)), 0.0);
}

// @test rgb2lch_test
// @expect rgb2lch_test[0] 53.2408 104.5518 39.999
kernel void rgb2lch_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(rgb2lch(float3(1.0, 0.0, 0.0)), 0.0);
}

// @test lab2lch_test
// @expect lab2lch_test[0] 50.0 35.3553 45.0
kernel void lab2lch_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(lab2lch(float3(50.0, 25.0, 25.0)), 0.0);
}

// @test lch2lab_test
// @expect lch2lab_test[0] 50.0 25.0033 25.0033
kernel void lch2lab_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(lch2lab(float3(50.0, 35.36, 45.0)), 0.0);
}

// @test lab2xyz_test
// @expect lab2xyz_test[0] 17.5061 18.4186 20.059
kernel void lab2xyz_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(lab2xyz(float3(50.0, 0.0, 0.0)), 0.0);
}

// @test xyz2lab_test
// @expect xyz2lab_test[0] 49.9777 0.0615 0.0653
kernel void xyz2lab_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(xyz2lab(float3(17.5, 18.4, 20.0)), 0.0);
}

// WGSL lab2lch4
// @test lab2lch4_test
// @expect lab2lch4_test[0] 50.0 35.3553 45.0 0.75
kernel void lab2lch4_test(device float4* results [[buffer(0)]]) {
    results[0] = lab2lch(float4(50.0, 25.0, 25.0, 0.75));
}

// WGSL lab2rgb4
// @test lab2rgb4_test
// @expect lab2rgb4_test[0] 1.0 0.0 0.0 0.4
kernel void lab2rgb4_test(device float4* results [[buffer(0)]]) {
    results[0] = lab2rgb(float4(53.24, 80.09, 67.20, 0.4));
}

// WGSL lab2srgb4
// @test lab2srgb4_test
// @expect lab2srgb4_test[0] 0.5524 0.413 0.634 0.85
kernel void lab2srgb4_test(device float4* results [[buffer(0)]]) {
    results[0] = lab2srgb(float4(50.0, 25.0, -25.0, 0.85));
}

// WGSL lab2xyz4
// @test lab2xyz4_test
// @expect lab2xyz4_test[0] 17.5061 18.4186 20.059 0.3
kernel void lab2xyz4_test(device float4* results [[buffer(0)]]) {
    results[0] = lab2xyz(float4(50.0, 0.0, 0.0, 0.3));
}

// WGSL lch2lab4
// @test lch2lab4_test
// @expect lch2lab4_test[0] 50.0 25.0033 25.0033 0.95
kernel void lch2lab4_test(device float4* results [[buffer(0)]]) {
    results[0] = lch2lab(float4(50.0, 35.36, 45.0, 0.95));
}

// WGSL lch2rgb4
// @test lch2rgb4_test
// @expect lch2rgb4_test[0] 1.0 0.0 0.0 0.2
kernel void lch2rgb4_test(device float4* results [[buffer(0)]]) {
    results[0] = lch2rgb(float4(53.24, 104.55, 40.0, 0.2));
}

// WGSL lch2srgb4
// @test lch2srgb4_test
// @expect lch2srgb4_test[0] 0.4277 0.4903 0.2895 0.65
kernel void lch2srgb4_test(device float4* results [[buffer(0)]]) {
    results[0] = lch2srgb(float4(50.0, 30.0, 120.0, 0.65));
}

// WGSL rgb2lab4
// @test rgb2lab4_test
// @expect rgb2lab4_test[0] 76.0693 0 0.00001 0.7
kernel void rgb2lab4_test(device float4* results [[buffer(0)]]) {
    results[0] = rgb2lab(float4(0.5, 0.5, 0.5, 0.7));
}

// WGSL rgb2lch4
// @test rgb2lch4_test
// @expect rgb2lch4_test[0] 53.2408 104.5518 39.999 0.8
kernel void rgb2lch4_test(device float4* results [[buffer(0)]]) {
    results[0] = rgb2lch(float4(1.0, 0.0, 0.0, 0.8));
}

// WGSL srgb2lab4
// @test srgb2lab4_test
// @expect srgb2lab4_test[0] 53.2408 80.0925 67.2032 0.75
kernel void srgb2lab4_test(device float4* results [[buffer(0)]]) {
    results[0] = srgb2lab(float4(1.0, 0.0, 0.0, 0.75));
}

// WGSL srgb2lch4
// @test srgb2lch4_test
// @expect srgb2lch4_test[0] 53.2408 104.5518 39.999 0.5
kernel void srgb2lch4_test(device float4* results [[buffer(0)]]) {
    results[0] = srgb2lch(float4(1.0, 0.0, 0.0, 0.5));
}

// WGSL xyz2lab4
// @test xyz2lab4_test
// @expect xyz2lab4_test[0] 49.9777 0.06151 0.06528 0.9
kernel void xyz2lab4_test(device float4* results [[buffer(0)]]) {
    results[0] = xyz2lab(float4(17.5, 18.4, 20.0, 0.9));
}
