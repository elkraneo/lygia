// Ported from test/wesl/color-adjust-basic.test.ts (same inputs and expected values).
// WGSL names encode the argument type; in MSL they're overloads:
//   WGSL desaturate/desaturate4          == MSL desaturate(float3|float4, float)
//   WGSL contrast/contrast3/contrast4    == MSL contrast(float|float3|float4, float)
//   WGSL vibrance3                       == MSL vibrance(float3, float)
// WGSL matrix[c][r] indexes columns, as does MSL float4x4[c][r].
// hueShift (WGSL tolerance 0.001) is in color_adjust_basic_eps.metal.
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/desaturate.msl"
#include "lygia/color/brightnessMatrix.msl"
#include "lygia/color/contrast.msl"
#include "lygia/color/contrastMatrix.msl"
#include "lygia/color/brightnessContrast.msl"
#include "lygia/color/exposure.msl"
#include "lygia/color/vibrance.msl"

// @test desaturate_test
// @expect desaturate_test[0] 0.7975 0.5475 0.2975
kernel void desaturate_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(desaturate(float3(1.0, 0.5, 0.0), 0.5), 0.0);
}

// WGSL desaturate4
// @test desaturate4_test
// @expect desaturate4_test[0] 0.7975 0.5475 0.2975 0.8
kernel void desaturate4_test(device float4* results [[buffer(0)]]) {
    results[0] = desaturate(float4(1.0, 0.5, 0.0, 0.8), 0.5);
}

// @test brightnessMatrix_test
// @expect brightnessMatrix_test[0] 0.2 0.2 0.2 1.0
kernel void brightnessMatrix_test(device float4* results [[buffer(0)]]) {
    float4x4 m = brightnessMatrix(0.2);
    results[0] = float4(m[3][0], m[3][1], m[3][2], m[3][3]);
}

// @test contrast_test
// @expect contrast_test[0] 0.8
kernel void contrast_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(contrast(0.7, 1.5), 0.0, 0.0, 0.0);
}

// WGSL contrast3
// @test contrast3_test
// @expect contrast3_test[0] 1.1 0.7 0.3
kernel void contrast3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(contrast(float3(0.8, 0.6, 0.4), 2.0), 0.0);
}

// @test contrastMatrix_test
// @expect contrastMatrix_test[0] 1.5 1.5 1.5 -0.25
kernel void contrastMatrix_test(device float4* results [[buffer(0)]]) {
    float4x4 m = contrastMatrix(1.5);
    results[0] = float4(m[0][0], m[1][1], m[2][2], m[3][0]);
}

// @test brightnessContrast_test
// @expect brightnessContrast_test[0] 0.9
kernel void brightnessContrast_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(brightnessContrast(0.7, 0.1, 1.5), 0.0, 0.0, 0.0);
}

// WGSL contrast4
// @test contrast4_test
// @expect contrast4_test[0] 0.95 0.65 0.35 0.9
kernel void contrast4_test(device float4* results [[buffer(0)]]) {
    results[0] = contrast(float4(0.8, 0.6, 0.4, 0.9), 1.5);
}

// @test exposure_test
// @expect exposure_test[0] 1.0
kernel void exposure_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(exposure(0.25, 2.0), 0.0, 0.0, 0.0);
}

// WGSL vibrance3
// @test vibrance_test
// @expect vibrance_test[0] 0.6258 0.4958 0.3658
kernel void vibrance_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(vibrance(float3(0.6, 0.5, 0.4), 0.5), 0.0);
}

// "vibrance - selective saturation boost"
// [0] = WGSL results[0] = (muted_sat_change, saturated_sat_change, desaturated.r, desaturated.g)
// [1] = desaturated.rg moved to .xy (annotations check leading components), expected close to (0.833, 0.393)
// [2] = relational flags: x = muted_sat_change > saturated_sat_change, y = muted_sat_change > 1.0
// @test vibranceSelective 3
// @expect vibranceSelective[1] 0.833 0.393
// @expect vibranceSelective[2] 1 1
kernel void vibranceSelective(device float4* results [[buffer(0)]]) {
    float3 muted = float3(0.6, 0.5, 0.4);
    float3 muted_boosted = vibrance(muted, 0.5);
    float3 saturated = float3(1.0, 0.1, 0.0);
    float3 saturated_boosted = vibrance(saturated, 0.5);
    float3 desaturated = vibrance(float3(0.8, 0.4, 0.2), -0.5);
    float muted_sat_change = (muted_boosted.r - muted_boosted.b) / (muted.r - muted.b);
    float saturated_sat_change = (saturated_boosted.r - saturated_boosted.g) / (saturated.r - saturated.g);
    results[0] = float4(muted_sat_change, saturated_sat_change, desaturated.r, desaturated.g);
    results[1] = float4(desaturated.r, desaturated.g, 0.0, 0.0);
    results[2] = float4(muted_sat_change > saturated_sat_change ? 1.0 : 0.0, muted_sat_change > 1.0 ? 1.0 : 0.0, 0.0, 0.0);
}
