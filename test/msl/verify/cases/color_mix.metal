// Ported from test/wesl/color-mix.test.ts (same inputs and expected values).
// Name mapping (WGSL names encode the type; MSL uses overloads):
//   mixOklab / mixOklab4           -> mixOklab(float3|float4, ...)
//   mixSpectral / mixSpectral4     -> mixSpectral(float3|float4, ...)
//   R = mixSpectral_linear_to_reflectance(rgb) -> float R[38]; mixSpectral_linear_to_reflectance(rgb, R)
//   mixSpectral_reflectance_to_xyz(R)          -> same name
// MIXOKLAB_SRGB / MIXSPECTRAL_SRGB are not defined, matching WESL's defaults (linear RGB inputs).
// Relational expects (toBeGreaterThan / toBeLessThan) are computed in-kernel as 1/0 flags and expected to be 1.
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/mixOklab.msl"
#include "lygia/color/mixSpectral.msl"

// mixOklab
// @test mixOklab_test
// @expect mixOklab_test[0] 0.2637 0.0866 0.3628
kernel void mixOklab_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(mixOklab(float3(1.0, 0.0, 0.0), float3(0.0, 0.0, 1.0), 0.5), 0.0);
}

// mixOklab4
// @test mixOklab4_test
// @expect mixOklab4_test[0] 0.2637 0.0866 0.3628 0.6
kernel void mixOklab4_test(device float4* results [[buffer(0)]]) {
    results[0] = mixOklab(float4(1.0, 0.0, 0.0, 0.8), float4(0.0, 0.0, 1.0, 0.4), 0.5);
}

// mixSpectral
// [0] = (mixed.rgb, linearMix.r); [1] and [2] are the relational checks as flags:
//   [1] = (mixed.r > 0, mixed.r < 0.15, mixed.g > 0, mixed.g < 0.1)
//   [2] = (mixed.b > 0, mixed.b < 0.1, max(mixed) < 0.2, |mixed.r - linearMix.r| > 0.3)
// @test mixSpectral_test 3
// @expect mixSpectral_test[0] 0.0673 0.0093 0.0241 0.5
// @expect mixSpectral_test[1] 1 1 1 1
// @expect mixSpectral_test[2] 1 1 1 1
kernel void mixSpectral_test(device float4* results [[buffer(0)]]) {
    float3 red = float3(1.0, 0.0, 0.0);
    float3 blue = float3(0.0, 0.0, 1.0);
    float3 mixed = mixSpectral(red, blue, 0.5);
    float3 linearMix = mix(red, blue, 0.5);
    results[0] = float4(mixed, linearMix.r);
    results[1] = float4(mixed.r > 0.0, mixed.r < 0.15, mixed.g > 0.0, mixed.g < 0.1);
    results[2] = float4(mixed.b > 0.0, mixed.b < 0.1, max(mixed.r, max(mixed.g, mixed.b)) < 0.2,
                        abs(mixed.r - linearMix.r) > 0.3);
}

// mixSpectral4
// [1] = flags (result.g > result.r, result.g > result.b)
// @test mixSpectral4_test 2
// @expect mixSpectral4_test[0] 0.0782 1.0272 0.0596 0.7
// @expect mixSpectral4_test[1] 1 1
kernel void mixSpectral4_test(device float4* results [[buffer(0)]]) {
    float4 result = mixSpectral(float4(1.0, 1.0, 0.0, 0.9), float4(0.0, 1.0, 1.0, 0.5), 0.5);
    results[0] = result;
    results[1] = float4(result.g > result.r, result.g > result.b, 0.0, 0.0);
}

// mixSpectral_linear_to_reflectance
// [1] = flags (R[0] < 0.2, R[37] > 0.8, R[37] > R[0])
// @test mixSpectral_reflectance 2
// @expect mixSpectral_reflectance[0] 0.0315 0.0318 0.9855
// @expect mixSpectral_reflectance[1] 1 1 1
kernel void mixSpectral_reflectance(device float4* results [[buffer(0)]]) {
    float R[MIXSPECTRAL_SIZE];
    mixSpectral_linear_to_reflectance(float3(1.0, 0.0, 0.0), R);
    results[0] = float4(R[0], R[19], R[37], 1.0);
    results[1] = float4(R[0] < 0.2, R[37] > 0.8, R[37] > R[0], 0.0);
}

// mixSpectral_reflectance_to_xyz
// [1] = flags (x > 0, y > 0, z > 0, x > 0.5y); [2] = flags (x < 1.5y, z > 0.5y, z < 1.5y)
// @test mixSpectral_xyz 3
// @expect mixSpectral_xyz[0] 0.4751 0.5 0.5441
// @expect mixSpectral_xyz[1] 1 1 1 1
// @expect mixSpectral_xyz[2] 1 1 1
kernel void mixSpectral_xyz(device float4* results [[buffer(0)]]) {
    float R[MIXSPECTRAL_SIZE];
    mixSpectral_linear_to_reflectance(float3(0.5), R);
    float3 xyz = mixSpectral_reflectance_to_xyz(R);
    results[0] = float4(xyz, 1.0);
    results[1] = float4(xyz.x > 0.0, xyz.y > 0.0, xyz.z > 0.0, xyz.x > xyz.y * 0.5);
    results[2] = float4(xyz.x < xyz.y * 1.5, xyz.z > xyz.y * 0.5, xyz.z < xyz.y * 1.5, 0.0);
}
