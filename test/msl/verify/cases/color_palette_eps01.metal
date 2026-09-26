// Ported from test/wesl/color-palette.test.ts (same inputs and expected values).
// saturationMatrix values, which the WGSL test checks with tolerance 0.1
// (its relational checks are in color_palette.metal).
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/saturationMatrix.msl"

// @eps 0.1
// @test saturationMatrix_test
// @expect saturationMatrix_test[0] 0.95 0.53 0.17
kernel void saturationMatrix_test(device float4* results [[buffer(0)]]) {
    float4x4 mat = saturationMatrix(1.5);
    float4 result = mat * float4(float3(0.8, 0.5, 0.3), 1.0);
    results[0] = float4(result.xyz, 0.0);
}
