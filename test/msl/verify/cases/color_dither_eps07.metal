// Ported from test/wesl/color-dither.test.ts (same inputs and expected values).
// "ditherVlachos - all wrapper functions", the precision-16 check the WGSL test runs with tolerance 0.07.
//   ditherVlachos3Precision(c, xy, p) -> ditherVlachos(float3, xy, p)
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/dither/vlachos.msl"

// @eps 0.07
// @test ditherVlachos_precision
// @expect ditherVlachos_precision[0] 0.5 0.625 0.4375 0.0
kernel void ditherVlachos_precision(device float4* results [[buffer(0)]]) {
    float3 color3 = float3(0.53, 0.62, 0.47);
    float2 xy = float2(2.0, 3.0);
    results[0] = float4(ditherVlachos(color3, xy, 16), 0.0);
}
