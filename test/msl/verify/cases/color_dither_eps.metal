// Ported from test/wesl/color-dither.test.ts (same inputs and expected values).
// "ditherVlachos - all wrapper functions", the checks the WGSL test runs with tolerance 0.004.
// (The 0.07 check is in color_dither_eps07.metal, the alpha check in color_dither.metal.)
//   ditherVlachos3(c, xy) -> ditherVlachos(float3, xy);  ditherVlachos4(c, xy) -> ditherVlachos(float4, xy)
// Default precision is 256 in both WESL and MSL.
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/dither/vlachos.msl"

// @eps 0.004
// @test ditherVlachos_default 2
// @expect ditherVlachos_default[0] 0.5273 0.6211 0.4688 0.0
// @expect ditherVlachos_default[1] 0.5273 0.6211 0.4688
kernel void ditherVlachos_default(device float4* results [[buffer(0)]]) {
    float3 color3 = float3(0.53, 0.62, 0.47);
    float4 color4 = float4(0.53, 0.62, 0.47, 0.85);
    float2 xy = float2(2.0, 3.0);
    results[0] = float4(ditherVlachos(color3, xy), 0.0);
    results[1] = ditherVlachos(color4, xy);
}
