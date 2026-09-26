// Ported from test/wesl/color-palette.test.ts (same inputs and expected values).
// hueShiftRYB4 rgba check, which the WGSL test runs with tolerance 0.15
// (its exact alpha check is in color_palette.metal).
//   hueShiftRYB4 -> hueShiftRYB(float4, float)
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/hueShiftRYB.msl"

// Note: MSL hueShiftRYB uses its angle like the WESL fix; GLSL still uses PI.
// hueShiftRYB.wesl:22 uses `a`. MSL gives (0, 0.66, 0.2); with `a` it gives (1, 1, 0).
// @eps 0.15
// @test hueShiftRYB4_test
// @expect hueShiftRYB4_test[0] 1.0 1.0 0.0 0.7
kernel void hueShiftRYB4_test(device float4* results [[buffer(0)]]) {
    results[0] = hueShiftRYB(float4(1.0, 0.0, 0.0, 0.7), 2.0944);
}
