// Ported from test/wesl/color-space-xyz.test.ts (same inputs and expected values).
// The second half of the WGSL "rgb2xyz" test, run with conditions { CIE_D50: true },
// which needs a different #define than color_space_xyz.metal.
#include <metal_stdlib>
using namespace metal;
#define CIE_D50
#include "lygia/color/space/rgb2xyz.msl"

// Note: MSL rgb2xyz returns XYZ in 0-100 like the WESL fix (#271); GLSL still returns 0-1.
// @test rgb2xyz_d50
// @expect rgb2xyz_d50[0] 68.9945 71.0127 43.6206
kernel void rgb2xyz_d50(device float4* results [[buffer(0)]]) {
    results[0] = float4(rgb2xyz(float3(0.8, 0.7, 0.5)), 0.0);
}
