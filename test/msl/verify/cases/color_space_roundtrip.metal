// Ported from test/wesl/color-space-roundtrip.test.ts (same inputs and expected values).
// rgb2xyY4 / xyY2rgb4 -> rgb2xyY(float4) / xyY2rgb(float4). The WGSL test uses tolerance 0.01.
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/space/rgb2xyY.msl"
#include "lygia/color/space/xyY2rgb.msl"

// @eps 0.01
// rgb2xyY4 -> xyY2rgb4 roundtrip
// Note: MSL rgb2xyz returns XYZ in 0-100 like the WESL fix (#271); GLSL still returns 0-1.
// (expect 0-100), so the MSL/GLSL roundtrip comes back 100x too small (0.009, 0.008, 0.007); WESL's
// rgb2xyz scales by 100 ("different from GLSL, see #271"), so only WESL roundtrips.
// @test xyY_roundtrip
// @expect xyY_roundtrip[0] 0.9 0.8 0.7 0.6
kernel void xyY_roundtrip(device float4* results [[buffer(0)]]) {
    float4 original = float4(0.9, 0.8, 0.7, 0.6);
    float4 xyY = rgb2xyY(original);
    results[0] = xyY2rgb(xyY);
}
