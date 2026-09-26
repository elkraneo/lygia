// Ported from test/wesl/color-palette.test.ts (same inputs and expected values).
// The checks the WGSL test runs with tolerance 0.05: paletteHue, hueDefault, hueShift4.
//   hue(x, ratio) / hueDefault(x) (color/palette/hue) -> hue(float, float) / hue(float)
//   hueShift4 -> hueShift(float4, float); TAU from math/const.msl
// HUESHIFT_AMOUNT is not defined, as in the WESL test.
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/palette/hue.msl"
#include "lygia/color/hueShift.msl"
#include "lygia/math/const.msl"

// @eps 0.05

// paletteHue
// @test paletteHue_test
// @expect paletteHue_test[0] 0.0 0.74 0.743
kernel void paletteHue_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(hue(0.5, 0.333), 0.0);
}

// hueDefault
// @test hueDefault_test
// @expect hueDefault_test[0] 0.0 0.74 0.743
kernel void hueDefault_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(hue(0.5), 0.0);
}

// hueShift4
// @test hueShift4_test
// @expect hueShift4_test[0] 0.0 1.0 0.0 0.85
kernel void hueShift4_test(device float4* results [[buffer(0)]]) {
    results[0] = hueShift(float4(1.0, 0.0, 0.0, 0.85), TAU * 0.3333);
}
