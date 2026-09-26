// Ported from test/wesl/color-palette.test.ts (same inputs and expected values).
// Checks with the default tolerance (0.0001). Checks the WGSL test runs with other tolerances are in
// color_palette_eps005.metal (0.05), color_palette_eps01.metal (0.1) and color_palette_eps015.metal (0.15).
// Name mapping (WGSL names encode the type; MSL uses overloads):
//   brightnessContrast3/4, exposure3/4, hueShiftRYB/4, whiteBalance3/4, vibrance4 -> same name, float3|float4
//   levelsOutputRange3 (color/levels/outputRange) -> levelsOutputRange(float3, float3, float3)
//   heatmap (color/palette/heatmap) -> heatmap(float)
// TARGET_MOBILE is not defined, so MSL whiteBalance uses the CIE/LMS path, the only one WESL has.
// Relational expects (toBeGreaterThan / toBeLessThan) are computed in-kernel as 1/0 flags and expected to be 1.
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/brightnessContrast.msl"
#include "lygia/color/exposure.msl"
#include "lygia/color/hueShiftRYB.msl"
#include "lygia/color/palette/heatmap.msl"
#include "lygia/color/whiteBalance.msl"
#include "lygia/color/saturationMatrix.msl"
#include "lygia/color/levels/outputRange.msl"
#include "lygia/color/vibrance.msl"

// brightnessContrast3
// @test brightnessContrast3_test
// @expect brightnessContrast3_test[0] 0.72 0.6 0.48
kernel void brightnessContrast3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(brightnessContrast(float3(0.6, 0.5, 0.4), 0.1, 1.2), 0.0);
}

// brightnessContrast4
// @test brightnessContrast4_test
// @expect brightnessContrast4_test[0] 0.72 0.6 0.48 0.8
kernel void brightnessContrast4_test(device float4* results [[buffer(0)]]) {
    results[0] = brightnessContrast(float4(0.6, 0.5, 0.4, 0.8), 0.1, 1.2);
}

// exposure3
// @test exposure3_test
// @expect exposure3_test[0] 1.0 1.0 1.0
kernel void exposure3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(exposure(float3(0.5, 0.5, 0.5), 1.0), 0.0);
}

// exposure4
// @test exposure4_test
// @expect exposure4_test[0] 1.0 1.0 1.0 0.7
kernel void exposure4_test(device float4* results [[buffer(0)]]) {
    results[0] = exposure(float4(0.5, 0.5, 0.5, 0.7), 1.0);
}

// hueShiftRYB
// Note: MSL hueShiftRYB uses its angle like the WESL fix; GLSL still uses PI.
// hueShiftRYB.wesl:22 uses `a`. MSL gives (0, 0.66, 0.2); with `a` it gives (1, 1, 0).
// @test hueShiftRYB_test
// @expect hueShiftRYB_test[0] 1.0 1.0 0.0
kernel void hueShiftRYB_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(hueShiftRYB(float3(1.0, 0.0, 0.0), 2.0944), 0.0);
}

// hueShiftRYB4: the "alpha exact" check (the rgba check with tolerance 0.15 is in color_palette_eps015.metal)
// @test hueShiftRYB4_alpha
// @expect hueShiftRYB4_alpha[0] 0.7
kernel void hueShiftRYB4_alpha(device float4* results [[buffer(0)]]) {
    float4 result = hueShiftRYB(float4(1.0, 0.0, 0.0, 0.7), 2.0944);
    results[0] = float4(result.a, 0.0, 0.0, 0.0);
}

// heatmap
// @test heatmap_test
// @expect heatmap_test[0] 0.4375 0.9919 0.4375
kernel void heatmap_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(heatmap(0.5), 0.0);
}

// whiteBalance3
// [0] = (warm.r, warm.b, cool.r, cool.b); [1] = flags (warm.r > 0.5, cool.b > warm.b, warm.r > cool.r)
// @test whiteBalance3_test 2
// @expect whiteBalance3_test[0] 0.5141 0.4585 0.4727 0.5919
// @expect whiteBalance3_test[1] 1 1 1
kernel void whiteBalance3_test(device float4* results [[buffer(0)]]) {
    float3 gray = float3(0.5, 0.5, 0.5);
    float3 warm = whiteBalance(gray, 0.2, 0.0);
    float3 cool = whiteBalance(gray, -0.2, 0.0);
    results[0] = float4(warm.r, warm.b, cool.r, cool.b);
    results[1] = float4(warm.r > 0.5, cool.b > warm.b, warm.r > cool.r, 0.0);
}

// whiteBalance4
// [0] = (tempShift.r, tempShift.b, tintMagenta.g, tintGreen.g)
// [1] = flags (tempShift.r > tempShift.b, green.g > magenta.g, magenta.g < 0.5, green.g > 0.5)
// @test whiteBalance4_test 2
// @expect whiteBalance4_test[0] 0.5141 0.4585 0.4872 0.5132
// @expect whiteBalance4_test[1] 1 1 1 1
kernel void whiteBalance4_test(device float4* results [[buffer(0)]]) {
    float4 gray = float4(0.5, 0.5, 0.5, 0.8);
    float4 tempShift = whiteBalance(gray, 0.2, 0.0);
    float4 tintMagenta = whiteBalance(gray, 0.0, 0.1);
    float4 tintGreen = whiteBalance(gray, 0.0, -0.1);
    results[0] = float4(tempShift.r, tempShift.b, tintMagenta.g, tintGreen.g);
    results[1] = float4(tempShift.r > tempShift.b, tintGreen.g > tintMagenta.g, tintMagenta.g < 0.5, tintGreen.g > 0.5);
}

// saturationMatrix: the relational checks (the values with tolerance 0.1 are in color_palette_eps01.metal)
// [0] = flags (result.r > 0.8, result.b < 0.3)
// @test saturationMatrix_flags
// @expect saturationMatrix_flags[0] 1 1
kernel void saturationMatrix_flags(device float4* results [[buffer(0)]]) {
    float4x4 mat = saturationMatrix(1.5);
    float4 result = mat * float4(float3(0.8, 0.5, 0.3), 1.0);
    results[0] = float4(result.x > 0.8, result.z < 0.3, 0.0, 0.0);
}

// levelsOutputRange3
// @test levelsOutputRange3_test
// @expect levelsOutputRange3_test[0] 0.5 0.5 0.5
kernel void levelsOutputRange3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(levelsOutputRange(float3(0.5), float3(0.2), float3(0.8)), 0.0);
}

// vibrance4
// @test vibrance4_test
// @expect vibrance4_test[0] 0.6258 0.4958 0.3658 0.8
kernel void vibrance4_test(device float4* results [[buffer(0)]]) {
    results[0] = vibrance(float4(0.6, 0.5, 0.4, 0.8), 0.5);
}
