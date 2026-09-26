// Ported from test/wesl/color-space-rgb.test.ts (same inputs and expected values).
// conditions: { HSV2RYB_FAST: true } -> #define HSV2RYB_FAST (same fast CMY-bias branch in WESL and MSL).
#include <metal_stdlib>
using namespace metal;
#define HSV2RYB_FAST
#include "lygia/color/space/hsv2ryb.msl"

// hsv2ryb - FAST mode
// @test hsv2ryb_fast
// @expect hsv2ryb_fast[0] 0.9 0.9 0.18
kernel void hsv2ryb_fast(device float4* results [[buffer(0)]]) {
    float3 hsv = float3(0.3333, 0.8, 0.9); // Greenish HSV
    results[0] = float4(hsv2ryb(hsv), 0.0);
}
