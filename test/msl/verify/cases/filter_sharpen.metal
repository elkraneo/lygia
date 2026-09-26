// Ported from test/wesl/filter-sharpen.test.ts (same inputs and expected values).
//
// Skipped (need a texture and a fragment render compared with an image snapshot):
// sharpenAdaptive, sharpenAdaptive4, sharpenContrastAdaptive, sharpenFast, sharpenFast4
// "- visual" tests. Their modules are still included below, so this file at least
// checks that they compile.
//
// Name mapping: WGSL sharpendAdaptiveControl4(vec4f) -> MSL sharpendAdaptiveControl(float4).
// toBeCloseTo(x, 2) means |got - x| < 0.005 and toBeCloseTo(x, 3) |got - x| < 0.0005;
// those are evaluated in the kernel and written as 1/0 flags next to the values.
#include <metal_stdlib>
using namespace metal;
#include "lygia/filter/sharpen/adaptive.msl"
#include "lygia/filter/sharpen/contrastAdaptive.msl"
#include "lygia/filter/sharpen/fast.msl"

static inline float flag(bool b) { return b ? 1.0 : 0.0; }

// [0] = gray, orange, black; [1] flags: gray ~ 0.25 (2 digits), orange ~ 0.318 (2 digits), black ~ 0 (3 digits)
// @test sharpendAdaptiveControl4 2
// @expect sharpendAdaptiveControl4[1] 1 1 1
kernel void sharpendAdaptiveControl4(device float4* results [[buffer(0)]]) {
    float r1 = sharpendAdaptiveControl(float4(0.5, 0.5, 0.5, 1.0));
    float r2 = sharpendAdaptiveControl(float4(0.8, 0.5, 0.2, 1.0));
    float r3 = sharpendAdaptiveControl(float4(0.0, 0.0, 0.0, 1.0));
    results[0] = float4(r1, r2, r3, 0.0);
    results[1] = float4(flag(abs(r1 - 0.25) < 0.005), flag(abs(r2 - 0.318) < 0.005), flag(abs(r3) < 0.0005), 0.0);
}
