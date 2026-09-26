// Ported from test/wesl/sprite.test.ts.
//
// Skipped: "sprite animation with Mega Man sprite sheet" samples a PNG texture and
// compares the render with an image snapshot, which this harness can't do. The module
// is included so this file at least checks that animation/spriteLoop.msl compiles, and
// the kernel checks the shader's non-texture part (ratio) on a square resolution.
#include <metal_stdlib>
using namespace metal;
#include "lygia/space/ratio.msl"
#include "lygia/animation/spriteLoop.msl"

// ratio(st, (128, 128)) is the identity for a square resolution
// @test ratio
// @expect ratio[0] 0.25 0.75
kernel void ratio(device float4* results [[buffer(0)]]) {
    results[0] = float4(ratio(float2(0.25, 0.75), float2(128.0)), 0.0, 0.0);
}
