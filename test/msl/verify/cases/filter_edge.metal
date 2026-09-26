// Ported from test/wesl/filter-edge.test.ts.
//
// Skipped: "edgePrewitt - visual" samples a texture and compares the render with an
// image snapshot, which this harness can't do. The module is included so this file at
// least checks that filter/edge/prewitt.msl compiles (the kernel itself checks nothing).
#include <metal_stdlib>
using namespace metal;
#include "lygia/filter/edge/prewitt.msl"

// @test compiles
kernel void compiles(device float4* results [[buffer(0)]]) {
    results[0] = float4(0.0);
}
