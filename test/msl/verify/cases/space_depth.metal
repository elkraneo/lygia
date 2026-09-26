// Ported from test/wesl/space-depth.test.ts (same inputs and expected values).
// Configuration: perspective projection (CAMERA_ORTHOGRAPHIC_PROJECTION undefined) in both WESL and MSL.
#include <metal_stdlib>
using namespace metal;
#include "lygia/space/linearizeDepth.msl"
#include "lygia/space/depth2viewZ.msl"
#include "lygia/space/viewZ2depth.msl"

// results[0] = (linearizeDepth, "depth2viewZ perspective", "viewZ2depth perspective")
// @test depth
// @expect depth[0] 0.1998 -1.9802 0.5
kernel void depth(device float4* results [[buffer(0)]]) {
    results[0] = float4(linearizeDepth(0.5, 0.1, 100.0), depth2viewZ(0.5, 1.0, 100.0), viewZ2depth(-1.9802, 1.0, 100.0), 0.0);
}
