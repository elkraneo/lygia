// Ported from test/wesl/draw.test.ts.
//
// Both tests render a shader and compare it with an image snapshot, which this harness
// can't do: "stroke and strokeEdge - grid pattern" (draw-stroke) and "2D SDF shapes -
// 14 shape grid" (draw-shapes). Approximation: the kernels evaluate the same functions
// the shaders use (WGSL strokeEdge = MSL stroke(x, size, w, edge); WGSL starSDF1 = MSL
// starSDF(st, V)) at a few points and check the results are finite and in [0, 1].
// No pixel values are compared.
#include <metal_stdlib>
using namespace metal;
#include "lygia/draw/stroke.msl"
#include "lygia/draw/fill.msl"
#include "lygia/space/ratio.msl"
#include "lygia/space/rotate.msl"
#include "lygia/sdf/circleSDF.msl"
#include "lygia/sdf/crossSDF.msl"
#include "lygia/sdf/flowerSDF.msl"
#include "lygia/sdf/gearSDF.msl"
#include "lygia/sdf/heartSDF.msl"
#include "lygia/sdf/hexSDF.msl"
#include "lygia/sdf/polySDF.msl"
#include "lygia/sdf/rectSDF.msl"
#include "lygia/sdf/raysSDF.msl"
#include "lygia/sdf/rhombSDF.msl"
#include "lygia/sdf/spiralSDF.msl"
#include "lygia/sdf/starSDF.msl"
#include "lygia/sdf/triSDF.msl"
#include "lygia/sdf/vesicaSDF.msl"

// draw-stroke: (stroke, strokeEdge) for 16 cells, as in the shader's 4x4 grid
// @test stroke 16
// @range stroke 0 1
kernel void stroke(device float4* results [[buffer(0)]]) {
    for (uint i = 0; i < 16; i++) {
        float col = float(i % 4u);
        float dist = 0.05 + 0.03 * float(i);
        float r1 = stroke(dist, 0.3, 0.05 + (col / 4.0) * 0.15);
        float r2 = stroke(dist, 0.3, 0.1, 0.001 + (col / 4.0) * 0.05);
        results[i] = float4(r1, r2, 0.0, 0.0);
    }
}

// draw-shapes: fill(sdf, 0.5) for the 14 shapes at a few rotated cell points (time 0.2155)
// @test shapes 16
// @range shapes 0 1
kernel void shapes(device float4* results [[buffer(0)]]) {
    const float2 pts[4] = {float2(0.5, 0.5), float2(0.3, 0.6), float2(0.75, 0.2), float2(0.1, 0.9)};
    for (uint k = 0; k < 4; k++) {
        float2 st = rotate(ratio(pts[k], float2(512.0)), -0.2155 * 0.4);
        results[k * 4 + 0] = float4(fill(circleSDF(st), 0.5), fill(vesicaSDF(st, 0.25), 0.5), fill(rhombSDF(st), 0.5), fill(triSDF(st), 0.5));
        results[k * 4 + 1] = float4(fill(rectSDF(st, float2(1.0)), 0.5), fill(polySDF(st, 5), 0.5), fill(hexSDF(st), 0.5), fill(starSDF(st, 5), 0.5));
        results[k * 4 + 2] = float4(fill(flowerSDF(st, 5), 0.5), fill(crossSDF(st, 1.0), 0.5), fill(gearSDF(st, 10.0, 10), 0.5), fill(heartSDF(st), 0.5));
        results[k * 4 + 3] = float4(fill(raysSDF(st, 14), 0.5), fill(spiralSDF(st, 0.1), 0.5), 0.0, 0.0);
    }
}
