// Ported from test/wesl/math-rotate.test.ts (same inputs and expected values).
// WGSL mat*vec and MSL mat*vec are the same (column-major matrices, column vectors).
//
// Rotation direction: commit 9a3a036c ("Fixed and optimised rotate functions") made the GLSL and
// HLSL rotate2d/3d/3dX/3dZ/4d/4dX/4dZ counterclockwise (transposed the old matrices), but the MSL
// files were never updated and still build the old, transposed (clockwise) matrices. The WESL
// port only updated rotate2d and rotate3dX, so the WGSL expectations agree with GLSL for those two
// and with the old convention (= MSL) for rotate3d/3dZ/4d/4dX/4dZ. rotate3dY/4dY are the same everywhere.
#include <metal_stdlib>
using namespace metal;
#include "lygia/math/const.msl"
#include "lygia/math/rotate2d.msl"
#include "lygia/math/rotate3d.msl"
#include "lygia/math/rotate3dX.msl"
#include "lygia/math/rotate3dY.msl"
#include "lygia/math/rotate3dZ.msl"
#include "lygia/math/rotate4d.msl"
#include "lygia/math/rotate4dX.msl"
#include "lygia/math/rotate4dY.msl"
#include "lygia/math/rotate4dZ.msl"

// "rotate2d - 90 degree rotation"
// @test rotate2d
// @expect rotate2d[0] 0.0 1.0 0.0 0.0
kernel void rotate2d(device float4* results [[buffer(0)]]) {
    float2 r = rotate2d(HALF_PI) * float2(1.0, 0.0);
    results[0] = float4(r.x, r.y, 0.0, 0.0);
}

// "rotate3d - rotation around axis"
// Passes, but only because WESL rotate3d still has the old convention: GLSL gives (0, 1, 0) here.
// @test rotate3d
// @xfail rotate3d[0] WGSL kept the rotation direction from before upstream commit 9a3a036c; MSL now matches GLSL
// @expect rotate3d[0] 0.0 -1.0 0.0 0.0
kernel void rotate3d(device float4* results [[buffer(0)]]) {
    float3 axis = normalize(float3(0.0, 0.0, 1.0));
    float3 r = rotate3d(axis, HALF_PI) * float3(1.0, 0.0, 0.0);
    results[0] = float4(r, 0.0);
}

// @test rotate3dX
// @expect rotate3dX[0] 0.0 0.0 1.0
kernel void rotate3dX(device float4* results [[buffer(0)]]) {
    results[0] = float4(rotate3dX(HALF_PI) * float3(0.0, 1.0, 0.0), 0.0);
}

// @test rotate3dY
// @expect rotate3dY[0] 0.0 0.0 -1.0
kernel void rotate3dY(device float4* results [[buffer(0)]]) {
    results[0] = float4(rotate3dY(HALF_PI) * float3(1.0, 0.0, 0.0), 0.0);
}

// Passes as in WGSL; GLSL (counterclockwise) gives (0, 1, 0).
// @test rotate3dZ
// @xfail rotate3dZ[0] WGSL kept the rotation direction from before upstream commit 9a3a036c; MSL now matches GLSL
// @expect rotate3dZ[0] 0.0 -1.0 0.0
kernel void rotate3dZ(device float4* results [[buffer(0)]]) {
    results[0] = float4(rotate3dZ(HALF_PI) * float3(1.0, 0.0, 0.0), 0.0);
}

// "rotate4d - axis-angle rotation"
// Passes as in WGSL; GLSL (counterclockwise) gives (0, 1, 0, 1).
// @test rotate4d
// @xfail rotate4d[0] WGSL kept the rotation direction from before upstream commit 9a3a036c; MSL now matches GLSL
// @expect rotate4d[0] 0.0 -1.0 0.0 1.0
kernel void rotate4d(device float4* results [[buffer(0)]]) {
    float3 axis = normalize(float3(0.0, 0.0, 1.0));
    results[0] = rotate4d(axis, HALF_PI) * float4(1.0, 0.0, 0.0, 1.0);
}

// Passes as in WGSL; GLSL (counterclockwise) gives (0, 0, 1, 1).
// @test rotate4dX
// @xfail rotate4dX[0] WGSL kept the rotation direction from before upstream commit 9a3a036c; MSL now matches GLSL
// @expect rotate4dX[0] 0.0 0.0 -1.0 1.0
kernel void rotate4dX(device float4* results [[buffer(0)]]) {
    results[0] = rotate4dX(HALF_PI) * float4(0.0, 1.0, 0.0, 1.0);
}

// @test rotate4dY
// @expect rotate4dY[0] 0.0 0.0 -1.0 1.0
kernel void rotate4dY(device float4* results [[buffer(0)]]) {
    results[0] = rotate4dY(HALF_PI) * float4(1.0, 0.0, 0.0, 1.0);
}

// Passes as in WGSL; GLSL (counterclockwise) gives (0, 1, 0, 1).
// @test rotate4dZ
// @xfail rotate4dZ[0] WGSL kept the rotation direction from before upstream commit 9a3a036c; MSL now matches GLSL
// @expect rotate4dZ[0] 0.0 -1.0 0.0 1.0
kernel void rotate4dZ(device float4* results [[buffer(0)]]) {
    results[0] = rotate4dZ(HALF_PI) * float4(1.0, 0.0, 0.0, 1.0);
}
