// Ported from test/wesl/geometry-triangle.test.ts (same inputs and expected values).
// Name mapping: WGSL barycentric(a,b,c) / barycentric2(tri) / barycentric3(tri, pos)
// are MSL barycentric(a,b,c) / barycentric(tri) / barycentric(tri, pos).
//
// The WGSL test's inequality and toBeCloseTo(x, 2) checks (|x - want| < 0.005) are
// evaluated in the kernel and written as 1/0 flags next to the values they check.
// Its expectCloseTo(..., 2) checks pass epsilon = 2, so they live in
// geometry_triangle_eps2.metal with @eps 2.
//
// barycentric3 on this branch divides by 2*area (the #321 fix); upstream WGSL divides
// by area. The WGSL checks on it are relative (ratios, < 0.01), so both pass them.
#include <metal_stdlib>
using namespace metal;
#include "lygia/geometry/triangle/triangle.msl"
#include "lygia/geometry/triangle/area.msl"
#include "lygia/geometry/triangle/barycentric.msl"
#include "lygia/geometry/triangle/normal.msl"

static inline float flag(bool b) { return b ? 1.0 : 0.0; }

// @test triangleStruct
// @expect triangleStruct[0] 3 5 4
kernel void triangleStruct(device float4* results [[buffer(0)]]) {
    Triangle tri = Triangle{float3(0.0), float3(3.0, 0.0, 0.0), float3(0.0, 4.0, 0.0)};
    results[0] = float4(length(tri.b - tri.a), length(tri.c - tri.b), length(tri.a - tri.c), 0.0);
}

// @test area
// @expect area[0] 6.5
kernel void area(device float4* results [[buffer(0)]]) {
    Triangle tri = Triangle{float3(0.0), float3(3.0, 0.0, 1.0), float3(0.0, 4.0, 1.0)};
    results[0] = float4(area(tri), 0.0, 0.0, 0.0);
}

// [0] = coords, sum; [1] = flags: sum toBeCloseTo(1, 2), reconstructed.x > -2, reconstructed.x < 3
// @test barycentricPoints 2
// @expect barycentricPoints[1] 1 1 1
kernel void barycentricPoints(device float4* results [[buffer(0)]]) {
    float3 a = float3(2.0, 1.0, -0.5), b = float3(-1.0, 3.0, 0.5), c = float3(1.5, -0.5, 2.0);
    float3 coords = barycentric(a, b, c);
    float sum = coords.x + coords.y + coords.z;
    float3 rec = coords.x * a + coords.y * b + coords.z * c;
    results[0] = float4(coords, sum);
    results[1] = float4(flag(abs(sum - 1.0) < 0.005), flag(rec.x > -2.0), flag(rec.x < 3.0), 1.0);
}

// WGSL barycentric2(tri); same checks as barycentricPoints
// @test barycentricTriangle 2
// @expect barycentricTriangle[1] 1 1 1
kernel void barycentricTriangle(device float4* results [[buffer(0)]]) {
    Triangle tri = Triangle{float3(2.0, 1.0, -0.5), float3(-1.0, 3.0, 0.5), float3(1.5, -0.5, 2.0)};
    float3 coords = barycentric(tri);
    float sum = coords.x + coords.y + coords.z;
    float3 rec = coords.x * tri.a + coords.y * tri.b + coords.z * tri.c;
    results[0] = float4(coords, sum);
    results[1] = float4(flag(abs(sum - 1.0) < 0.005), flag(rec.x > -2.0), flag(rec.x < 3.0), 1.0);
}

// WGSL barycentric3(tri, tri.a): r0 > r1, r0 > r2, r1 < 0.01, r2 < 0.01
// @test barycentricVertex 2
// @expect barycentricVertex[1] 1 1 1 1
kernel void barycentricVertex(device float4* results [[buffer(0)]]) {
    Triangle tri = Triangle{float3(0.0), float3(1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0)};
    float3 r = barycentric(tri, tri.a);
    results[0] = float4(r, 0.0);
    results[1] = float4(flag(r.x > r.y), flag(r.x > r.z), flag(r.y < 0.01), flag(r.z < 0.01));
}

// WGSL barycentric3(tri, midpoint of a-b): |r0 - r1| < 0.01, r2 < 0.01
// @test barycentricMidpoint 2
// @expect barycentricMidpoint[1] 1 1
kernel void barycentricMidpoint(device float4* results [[buffer(0)]]) {
    Triangle tri = Triangle{float3(0.0), float3(2.0, 0.0, 0.0), float3(0.0, 2.0, 0.0)};
    float3 r = barycentric(tri, float3(1.0, 0.0, 0.0));
    results[0] = float4(r, 0.0);
    results[1] = float4(flag(abs(r.x - r.y) < 0.01), flag(r.z < 0.01), 0.0, 0.0);
}

// normal: length toBeCloseTo(1, 2)
// @test normalLength 2
// @expect normalLength[1] 1
kernel void normalLength(device float4* results [[buffer(0)]]) {
    Triangle tri = Triangle{float3(0.0), float3(1.0, 0.0, 1.0), float3(0.0, 1.0, 1.0)};
    float3 n = normal(tri);
    results[0] = float4(n, length(n));
    results[1] = float4(flag(abs(length(n) - 1.0) < 0.005), 0.0, 0.0, 0.0);
}
