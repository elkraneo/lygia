// Ported from test/wesl/geometry-triangle.test.ts: the checks that call
// expectCloseTo(want, got, 2), i.e. with epsilon 2 (probably meant as "2 digits", but
// kept as written). The rest of the file is in geometry_triangle.metal.
//
// Note: at a tight tolerance the barycentric expectation (1/3, 1/3, 1/3) would not hold
// in any language: barycentric(a, b, c) for these points is (0.9014, 0.3169, -0.2183)
// (computed by hand from the formula, identical in GLSL, WGSL and MSL).
#include <metal_stdlib>
using namespace metal;
#include "lygia/geometry/triangle/triangle.msl"
#include "lygia/geometry/triangle/barycentric.msl"
#include "lygia/geometry/triangle/centroid.msl"
#include "lygia/geometry/triangle/normal.msl"

// @eps 2

// WGSL barycentric(a, b, c)
// @test barycentricPoints
// @expect barycentricPoints[0] 0.333 0.333 0.333
kernel void barycentricPoints(device float4* results [[buffer(0)]]) {
    results[0] = float4(barycentric(float3(2.0, 1.0, -0.5), float3(-1.0, 3.0, 0.5), float3(1.5, -0.5, 2.0)), 0.0);
}

// WGSL barycentric2(tri)
// @test barycentricTriangle
// @expect barycentricTriangle[0] 0.333 0.333 0.333
kernel void barycentricTriangle(device float4* results [[buffer(0)]]) {
    Triangle tri = Triangle{float3(2.0, 1.0, -0.5), float3(-1.0, 3.0, 0.5), float3(1.5, -0.5, 2.0)};
    results[0] = float4(barycentric(tri), 0.0);
}

// @test centroid
// @expect centroid[0] 1.0 1.333 0.667
kernel void centroid(device float4* results [[buffer(0)]]) {
    Triangle tri = Triangle{float3(1.0, 2.0, -1.0), float3(4.0, -1.0, 2.0), float3(-2.0, 3.0, 1.0)};
    results[0] = float4(centroid(tri), 0.0);
}

// @test normal
// @expect normal[0] -0.577 -0.577 0.577
kernel void normal(device float4* results [[buffer(0)]]) {
    Triangle tri = Triangle{float3(0.0), float3(1.0, 0.0, 1.0), float3(0.0, 1.0, 1.0)};
    results[0] = float4(normal(tri), 0.0);
}
