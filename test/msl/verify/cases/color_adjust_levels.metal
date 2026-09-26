// Ported from test/wesl/color-adjust-levels.test.ts (same inputs and expected values).
// WGSL names encode the argument types; in MSL they're overloads:
//   WGSL levelsInputRange3/4[Float]   == MSL levelsInputRange(float3|float4, float3|float, float3|float)
//   WGSL levelsGamma3/4[Float]        == MSL levelsGamma(float3|float4, float3|float)
//   WGSL levelsOutputRange3/4[Float]  == MSL levelsOutputRange(float3|float4, float3|float, float3|float)
//   WGSL levels3/4[Float]             == MSL levels(float3|float4, iMin, g, iMax, oMin, oMax) (float3 or float args)
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/levels.msl"

// WGSL levelsInputRange3
// @test levelsInputRange3_test
// @expect levelsInputRange3_test[0] 0.1667 0.5 0.8333
kernel void levelsInputRange3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(levelsInputRange(float3(0.3, 0.5, 0.7), float3(0.2), float3(0.8)), 0.0);
}

// WGSL levelsGamma3; expected y is Math.SQRT1_2
// @test levelsGamma3_test
// @expect levelsGamma3_test[0] 0.5 0.70710678 0.866
kernel void levelsGamma3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(levelsGamma(float3(0.25, 0.5, 0.75), float3(2.0)), 0.0);
}

// WGSL levels3Float
// @test levels3Float_test
// @expect levels3Float_test[0] 0.4266 0.6657 0.8303
kernel void levels3Float_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(levels(float3(0.3, 0.5, 0.7), 0.2, 2.0, 0.8, 0.1, 0.9), 0.0);
}

// WGSL levels3
// @test levels3_test
// @expect levels3_test[0] 0.4742 0.6292 0.7481
kernel void levels3_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(levels(float3(0.4, 0.6, 0.8), float3(0.2), float3(2.0), float3(0.9), float3(0.1), float3(0.8)), 0.0);
}

// WGSL levels4
// @test levels4_test
// @expect levels4_test[0] 0.4742 0.6292 0.7481 0.75
kernel void levels4_test(device float4* results [[buffer(0)]]) {
    results[0] = levels(float4(0.4, 0.6, 0.8, 0.75), float3(0.2), float3(2.0), float3(0.9), float3(0.1), float3(0.8));
}

// WGSL levels4Float
// @test levels4Float_test
// @expect levels4Float_test[0] 0.58 0.8032 0.2 0.85
kernel void levels4Float_test(device float4* results [[buffer(0)]]) {
    results[0] = levels(float4(0.5, 0.7, 0.3, 0.85), 0.3, 1.5, 0.8, 0.2, 0.9);
}

// WGSL levelsGamma3Float
// @test levelsGamma3Float_test
// @expect levelsGamma3Float_test[0] 0.4 0.6 0.8
kernel void levelsGamma3Float_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(levelsGamma(float3(0.16, 0.36, 0.64), 2.0), 0.0);
}

// WGSL levelsGamma4
// @test levelsGamma4_test
// @expect levelsGamma4_test[0] 0.5 0.63 0.9086 0.9
kernel void levelsGamma4_test(device float4* results [[buffer(0)]]) {
    results[0] = levelsGamma(float4(0.25, 0.5, 0.75, 0.9), float3(2.0, 1.5, 3.0));
}

// WGSL levelsGamma4Float
// @test levelsGamma4Float_test
// @expect levelsGamma4Float_test[0] 0.3 0.5 0.7 0.85
kernel void levelsGamma4Float_test(device float4* results [[buffer(0)]]) {
    results[0] = levelsGamma(float4(0.09, 0.25, 0.49, 0.85), 2.0);
}

// WGSL levelsInputRange3Float
// @test levelsInputRange3Float_test
// @expect levelsInputRange3Float_test[0] 0.125 0.5 0.875
kernel void levelsInputRange3Float_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(levelsInputRange(float3(0.2, 0.5, 0.8), 0.1, 0.9), 0.0);
}

// WGSL levelsInputRange4
// @test levelsInputRange4_test
// @expect levelsInputRange4_test[0] 0.1667 0.4 0.8 0.75
kernel void levelsInputRange4_test(device float4* results [[buffer(0)]]) {
    results[0] = levelsInputRange(float4(0.3, 0.6, 0.9, 0.75), float3(0.2, 0.4, 0.5), float3(0.8, 0.9, 1.0));
}

// WGSL levelsInputRange4Float
// @test levelsInputRange4Float_test
// @expect levelsInputRange4Float_test[0] 0.0714 0.5 0.9286 0.95
kernel void levelsInputRange4Float_test(device float4* results [[buffer(0)]]) {
    results[0] = levelsInputRange(float4(0.15, 0.45, 0.75, 0.95), 0.1, 0.8);
}

// WGSL levelsOutputRange3Float
// @test levelsOutputRange3Float_test
// @expect levelsOutputRange3Float_test[0] 0.2 0.55 0.9
kernel void levelsOutputRange3Float_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(levelsOutputRange(float3(0.0, 0.5, 1.0), 0.2, 0.9), 0.0);
}

// WGSL levelsOutputRange4
// @test levelsOutputRange4_test
// @expect levelsOutputRange4_test[0] 0.275 0.55 0.825 0.8
kernel void levelsOutputRange4_test(device float4* results [[buffer(0)]]) {
    results[0] = levelsOutputRange(float4(0.25, 0.5, 0.75, 0.8), float3(0.1, 0.2, 0.3), float3(0.8, 0.9, 1.0));
}

// WGSL levelsOutputRange4Float
// @test levelsOutputRange4Float_test
// @expect levelsOutputRange4Float_test[0] 0.43 0.69 0.82 0.7
kernel void levelsOutputRange4Float_test(device float4* results [[buffer(0)]]) {
    results[0] = levelsOutputRange(float4(0.2, 0.6, 0.8, 0.7), 0.3, 0.95);
}
