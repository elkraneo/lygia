// Ported from test/wesl/math.test.ts (same inputs and expected values).
// Name mapping (WGSL -> MSL):
//   saturate3, pow22, sum2/sum3, inside2, within2    -> the MSL overloads saturate/pow2/sum/inside/within
//   fmod2/fmod3/fmod4 (floored: x - y * floor(x / y)) -> MSL mod() from math/mod.msl (same formula;
//                                                      not Metal's builtin fmod, which truncates)
//   atan2Custom(y, x) = (atan2(y, x) + PI) % TAU     -> math/atan2.msl defines nothing (Metal's native
//                                                      atan2 is used instead) and documents
//                                                      mod(atan2(y, x) + PI, TAU) as the equivalent of
//                                                      the GLSL atan2, so that expression is used here.
//   bump2(vec2f, vec2f)                              -> MSL (like GLSL) only has float and float3
//                                                      overloads; checked with bump(float3, float3).xy.
//   mod2(&p, s)                                      -> mod2(thread float2&, float)
// Configuration: saturateMediump without TARGET_MOBILE is a pass-through in both WESL and MSL, so the
// v4 check (WGSL: 100000 or 65504) expects 100000.
// Inequality checks in the WGSL tests (atan2Custom in [0, TAU)) are implied by the exact values.
// Tolerances: the mod2 test checks p with toBeCloseTo(1.0, 1) (|diff| < 0.05); here the default
// 0.0001 is used (stricter).
// saturate, absi and saturateMediump are macros in MSL, so those kernels have other names.
#include <metal_stdlib>
using namespace metal;
#include "lygia/math/const.msl"
#include "lygia/math/saturate.msl"
#include "lygia/math/pow2.msl"
#include "lygia/math/pow3.msl"
#include "lygia/math/pow5.msl"
#include "lygia/math/pow7.msl"
#include "lygia/math/absi.msl"
#include "lygia/math/cubicMix.msl"
#include "lygia/math/smootherstep.msl"
#include "lygia/math/mod.msl"
#include "lygia/math/map.msl"
#include "lygia/math/mirror.msl"
#include "lygia/math/decimate.msl"
#include "lygia/math/taylorInvSqrt.msl"
#include "lygia/math/adaptiveThreshold.msl"
#include "lygia/math/atan2.msl"
#include "lygia/math/bump.msl"
#include "lygia/math/highPass.msl"
#include "lygia/math/inside.msl"
#include "lygia/math/mod2.msl"
#include "lygia/math/mod289.msl"
#include "lygia/math/powFast.msl"
#include "lygia/math/round.msl"
#include "lygia/math/saturateMediump.msl"
#include "lygia/math/sum.msl"
#include "lygia/math/within.msl"

// "saturate" and "saturate clamped upper": (saturate(-0.5), saturate(1.5))
// @test saturateScalar
// @expect saturateScalar[0] 0.0 1.0
kernel void saturateScalar(device float4* results [[buffer(0)]]) {
    results[0] = float4(saturate(-0.5), saturate(1.5), 0.0, 0.0);
}

// @test saturate3
// @expect saturate3[0] 0.0 0.5 1.0
kernel void saturate3(device float4* results [[buffer(0)]]) {
    results[0] = float4(saturate(float3(-0.5, 0.5, 1.5)), 0.0);
}

// (pow2(3), pow22(2, 3).xy, pow3(2)) and (pow5(2), pow7(2))
// @test pow 2
// @expect pow[0] 9.0 4.0 9.0 8.0
// @expect pow[1] 32.0 128.0
kernel void pow(device float4* results [[buffer(0)]]) {
    float2 p22 = pow2(float2(2.0, 3.0));
    results[0] = float4(pow2(3.0), p22.x, p22.y, pow3(2.0));
    results[1] = float4(pow5(2.0), pow7(2.0), 0.0, 0.0);
}

// "absi" and "absi negative" (results[1].x)
// @test absiInt 2
// @expect absiInt[0] 5.0 5.0 0.0 12.0
// @expect absiInt[1] 5.0
kernel void absiInt(device float4* results [[buffer(0)]]) {
    results[0] = float4(float(absi(5)), float(absi(-5)), float(absi(0)), float(absi(-12)));
    results[1] = float4(float(absi(-5)), 0.0, 0.0, 0.0);
}

// @test cubicMix
// @expect cubicMix[0] 0.0 0.15625 0.84375 1.0
kernel void cubicMix(device float4* results [[buffer(0)]]) {
    results[0] = float4(cubicMix(0.0, 1.0, 0.0), cubicMix(0.0, 1.0, 0.25), cubicMix(0.0, 1.0, 0.75), cubicMix(0.0, 1.0, 1.0));
}

// @test smootherstep
// @expect smootherstep[0] 0.0 0.10352 0.89648 1.0
kernel void smootherstep(device float4* results [[buffer(0)]]) {
    results[0] = float4(smootherstep(0.0, 1.0, 0.0), smootherstep(0.0, 1.0, 0.25), smootherstep(0.0, 1.0, 0.75), smootherstep(0.0, 1.0, 1.0));
}

// WGSL result[0..3] = fmod2(5, 7 / 3, 4).xy, fmod2(-5, -7 / 3, 4).xy
// @test fmod2
// @expect fmod2[0] 2.0 3.0 1.0 1.0
kernel void fmod2(device float4* results [[buffer(0)]]) {
    float2 r1 = mod(float2(5.0, 7.0), float2(3.0, 4.0));
    float2 r2 = mod(float2(-5.0, -7.0), float2(3.0, 4.0));
    results[0] = float4(r1, r2);
}

// @test fmod3
// @expect fmod3[0] 0.5 3.3 1.9
kernel void fmod3(device float4* results [[buffer(0)]]) {
    results[0] = float4(mod(float3(-5.5, 7.3, -2.1), float3(3.0, 4.0, 2.0)), 0.0);
}

// @test fmod4
// @expect fmod4[0] 1.0 2.0 0.0 0.0
kernel void fmod4(device float4* results [[buffer(0)]]) {
    results[0] = mod(float4(10.0, -10.0, 7.5, -7.5), float4(3.0, 3.0, 2.5, 2.5));
}

// "map - remap value between ranges"
// @test map
// @expect map[0] 50.0 150.0 0.0 0.0
kernel void map(device float4* results [[buffer(0)]]) {
    results[0] = float4(map(0.5, 0.0, 1.0, 0.0, 100.0), map(5.0, 0.0, 10.0, 100.0, 200.0), 0.0, 0.0);
}

// "mirror - triangle wave"
// @test mirror
// @expect mirror[0] 0.5 0.5 0.5 0.5
kernel void mirror(device float4* results [[buffer(0)]]) {
    results[0] = float4(mirror(0.5), mirror(1.5), mirror(2.5), mirror(3.5));
}

// "decimate - quantize value"
// @test decimate
// @expect decimate[0] 0.5
kernel void decimate(device float4* results [[buffer(0)]]) {
    results[0] = float4(decimate(0.567, 10.0), 0.0, 0.0, 0.0);
}

// @test taylorInvSqrt
// @expect taylorInvSqrt[0] 0.9391 -1.6221 1.5794 0.0854
kernel void taylorInvSqrt(device float4* results [[buffer(0)]]) {
    results[0] = float4(taylorInvSqrt(1.0), taylorInvSqrt(4.0), taylorInvSqrt(0.25), taylorInvSqrt(2.0));
}

// @test adaptiveThreshold
// @expect adaptiveThreshold[0] 1.0 0.0 0.0 0.0
kernel void adaptiveThreshold(device float4* results [[buffer(0)]]) {
    results[0] = float4(adaptiveThreshold(0.8, 0.5, 0.1), adaptiveThreshold(0.4, 0.5, 0.1), 0.0, 0.0);
}

// "atan2Custom" (3PI/2, PI, PI/2, 0) and "atan2Custom - additional angles" (results[1].x = 5PI/4)
// @test atan2Custom 2
// @expect atan2Custom[0] 4.71238898 3.14159265 1.57079633 0.0
// @expect atan2Custom[1] 3.92699082
// @range atan2Custom 0 6.2831852
kernel void atan2Custom(device float4* results [[buffer(0)]]) {
    results[0] = float4(mod(atan2(1.0, 0.0) + PI, TAU), mod(atan2(0.0, 1.0) + PI, TAU),
                        mod(atan2(-1.0, 0.0) + PI, TAU), mod(atan2(0.0, -1.0) + PI, TAU));
    results[1] = float4(mod(atan2(1.0, 1.0) + PI, TAU), 0.0, 0.0, 0.0);
}

// "bump" (results[0].xyz) and "bump2" (results[1].xy)
// @test bump 2
// @expect bump[0] 1.0 0.0 0.75
// @expect bump[1] 1.0 0.75
kernel void bump(device float4* results [[buffer(0)]]) {
    results[0] = float4(bump(0.0, 0.0), bump(1.0, 0.0), bump(0.5, 0.0), 0.0);
    float3 b2 = bump(float3(0.0, 0.5, 0.0), float3(0.0));
    results[1] = float4(b2.x, b2.y, 0.0, 0.0);
}

// @test highPass
// @expect highPass[0] 0.6 0.0
kernel void highPass(device float4* results [[buffer(0)]]) {
    results[0] = float4(highPass(0.8, 0.5), highPass(0.3, 0.5), 0.0, 0.0);
}

// "inside - scalar" (results[0].xyz) and "inside2" (results[1].xy)
// @test inside 2
// @expect inside[0] 1.0 0.0 0.0
// @expect inside[1] 1.0 0.0
kernel void inside(device float4* results [[buffer(0)]]) {
    results[0] = float4(float(inside(5.0, 0.0, 10.0)), float(inside(-1.0, 0.0, 10.0)), float(inside(11.0, 0.0, 10.0)), 0.0);
    results[1] = float4(float(inside(float2(5.0, 5.0), float2(0.0), float2(10.0))),
                        float(inside(float2(-1.0, 5.0), float2(0.0), float2(10.0))), 0.0, 0.0);
}

// "mod2 - mutates pointer": only p.xy is checked, as in the WGSL test
// @test mod2
// @expect mod2[0] 1.0 1.0
kernel void mod2(device float4* results [[buffer(0)]]) {
    float2 p = float2(7.0, 10.0);
    float2 c = mod2(p, 3.0);
    results[0] = float4(p.x, p.y, c.x, c.y);
}

// @test mod289
// @expect mod289[0] 11.0 0.0 100.0
kernel void mod289(device float4* results [[buffer(0)]]) {
    results[0] = float4(mod289(300.0), mod289(289.0), mod289(100.0), 0.0);
}

// @test powFast
// @expect powFast[0] 0.6667 0.9302 0.3077 1.0
kernel void powFast(device float4* results [[buffer(0)]]) {
    results[0] = float4(powFast(0.5, 0.5), powFast(0.8, 0.3), powFast(0.25, 0.75), powFast(1.0, 0.5));
}

// math/round.msl only documents Metal's builtin round, which rounds halfway cases away from zero like LYGIA's GLSL round.
// @test roundHalfAway
// @expect roundHalfAway[0] 2.0 3.0 -2.0 -3.0
kernel void roundHalfAway(device float4* results [[buffer(0)]]) {
    results[0] = float4(round(2.3), round(2.7), round(-2.3), round(-2.7));
}

// @test saturateMediumpDesktop
// @expect saturateMediumpDesktop[0] -0.5 0.5 1000.0 100000.0
kernel void saturateMediumpDesktop(device float4* results [[buffer(0)]]) {
    results[0] = float4(saturateMediump(-0.5), saturateMediump(0.5), saturateMediump(1000.0), saturateMediump(100000.0));
}

// @test sum2
// @expect sum2[0] 10.0 3.0 -5.0 0.75
kernel void sum2(device float4* results [[buffer(0)]]) {
    results[0] = float4(sum(float2(3.0, 7.0)), sum(float2(-5.0, 8.0)), sum(float2(-2.0, -3.0)), sum(float2(0.5, 0.25)));
}

// @test sum3
// @expect sum3[0] 15.0 3.0 -6.0 1.5
kernel void sum3(device float4* results [[buffer(0)]]) {
    results[0] = float4(sum(float3(3.0, 7.0, 5.0)), sum(float3(-2.0, 6.0, -1.0)), sum(float3(-1.0, -2.0, -3.0)), sum(float3(0.25, 0.5, 0.75)));
}

// "within - scalar" (results[0].xyz) and "within2" (results[1].xy)
// @test within 2
// @expect within[0] 1.0 0.0 0.0
// @expect within[1] 1.0 0.0
kernel void within(device float4* results [[buffer(0)]]) {
    results[0] = float4(within(5.0, 0.0, 10.0), within(-1.0, 0.0, 10.0), within(11.0, 0.0, 10.0), 0.0);
    results[1] = float4(within(float2(5.0, 5.0), float2(0.0), float2(10.0)), within(float2(-1.0, 5.0), float2(0.0), float2(10.0)), 0.0, 0.0);
}
