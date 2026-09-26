// Ported from test/wesl/lighting-specular.test.ts (same inputs and expected values).
// Name mapping (WGSL -> MSL): fresnel(vec3f, f32) -> fresnel(float3, float);
// fresnelF32 -> fresnel(float, float); fresnelFromVectors -> fresnel(float3, float3, float3);
// fresnelRoughness -> fresnel(float3, float, float). No TARGET_MOBILE / PLATFORM_RPI,
// as in the WGSL test (desktop branch: GGX with 1 - NoH^2, smithGGXCorrelated).
// Inequality checks and the toShininess checks with a custom epsilon (2.0, 0.1, 0.1,
// 0.01) are evaluated in the kernel and written as 1/0 flags next to the values.
#include <metal_stdlib>
using namespace metal;
#include "lygia/lighting/fresnel.msl"
#include "lygia/lighting/specular/cookTorrance.msl"
#include "lygia/lighting/toShininess.msl"

static inline float flag(bool b) { return b ? 1.0 : 0.0; }

// WGSL fresnel(vec3f(0.04), 1.0).x
// @test fresnelVec3
// @expect fresnelVec3[0] 0.04
kernel void fresnelVec3(device float4* results [[buffer(0)]]) {
    results[0] = float4(fresnel(float3(0.04), 1.0).x, 0.0, 0.0, 0.0);
}

// WGSL fresnelF32(0.04, 1.0)
// @test fresnelF32
// @expect fresnelF32[0] 0.04
kernel void fresnelF32(device float4* results [[buffer(0)]]) {
    results[0] = float4(fresnel(0.04, 1.0), 0.0, 0.0, 0.0);
}

// WGSL fresnelFromVectors(f0, normal, view).x
// @test fresnelFromVectors
// @expect fresnelFromVectors[0] 0.04
kernel void fresnelFromVectors(device float4* results [[buffer(0)]]) {
    results[0] = float4(fresnel(float3(0.04), float3(0.0, 0.0, 1.0), float3(0.0, 0.0, 1.0)).x, 0.0, 0.0, 0.0);
}

// WGSL fresnelRoughness; [1] flags: r1 > 0.4, r2 < 0.3, r1 > 1.5*r2, r3 > r0; [2] flags: r3 < r1, |r1 - r2| > 0.4
// @test fresnelRoughness 3
// @expect fresnelRoughness[0] 0.04 0.54782 0.07543 0.06687
// @expect fresnelRoughness[1] 1 1 1 1
// @expect fresnelRoughness[2] 1 1
kernel void fresnelRoughness(device float4* results [[buffer(0)]]) {
    float3 f0 = float3(0.04);
    float n = fresnel(f0, 1.0, 0.1).x;
    float gs = fresnel(f0, 0.1, 0.1).x;
    float gr = fresnel(f0, 0.1, 0.9).x;
    float ms = fresnel(f0, 0.5, 0.1).x;
    results[0] = float4(n, gs, gr, ms);
    results[1] = float4(flag(gs > 0.4), flag(gr < 0.3), flag(gs > gr * 1.5), flag(ms > n));
    results[2] = float4(flag(ms < gs), flag(abs(gs - gr) > 0.4), 0.0, 0.0);
}

// [1] flags: r0 > 0.1, r0 > r1, r2 < r0, smoothRatio > roughRatio
// @test specularCookTorrance 2
// @expect specularCookTorrance[0] 0.31831 0.00393 0.00918 0.00409
// @expect specularCookTorrance[1] 1 1 1 1
kernel void specularCookTorrance(device float4* results [[buffer(0)]]) {
    float3 specularColor = float3(0.04);
    float3 N = float3(0.0, 0.0, 1.0);
    float3 L1 = N, H1 = N;
    float perfectSmooth = specularCookTorrance(L1, N, H1, 1.0, 1.0, 1.0, 0.1, specularColor).x;
    float perfectRough = specularCookTorrance(L1, N, H1, 1.0, 1.0, 1.0, 0.9, specularColor).x;
    float3 L3 = normalize(float3(0.5, 0.0, 1.0));
    float3 V3 = float3(0.0, 0.0, 1.0);
    float3 H3 = normalize(L3 + V3);
    float NoV3 = dot(N, V3), NoL3 = dot(N, L3), NoH3 = dot(N, H3);
    float offSpecSmooth = specularCookTorrance(L3, N, H3, NoV3, NoL3, NoH3, 0.1, specularColor).x;
    float offSpecRough = specularCookTorrance(L3, N, H3, NoV3, NoL3, NoH3, 0.9, specularColor).x;
    results[0] = float4(perfectSmooth, perfectRough, offSpecSmooth, offSpecRough);
    float smoothRatio = perfectSmooth / (offSpecSmooth + 0.001);
    float roughRatio = perfectRough / (offSpecRough + 0.001);
    results[1] = float4(flag(perfectSmooth > 0.1), flag(perfectSmooth > perfectRough),
                        flag(offSpecSmooth < perfectSmooth), flag(smoothRatio > roughRatio));
}

// [0] = verySmooth, veryRough, midRough, metallic
// [1] flags: |r0 - 194.4| < 2.0, |r1 - 9.8| < 0.1, |r2 - 57.6| < 0.1, |r3 - 32.77| < 0.01
// [2] flags: r0 > 150, r1 < 15, r0 > 10*r1, r1 < r2 < r0
// [3] flags: 25 < r3 < 40, r0 < 250, r1 > 0
// @test toShininess 4
// @expect toShininess[1] 1 1 1 1
// @expect toShininess[2] 1 1 1 1
// @expect toShininess[3] 1 1 1
kernel void toShininess(device float4* results [[buffer(0)]]) {
    float verySmooth = toShininess(0.0, 0.0);
    float veryRough = toShininess(1.0, 0.0);
    float midRough = toShininess(0.5, 0.0);
    float metallic = toShininess(0.3, 1.0);
    results[0] = float4(verySmooth, veryRough, midRough, metallic);
    results[1] = float4(flag(abs(verySmooth - 194.4) < 2.0), flag(abs(veryRough - 9.8) < 0.1),
                        flag(abs(midRough - 57.6) < 0.1), flag(abs(metallic - 32.77) < 0.01));
    results[2] = float4(flag(verySmooth > 150.0), flag(veryRough < 15.0), flag(verySmooth > veryRough * 10.0),
                        flag(midRough > veryRough && midRough < verySmooth));
    results[3] = float4(flag(metallic > 25.0 && metallic < 40.0), flag(verySmooth < 250.0), flag(veryRough > 0.0), 0.0);
}
