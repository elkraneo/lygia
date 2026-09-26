// Ported from test/wesl/lighting-diffuse.test.ts (same inputs and expected values).
// The WGSL test computes the expectations in JS (double precision) and compares with
// epsilon 1e-5; the values below are those JS expressions evaluated:
//   NoL = 1/sqrt(3); A = 1 + (1/1.13 + 0.5/1.33); B = 0.45/1.09
//   smooth = NoL, rough = NoL*A, retro = NoL*(A + B*(1 - NoL^2)/NoL)
// Inequality checks are evaluated in the kernel and written as 1/0 flags.
#include <metal_stdlib>
using namespace metal;
#include "lygia/lighting/diffuse/orenNayar.msl"

// @eps 0.00001

static inline float flag(bool b) { return b ? 1.0 : 0.0; }

// [1] flags: rough > smooth, retro > rough
// @test diffuseOrenNayar 2
// @expect diffuseOrenNayar[0] 0.5773502691896258 1.3053286843299154 1.5805580421280803
// @expect diffuseOrenNayar[1] 1 1
kernel void diffuseOrenNayar(device float4* results [[buffer(0)]]) {
    float3 L = normalize(float3(1.0, 1.0, 1.0));
    float3 N = float3(0.0, 0.0, 1.0);
    float3 V = float3(0.0, 0.0, 1.0);
    float NoV = dot(N, V);
    float NoL = dot(N, L);
    float smoothResult = diffuseOrenNayar(L, N, V, NoV, NoL, 0.0);
    float roughResult = diffuseOrenNayar(L, N, V, NoV, NoL, 1.0);
    float3 V2 = L;
    float NoV2 = dot(N, V2);
    float retroResult = diffuseOrenNayar(L, N, V2, NoV2, NoL, 1.0);
    results[0] = float4(smoothResult, roughResult, retroResult, 0.0);
    results[1] = float4(flag(roughResult > smoothResult), flag(retroResult > roughResult), 0.0, 0.0);
}
