// Ported from test/wesl/shaders/color_space_gamma.test.wesl (same inputs and expected values).
// The WESL test uses expectNear/expectNearVec3/Vec4 (relative 1e-3, absolute 1e-6); here the
// default absolute 0.0001 is used, which is stricter for these values.
// WGSL gamma2linear/gamma2linear3/gamma2linear4 (and linear2gamma*) == MSL overloads for float/float3/float4.
// WESL uses const GAMMA = 2.2; MSL defines GAMMA 2.2 by default (not TARGET_MOBILE/PLATFORM_*), so the configs match.
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/space/gamma2linear.msl"
#include "lygia/color/space/linear2gamma.msl"

// @test gamma2linearVec3
// @expect gamma2linearVec3[0] 0.2176 0.2176 0.2176
kernel void gamma2linearVec3(device float4* results [[buffer(0)]]) {
    results[0] = float4(gamma2linear(float3(0.5, 0.5, 0.5)), 0.0);
}

// @test linear2gammaVec3
// @expect linear2gammaVec3[0] 0.5325 0.5325 0.5325
kernel void linear2gammaVec3(device float4* results [[buffer(0)]]) {
    results[0] = float4(linear2gamma(float3(0.25, 0.25, 0.25)), 0.0);
}

// @test gamma2linearScalar
// @expect gamma2linearScalar[0] 0.2176
kernel void gamma2linearScalar(device float4* results [[buffer(0)]]) {
    results[0] = float4(gamma2linear(0.5), 0.0, 0.0, 0.0);
}

// @test gamma2linearVec4Alpha
// @expect gamma2linearVec4Alpha[0] 0.2176 0.2176 0.2176 0.7
kernel void gamma2linearVec4Alpha(device float4* results [[buffer(0)]]) {
    results[0] = gamma2linear(float4(0.5, 0.5, 0.5, 0.7));
}

// @test linear2gammaScalar
// @expect linear2gammaScalar[0] 0.5325
kernel void linear2gammaScalar(device float4* results [[buffer(0)]]) {
    results[0] = float4(linear2gamma(0.25), 0.0, 0.0, 0.0);
}

// @test linear2gammaVec4Alpha
// @expect linear2gammaVec4Alpha[0] 0.5325 0.5325 0.5325 0.4
kernel void linear2gammaVec4Alpha(device float4* results [[buffer(0)]]) {
    results[0] = linear2gamma(float4(0.25, 0.25, 0.25, 0.4));
}
