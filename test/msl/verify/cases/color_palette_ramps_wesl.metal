// Ported from test/wesl/shaders/color_palette_ramps.test.wesl (same inputs and expected values).
// The WESL test uses expectNearVec3 (relative 1e-3, absolute 1e-6); here the default
// absolute 0.0001 is used. That is stricter for most values; for fire(1.0) = 20.09 WESL
// allows 0.02, and the expected values are given to 4 decimals, which 0.0001 still accepts.
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/palette/fire.msl"
#include "lygia/color/palette/water.msl"

// @test fireCool
// @expect fireCool[0] 0.3679 0.0920 0.0230
kernel void fireCool(device float4* results [[buffer(0)]]) { results[0] = float4(fire(0.0), 0.0); }

// @test fireNeutral
// @expect fireNeutral[0] 1.0 0.25 0.0625
kernel void fireNeutral(device float4* results [[buffer(0)]]) { results[0] = float4(fire(0.25), 0.0); }

// @test fireWarm
// @expect fireWarm[0] 2.7183 0.6796 0.1699
kernel void fireWarm(device float4* results [[buffer(0)]]) { results[0] = float4(fire(0.5), 0.0); }

// @test fireHot
// @expect fireHot[0] 20.0855 5.0214 1.2553
kernel void fireHot(device float4* results [[buffer(0)]]) { results[0] = float4(fire(1.0), 0.0); }

// @test waterDeep
// @expect waterDeep[0] 0.0001 0.2401 0.4096
kernel void waterDeep(device float4* results [[buffer(0)]]) { results[0] = float4(water(0.0), 0.0); }

// @test waterMid
// @expect waterMid[0] 0.01 0.49 0.64
kernel void waterMid(device float4* results [[buffer(0)]]) { results[0] = float4(water(0.5), 0.0); }

// @test waterSurface
// @expect waterSurface[0] 1.0 1.0 1.0
kernel void waterSurface(device float4* results [[buffer(0)]]) { results[0] = float4(water(1.0), 0.0); }

// @test waterClamped
// @expect waterClamped[0] 1.0 1.0 1.0
kernel void waterClamped(device float4* results [[buffer(0)]]) { results[0] = float4(water(1.5), 0.0); }
